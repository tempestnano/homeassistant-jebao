"""Data update coordinator for Jebao."""
from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from datetime import datetime, timedelta
import logging
from typing import Optional

from jebao import JebaoError, MDP20000Device, PumpStatus, discover_devices

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_HOST
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator, UpdateFailed
from homeassistant.util import dt as dt_util

from .const import CONF_PASSCODE, DEFAULT_SCAN_INTERVAL, DOMAIN

_LOGGER = logging.getLogger(__name__)

# The pumps' Wi-Fi drops for a minute or so at a time. Keep showing the last status
# through that; commands sent meanwhile still fail, so nothing is silently lost.
OFFLINE_GRACE = timedelta(seconds=90)

# HA logs only the first failed poll, which hides a pump that stays unreachable
OFFLINE_LOG_INTERVAL = timedelta(minutes=15)


class JebaoDataUpdateCoordinator(DataUpdateCoordinator[PumpStatus]):
    """Polls one pump; data is its decoded status (every datapoint)."""

    def __init__(
        self,
        hass: HomeAssistant,
        device: MDP20000Device,
        entry: ConfigEntry,
        device_id: str,
        scan_interval: int = DEFAULT_SCAN_INTERVAL,
    ) -> None:
        """Initialize coordinator."""
        self.device = device
        self.entry = entry
        self.device_id = device_id
        self._discovery_attempted = False
        self._failing_since: Optional[datetime] = None
        self._next_offline_log: Optional[datetime] = None
        # Polls and commands share one session; keep them from interleaving
        self._lock = asyncio.Lock()

        super().__init__(
            hass,
            _LOGGER,
            name=DOMAIN,
            update_interval=timedelta(seconds=scan_interval),
        )

    async def _async_update_data(self) -> PumpStatus:
        """Fetch data from device."""
        async with self._lock:
            try:
                await self._ensure_connected()
                await self.device.update()
            except JebaoError as err:
                # Start the next poll from a fresh session rather than a half-dead one
                await self.device.disconnect()
                return self._poll_failed(err)
        self._save_passcode()
        self._poll_succeeded()

        status = self.device.status
        if status is None:
            raise UpdateFailed("Pump sent a status the integration couldn't decode")
        return status

    def _poll_failed(self, err: JebaoError) -> PumpStatus:
        """Ride out a short drop; after that, report the pump as offline."""
        now = dt_util.utcnow()
        if self._failing_since is None:
            self._failing_since = now
            self._next_offline_log = now + OFFLINE_LOG_INTERVAL
        if self.data is not None and now - self._failing_since < OFFLINE_GRACE:
            _LOGGER.debug("Poll of %s failed, keeping last status: %s", self.entry.title, err)
            return self.data
        if now >= self._next_offline_log:
            self._next_offline_log = now + OFFLINE_LOG_INTERVAL
            _LOGGER.warning(
                "%s has been offline since %s: %s",
                self.entry.title,
                dt_util.as_local(self._failing_since).isoformat(timespec="seconds"),
                err,
            )
        raise UpdateFailed(f"Error communicating with device: {err}") from err

    def _poll_succeeded(self) -> None:
        if self._failing_since is not None and self.last_update_success is False:
            _LOGGER.info(
                "%s is back after %s offline",
                self.entry.title,
                dt_util.utcnow() - self._failing_since,
            )
        self._failing_since = None

    async def async_command(
        self, command: Callable[[MDP20000Device], Awaitable[None]]
    ) -> None:
        """Run a control command and publish the pump's new state.

        Raises:
            HomeAssistantError: The pump didn't take the command
        """
        async with self._lock:
            try:
                await self._ensure_connected()
                await command(self.device)
            except ValueError as err:
                raise HomeAssistantError(str(err)) from err
            except JebaoError as err:
                await self.device.disconnect()
                raise HomeAssistantError(f"Pump command failed: {err}") from err
            # Read the result straight away (the new value reads back at once).
            # Publishing also restarts the poll timer, so publishing the status from
            # before the write would hide the change for a whole interval.
            try:
                await self.device.update()
                status = self.device.status
            except JebaoError as err:
                # The pump took the command, so don't report it as failed
                _LOGGER.debug("Read-back from %s failed: %s", self.entry.title, err)
                await self.device.disconnect()
                status = None
        if status is not None:
            self._poll_succeeded()
            self.async_set_updated_data(status)
        else:
            await self.async_request_refresh()

    async def async_sync_clock(self) -> None:
        """Set the pump's clock (which its schedule runs on) to HA's local time."""
        now = dt_util.now().replace(tzinfo=None)
        await self.async_command(lambda device: device.set_clock(now))

    def _save_passcode(self) -> None:
        """Keep the pump's passcode in the entry, for logging in after a restart.

        Pumps only hand it out for a while after power-up (GAgent's binding
        window); after that, a client that doesn't know it can't log in until
        the pump is power-cycled. Login with it works any time.
        """
        passcode = self.device.passcode
        if passcode and self.entry.data.get(CONF_PASSCODE) != passcode:
            self.hass.config_entries.async_update_entry(
                self.entry, data={**self.entry.data, CONF_PASSCODE: passcode}
            )

    async def _ensure_connected(self) -> None:
        if self.device.is_connected:
            return
        _LOGGER.debug("Connecting to %s", self.device.host)
        try:
            await self.device.connect(timeout=5.0)
            self._discovery_attempted = False
        except JebaoError as err:
            # Try discovery to find device with new IP
            new_ip = await self._try_discovery_recovery()
            if not new_ip:
                raise
            _LOGGER.debug("Reconnect to %s failed (%s), trying %s", self.device.host, err, new_ip)
            await self._reconnect_with_new_ip(new_ip)

    async def _try_discovery_recovery(self) -> Optional[str]:
        """Try to find device via discovery if IP changed.

        Returns:
            New IP address if found, None otherwise
        """
        if self._discovery_attempted:
            _LOGGER.debug("Discovery already attempted this cycle, skipping")
            return None

        self._discovery_attempted = True
        current_ip = self.entry.data[CONF_HOST]

        _LOGGER.info("Attempting discovery to find device %s (current IP: %s)", self.device_id, current_ip)

        try:
            devices = await discover_devices(timeout=10.0)

            # Find our device by device_id
            for device in devices:
                if device.device_id == self.device_id:
                    if device.ip_address != current_ip:
                        _LOGGER.warning(
                            "Device %s found at new IP: %s (was: %s)",
                            self.device_id,
                            device.ip_address,
                            current_ip
                        )
                        return device.ip_address
                    _LOGGER.info("Device found at same IP %s", current_ip)
                    return current_ip

            _LOGGER.error("Device %s not found in discovery", self.device_id)
            return None

        except Exception as err:  # pylint: disable=broad-except
            _LOGGER.error("Discovery failed: %s", err)
            return None

    async def _reconnect_with_new_ip(self, new_ip: str) -> None:
        """Update config entry and reconnect with new IP.

        Args:
            new_ip: New IP address for the device

        Raises:
            JebaoError: If reconnection fails
        """
        current_ip = self.entry.data[CONF_HOST]

        # Update config entry with new IP
        _LOGGER.info("Updating config entry IP from %s to %s", current_ip, new_ip)
        new_data = dict(self.entry.data)
        new_data[CONF_HOST] = new_ip
        self.hass.config_entries.async_update_entry(self.entry, data=new_data)

        # Update device instance with new IP
        self.device.host = new_ip
        self.device._protocol.host = new_ip  # pylint: disable=protected-access

        await self.device.disconnect()
        await self.device.connect(timeout=5.0)
        _LOGGER.info("Successfully reconnected to device at new IP %s", new_ip)
        self._discovery_attempted = False
