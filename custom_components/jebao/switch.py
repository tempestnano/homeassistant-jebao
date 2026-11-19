"""Switch platform for Jebao: the pump's own schedule, and the cycle."""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from jebao import parse_slots
import voluptuous as vol

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, callback
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import config_validation as cv, entity_platform
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.components.switch import SwitchEntity
from homeassistant.helpers.restore_state import RestoreEntity
from homeassistant.util import dt as dt_util

from .const import ATTR_ENABLE, ATTR_SLOTS, CONF_SYNC_CLOCK, DEFAULT_SYNC_CLOCK, DOMAIN, SERVICE_SET_SCHEDULE
from .coordinator import JebaoDataUpdateCoordinator
from .cycle import MAX_SPEED, MIN_SPEED, POWER, SCHEDULE, SPEED, CycleSettings, PumpCycle
from .entity import JebaoEntity

_LOGGER = logging.getLogger(__name__)

SERVICE_SET_CYCLE = "set_cycle"
WRITE_ATTEMPTS = 3  # per cycle write (see _CoordinatorPump.async_cycle_write)
WRITE_RETRY_DELAY = 2.0  # seconds
_SPEED = vol.All(vol.Coerce(int), vol.Range(min=MIN_SPEED, max=MAX_SPEED))
SET_CYCLE_SCHEMA = {
    vol.Optional("normal_speed"): _SPEED,
    vol.Optional("run_minutes"): vol.All(vol.Coerce(int), vol.Range(min=1, max=1440)),
    vol.Optional("pause_speed"): vol.Any(0, _SPEED),
    vol.Optional("pause_seconds"): vol.All(vol.Coerce(int), vol.Range(min=0, max=3600)),
    vol.Optional("boost_speed"): _SPEED,
    vol.Optional("boost_seconds"): vol.All(vol.Coerce(int), vol.Range(min=1, max=3600)),
    vol.Optional(ATTR_ENABLE): cv.boolean,
}

SLOT_SCHEMA = vol.Schema(
    {
        vol.Required("start"): cv.string,
        vol.Required("end"): cv.string,
        vol.Optional("mode", default="run"): vol.In(["run", "feed", "off"]),
        vol.Optional("speed"): vol.All(vol.Coerce(int), vol.Range(min=30, max=100)),
    }
)
SET_SCHEDULE_SCHEMA = {
    vol.Required(ATTR_SLOTS): vol.All(cv.ensure_list, [SLOT_SCHEMA], vol.Length(max=48)),
    vol.Optional(ATTR_ENABLE): cv.boolean,
}


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up the schedule switch and the set_schedule service."""
    coordinator: JebaoDataUpdateCoordinator = hass.data[DOMAIN][entry.entry_id]["coordinator"]
    async_add_entities([JebaoScheduleSwitch(coordinator), JebaoCycleSwitch(coordinator)])

    platform = entity_platform.async_get_current_platform()
    platform.async_register_entity_service(
        SERVICE_SET_SCHEDULE, SET_SCHEDULE_SCHEMA, "async_set_schedule"
    )
    platform.async_register_entity_service(
        SERVICE_SET_CYCLE, SET_CYCLE_SCHEMA, "async_set_cycle"
    )


class JebaoScheduleSwitch(JebaoEntity, SwitchEntity):
    """Whether the pump follows its own daily schedule (the app's "Program")."""

    _attr_translation_key = "schedule"
    _attr_icon = "mdi:calendar-clock"

    def __init__(self, coordinator: JebaoDataUpdateCoordinator) -> None:
        """Initialize switch."""
        super().__init__(coordinator, "schedule")

    @property
    def is_on(self) -> bool:
        """Return true if the schedule is running."""
        return self.coordinator.data.timer_on

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """The schedule itself, in the form set_schedule takes."""
        return {ATTR_SLOTS: [slot.as_dict() for slot in self.coordinator.data.schedule_slots]}

    async def async_turn_on(self, **kwargs: Any) -> None:
        """Start the schedule."""
        await self.coordinator.async_set_schedule_enabled(True)

    async def async_turn_off(self, **kwargs: Any) -> None:
        """Stop the schedule; the pump keeps its current speed."""
        await self.coordinator.async_set_schedule_enabled(False)

    async def async_set_schedule(self, slots: list[dict], enable: bool | None = None) -> None:
        """Replace the pump's schedule (the set_schedule service)."""
        try:
            parsed = parse_slots(
                {**slot, "speed": slot.get("speed", 100)} if slot.get("mode", "run") == "run" else slot
                for slot in slots
            )
        except (KeyError, ValueError) as err:
            raise HomeAssistantError(f"Invalid schedule: {err}") from err
        self.coordinator.cancel_schedule_resume()
        if enable or (enable is None and self.coordinator.data.timer_on):
            await self._async_sync_clock_if_enabled()
        now = dt_util.now().replace(tzinfo=None)
        await self.coordinator.async_command(
            lambda device: device.set_schedule(parsed, enabled=enable, now=now),
            takes_over=bool(enable),
        )

    async def async_set_cycle(self, **kwargs: Any) -> None:
        """Only the Cycle switch has a cycle."""
        raise HomeAssistantError("Target the pump's Cycle switch")

    async def _async_sync_clock_if_enabled(self) -> None:
        # The schedule runs on the pump's clock, so set it before relying on it
        if self.coordinator.entry.options.get(CONF_SYNC_CLOCK, DEFAULT_SYNC_CLOCK):
            await self.coordinator.async_sync_clock()


class _CoordinatorPump:
    """The pump as the cycle sees it (cycle.CyclePump), through the coordinator."""

    def __init__(self, coordinator: JebaoDataUpdateCoordinator) -> None:
        self._coordinator = coordinator

    @property
    def available(self) -> bool:
        return self._coordinator.last_update_success and self._coordinator.data is not None

    def get_attribute(self, name: str) -> Any:
        data = self._coordinator.data
        return None if data is None else getattr(data, name)

    async def async_cycle_write(self, name: str, value: Any) -> None:
        if name == POWER:
            command = (lambda d: d.turn_on()) if value else (lambda d: d.turn_off())
        elif name == SPEED:
            command = lambda d: d.set_speed(int(value))  # noqa: E731
        elif name == SCHEDULE and not value:
            command = lambda d: d.set_schedule_enabled(False)  # noqa: E731
        else:
            raise ValueError(f"The cycle doesn't set {name}={value}")
        # Each write opens a new session, and the pump's Wi-Fi module often doesn't
        # answer for some seconds after the motor starts. These writes set absolute
        # values, so sending one again is harmless: try a few times before the
        # cycle gives up and starts over.
        for attempt in range(1, WRITE_ATTEMPTS + 1):
            try:
                await self._coordinator.async_command(command)
                return
            except HomeAssistantError:
                if attempt == WRITE_ATTEMPTS:
                    raise
                await asyncio.sleep(WRITE_RETRY_DELAY)

    async def async_refresh_status(self) -> None:
        await self._coordinator.async_refresh_status()


class JebaoCycleSwitch(JebaoEntity, SwitchEntity, RestoreEntity):
    """Runs the pump in a repeating run, pause and boost cycle (see cycle.py).

    Home Assistant times it, so it stays available while the pump is offline (the
    cycle waits for it), and its settings and on/off survive a restart.
    """

    _attr_translation_key = "cycle"
    _attr_icon = "mdi:sync"

    def __init__(self, coordinator: JebaoDataUpdateCoordinator) -> None:
        """Initialize switch."""
        super().__init__(coordinator, "cycle")
        self._cycle = PumpCycle(_CoordinatorPump(coordinator), self._cycle_changed)

    @property
    def available(self) -> bool:
        """Always: the cycle waits for the pump by itself."""
        return True

    @property
    def is_on(self) -> bool:
        """Whether the cycle is running."""
        return self._cycle.running

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """The settings, and what the cycle is doing until when."""
        ends = self._cycle.phase_ends
        return {
            **self._cycle.settings.as_dict(),
            "phase": str(self._cycle.phase),
            "phase_ends": ends.isoformat() if ends else None,
        }

    async def async_turn_on(self, **kwargs: Any) -> None:
        """Start the cycle."""
        self._cycle.start()
        self.async_write_ha_state()

    async def async_turn_off(self, **kwargs: Any) -> None:
        """Stop the cycle, back at the normal speed."""
        await self._cycle.stop()
        self.async_write_ha_state()

    async def async_set_cycle(self, enable: bool | None = None, **changes: Any) -> None:
        """Change the settings (the set_cycle service); a running cycle starts over."""
        try:
            settings = self._cycle.settings.replace(changes)
        except ValueError as err:
            raise HomeAssistantError(f"Invalid cycle: {err}") from err
        if enable or (enable is None and self._cycle.running):
            self._cycle.start(settings)
        else:
            self._cycle.settings = settings
            if enable is False:
                await self._cycle.stop()
        self.async_write_ha_state()

    async def async_set_schedule(self, slots: list[dict], enable: bool | None = None) -> None:
        """The schedule is on the Schedule switch."""
        raise HomeAssistantError("Target the pump's Schedule switch")

    async def async_added_to_hass(self) -> None:
        """Take over the cycle from before a restart."""
        await super().async_added_to_hass()
        self.coordinator.cycle = self._cycle
        last = await self.async_get_last_state()
        if last is None:
            return
        try:
            self._cycle.settings = CycleSettings.from_dict(last.attributes)
        except (TypeError, ValueError):
            _LOGGER.warning("Ignoring the saved cycle settings of %s", self.entity_id)
        if last.state == "on":
            self._cycle.start()

    async def async_will_remove_from_hass(self) -> None:
        """Stop the cycle where it is."""
        await super().async_will_remove_from_hass()
        self._cycle.close()
        if self.coordinator.cycle is self._cycle:
            self.coordinator.cycle = None

    @callback
    def _cycle_changed(self) -> None:
        if self.hass is not None and self.entity_id:
            self.async_write_ha_state()
