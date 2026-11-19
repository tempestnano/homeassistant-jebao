"""A repeating run, pause and boost cycle for return pumps, timed by Home Assistant.

The pump's own schedule can't do this: it has minute resolution and 48 slots a day.
The cycle runs at the normal speed for a while, pauses (power off, or a slow speed)
for some seconds, boosts for some seconds, and repeats.

What running it from automations taught:
- Setting a speed straight after power on can leave the pump off, so wait for the
  pump to report power on, then a little longer, and check power again afterwards.
- A pump on weak Wi-Fi drops out often. Writes then fail; the pump keeps running at
  whatever it was last told. Wait for it to come back, return it to normal, and
  start the run period again.
- Don't write while it's feeding: the feed runs on the pump's own timer, and the
  cycle carries on when the feed is over.

This module doesn't know about Home Assistant: the pump is anything with the
CyclePump methods, so the cycle can be tested on its own. Ported from the
jebao_aqua integration, where it has run a pond skimmer since 2026-11-12.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import asdict, dataclass, fields
from datetime import UTC, datetime, timedelta
from enum import StrEnum
import logging
from typing import Any, Protocol

_LOGGER = logging.getLogger(__name__)

POWER = "switch_on"
SPEED = "motor_speed"
FEEDING = "feed_switch"
SCHEDULE = "timer_on"

MIN_SPEED = 30
MAX_SPEED = 100

# Datapoints a person sets that take the pump out of the cycle's hands
TAKES_OVER = (POWER, SPEED, SCHEDULE)


class CyclePump(Protocol):
    """What the cycle needs from a pump."""

    @property
    def available(self) -> bool:
        """Whether it's connected."""

    def get_attribute(self, name: str) -> Any:
        """The last value the pump reported (a PumpStatus field)."""

    async def async_cycle_write(self, name: str, value: Any) -> None:
        """Set the power, speed or schedule. Raises when it fails."""

    async def async_refresh_status(self) -> None:
        """Read the pump's status now, while waiting on it. Raises when it fails."""


@dataclass(frozen=True)
class CycleSettings:
    """How the cycle runs. A pause speed of 0 turns the power off."""

    normal_speed: int = 60
    run_minutes: int = 10
    pause_speed: int = 0
    pause_seconds: int = 10
    boost_speed: int = 100
    boost_seconds: int = 30

    def __post_init__(self) -> None:
        """Refuse settings the pump can't run."""
        for name in ("normal_speed", "boost_speed"):
            if not MIN_SPEED <= getattr(self, name) <= MAX_SPEED:
                raise ValueError(f"{name} must be {MIN_SPEED}-{MAX_SPEED}")
        if self.pause_speed and not MIN_SPEED <= self.pause_speed <= MAX_SPEED:
            raise ValueError(f"pause_speed must be 0 (off) or {MIN_SPEED}-{MAX_SPEED}")
        if not 1 <= self.run_minutes <= 24 * 60:
            raise ValueError("run_minutes must be 1-1440")
        if not 0 <= self.pause_seconds <= 3600:
            raise ValueError("pause_seconds must be 0-3600")
        if not 1 <= self.boost_seconds <= 3600:
            raise ValueError("boost_seconds must be 1-3600")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CycleSettings:
        """Settings from a dict, keeping the defaults for what's missing."""
        names = {field.name for field in fields(cls)}
        return cls(**{key: int(value) for key, value in data.items() if key in names})

    def replace(self, changes: Mapping[str, Any]) -> CycleSettings:
        """These settings with some changed."""
        return self.from_dict({**asdict(self), **changes})

    def as_dict(self) -> dict[str, int]:
        """The settings as a dict."""
        return asdict(self)


class Phase(StrEnum):
    """What the cycle is doing."""

    OFF = "off"
    WAITING = "waiting"  # For the pump to come back, or a feed to end
    NORMAL = "normal"
    PAUSE = "pause"
    BOOST = "boost"


class WriteFailed(Exception):
    """A write to the pump didn't go through."""


class PumpCycle:
    """Runs the cycle on one pump while it's started."""

    POLL = 1.0  # How often to look at the pump while waiting on it
    POWER_ON_TIMEOUT = 20.0  # For the pump to report power on after it's turned on
    SETTLE = 3.0  # After power on, before setting a speed; after a speed, before checking power
    RETRY = 5.0  # After a failed write

    def __init__(
        self,
        pump: CyclePump,
        on_change: Callable[[], None],
        settings: CycleSettings | None = None,
        sleep: Callable[[float], Awaitable[Any]] = asyncio.sleep,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        """Set up a stopped cycle. on_change is called whenever the phase changes."""
        self.pump = pump
        self.settings = settings or CycleSettings()
        self.phase = Phase.OFF
        self.phase_ends: datetime | None = None
        self._on_change = on_change
        self._sleep = sleep
        self._now = now
        self._task: asyncio.Task | None = None

    @property
    def running(self) -> bool:
        """Whether the cycle is started."""
        return self._task is not None

    def start(self, settings: CycleSettings | None = None) -> None:
        """Start the cycle, or restart it with new settings, from the normal phase."""
        if settings is not None:
            self.settings = settings
        self._cancel()
        self._task = asyncio.get_running_loop().create_task(self._run())

    async def stop(self, restore: bool = True) -> None:
        """Stop the cycle; with restore, put the pump back to normal if it's mid-cycle."""
        was = self.phase
        self._cancel()
        self._set_phase(Phase.OFF)
        if restore and was in (Phase.PAUSE, Phase.BOOST) and self.pump.available:
            try:
                await self._normal()
            except WriteFailed as err:
                _LOGGER.warning("Couldn't put the pump back to normal: %s", err)

    def taken_over(self) -> None:
        """Someone set the power, speed or schedule: leave the pump as they set it."""
        if self.running:
            _LOGGER.info("Stopping the cycle: the pump was set by hand")
            self._cancel()
            self._set_phase(Phase.OFF)

    def close(self) -> None:
        """Stop without touching the pump or reporting it, when unloading."""
        self._cancel()

    def _cancel(self) -> None:
        if self._task is not None:
            self._task.cancel()
            self._task = None

    def _set_phase(self, phase: Phase, seconds: float | None = None) -> None:
        ends = self._now() + timedelta(seconds=seconds) if seconds is not None else None
        if phase == self.phase and ends is None and self.phase_ends is None:
            return
        self.phase = phase
        self.phase_ends = ends
        self._on_change()

    def _ready(self) -> bool:
        return self.pump.available and not self.pump.get_attribute(FEEDING)

    async def _write(self, name: str, value: Any) -> None:
        if not self._ready():
            raise WriteFailed("the pump is offline or feeding")
        try:
            await self.pump.async_cycle_write(name, value)
        except Exception as err:  # noqa: BLE001 - any failure means try again later
            raise WriteFailed(f"{name}={value}: {err}") from err

    async def _power_on(self) -> None:
        """Turn the pump on and wait until it says it is, then a little longer."""
        if not self.pump.get_attribute(POWER):
            await self._write(POWER, True)
            waited = 0.0
            while not self.pump.get_attribute(POWER) and waited < self.POWER_ON_TIMEOUT:
                await self._sleep(self.POLL)
                waited += self.POLL
                try:
                    await self.pump.async_refresh_status()
                except Exception as err:  # noqa: BLE001 - the poll will catch up
                    _LOGGER.debug("Status read while waiting for power on failed: %s", err)
            await self._sleep(self.SETTLE)

    async def _set_speed(self, speed: int) -> float:
        """Run at this speed, with the power on; how long ago the speed was set."""
        await self._power_on()
        since = 0.0
        if self.pump.get_attribute(SPEED) != speed:
            await self._write(SPEED, speed)
            await self._sleep(self.SETTLE)
            since = self.SETTLE
        if not self.pump.get_attribute(POWER):
            await self._write(POWER, True)
        return since

    async def _normal(self) -> None:
        if self.pump.get_attribute(SCHEDULE):
            await self._write(SCHEDULE, False)
        await self._set_speed(self.settings.normal_speed)

    async def _run(self) -> None:
        try:
            while True:
                if not self._ready():
                    self._set_phase(Phase.WAITING)
                    await self._sleep(self.POLL)
                    continue
                try:
                    await self._one_cycle()
                except WriteFailed as err:
                    _LOGGER.info("Cycle interrupted, starting over: %s", err)
                    self._set_phase(Phase.WAITING)
                    await self._sleep(self.RETRY)
        except asyncio.CancelledError:
            raise
        except Exception:
            _LOGGER.exception("The pump cycle stopped")
            self._task = None
            self._set_phase(Phase.OFF)

    async def _one_cycle(self) -> None:
        settings = self.settings
        await self._normal()
        self._set_phase(Phase.NORMAL, settings.run_minutes * 60)
        await self._sleep(settings.run_minutes * 60)

        if settings.pause_seconds:
            if settings.pause_speed:
                await self._write(SPEED, settings.pause_speed)
            else:
                await self._write(POWER, False)
            self._set_phase(Phase.PAUSE, settings.pause_seconds)
            await self._sleep(settings.pause_seconds)

        boosting = await self._set_speed(settings.boost_speed)
        left = max(settings.boost_seconds - boosting, 0)
        self._set_phase(Phase.BOOST, left)
        await self._sleep(left)
