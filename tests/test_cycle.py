"""Tests for the run, pause and boost cycle."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
import importlib.util
from pathlib import Path
import sys
from typing import Any
import unittest

PATH = Path(__file__).parents[1] / "custom_components" / "jebao" / "cycle.py"
SPEC = importlib.util.spec_from_file_location("jebao_cycle", PATH)
assert SPEC and SPEC.loader
cycle = importlib.util.module_from_spec(SPEC)
sys.modules["jebao_cycle"] = cycle
SPEC.loader.exec_module(cycle)

START = datetime(2026, 11, 12, 12, 0, tzinfo=UTC)


class Stop(BaseException):
    """Ends a test run from inside the fake clock."""


class FakePump:
    """A pump that applies writes at once, on a fake clock."""

    def __init__(self, **status: Any) -> None:
        self.status = {"switch_on": True, "motor_speed": 60, "feed_switch": False,
                       "timer_on": False, **status}
        self.available = True
        self.writes: list[tuple[float, str, Any]] = []
        self.time = 0.0
        self.until = 0.0
        self.events: dict[float, Any] = {}  # time -> callable, run when the clock passes it
        self.fail_writes = False

    def get_attribute(self, name: str) -> Any:
        return self.status.get(name)

    async def async_refresh_status(self) -> None:
        pass

    async def async_cycle_write(self, name: str, value: Any) -> None:
        if self.fail_writes:
            raise RuntimeError("no answer")
        self.writes.append((self.time, name, value))
        self.status[name] = value
        if name == "motor_speed" and value != 0:
            self.status["timer_on"] = False

    async def sleep(self, seconds: float) -> None:
        target = self.time + seconds
        for at in sorted(self.events):
            if self.time < at <= target:
                self.events.pop(at)()
        self.time = target
        if self.time >= self.until:
            raise Stop
        await asyncio.sleep(0)

    def now(self) -> datetime:
        return START + timedelta(seconds=self.time)


def run(pump: FakePump, until: float, settings=None) -> cycle.PumpCycle:
    """Run a cycle on the pump until the fake clock reaches this time."""
    pump.until = until
    changes: list[str] = []

    async def go() -> cycle.PumpCycle:
        runner = cycle.PumpCycle(
            pump, lambda: changes.append(runner.phase), settings, pump.sleep, pump.now
        )
        runner.start()
        try:
            await runner._task
        except Stop:
            pass
        runner.changes = changes
        return runner

    return asyncio.run(go())


def names(writes):
    return [(name, value) for _, name, value in writes]


class CycleTest(unittest.TestCase):
    def test_one_cycle(self) -> None:
        pump = FakePump()
        run(pump, 700)
        self.assertEqual(
            [(600, "switch_on", False), (610, "switch_on", True),
             (613, "motor_speed", 100), (643, "motor_speed", 60)],
            pump.writes,
        )

    def test_takes_the_pump_from_the_schedule_and_off(self) -> None:
        pump = FakePump(switch_on=False, motor_speed=80, timer_on=True)
        run(pump, 100)
        self.assertEqual(
            [("timer_on", False), ("switch_on", True), ("motor_speed", 60)],
            names(pump.writes),
        )

    def test_waits_for_power_on_before_the_boost(self) -> None:
        pump = FakePump()
        real_write = pump.async_cycle_write

        async def slow_power(name, value):
            await real_write(name, value)
            if (name, value) == ("switch_on", True):
                pump.status["switch_on"] = False  # Reports on 4 s later
                pump.events[pump.time + 4] = lambda: pump.status.update(switch_on=True)

        pump.async_cycle_write = slow_power
        run(pump, 640)
        boost = [t for t, name, value in pump.writes if (name, value) == ("motor_speed", 100)]
        self.assertEqual([617], boost)  # 610 on, reports at 614, 3 s more

    def test_slow_pause_instead_of_off(self) -> None:
        pump = FakePump()
        run(pump, 700, cycle.CycleSettings(pause_speed=30, pause_seconds=20))
        self.assertEqual(
            [("motor_speed", 30), ("motor_speed", 100), ("motor_speed", 60)],
            names(pump.writes),
        )

    def test_no_pause(self) -> None:
        pump = FakePump()
        run(pump, 700, cycle.CycleSettings(pause_seconds=0))
        self.assertEqual([("motor_speed", 100), ("motor_speed", 60)], names(pump.writes))

    def test_offline_pump_starts_over_when_back(self) -> None:
        pump = FakePump()
        pump.events[300] = lambda: setattr(pump, "available", False)
        pump.events[800] = lambda: setattr(pump, "available", True)
        runner = run(pump, 1000)
        self.assertEqual([], pump.writes)  # It never left normal
        self.assertIn(cycle.Phase.WAITING, runner.changes)
        self.assertEqual(cycle.Phase.NORMAL, runner.phase)
        self.assertEqual(START + timedelta(seconds=1400), runner.phase_ends)

    def test_back_to_normal_after_an_outage_mid_boost(self) -> None:
        pump = FakePump()
        pump.events[620] = lambda: setattr(pump, "available", False)
        pump.events[700] = lambda: setattr(pump, "available", True)
        run(pump, 800)
        self.assertEqual(("motor_speed", 60), names(pump.writes)[-1])
        self.assertLessEqual(700, pump.writes[-1][0])

    def test_no_writes_while_feeding(self) -> None:
        pump = FakePump()
        pump.events[500] = lambda: pump.status.update(feed_switch=True)
        pump.events[900] = lambda: pump.status.update(feed_switch=False)
        run(pump, 1000)
        self.assertTrue(all(t > 900 for t, _, _ in pump.writes))

    def test_failed_writes_retry(self) -> None:
        pump = FakePump()
        pump.fail_writes = True
        pump.events[650] = lambda: setattr(pump, "fail_writes", False)
        runner = run(pump, 700)
        self.assertEqual([], pump.writes)
        self.assertEqual(cycle.Phase.NORMAL, runner.phase)


class StopTest(unittest.TestCase):
    def _mid_boost(self, action):
        pump = FakePump()
        pump.until = 10**9

        async def go():
            runner = cycle.PumpCycle(pump, lambda: None, None, pump.sleep, pump.now)
            runner.start()
            while runner.phase != cycle.Phase.BOOST:
                await asyncio.sleep(0)
            pump.writes.clear()
            await action(runner)
            await asyncio.sleep(0)
            return runner

        return pump, asyncio.run(go())

    def test_stop_returns_to_normal(self) -> None:
        async def stop(runner):
            await runner.stop()

        pump, runner = self._mid_boost(stop)
        self.assertFalse(runner.running)
        self.assertEqual(cycle.Phase.OFF, runner.phase)
        self.assertEqual([("motor_speed", 60)], names(pump.writes))

    def test_taken_over_leaves_the_pump_alone(self) -> None:
        async def take(runner):
            runner.taken_over()

        pump, runner = self._mid_boost(take)
        self.assertFalse(runner.running)
        self.assertEqual([], pump.writes)


class SettingsTest(unittest.TestCase):
    def test_defaults_are_the_pond_skimmer_cycle(self) -> None:
        self.assertEqual(
            {"normal_speed": 60, "run_minutes": 10, "pause_speed": 0,
             "pause_seconds": 10, "boost_speed": 100, "boost_seconds": 30},
            cycle.CycleSettings().as_dict(),
        )

    def test_from_saved_attributes(self) -> None:
        settings = cycle.CycleSettings.from_dict(
            {"boost_speed": 90, "phase": "normal", "friendly_name": "Cycle"}
        )
        self.assertEqual(90, settings.boost_speed)
        self.assertEqual(60, settings.normal_speed)

    def test_refuses_what_the_pump_cant_run(self) -> None:
        for bad in ({"normal_speed": 20}, {"pause_speed": 10}, {"boost_speed": 101},
                    {"run_minutes": 0}, {"boost_seconds": 0}):
            with self.subTest(bad), self.assertRaises(ValueError):
                cycle.CycleSettings().replace(bad)


if __name__ == "__main__":
    unittest.main()
