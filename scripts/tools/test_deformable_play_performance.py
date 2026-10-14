"""Play display changes must preserve the saved control/history clock."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


path = Path(__file__).resolve().parents[1] / "rsl_rl/deformable_play_performance.py"
spec = importlib.util.spec_from_file_location("play_performance_test", path)
performance = importlib.util.module_from_spec(spec)
spec.loader.exec_module(performance)


def test_render_rate_preserves_physics_control_and_history_clock():
    cfg = SimpleNamespace(sim=SimpleNamespace(dt=.001, render_interval=5),
                          decimation=5, policy_history_period_s=.01, policy_history_length=5)
    performance.configure_render_rate(cfg, 50.)
    assert cfg.sim.render_interval == 20
    assert cfg.sim.dt == .001 and cfg.decimation == 5
    assert cfg.policy_history_period_s == .01 and cfg.policy_history_length == 5
    performance.configure_render_rate(cfg, 10000.)
    assert cfg.sim.render_interval == cfg.decimation


@pytest.mark.parametrize("hz", [0., -1., float("nan"), float("inf")])
def test_invalid_render_rate_does_not_mutate_the_saved_clock(hz):
    cfg = SimpleNamespace(sim=SimpleNamespace(dt=.001, render_interval=5), decimation=5)
    with pytest.raises(ValueError, match="positive and finite"):
        performance.configure_render_rate(cfg, hz)
    assert cfg.sim.render_interval == 5 and cfg.sim.dt == .001 and cfg.decimation == 5


def test_real_time_factor_uses_policy_time_and_excludes_startup(monkeypatch):
    clock = [80.]
    monkeypatch.setattr(performance.time, "perf_counter", lambda: clock[0])
    env = SimpleNamespace(device="cpu", step_dt=.005, physics_dt=.001,
                          cfg=SimpleNamespace(sim=SimpleNamespace(render_interval=20)))
    probe = performance.PlayPerformance(env)
    probe.start()
    probe.steps = 100
    clock[0] = 82.
    report = probe.finish()
    assert report["wall_s"] == 2.
    assert report["simulation_s"] == .5 and report["realtime_factor"] == .25
