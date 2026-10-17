"""Regression checks for asset relocation and the source/artifact boundary."""
import importlib.util
from pathlib import Path

import pytest

from check_repository import artifact_reason

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_robot_and_terrain_roots_follow_checkout(monkeypatch):
    monkeypatch.delenv("WHEELED_LEGGED_ASSETS_DIR", raising=False)
    world = load("layout_world", "src/agent_world/__init__.py")
    assert Path(world.RobotAssetPath) == ROOT / "assets/robots"
    assert Path(world.TerrainAssetPath) == ROOT / "assets/terrains"


def test_asset_override_works_without_working_directory_dependency(monkeypatch, tmp_path):
    external = tmp_path / "external assets"
    monkeypatch.setenv("WHEELED_LEGGED_ASSETS_DIR", str(external))
    monkeypatch.chdir(tmp_path)
    world = load("layout_external_world", "src/agent_world/__init__.py")
    assert Path(world.RobotAssetPath) == external / "robots"
    assert Path(world.TerrainAssetPath) == external / "terrains"


@pytest.mark.parametrize("name", ["own_wheel_leg_v2.json", "own_wheel_leg_v2_round2.json"])
def test_frozen_contract_resolves_relocated_robot_data(name):
    contract = load("layout_contract", "src/agent_tasks/direct/wheel_leg_v2/contract.py")
    value = contract.load_contract(contract.PACKAGE_DIR / "contracts" / name)
    original_digest = contract.contract_digest(value)
    paths = contract.asset_paths(value)
    assert paths["directory"] == ROOT / "assets/robots/Wheel_leg_V2"
    assert all(path.is_file() for key, path in paths.items() if key != "directory")
    assert contract.contract_digest(value) == original_digest


@pytest.mark.parametrize("name", [
    "参考例程/src/example.py", "references/example/src/module.py", "models/policy.pt",
    "runs_v5/run/startup.json", "outputs/legacy/source/main.py", "logs/events.out.tfevents.1",
])
def test_local_material_cannot_enter_tracked_tree(name):
    assert artifact_reason(name) is not None


@pytest.mark.parametrize("name", [
    "src/agent_rl/rsl_rl/env/vec_env.py", "assets/robots/robot/robot.usd",
    "models/README.md", "references/README.md",
])
def test_source_environment_and_owned_assets_are_allowed(name):
    assert artifact_reason(name) is None
