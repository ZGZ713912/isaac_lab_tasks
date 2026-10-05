"""The rendered mesh and runtime contact/reset profile must use identical grades."""
import ast
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
import trimesh

ROOT = Path(__file__).resolve().parents[2]


def terrain_functions():
    path = ROOT / "source/agent_world/agent_world/terrains/height_field.py"
    names = {"periodic_slope_angle", "_periodic_slope_angle_table", "periodic_slope_height", "periodic_slope_terrain"}
    tree = ast.parse(path.read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = {"np": np, "trimesh": trimesh}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), namespace)
    return SimpleNamespace(**{name: namespace[name] for name in names})


def runtime_utilities(terrain):
    module = ModuleType("agent_world.terrains")
    module.periodic_slope_angle = terrain.periodic_slope_angle
    spec = importlib.util.spec_from_file_location(
        "deformable_grade_runtime", ROOT / "source/agent_tasks/agent_tasks/direct/deformable_suspension/cfg_utils.py")
    result = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"agent_world.terrains": module}):
        spec.loader.exec_module(result)
    return result


def test_discrete_grade_mesh_matches_runtime_height_and_retains_all_grades():
    terrain = terrain_functions()
    du = runtime_utilities(terrain)
    choices = (5., 10., 17., 20., 20.)
    cfg = SimpleNamespace(horizontal_scale=.1, size=(150., 1.), segment_length=6.,
                          angle_range=(5., 20.), angle_seed=13, angle_choices=choices)
    meshes, _ = terrain.periodic_slope_terrain(0., cfg)
    vertices = torch.tensor(np.array(meshes[0].vertices), dtype=torch.float32)
    table = du.build_periodic_slope_angle_table(6., (5., 20.), 13, 10, "cpu", choices)
    np.testing.assert_allclose(table.numpy()[:5], [20., 20., 5., 10., 17.])
    heights = du.periodic_slope_height_torch(vertices[:, 0], 6., table)
    torch.testing.assert_close(heights, vertices[:, 2], atol=1.e-5, rtol=1.e-5)
    # Cyclic repetition is a grade quota per full sequence, not per transition.
    assert (table == 20).sum() == 4


def test_default_continuous_sampler_is_unchanged_and_invalid_choices_fail():
    terrain = terrain_functions()
    for k in range(8):
        expected = np.random.default_rng([13, k]).uniform(17., 20.)
        assert terrain.periodic_slope_angle(k, (17., 20.), 13) == expected
    for choices in [(), (float("nan"),), (-1.,), (90.,)]:
        with pytest.raises(ValueError, match="choices"):
            terrain.periodic_slope_angle(0, (0., 20.), 13, choices)
