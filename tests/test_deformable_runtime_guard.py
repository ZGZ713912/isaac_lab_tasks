"""Sensor-only flat admission, acceleration rejection and temporal contracts."""

import math

import pytest
import torch

from test_deformable_dynamic import load_file

guard_module = load_file("deformable_runtime_guard_test",
    "src/agent_tasks/direct/deformable_suspension/runtime_guard.py")


@pytest.mark.parametrize("grade,expected", [(0., True), (2., True), (2.9, True), (5., False), (10., False), (20., False)])
def test_sensor_grade_admission(grade, expected):
    guard = guard_module.SuspensionRuntimeGuard(1, "cpu")
    angle = math.radians(grade)
    gravity = torch.tensor([[math.sin(angle), 0., -math.cos(angle)]])
    q = torch.full((1, 4), math.atan2(.13694, .029108)-math.radians(19.))
    for _ in range(25):
        guard.update(q, torch.zeros(1, 3), -guard_module.GRAVITY*gravity, torch.zeros(1, 3), gravity, .005)
    assert guard.flat_hold.item() is expected
    assert guard.grade_deg.item() == pytest.approx(grade, abs=.03)


def test_horizontal_body_on_unequal_legs_is_not_flat_ground():
    guard = guard_module.SuspensionRuntimeGuard(1, "cpu")
    q = torch.tensor([[.3, 1.04, 1.04, .3]])
    gravity = torch.tensor([[0., 0., -1.]])
    for _ in range(30):
        guard.update(q, torch.zeros(1, 3), -guard_module.GRAVITY*gravity, torch.zeros(1, 3), gravity, .005)
    assert guard.grade_deg.item() > 3.5
    assert not guard.flat_hold.item()


def test_flat_learning_starts_before_confirmation_but_runtime_waits():
    guard = guard_module.SuspensionRuntimeGuard(2, "cpu")
    angles = torch.deg2rad(torch.tensor([2.9, 5.]))
    gravity = torch.stack((angles.sin(), torch.zeros(2), -angles.cos()), -1)
    q = torch.full((2, 4), 1.0646473)
    zeros = torch.zeros(2, 3)
    guard.update(q, zeros, -guard_module.GRAVITY*gravity, zeros, gravity, .005)
    assert guard.flat_candidate.tolist() == [True, False]
    assert not guard.flat_hold.any()
    guard.reset(torch.tensor([0]))
    assert not guard.flat_candidate.any()


def test_flat_hold_survives_acceleration_braking_and_turning():
    guard = guard_module.SuspensionRuntimeGuard(2, "cpu")
    q = torch.full((2, 4), 1.0646473192622632)
    gravity = torch.tensor([[0., 0., -1.]]).expand(2, -1)
    velocity = torch.zeros(2, 3)
    for tick in range(800):
        accel = 0. if tick < 40 else (2. if tick < 240 else (-2. if tick < 640 else 0.))
        velocity[:, 0] += accel * .005
        gyro = torch.tensor([[0., 0., 0.], [0., 0., 1.]])
        force = torch.cross(gyro, velocity, dim=-1) - guard_module.GRAVITY*gravity
        force[:, 0] += accel
        compensated, _ = guard.update(q, gyro, force, velocity, gravity, .005)
        if tick >= 20:
            assert guard.flat_hold.all()
            assert compensated[:, :2].abs().max().item() < .0005
    guard.update(q, gyro, torch.zeros_like(force), velocity, gravity, .005)
    assert not guard.flat_hold.any()
    assert not guard.valid.any()


def test_nonplanar_and_bad_sensor_frames_release_then_reconfirm():
    guard = guard_module.SuspensionRuntimeGuard(2, "cpu")
    q = torch.full((2, 4), 1.0646473192622632)
    g = torch.tensor([[0., 0., -1.]]).expand(2, -1)
    zeros = torch.zeros(2, 3)
    for _ in range(20):
        guard.update(q, zeros, -guard_module.GRAVITY*g, zeros, g, .005)
    assert guard.flat_hold.all()
    q[0] = torch.tensor([.2, 1.04, .2, 1.04])
    q[1, 0] = float("nan")
    _, changed = guard.update(q, zeros, -guard_module.GRAVITY*g, zeros, g, .005)
    assert changed.all()
    assert not guard.flat_hold.any()
    assert guard.plane_error_m[0] > .006
    assert not guard.valid[1]


def test_history_retains_40ms_span_and_10ms_action_feedback():
    history = guard_module.CausalPolicyHistory(2, 5, .005, .01, "cpu")
    for tick in range(12):
        frame = torch.full((2, 32), float(tick))
        result = history.push(frame, torch.full((2, 4), float(tick)))
    torch.testing.assert_close(result[0, :, 10], torch.tensor([3., 5., 7., 9., 11.]))
    torch.testing.assert_close(result[0, -1, 22:26], torch.full((4,), 10.))
    history.reset(torch.tensor([0]))
    result = history.push(torch.full((2, 32), 12.), torch.full((2, 4), 12.))
    torch.testing.assert_close(result[0, :, 10], torch.full((5,), 12.))
    assert result[0, :, 22:26].count_nonzero() == 0
    torch.testing.assert_close(result[1, :, 10], torch.tensor([4., 6., 8., 10., 12.]))
    with pytest.raises(ValueError):
        guard_module.CausalPolicyHistory(1, 5, .006, .01, "cpu")


def test_flat_return_brakes_before_baseline_and_clears_velocity():
    target = torch.tensor([[.9, 1.02, 1.0646473, 1.08]])
    baseline = torch.full_like(target, 1.0646473)
    velocity = torch.zeros_like(target)
    peak_speed = 0.
    for _ in range(220):
        previous = target.clone()
        before = velocity.clone()
        target, velocity = guard_module.flat_return_target(target, baseline, velocity, .005, .5, 2.)
        assert (target-baseline).abs().max() <= (previous-baseline).abs().max()+1.e-7
        peak_speed = max(peak_speed, velocity.abs().max().item())
        # Only the final sampled crossing may round its remaining velocity to zero.
        moving = (target-baseline).abs() > 1.e-7
        assert (velocity-before).abs()[moving].max().item() <= .010001 if moving.any() else True
    torch.testing.assert_close(target, baseline)
    assert velocity.count_nonzero() == 0
    assert .1 < peak_speed <= .500001


def test_terminal_reference_brakes_at_measured_baseline_per_leg():
    baseline = torch.ones(1, 4)
    measured = torch.tensor([[.9, .995, 1., 1.02]])
    velocity = torch.full_like(measured, .5)
    limited = guard_module.flat_terminal_velocity(measured, baseline, velocity, 2.)
    torch.testing.assert_close(limited, torch.tensor([[.5, math.sqrt(.02), 0., 0.]]))
    # The opposite travel direction has the same physical stopping contract.
    limited = guard_module.flat_terminal_velocity(2.-measured, baseline, -velocity, 2.)
    torch.testing.assert_close(limited, torch.tensor([[-.5, -math.sqrt(.02), 0., 0.]]))


def test_sensor_confirmation_rejects_one_plausible_sample_before_free_fall():
    guard = guard_module.SuspensionRuntimeGuard(1, "cpu")
    q = torch.full((1, 4), math.atan2(.13694, .029108) - math.radians(17.))
    gyro = torch.zeros(1, 3)
    twist = torch.zeros(1, 3)
    gravity = torch.tensor([[0., 0., -1.]])
    force = -gravity * guard_module.GRAVITY
    guard.update(q, gyro, force, twist, gravity, .005)
    assert not guard.sensor_ready.item()
    guard.update(q, gyro, torch.zeros_like(force), twist, gravity, .005)
    assert guard.sensor_valid_time.item() == 0.
    assert not guard.sensor_ready.item()
    assert guard.encoder_flat_candidate.item() and not guard.flat_hold.item()
    for _ in range(19):
        guard.update(q, gyro, force, twist, gravity, .005)
        assert not guard.sensor_ready.item()
    guard.update(q, gyro, force, twist, gravity, .005)
    assert guard.sensor_ready.item() and guard.flat_hold.item()
    guard.update(q, gyro, torch.zeros_like(force), twist, gravity, .005)
    assert not guard.sensor_ready.item() and not guard.flat_hold.item()


def test_coordinated_return_preserves_shared_progress_and_finishes_all_legs():
    start = torch.tensor([[.8, 1., 1.0646473, .95]])
    baseline = torch.full_like(start, 1.0646473)
    for seconds in (.1, .3, .6, .9):
        target, velocity = guard_module.coordinated_flat_return(start, baseline, torch.tensor([seconds]), 1.2, .5, 2.)
        moving = (baseline - start).abs() > 1.e-6
        progress = ((target - start) / (baseline - start))[moving]
        assert progress.max() - progress.min() <= 1.e-5
        assert velocity.abs().max() <= .5
    target, velocity = guard_module.coordinated_flat_return(start, baseline, torch.tensor([1.2]), 1.2, .5, 2.)
    torch.testing.assert_close(target, baseline)
    assert velocity.count_nonzero() == 0


def test_startup_candidate_uses_initial_attitude_without_accelerometer_hold():
    guard = guard_module.SuspensionRuntimeGuard(1, "cpu")
    q = torch.full((1, 4), math.atan2(.13694, .029108) - math.radians(17.))
    gyro, twist, force = torch.zeros(1, 3), torch.zeros(1, 3), torch.zeros(1, 3)
    gravity = torch.tensor([[math.sin(math.radians(10.)), 0., -math.cos(math.radians(10.))]])
    guard.update(q, gyro, force, twist, gravity, .005)
    assert not guard.encoder_flat_candidate.item() and not guard.flat_hold.item()
    gravity = torch.tensor([[0., 0., -1.]])
    guard.update(q, gyro, force, twist, gravity, .005)
    assert guard.encoder_flat_candidate.item() and not guard.flat_hold.item()


def test_flat_learning_supervises_each_leg_and_excludes_reference_conflict():
    from test_deformable_policy_preservation import module, Policy
    algorithm = module.DiagnosticPPO(Policy(), device="cpu", flat_hold_weight=1.)
    obs = {"training_flat_hold": torch.tensor([[1.], [0.]])}
    actions = torch.tensor([[-.03, .05, .2, -.1], [.2, -.2, .3, -.4]], requires_grad=True)
    loss, fraction = algorithm._flat_hold_loss(obs, actions)
    loss.backward()
    assert fraction.item() == .5
    assert (actions.grad[0] * actions[0] > 0).all()
    assert actions.grad[1].count_nonzero() == 0
    with pytest.raises(ValueError):
        algorithm._flat_hold_loss({}, actions)


def test_flat_selector_cannot_preserve_the_old_raised_reference():
    from test_deformable_policy_preservation import module, Policy, histories
    policy = Policy()
    algorithm = module.DiagnosticPPO(policy, device="cpu", steep_preservation_weight=1.,
                                     reference_all_postures=True, flat_hold_weight=1.)
    algorithm.initialize_steep_reference()
    obs = histories([0., 0.])
    obs["training_flat_hold"] = torch.tensor([[1.], [0.]])
    reference = policy.actor(obs["policy"]).detach()
    actions = (reference+.1).requires_grad_()
    loss, fraction = algorithm._steep_preservation_loss(obs, actions)
    loss.backward()
    assert fraction.item() == .5
    assert actions.grad[0].count_nonzero() == 0
    assert (actions.grad[1] > 0).all()


def test_training_floor_labels_cover_near_flat_and_reject_steps():
    module = load_file("flat_supervision_test", "src/agent_tasks/direct/deformable_suspension/training_supervision.py")
    points = torch.tensor([[[.3,-.3,0.],[.3,.3,0.],[-.3,.3,0.],[-.3,-.3,0.]]]).repeat(5,1,1)
    normals = torch.zeros_like(points)
    normals[...,2] = 1.
    height = torch.zeros(5,4)
    for i, grade in ((1,2.9),(2,5.)):
        angle = math.radians(grade)
        height[i] = points[i,:,0]*math.tan(angle)
        normals[i,:,0] = -math.sin(angle)
        normals[i,:,2] = math.cos(angle)
    height[3,0] = .08  # A level corner obstacle is not a flat support plane.
    height[4,:2] = .08  # A two-wheel ledge fits a plane, but its grade exceeds 3deg.
    assert module.flat_ground_label(points,normals,height).tolist() == [True,True,False,False,False]
    points[0,0,0] = float("nan")
    assert not module.flat_ground_label(points,normals,height)[0]


def test_startup_samples_receive_more_learning_without_changing_the_actor():
    from test_deformable_policy_preservation import module, Policy
    algorithm = module.DiagnosticPPO(Policy(), device="cpu", flat_hold_weight=1., flat_hold_startup_weight=50.)
    obs = {"training_flat_hold":torch.ones(2,1), "training_flat_startup":torch.tensor([[1.],[0.]])}
    actions = torch.full((2,4),-.1,requires_grad=True)
    loss,fraction = algorithm._flat_hold_loss(obs,actions)
    loss.backward()
    assert fraction.item() == 1.
    torch.testing.assert_close(actions.grad[0],50.*actions.grad[1])
    with pytest.raises(ValueError,match="startup"):
        algorithm._flat_hold_loss({"training_flat_hold":torch.ones(2,1)},actions)


def test_ground_and_startup_labels_survive_storage_without_entering_h5_actor():
    from tensordict import TensorDict
    from rsl_rl.storage.rollout_storage import RolloutStorage
    module = load_file("flat_selector_actor", "src/agent_rl/rsl_rl/modules/actor_critic_suspension_mlp.py")
    labels = torch.arange(4.)[:,None]/3.
    obs = TensorDict({"policy":torch.zeros(4,160), "critic":torch.zeros(4,40),
                      "training_flat_hold":labels, "training_flat_startup":1.-labels},batch_size=[4])
    obs["policy"][:,0] = labels[:,0]
    actor = module.ActorCriticSuspensionMLP(obs,{"policy":["policy"],"critic":["critic"]},4,
                                          actor_hidden_dims=[8],critic_hidden_dims=[8],history_length=5)
    original = actor.act_inference(obs)
    changed = obs.clone()
    changed["training_flat_hold"][:] = 100.
    changed["training_flat_startup"][:] = -100.
    torch.testing.assert_close(actor.act_inference(changed),original,rtol=0,atol=0)
    assert actor.get_actor_obs(obs).shape == (4,160)
    assert actor.get_critic_obs(obs).shape == (4,40)
    storage = RolloutStorage("rl",4,2,obs,[4],device="cpu")
    storage.observations[0].copy_(obs);storage.observations[1].copy_(obs)
    for batch,*_ in storage.mini_batch_generator(2,1):
        torch.testing.assert_close(batch["training_flat_hold"][:,0],batch["policy"][:,0])
        torch.testing.assert_close(batch["training_flat_startup"][:,0],1.-batch["policy"][:,0])


def test_acceptance_cannot_hide_one_raised_leg_in_a_mean(tmp_path):
    import json
    import numpy as np
    from deformable_guarded_assess import flat_assessment
    (tmp_path/"traces").mkdir()
    checkpoint = tmp_path/"model.pt"
    checkpoint.write_bytes(b"test-checkpoint")
    row = dict(failure_adjusted_all_contact_rate=1., tilt_deg=dict(abs_p95=0.),
               terminated_resets=0, physical_terminated_resets=0, terrain_boundary_violations=0,
               mean_abs_slip_m_s=0.)
    report = dict(checkpoint=str(checkpoint), seed=1, runtime_guard=dict(policy_rate_hz=200.),
                  terrain=dict(grade_deg=0.), results=dict(POLICY=dict(static=row)))
    (tmp_path/"run.log").write_text(json.dumps(report)+'\n')
    q = np.full((300, 1, 4), math.atan2(.13694,.029108)-math.radians(17.))
    q[..., 0] -= math.radians(2.)
    guard = np.zeros((300, 1, 9))
    guard[..., 0] = 1.
    np.savez(tmp_path/"traces/policy_static.npz", episode_age_s=np.full((300,1),2.), runtime_guard=guard,
             leg_target=q, joint_q=q, raw_policy_actions=np.zeros_like(q),
             time_s=np.linspace(1.,2.5,300), columns=np.array(("cmd_vx_body", "cmd_vy_body", "cmd_wz")),
             metrics=np.zeros((300,1,3)))
    assessment = flat_assessment(tmp_path)
    assert not assessment["passed"]
    assert not assessment["cases"]["static"]["checks"]["strict_minangle_target"]
    assert assessment["cases"]["static"]["target_error_max_deg"] == pytest.approx(2.)
    with np.load(tmp_path/"traces/policy_static.npz") as source:
        data = {name: source[name] for name in source.files}
    data["episode_age_s"][:] = .2
    np.savez(tmp_path/"traces/policy_static.npz", **data)
    assessment = flat_assessment(tmp_path)
    assert not assessment["passed"]
    assert assessment["cases"]["static"]["physical_error_p95_deg"] is None
    assert not assessment["cases"]["static"]["checks"]["settled_samples"]


def test_acceptance_keeps_transient_error_even_if_last_second_is_perfect(tmp_path):
    import json
    import numpy as np
    from deformable_guarded_assess import flat_assessment
    (tmp_path/"traces").mkdir()
    checkpoint = tmp_path/"model.pt"
    checkpoint.write_bytes(b"test-checkpoint")
    row = dict(failure_adjusted_all_contact_rate=1., tilt_deg=dict(abs_p95=0.),
               terminated_resets=0, physical_terminated_resets=0, terrain_boundary_violations=0)
    report = dict(checkpoint=str(checkpoint), seed=1, runtime_guard=dict(policy_rate_hz=200.),
                  terrain=dict(grade_deg=2.), results=dict(POLICY=dict(static=row)))
    (tmp_path/"run.log").write_text(json.dumps(report)+'\n')
    q = np.full((600, 1, 4), math.atan2(.13694,.029108)-math.radians(17.))
    measured = q.copy()
    measured[:200] -= math.radians(1.)
    guard = np.zeros((600, 1, 9))
    guard[..., 0] = 1.
    np.savez(tmp_path/"traces/policy_static.npz", episode_age_s=np.full((600,1),2.), runtime_guard=guard,
             leg_target=q, joint_q=measured, raw_policy_actions=np.zeros_like(q),
             time_s=np.linspace(1.,4.,600), columns=np.array(("cmd_vx_body", "cmd_vy_body", "cmd_wz")),
             metrics=np.zeros((600,1,3)))
    result = flat_assessment(tmp_path)
    assert result["cases"]["static"]["physical_steady_error_max_deg"] == pytest.approx(0.)
    assert not result["passed"]
    assert not result["cases"]["static"]["checks"]["measured_minangle"]
    from deformable_guarded_assess import baseline_safety
    assert baseline_safety(result)["passed"]
    result["cases"]["static"]["checks"]["no_physical_reset"] = False
    assert not baseline_safety(result)["passed"]


@pytest.mark.parametrize("fault", [None, "runtime_source", "snapshot", "active", "incomplete", "weights", "actor"])
def test_reusing_short_run_requires_same_training_and_exact_completed_weights(tmp_path, monkeypatch, fault):
    import hashlib
    import json
    import deformable_guarded_reuse as reuse
    reference = tmp_path / "reference.pt"
    reference.write_bytes(b"qualified-reference")
    candidate = tmp_path / "short.pt"
    candidate.write_bytes(b"updated-short-policy")
    runtime_name = "src/agent_tasks/direct/deformable_suspension/dynamic_env.py"
    snapshot = tmp_path / "source" / runtime_name
    snapshot.parent.mkdir(parents=True)
    snapshot.write_bytes(b"runtime-source")
    source_sha = hashlib.sha256(snapshot.read_bytes()).hexdigest()
    weight_sha = hashlib.sha256(candidate.read_bytes()).hexdigest()
    sources = {runtime_name: source_sha, "tools/deformable_guarded_assess.py": "new-2s-criterion"}
    state = dict(status="needs_iteration", checkpoint=str(reference),
        checkpoint_sha256=hashlib.sha256(reference.read_bytes()).hexdigest(), sources={runtime_name: source_sha},
        training=[dict(checkpoint=str(candidate), checkpoint_sha256=weight_sha, updates=100, seed=89)],
        jobs=[dict(name="iteration_03_short", status="completed", exit_code=0,
                   command=["train.py", "--max_iterations", "100", "--seed", "89", "--checkpoint", str(reference)])])
    if fault == "runtime_source":
        state["sources"][runtime_name] = "old-runtime"
    elif fault == "snapshot":
        snapshot.write_bytes(b"changed-runtime")
    elif fault == "active":
        state["status"] = "running"
    elif fault == "incomplete":
        state["jobs"][0]["status"] = "failed"
    elif fault == "weights":
        state["training"][0]["checkpoint_sha256"] = "different-weights"
    (tmp_path / "status.json").write_text(json.dumps(state))
    monkeypatch.setattr(reuse, "checkpoint_info", lambda path: dict(sha256=weight_sha, iteration=99))
    monkeypatch.setattr(reuse, "actor_update_info", lambda old, new: dict(policy_updated=fault != "actor"))
    if fault is None:
        path, record = reuse.reuse_short_run(reference, tmp_path, sources, 100, 89)
        assert path == candidate
        assert record["provenance"]["additional_short_updates"] == 0
        assert record["updates"] == 100
    else:
        with pytest.raises(ValueError):
            reuse.reuse_short_run(reference, tmp_path, sources, 100, 89)


@pytest.mark.parametrize("fault", [None, "missing_stamp", "other_experiment", "incomplete_run"])
def test_checkpoint_collection_uses_the_subprocess_timestamp(tmp_path, fault):
    from deformable_guarded_reuse import training_checkpoint
    experiment = tmp_path / "logs/rsl_rl/deformable_guarded_mlp5_v4"
    old = experiment / "2026-10-12_23-39-34_iteration_02_training_short"
    new = experiment / "2026-10-13_20-43-09_training_short"
    for directory in (old, new):
        (directory / "params").mkdir(parents=True)
        (directory / "params/agent.yaml").write_text("policy: {}")
        (directory / "model_99.pt").write_bytes(b"checkpoint")
    stamp = "Exact experiment name requested from command line: 2026-10-13_20-43-09\n"
    reported = experiment if fault != "other_experiment" else tmp_path / "wrong_experiment"
    log = tmp_path / "training.log"
    log.write_text(f"[INFO] Logging experiment in directory: {reported}\n"
                   + (stamp if fault != "missing_stamp" else ""))
    if fault == "incomplete_run":
        (new / "params/agent.yaml").unlink()
    if fault is None:
        assert training_checkpoint(log, "deformable_guarded_mlp5_v4", "training_short", tmp_path) == new / "model_99.pt"
    else:
        with pytest.raises(ValueError):
            training_checkpoint(log, "deformable_guarded_mlp5_v4", "training_short", tmp_path)


def test_paired_slope_early_rejection_cannot_count_as_full_coverage(tmp_path):
    import json
    from deformable_guarded_assess import paired_slopes
    old, new = tmp_path / "old", tmp_path / "new"
    for grade in (0, 5, 10, 17, 20):
        for root in (old, new):
            directory = root / f"grade_{grade}"
            directory.mkdir(parents=True)
            row = dict(initial_state_sha256="same-state", failure_adjusted_all_contact_rate=1.,
                       tilt_deg=dict(abs_p95=1.), terminated_resets=0)
            report = dict(checkpoint="probe", seed=1234, num_envs=16, steps=1200, step_dt_s=.005,
                          command_profile="play", command_frame="body", history=dict(length=5),
                          results=dict(POLICY=dict(static=row)))
            if root == new and grade == 5:
                row["tilt_deg"]["abs_p95"] = 4.
                row["terminated_resets"] = 1
            (directory / "run.log").write_text(json.dumps(report) + '\n')
    first = paired_slopes(old, new, grades=(0,))
    assert first["passed"] and not first["complete"]
    early = paired_slopes(old, new, grades=(5,))
    assert not early["passed"] and not early["complete"]
    full = paired_slopes(old, new)
    assert full["complete"] and not full["passed"]
    assert full["grades"]["5"]["cases"]["static"]["tilt_p95_delta_deg"] == 3.
    with pytest.raises(ValueError, match="coverage"):
        paired_slopes(old, new, grades=())


@pytest.mark.parametrize("fault", [None, "architecture", "runtime", "snapshot", "nonliteral"])
def test_fixed_checkpoint_physics_cache_accepts_only_literal_training_changes(tmp_path, fault):
    import hashlib
    from deformable_guarded_reuse import TRAINING_CONFIG, evaluation_sources_match
    root, saved = tmp_path / "current", tmp_path / "previous"
    runtime = "src/agent_tasks/direct/deformable_suspension/dynamic_env.py"
    old = ("class DeformableGuardedRoutedMLPPPORunnerCfg:\n"
           "    observation_space = 160\n"
           "    def __post_init__(self):\n"
           "        self.algorithm.learning_rate = 1.e-5\n"
           "        self.algorithm.flat_hold_startup_weight = 1.\n"
           "        self.policy.history_length = 5\n")
    new = old.replace("1.e-5", "1.e-6")
    if fault == "architecture":
        new = new.replace("history_length = 5", "history_length = 8")
    elif fault == "nonliteral":
        new = new.replace("1.e-6", "get_learning_rate()")
    previous, current = {}, {}
    for name, a, b in ((TRAINING_CONFIG, old, new), (runtime, "runtime", "runtime")):
        if name == runtime and fault == "runtime":
            b = "changed-runtime"
        snapshot, actual = saved / "source" / name, root / name
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        actual.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_text(a)
        actual.write_text(b)
        previous[name] = hashlib.sha256(snapshot.read_bytes()).hexdigest()
        current[name] = hashlib.sha256(actual.read_bytes()).hexdigest()
        if name == TRAINING_CONFIG and fault == "snapshot":
            snapshot.write_text("corrupt-snapshot")
    assert evaluation_sources_match(saved, previous, current, root) is (fault is None)


def test_two_second_return_excludes_only_encoder_transient_and_keeps_all_resets(tmp_path):
    import json
    import numpy as np
    from deformable_guarded_assess import flat_assessment
    (tmp_path / "traces").mkdir()
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"test-checkpoint")
    row = dict(failure_adjusted_all_contact_rate=1., tilt_deg=dict(abs_p95=0.),
               terminated_resets=0, physical_terminated_resets=0, terrain_boundary_violations=0)
    report = dict(checkpoint=str(checkpoint), seed=1, runtime_guard=dict(policy_rate_hz=200.),
                  terrain=dict(grade_deg=2.9), results=dict(POLICY=dict(static=row)))
    age = np.linspace(1.5, 3.5, 401)[:, None]
    q = np.full((401, 1, 4), math.atan2(.13694, .029108) - math.radians(17.))
    target, measured = q.copy(), q.copy()
    target[age[:, 0] < 2.] -= math.radians(10.)
    measured[age[:, 0] < 2.] -= math.radians(10.)
    guard = np.zeros((401, 1, 9))
    guard[age[:, 0] >= 2., :, 0] = 1.
    data = dict(episode_age_s=age, runtime_guard=guard, leg_target=target, joint_q=measured,
                raw_policy_actions=np.zeros_like(q), time_s=age[:, 0],
                columns=np.array(("cmd_vx_body", "cmd_vy_body", "cmd_wz")), metrics=np.zeros((401, 1, 3)))
    def assess():
        (tmp_path / "run.log").write_text(json.dumps(report) + '\n')
        np.savez(tmp_path / "traces/policy_static.npz", **data)
        return flat_assessment(tmp_path)
    result = assess()
    assert result["passed"]
    assert result["limits"]["target_check_after_s"] == 2.
    assert result["cases"]["static"]["settled_samples"] == 301
    measured[100, 0, 0] -= math.radians(1.)
    assert not assess()["cases"]["static"]["checks"]["measured_minangle"]
    measured[:] = q
    row["physical_terminated_resets"] = 1
    assert not assess()["cases"]["static"]["checks"]["no_physical_reset"]


def test_guarded_checkpoint_requires_saved_timing_and_explicit_transition(tmp_path):
    import yaml
    from types import SimpleNamespace
    contract = load_file("guarded_checkpoint_test", "scripts/utils/deformable_checkpoint.py")
    checkpoint = tmp_path/"model.pt"
    checkpoint.write_bytes(b"checkpoint")
    params = tmp_path/"params"
    params.mkdir()
    fields = dict(runtime_guard_version="sensor_plane_imu_v8", policy_history_period_s=.01,
        guard_enter_deg=3., guard_exit_deg=3.5, guard_plane_tolerance_m=.006, guard_confirmation_s=.1,
        guard_attitude_tau_s=.25, guard_acceleration_tau_s=.05, guard_acceleration_fusion_scale=.5,
        guard_flat_return_rate_limit=.5, guard_flat_return_acceleration_limit=2., guard_flat_velocity_reference=True,
        guard_flat_terminal_braking=True, guard_flat_terminal_acceleration_limit=1., guard_startup_trajectory=True,
        guard_coordinated_return=True, guard_flat_return_minimum_duration_s=1.2)
    cfg = SimpleNamespace(action_contract_version="minangle_physical_v3", real2sim_enabled=False,
                          runtime_guard_enabled=True, decimation=5, sim=SimpleNamespace(dt=.001), **fields)
    saved = dict(action_contract_version=cfg.action_contract_version, real2sim_enabled=False,
                 runtime_guard_enabled=True, policy_history_length=5, decimation=5, sim=dict(dt=.001), **fields)
    path = params/"env.yaml"
    path.write_text(yaml.safe_dump(saved))
    contract.validate_deformable_checkpoint(cfg,checkpoint)
    saved["decimation"]=10
    path.write_text(yaml.safe_dump(saved))
    with pytest.raises(ValueError,match="timing"):
        contract.validate_deformable_checkpoint(cfg,checkpoint)
    saved["runtime_guard_enabled"]=False
    path.write_text(yaml.safe_dump(saved))
    with pytest.raises(ValueError,match="explicit"):
        contract.validate_deformable_checkpoint(cfg,checkpoint)
    contract.validate_deformable_checkpoint(cfg,checkpoint,allow_runtime_guard_transition=True)
    saved["decimation"]=5
    path.write_text(yaml.safe_dump(saved))
    with pytest.raises(ValueError,match="timing"):
        contract.validate_deformable_checkpoint(cfg,checkpoint,allow_runtime_guard_transition=True)
