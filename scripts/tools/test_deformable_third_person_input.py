"""Kit input routing and reset regression, without launching the renderer."""

import importlib.util
import math
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as NS

import pytest
import torch


@pytest.fixture
def adapter(monkeypatch):
    root = Path(__file__).resolve().parents[1] / "rsl_rl"
    monkeypatch.syspath_prepend(str(root))
    app = NS(focused=True, get_window=lambda: "window")
    app.is_focused = lambda: app.focused
    modes, restored, unsubscribed = [], [], []
    windowing = NS(get_cursor_position=lambda window: (31, 47),
                   set_cursor_mode=lambda window, mode: modes.append(mode),
                   set_cursor_position=lambda window, position: restored.append(position))
    input_api = NS(subscribe_to_input_events=lambda callback, **kwargs: 123,
                   unsubscribe_to_input_events=lambda token: unsubscribed.append(token))
    enums = NS(DeviceType=NS(KEYBOARD="keyboard", MOUSE="mouse"),
               KeyboardEventType=NS(KEY_PRESS="press", KEY_RELEASE="release"),
               MouseEventType=NS(RIGHT_BUTTON_DOWN="right-down", RIGHT_BUTTON_UP="right-up",
                                MOVE="move", SCROLL="scroll", LEFT_BUTTON_DOWN="left-down"),
               acquire_input_interface=lambda: input_api)
    widget = NS(screen_position_x=10., screen_position_y=20., computed_width=500., computed_height=400.)
    modules = {name: ModuleType(name) for name in
               ("carb", "carb.input", "carb.windowing", "omni", "omni.appwindow", "omni.kit",
                "omni.kit.viewport", "omni.kit.viewport.utility")}
    modules["carb.input"].__dict__.update(vars(enums))
    modules["carb.windowing"].__dict__.update(CursorMode=NS(DISABLED="disabled", NORMAL="normal"),
                                             acquire_windowing_interface=lambda: windowing)
    modules["carb"].__dict__.update(input=modules["carb.input"], windowing=modules["carb.windowing"])
    modules["omni"].appwindow = modules["omni.appwindow"]
    modules["omni.appwindow"].get_default_app_window = lambda: app
    modules["omni.kit.viewport.utility"].get_active_viewport_window = lambda: NS(
        viewport_widget=object(), get_frame=lambda name: widget)
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    spec = importlib.util.spec_from_file_location("third_person_input_test", root / "deformable_third_person.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    data = NS(root_link_quat_w=torch.tensor([[1., 0., 0., 0.]]), root_ang_vel_b=torch.zeros(1, 3))
    env = NS(robot=NS(data=data), cmd_buf=torch.zeros(1, 3))
    wrapper = NS(unwrapped=env)
    controller = module.ThirdPersonController(wrapper)
    yield NS(controller=controller, app=app, modes=modes, restored=restored,
             unsubscribed=unsubscribed, wrapper=wrapper)
    controller.close()


def key(name, down=True):
    return NS(deviceType="keyboard", event=NS(input=NS(name=name), type="press" if down else "release"))


def mouse(kind, xy=(200., 200.), scroll=0.):
    return NS(deviceType="mouse", event=NS(type=kind, pixel_coords=xy, scrollDelta=(0., scroll)))


def test_viewport_consumes_driving_mouse_and_release_keeps_target(adapter):
    c = adapter.controller
    assert c._on_input(key("W")) is True  # A UI text field must retain its key.
    assert c._on_input(mouse("right-down", (0., 0.))) is True
    assert c._on_input(mouse("right-down")) is False
    assert adapter.modes == ["disabled"]
    assert c._on_input(mouse("move")) is False
    assert c._on_input(mouse("move", (300., 190.))) is False
    assert c.state.target_yaw == pytest.approx(math.radians(-15.))
    assert c.state.pitch_deg == 21.5
    assert c._on_input(mouse("scroll", scroll=1.)) is False
    assert c.state.distance < 3.
    assert c._on_input(mouse("right-up")) is False
    assert adapter.modes == ["disabled", "normal"] and adapter.restored == [(31, 47)]
    c.apply()
    assert c.env.cmd_buf[0, 2] < 0.  # Pending physical turn survives release.


def test_focus_loss_clears_commands_and_releases_mouse(adapter):
    c = adapter.controller
    c._on_input(mouse("right-down"))
    c._on_input(key("W"))
    c.apply()
    assert c.env.cmd_buf[0, 0] > 0.
    adapter.app.focused = False
    c.apply()
    assert not c.env.cmd_buf.any() and c.state.suspended and not c.state.right_held
    assert adapter.modes[-1] == "normal"


def test_r_reset_can_write_articulation_inference_buffers(adapter):
    c = adapter.controller
    with torch.inference_mode():
        articulation_buffer = torch.ones(1)

    def reset():
        articulation_buffer.zero_()
        return "fresh-observation", {}

    adapter.wrapper.reset = reset
    c.dispatch(dict(type="key", name="R", down=True))
    assert c.apply() == "fresh-observation"
    assert not articulation_buffer.any() and not c.state.reset_requested
    c.close()
    c.close()
    assert adapter.unsubscribed == [123]
