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
    root = Path(__file__).resolve().parents[1] / "scripts/rsl_rl"
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
               KeyboardEventType=NS(KEY_PRESS="press", KEY_RELEASE="release", KEY_REPEAT="repeat", CHAR="char"),
               MouseEventType=NS(RIGHT_BUTTON_DOWN="right-down", RIGHT_BUTTON_UP="right-up",
                                MOVE="move", SCROLL="scroll", LEFT_BUTTON_DOWN="left-down"),
               acquire_input_interface=lambda: input_api)
    widget = NS(screen_position_x=10., screen_position_y=20., computed_width=500., computed_height=400.)
    ui_callbacks = {}
    for kind in ("pressed", "released", "moved", "wheel"):
        setattr(widget, "set_mouse_" + kind + "_fn", lambda callback, kind=kind: ui_callbacks.update({kind: callback}))
    viewport = NS(focused=True, viewport_widget=object(), get_frame=lambda name: widget)
    modules = {name: ModuleType(name) for name in
               ("carb", "carb.input", "carb.windowing", "omni", "omni.appwindow", "omni.kit",
                "omni.kit.viewport", "omni.kit.viewport.utility")}
    modules["carb.input"].__dict__.update(vars(enums))
    modules["carb.windowing"].__dict__.update(CursorMode=NS(DISABLED="disabled", NORMAL="normal"),
                                             acquire_windowing_interface=lambda: windowing)
    modules["carb"].__dict__.update(input=modules["carb.input"], windowing=modules["carb.windowing"])
    modules["omni"].appwindow = modules["omni.appwindow"]
    modules["omni.appwindow"].get_default_app_window = lambda: app
    modules["omni.kit.viewport.utility"].get_active_viewport_window = lambda: viewport
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    spec = importlib.util.spec_from_file_location("third_person_input_test", root / "deformable_third_person.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    data = NS(root_link_quat_w=torch.tensor([[1., 0., 0., 0.]]), root_ang_vel_b=torch.zeros(1, 3))
    env = NS(robot=NS(data=data), cmd_buf=torch.zeros(1, 3), step_dt=.005)
    wrapper = NS(unwrapped=env)
    controller = module.ThirdPersonController(wrapper)
    yield NS(controller=controller, app=app, modes=modes, restored=restored,
             unsubscribed=unsubscribed, wrapper=wrapper, ui_callbacks=ui_callbacks, viewport=viewport)
    controller.close()


def key(name, down=True):
    return NS(deviceType="keyboard", event=NS(input=NS(name=name), type="press" if down else "release"))


def mouse(kind, xy=(200., 200.), scroll=0.):
    return NS(deviceType="mouse", event=NS(type=kind, pixel_coords=xy, scrollDelta=(0., scroll)))


def character(text):
    # Kit sends CHAR separately from KEY_PRESS; its input is a str, not an enum.
    return NS(deviceType="keyboard", event=NS(input=text, type="char"))


@pytest.mark.parametrize("text", ["w", "W", "q", "r", "中文"])
@pytest.mark.parametrize("active", [False, True])
def test_character_events_preserve_ui_and_never_dispatch_drive_keys(adapter, text, active):
    c = adapter.controller
    if active:
        c._on_input(mouse("left-down"))
    assert c._on_input(character(text)) is (not active or text == "中文")
    assert c.input_counts["key"] == 0
    assert not c.state.pressed and not c.state.reset_requested
    assert c.state.target_yaw == 0.


@pytest.mark.parametrize("name,index,sign", [("W", 0, 1), ("S", 0, -1), ("A", 1, 1),
                                           ("D", 1, -1), ("Q", 2, 1), ("E", 2, -1)])
def test_real_key_character_repeat_release_sequence(adapter, name, index, sign):
    c = adapter.controller
    c._on_input(mouse("left-down"))
    assert c._on_input(key(name)) is False
    assert c._on_input(character(name.lower())) is False
    repeated = key(name)
    repeated.event.type = "repeat"
    assert c._on_input(repeated) is False
    assert c._on_input(character(name.upper())) is False
    for _ in range(20):
        c.apply()
    assert sign * c.env.cmd_buf[0, index] > 0.
    assert c.input_counts["key"] == 1
    assert c._on_input(key(name, down=False)) is False
    assert not c.state.pressed
    assert c.input_counts["key"] == 2
    c._on_input(key("L"))
    c.apply()
    assert not c.env.cmd_buf.any()


def test_viewport_ui_click_activates_keyboard_without_native_mouse_event(adapter):
    c = adapter.controller
    adapter.ui_callbacks["pressed"](200., 200., 0, 0)
    assert c._keyboard_active
    assert c._on_input(key("W")) is False
    assert c._on_input(character("w")) is False
    c.apply()
    assert c.env.cmd_buf[0, 0] == pytest.approx(.8)
    # Kit can also consume clicks into another panel; check actual UI focus.
    adapter.viewport.focused = False
    assert c._on_input(key("W")) is True
    assert c._on_input(character("w")) is True
    c.apply()
    assert not c._keyboard_active and not c.env.cmd_buf.any()


def test_viewport_ui_right_drag_zoom_and_cleanup(adapter):
    c = adapter.controller
    adapter.ui_callbacks["pressed"](200., 200., 1, 0)
    adapter.ui_callbacks["moved"](200., 200., 0, True)
    adapter.ui_callbacks["moved"](300., 190., 0, True)
    assert c.state.target_yaw == pytest.approx(math.radians(-15.))
    assert c.state.pitch_deg == 21.5
    adapter.ui_callbacks["wheel"](0., 1., 0)
    assert c.state.distance < 3.
    adapter.ui_callbacks["released"](300., 190., 1, 0)
    assert not c.state.right_held and c._restore_cursor is None
    assert c.state.command(c.yaw())[2] < 0.
    c.close()
    assert all(callback is None for callback in adapter.ui_callbacks.values())


def test_leaving_viewport_stops_held_keys_without_another_key_event(adapter):
    c = adapter.controller
    adapter.ui_callbacks["pressed"](200., 200., 1, 0)
    c._on_input(key("W"))
    c.apply()
    assert c.env.cmd_buf[0, 0] > 0.
    adapter.viewport.focused = False
    c.apply()
    assert not c.env.cmd_buf.any() and not c.state.pressed
    assert not c._keyboard_active and c._restore_cursor is None


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


def test_keyboard_heading_works_after_viewport_click_without_mouse_capture(adapter):
    c = adapter.controller
    c._on_input(mouse("left-down"))
    assert c._on_input(key("W")) is False
    assert c._on_input(key("E")) is False
    repeated = key("E")
    repeated.event.type = "repeat"
    assert c._on_input(repeated) is False
    for _ in range(20):
        c.apply()
    assert c.env.cmd_buf[0, 0] > 0. and c.env.cmd_buf[0, 2] < 0.
    assert c.state.target_yaw == pytest.approx(math.radians(-6.))
    assert not adapter.modes  # Keyboard driving never captures the cursor.
    assert c._on_input(key("E", down=False)) is False
    target = c.state.target_yaw
    c.apply()
    assert c.state.target_yaw == target and c.env.cmd_buf[0, 2] < 0.
    assert c._on_input(key("L")) is False
    c.apply()
    assert not c.env.cmd_buf.any() and not c.state.pressed


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
