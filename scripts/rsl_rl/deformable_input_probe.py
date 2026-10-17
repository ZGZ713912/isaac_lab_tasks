"""Bounded diagnostic of the installed Kit input provider and cursor API."""

import copy
import math
from types import SimpleNamespace

import carb.input
from omni.kit.viewport.utility import get_active_viewport_window


def probe_kit_input(controller):
    """Inject private virtual devices; consume their events before UI navigation."""
    c = controller
    if c._subscription is not None:
        raise ValueError("Input API probe requires isolated --teleop-replay")
    provider = carb.input.acquire_input_provider()
    keyboard = provider.create_keyboard("deformable-probe-keyboard")
    mouse = provider.create_mouse("deformable-probe-mouse")
    saved = (c.app_window, c.live_input, copy.deepcopy(c.state), c._keyboard_active, c.input_counts.copy())
    focused = SimpleNamespace(value=True)
    c.app_window = SimpleNamespace(is_focused=lambda: focused.value, get_window=saved[0].get_window)
    # Virtual devices do not change the real UI's focus. Exercise a controlled
    # viewport focus here; physical window focus needs a separate GUI check.
    c._viewport_focused = lambda: focused.value
    c.live_input = True
    cursor_before = tuple(c.windowing.get_cursor_position(c.app_window.get_window()))
    propagated = []
    keyboard_event_types = {}
    character_input_types = set()
    drive_commands = {}

    def own(event):
        return event.device == keyboard or event.device == mouse

    def handle(event):
        if not own(event):
            return True
        if event.deviceType == carb.input.DeviceType.KEYBOARD:
            raw = event.event
            keyboard_event_types[raw.type.name] = keyboard_event_types.get(raw.type.name, 0) + 1
            if raw.type == carb.input.KeyboardEventType.CHAR:
                character_input_types.add(type(raw.input).__name__)
        return c._on_input(event)

    handler = c.input.subscribe_to_input_events(handle, order=0)
    observer = c.input.subscribe_to_input_events(lambda event: (propagated.append(event) or True)
                                               if own(event) else True, order=1)
    try:
        widget = get_active_viewport_window().get_frame("deformable-third-person-input")
        assert widget.computed_width > 0 and widget.computed_height > 0, "Viewport has no laid-out bounds"
        xy = (widget.screen_position_x + widget.computed_width/2,
              widget.screen_position_y + widget.computed_height/2)
        assert c._over_viewport(xy)

        def send_mouse(kind, position=xy, amount=None):
            values = (0., amount) if amount is not None else (.5, .5)
            provider.buffer_mouse_event(mouse, kind, values, 0, position)
            c.input.distribute_buffered_events()

        def send_key(key, down):
            kind = carb.input.KeyboardEventType.KEY_PRESS if down else carb.input.KeyboardEventType.KEY_RELEASE
            provider.buffer_keyboard_key_event(keyboard, kind, key, 0)
            if down and len(key.name) == 1:
                provider.buffer_keyboard_char_event(keyboard, key.name.lower(), 0)
            c.input.distribute_buffered_events()

        c.state.reset(c.yaw())
        start_yaw = c.state.target_yaw
        send_mouse(carb.input.MouseEventType.RIGHT_BUTTON_DOWN)
        assert c.state.right_held and c._restore_cursor is not None
        send_mouse(carb.input.MouseEventType.MOVE)
        send_mouse(carb.input.MouseEventType.MOVE, (xy[0]+100., xy[1]-10.))
        assert abs(c.state.target_yaw - start_yaw + math.radians(15.)) < 1.e-6
        assert c.state.pitch_deg == saved[2].pitch_deg + 1.5
        send_mouse(carb.input.MouseEventType.RIGHT_BUTTON_UP)
        assert not c.state.right_held and c._restore_cursor is None
        assert c.state.command(c.yaw())[2] < 0.
        send_mouse(carb.input.MouseEventType.SCROLL, amount=1.)
        assert c.state.distance < saved[2].distance
        # A plain viewport click must enable keyboard driving without RMB.
        c._keyboard_active = False
        send_mouse(carb.input.MouseEventType.LEFT_BUTTON_DOWN)
        assert c._keyboard_active
        propagated.clear()  # The activation click may also focus the viewport.
        for name, index, sign in (("W", 0, 1), ("S", 0, -1), ("A", 1, 1),
                                  ("D", 1, -1), ("Q", 2, 1), ("E", 2, -1)):
            c.state.reset(c.yaw())
            key = getattr(carb.input.KeyboardInput, name)
            before = c.input_counts["key"]
            send_key(key, True)
            provider.buffer_keyboard_key_event(keyboard, carb.input.KeyboardEventType.KEY_REPEAT, key, 0)
            provider.buffer_keyboard_char_event(keyboard, name, 0)
            c.input.distribute_buffered_events()
            c.state.advance(.1)
            command = c.state.command(c.yaw())
            assert sign * command[index] > 0., (name, command)
            assert c.input_counts["key"] == before + 1, "CHAR/repeat dispatched an extra command"
            drive_commands[name] = list(command)
            send_key(key, False)
            assert not c.state.pressed
        c.state.reset(c.yaw())
        send_key(carb.input.KeyboardInput.Q, True)
        c.state.advance(.1)
        assert c.state.command(c.yaw())[2] > 0.
        send_key(carb.input.KeyboardInput.Q, False)
        target = c.state.target_yaw
        c.state.advance(.1)
        assert c.state.target_yaw == target
        send_key(carb.input.KeyboardInput.E, True)
        c.state.advance(.2)
        assert c.state.command(c.yaw())[2] < 0.
        send_key(carb.input.KeyboardInput.E, False)
        send_key(carb.input.KeyboardInput.W, True)
        send_key(carb.input.KeyboardInput.W, False)
        send_key(carb.input.KeyboardInput.R, True)
        assert c.state.reset_requested
        send_key(carb.input.KeyboardInput.R, False)
        send_key(carb.input.KeyboardInput.ESCAPE, True)
        assert c.state.suspended and not c.state.reset_requested
        assert not propagated, "A driving event leaked to subsequent viewport navigation handlers"
        return dict(passed=True, virtual_device_events=c.input_counts.copy(),
                    keyboard_event_types=keyboard_event_types,
                    character_input_types=sorted(character_input_types),
                    keyboard_only_viewport_activation=True, drive_commands=drive_commands,
                    later_navigation_events=len(propagated),
                    cursor_restored=tuple(c.windowing.get_cursor_position(c.app_window.get_window())) == cursor_before,
                    viewport_bounds=[widget.screen_position_x, widget.screen_position_y,
                                     widget.computed_width, widget.computed_height])
    finally:
        c.release_mouse()
        c.input.unsubscribe_to_input_events(observer)
        c.input.unsubscribe_to_input_events(handler)
        provider.destroy_mouse(mouse)
        provider.destroy_keyboard(keyboard)
        del c._viewport_focused
        c.app_window, c.live_input, c.state, c._keyboard_active, c.input_counts = saved
