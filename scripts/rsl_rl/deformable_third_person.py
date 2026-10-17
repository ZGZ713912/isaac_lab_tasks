"""Kit input adapter and world-up camera for physical third-person driving."""

import weakref
import torch

import carb.input
import carb.windowing
import omni.appwindow
from omni.kit.viewport.utility import get_active_viewport_window

from deformable_third_person_state import ThirdPersonState, quaternion_yaw


class ThirdPersonController:
    def __init__(self, env, *, vx_max=.8, vy_max=.5, wz_max=1.5, live_input=True):
        self.wrapper = env
        self.env = env.unwrapped
        self.state = ThirdPersonState(vx_max=vx_max, vy_max=vy_max, wz_max=wz_max)
        self.app_window = omni.appwindow.get_default_app_window()
        self.windowing = carb.windowing.acquire_windowing_interface()
        self.input = carb.input.acquire_input_interface()
        self.live_input = live_input
        self._subscription = None
        self._last_mouse = self._restore_cursor = None
        self._keyboard_active = False
        self._viewport_frame = None
        viewport = get_active_viewport_window()
        if viewport is not None:
            self._viewport_frame = viewport.get_frame("deformable-third-person-input")
            if live_input:
                # Kit's UI consumes physical viewport clicks before they reach
                # carb.input. Bind that UI path as well as the native input path.
                self._viewport_frame.set_mouse_pressed_fn(
                    lambda x, y, button, flags, obj=weakref.proxy(self): obj._on_viewport_press(x, y, button, flags))
                self._viewport_frame.set_mouse_released_fn(
                    lambda x, y, button, flags, obj=weakref.proxy(self): obj._on_viewport_release(x, y, button, flags))
                self._viewport_frame.set_mouse_moved_fn(
                    lambda x, y, *args, obj=weakref.proxy(self): obj._move_mouse((x, y)))
                self._viewport_frame.set_mouse_wheel_fn(
                    lambda x, y, flags, obj=weakref.proxy(self): obj.dispatch(dict(type="scroll", amount=y)))
        self._closed = False
        self.camera_samples = []
        self._camera_sample_step = -1000
        self.input_counts = dict(key=0, move=0, right=0, scroll=0)
        self.reset()
        if live_input:
            self._subscription = self.input.subscribe_to_input_events(
                lambda event, obj=weakref.proxy(self): obj._on_input(event), order=0)

    def __str__(self):
        return ("第三人称驾驶：W/S 前后，A/D 横移；按住 Q/E 向左/右转车头，松开后完成最后朝向；"
                "相机始终跟随实际车头。右键鼠标也可转向/调俯仰；滚轮缩放；"
                "L 停止，R 重置，Esc 释放输入。请先点击场地视口。")

    def yaw(self):
        return quaternion_yaw(self.env.robot.data.root_link_quat_w[0].tolist())

    def reset(self):
        self.release_mouse()
        self.state.reset(self.yaw())

    def capture_mouse(self):
        if not self.live_input or self._restore_cursor is not None:
            return
        window = self.app_window.get_window()
        self._restore_cursor = self.windowing.get_cursor_position(window)
        self.windowing.set_cursor_mode(window, carb.windowing.CursorMode.DISABLED)
        self._last_mouse = None

    def release_mouse(self):
        if self._restore_cursor is not None:
            window = self.app_window.get_window()
            self.windowing.set_cursor_mode(window, carb.windowing.CursorMode.NORMAL)
            self.windowing.set_cursor_position(window, self._restore_cursor)
            self._restore_cursor = None
        self._last_mouse = None

    def _over_viewport(self, position):
        window = get_active_viewport_window()
        if window is None or window.viewport_widget is None:
            return False
        # ViewportWidget wraps a private ui.Frame. Use Kit's public overlay
        # frame API to get the actual viewport's screen bounds instead.
        widget = window.get_frame("deformable-third-person-input")
        x, y = position
        return (widget.screen_position_x <= x <= widget.screen_position_x + widget.computed_width
                and widget.screen_position_y <= y <= widget.screen_position_y + widget.computed_height)

    def _viewport_focused(self):
        viewport = get_active_viewport_window()
        return viewport is not None and viewport.focused

    def _on_viewport_press(self, x, y, button, flags):
        if button in (0, 1):
            self._keyboard_active = True
        if button == 1 and not self.state.right_held:
            self.dispatch(dict(type="right", down=True))
            self.capture_mouse()

    def _on_viewport_release(self, x, y, button, flags):
        if button == 1 and self.state.right_held:
            self.dispatch(dict(type="right", down=False))
            self.release_mouse()

    def _move_mouse(self, position):
        if not self.state.right_held:
            return
        if self._last_mouse is not None:
            self.dispatch(dict(type="move", dx=position[0]-self._last_mouse[0],
                               dy=position[1]-self._last_mouse[1]))
        self._last_mouse = tuple(position)

    def _on_input(self, event):
        if not self.app_window.is_focused():
            return True
        if event.deviceType == carb.input.DeviceType.KEYBOARD:
            if self._keyboard_active and not self._viewport_focused():
                self._keyboard_active = False
                self.state.stop(self.yaw(), suspend=True)
                self.release_mouse()
            raw = event.event
            # CHAR carries text, whereas key press/repeat/release carry a
            # KeyboardInput enum. Physical typing emits both kinds of event.
            name = raw.input.upper() if isinstance(raw.input, str) else raw.input.name
            down = raw.type == carb.input.KeyboardEventType.KEY_PRESS
            if name in ("W", "S", "A", "D", "Q", "E", "L", "R", "ESCAPE", "X", "Z") and self._keyboard_active:
                if raw.type in (carb.input.KeyboardEventType.KEY_PRESS, carb.input.KeyboardEventType.KEY_RELEASE):
                    self.dispatch(dict(type="key", name=name, down=down))
                    if name in ("ESCAPE", "L", "R") and down:
                        self.release_mouse()
                # Consume characters and auto-repeat too, without dispatching a
                # second command or allowing the viewport's shortcuts to run.
                return False
        elif event.deviceType == carb.input.DeviceType.MOUSE:
            raw = event.event
            et = raw.type
            if et == carb.input.MouseEventType.RIGHT_BUTTON_DOWN and self._over_viewport(raw.pixel_coords):
                self._keyboard_active = True
                self.dispatch(dict(type="right", down=True))
                self.capture_mouse()
                return False
            if et == carb.input.MouseEventType.RIGHT_BUTTON_UP and self.state.right_held:
                self.dispatch(dict(type="right", down=False))
                self.release_mouse()
                return False
            if et == carb.input.MouseEventType.MOVE and self.state.right_held:
                self._move_mouse(raw.pixel_coords)
                return False
            if et == carb.input.MouseEventType.SCROLL and (self.state.right_held or self._over_viewport(raw.pixel_coords)):
                self.dispatch(dict(type="scroll", amount=raw.scrollDelta[1]))
                return False
            if et == carb.input.MouseEventType.LEFT_BUTTON_DOWN:
                self._keyboard_active = self._over_viewport(raw.pixel_coords)
                if not self._keyboard_active:
                    self.state.stop(self.yaw(), suspend=True)
                    self.release_mouse()
        return True

    def dispatch(self, event):
        """Live events and bounded replay share this exact control state path."""
        kind = event["type"]
        self.input_counts[kind] += 1
        if kind == "key":
            self.state.key(event["name"], event["down"], self.yaw())
        elif kind == "right":
            self.state.right_button(event["down"])
        elif kind == "move":
            self.state.mouse_move(event["dx"], event["dy"])
        elif kind == "scroll":
            self.state.scroll(event["amount"])

    def apply(self):
        if self.live_input and (not self.app_window.is_focused()
                               or (self._keyboard_active and not self._viewport_focused())):
            self._keyboard_active = False
            self.state.stop(self.yaw(), suspend=True)
            self.release_mouse()
        if self.state.reset_requested:
            self.state.reset_requested = False
            # IsaacLab materializes some articulation buffers inside env.step's
            # inference context; resetting those tensors needs the same context.
            with torch.inference_mode():
                obs, _ = self.wrapper.reset()
            self.reset()
            return obs
        self.state.advance(self.env.step_dt)
        yaw_rate = float(self.env.robot.data.root_ang_vel_b[0, 2])
        self.env.cmd_buf[:] = self.env.cmd_buf.new_tensor(self.state.command(self.yaw(), yaw_rate))
        return None

    def update_camera(self):
        cast = self.env.field_surface.segment if hasattr(self.env, "field_surface") else None
        position = self.env.robot.data.root_link_pos_w[0].tolist()
        yaw = self.yaw()
        eye, pivot = self.state.camera_pose(position, yaw, cast)
        self.env.viewport_camera_controller.update_view_location(eye=eye, lookat=pivot)
        step = self.env.common_step_counter
        sample_period = max(1, round(.1 / self.env.step_dt))
        if len(self.camera_samples) < 256 and step - self._camera_sample_step >= sample_period:
            self._camera_sample_step = step
            self.camera_samples.append(dict(step=self.env.common_step_counter, yaw=yaw,
                target_yaw=self.state.target_yaw, eye=eye.tolist(), pivot=pivot.tolist(),
                pitch_deg=self.state.pitch_deg, distance_requested=self.state.distance,
                right_held=self.state.right_held))

    def audit(self):
        return dict(input_counts=self.input_counts, target_yaw=self.state.target_yaw,
                    camera_samples=self.camera_samples, live_input=self.live_input,
                    keyboard_turn_rate_rad_s=min(self.state.keyboard_turn_rate, self.state.wz_max),
                    behavior="Hold Q/E to change heading; release keeps the final target; camera follows actual yaw")

    def close(self):
        if self._closed:
            return
        self.release_mouse()
        if self._subscription is not None:
            self.input.unsubscribe_to_input_events(self._subscription)
            self._subscription = None
        if self._viewport_frame is not None and self.live_input:
            self._viewport_frame.set_mouse_pressed_fn(None)
            self._viewport_frame.set_mouse_released_fn(None)
            self._viewport_frame.set_mouse_moved_fn(None)
            self._viewport_frame.set_mouse_wheel_fn(None)
        self._closed = True
