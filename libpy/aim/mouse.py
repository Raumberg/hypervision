import ctypes
import pynput
from ctypes import (
    c_ulong     as ulong,
    c_long      as long,
    c_ushort    as ushort,
    c_short     as short,
    Structure   as struct,
    Union       as union,
    pointer,
    c_void_p    as voidptr,

    byref,
    sizeof,

    POINTER     as PTR,
    cast
)

ptr = PTR(ulong)
SendInput = pynput._util.win32.SendInput

class KeyBdInput(struct):
    _fields_ = [("wVk",         ushort),
                ("wScan",       ushort),
                ("dwFlags",     ulong),
                ("time",        ulong),
                ("dwExtraInfo", voidptr)]
    
class HardwareInput(struct):
    _fields_ = [("uMsg",        ulong),
                ("wParamL",     short),
                ("wParamH",     ushort)]
    
class MouseInput(struct):
    _fields_ = [("dx",          long),
                ("dy",          long),
                ("mouseData",   ulong),
                ("dwFlags",     ulong),
                ("time",        ulong),
                ("dwExtraInfo", voidptr)]
    
class Input_I(union):
    _fields_ = [("ki",          KeyBdInput),
                ("mi",          MouseInput),
                ("hi",          HardwareInput)]
    
class Input(struct):
    _fields_ = [("type",        ulong),
                ("ii",          Input_I)]


class MouseController:
    """
    Handles mouse input simulation using Windows API
    """
    
    def __init__(self, screen_width: int, screen_height: int):
        """
        Initialize mouse controller
        
        Args:
            screen_width: Screen width in pixels
            screen_height: Screen height in pixels
        """
        self.screen_width = screen_width
        self.screen_height = screen_height

    def set_abs_position(self, x: int, y: int) -> None:
        """
        Set absolute mouse position
        
        Args:
            x: Target X coordinate
            y: Target Y coordinate
        """
        absolute_x = int((x / self.screen_width) * 65535)
        absolute_y = int((y / self.screen_height) * 65535)

        nullptr = ulong(0)
        mouse_input = MouseInput(
            dx=absolute_x,
            dy=absolute_y,
            mouseData=0,
            dwFlags=0x0001 | 0x8000,  # MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_MOVE
            time=0,
            dwExtraInfo=cast(pointer(nullptr), voidptr)
        )

        input_union = Input_I(mi=mouse_input)
        input_struct = Input(type=0, ii=input_union)
        SendInput(1, byref(input_struct), sizeof(input_struct))

    def set_rel_position(self, x: int, y: int, sensitivity: float) -> None:
        """Move mouse relative to screen center using deltas"""
        screen_center_x = self.screen_width // 2
        screen_center_y = self.screen_height // 2
        
        # # # Calculate delta from screen center to target
        delta_x = int((x - screen_center_x) * sensitivity if sensitivity else 1)
        delta_y = int((y - screen_center_y) * sensitivity if sensitivity else 1)

        # Prepare relative mouse movement input
        nullptr = ulong(0)
        mouse_input = MouseInput(
            dx=delta_x,
            dy=delta_y,
            mouseData=0,
            dwFlags=0x0001,  # MOUSEEVENTF_MOVE (relative movement)
            time=0,
            dwExtraInfo=cast(pointer(nullptr), voidptr)
        )

        input_union = Input_I(mi=mouse_input)
        input_struct = Input(type=0, ii=input_union)
        SendInput(1, byref(input_struct), sizeof(input_struct))