import numpy as np
import win32gui, win32ui, win32con, win32api
import cv2

class VisionSystem:
    """
    Handles screen capture and image processing using Windows API
    """
    
    def __init__(self, screen_width: int, screen_height: int, activation_range: int):
        """
        Initialize vision system
        
        Args:
            screen_width: Screen width in pixels
            screen_height: Screen height in pixels
            activation_range: Detection area size
        """
        self.region = self._calculate_region(screen_width, screen_height, activation_range)

    def _calculate_region(self, width: int, height: int, activation_range: int) -> tuple:
        """Calculate screen capture region as tuple of integers"""
        left = width//2 - activation_range//2
        top = height//2 - activation_range//2
        return (
            left,
            top,
            left + activation_range,
            top + activation_range
        )

    def capture_frame(self) -> np.ndarray:
        """Capture and preprocess game frame using Windows API"""
        return self.grab_screen(self.region)

    def grab_screen(self, region: tuple) -> np.ndarray:
        """
        Capture screen region using Windows API (matches original implementation)
        Args:
            region: Tuple (left, top, right, bottom)
        Returns:
            numpy array in BGR format
        """
        left, top, right, bottom = region
        width = right - left
        height = bottom - top

        hwin = win32gui.GetDesktopWindow()
        hwindc = win32gui.GetWindowDC(hwin)
        srcdc = win32ui.CreateDCFromHandle(hwindc)
        memdc = srcdc.CreateCompatibleDC()
        bmp = win32ui.CreateBitmap()
        bmp.CreateCompatibleBitmap(srcdc, width, height)
        memdc.SelectObject(bmp)
        memdc.BitBlt((0, 0), (width, height), srcdc, (left, top), win32con.SRCCOPY)

        signed_ints_array = bmp.GetBitmapBits(True)
        img = np.frombuffer(signed_ints_array, dtype='uint8')
        img.shape = (height, width, 4)

        # Cleanup
        srcdc.DeleteDC()
        memdc.DeleteDC()
        win32gui.ReleaseDC(hwin, hwindc)
        win32gui.DeleteObject(bmp.GetHandle())

        # Convert BGRA to BGR to match original implementation
        return cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)