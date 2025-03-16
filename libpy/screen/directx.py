# Windows DirectX example using DXCam
import dxcam
camera = dxcam.create(output_idx=0, output_color="BGR")
camera.start(target_fps=240, video_mode=True)  # Direct GPU texture access

def capture_frame():
    return camera.get_latest_frame()  # Returns GPU-resident texture