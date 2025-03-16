<div align="center">
  <h1 style="background: linear-gradient(to right, black, white); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin: 0;">
        [Hypervision 📟]
    </h1>
    <br>
    <br>
    <p align="center">
        <img src="https://img.shields.io/github/issues/Raumberg/hypervision?style=for-the-badge">
        <br>
        <img src="https://img.shields.io/github/languages/count/Raumberg/hypervision?style=for-the-badge">
        <img src="https://img.shields.io/github/repo-size/Raumberg/hypervision?style=for-the-badge">
        <br>
    </p>
</div>

**Hypervision** is a realtime object detection system, utilized for target autoaiming.  
In other words, it is neural network based aimbot, capturing screen region, detecting targets and performing mouse actions if needed.  

### ✨ What Does Hypervision Offer?
- 🚀 **Production-Ready Project:** A fully functional, out-of-the-box solution for real-time object detection and auto-aiming. 
- 🔧 **ONNX → TensorRT Conversion Script:** Easily convert your ONNX models to TensorRT for optimized performance.
- ⚙️ **Basic Configuration:** Simple YAML-based setup to get you started in minutes.
- 🧠 **Pre-Trained YOLOv11 Model:** Jumpstart your project with a state-of-the-art, pre-trained detection model.
- **🔗 Extensions**:
  - **💻 C Foreign Function Interfaces (FFI)**: Direct `Windows.h` integration for advanced, low-latency aiming techniques.  
  - **📟 Custom Fused CUDA Kernel**: Zero-copy GPU operations for maximum efficiency and minimal overhead. 

## 🎥 Demo
Check out Hypervision in action:

![Hypervision](assets/demo.gif?raw=true)

### 🛠 Key Features
- 🔭 **YOLOv11 Powered**: Built on the robust YOLOv11 architecture for high accuracy and speed.
- 🚀 **Optimized Performance**: Utilizes TensorRT, kernel fusion, and vectorized operations for lightning-fast processing.
- 📚 **Easy Adaptability**: No darknet or other "magical" libraries! Just pure YOLOv11 from Ultralytics. Train a different model and seamlessly integrate it into the existing script.
- 🕶 **All-in-One Visualization**: A dedicated window displays FPS, detections, confidences, and a central aim dot for enhanced user experience.
- ⚙️ **Out-of-the-Box Usage**: Leverage a simple YAML configuration system for quick setup and customization.

> **⚠️ Disclaimer:**  
> **This software is for educational and experimental purposes only. Use it only where you have explicit permission. Misuse may violate terms of service or laws. The author is not responsible for any consequences.**

### 🛠 Technical Details
- Model was trained on 2x **NVIDIA A100 80GB GPU** for 100 epochs on a custom dataset *(training time: approx. 30 min)*
- Inference/Total script time -  **NVIDIA RTX 4060, Intel Core i5 9600KF**:  
*Inference: 5-8ms*  
*Total: 20-30ms*  
*FPS: 35-40*

## 🎯 Example config:
```yaml
# screen configuration
screen_width: 1920  
screen_height: 1080
# neural network configuration
activation_range: 640 # <- how far from the center we want to detect 
confidence_threshold: 0.2 # <- what is minimal confidence to consider detection a target
nms_threshold: 0.45 # <- threshold for non-maximum suppression
model_path: "models/trt/yolov11.engine" # <- self-explainatory
enable_aim: False # <- whether to enable autoaiming
display: True # <- whether to display separate window with detector info (can be slightly slower)
toggle_button: 'ctrl' # <- what button to use for aiming to target
mouse: 'rel' # <- mouse movement directive (support for: relative (rel) and absolute (abs))
```

## 🔧 Getting started
To get started with **Hypervision**, follow these steps:
1. Obtain `CUDA` and `CUDNN` drivers by following NVIDIA links.
2. Obtain and build `OpenCV` from source (v4.12-dev used by author), linking against CUDA library to enable CUDA processing within OpenCV.
3. Syncronize dependancies in UV venv:
```bash
uv sync
```
4. Check the configuration in config.yaml (you can provide your own by passing an argument to the main):
```bash
python main.py <path-to-your-config.yaml>
```
5. Run the script by
```bash
uv run main.py
```

## 🔧 How to train my own model for my own detection task?
To get started with adapting **Hypervision** to your specific task, the following steps needed:
### 1. Obtain ultralytics library
```bash
pip install ultralytics
```  
Now you will have access not only to the library itself, but also to the yolo CLI.  
### 2. Obtain labeled (or label yourself) a dataset (We used `Roboflow`)  
### 3. Export dataset to a specific format (`YOLOvN` / `RT-DETR` if you want to use RT-DETR)  
### 4. Use `yolo` CLI to train your own model:  
```bash
yolo detect train data=<path-to-your-data.yaml> model=<desired-model> epochs=100 imgsz=640
```  
### 5. Then, export your best.pt model to the ONNX format using:
```bash
yolo export model=<path-to-your-best.pt> format=ONNX
```  
### 6. After obtaining ONNX model, convert it to the .engine (TensorRT) format using the script in:
```bash
# hypervision/scripts/build_engine.py  
# --- \\ ---
python build_engine.py --onnx ../models/onnx/yolov11.onnx --engine ../models/temp/yolov11.engine
```
### 7. Now, provide the path of your TensorRT model to the config.yaml
### 8. Enjoy the ride!

# 🔧 Extensions:
You can also try use the following test extensions of the **Hypervision** if you are really curious:
- **Extern C function calls**: in the `./libc` section there is rewritten mouse aim function calls which utilize direct windows api.  
*You can build those extensions and speed the aim up or use more precise aim processing:*  
```bash
cd hypervision/libc/aim && python ./setup.py build_ext --inplace
```  
Note:  
When building C extension, the compiler will scream for **reinterpret_casts**, but no worries, everything is alright.  
After that, you can use the module in the main Hypervision logic sections.  
- **Custom CUDA kernels**: We built `custom CUDA kernels` to minimize CPU-GPU overhead and provide GPU zero-copy operations.  
*You can build custom cuda kernel using NVCC:*  
```bash
nvcc -o fusion.o -c -arch=<your-GPU-architectuse> --use_fast_math rtFusion.cu
```
To obtain your gpu architecture, please navigate to NVIDIA website.