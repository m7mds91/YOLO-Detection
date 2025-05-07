# Advanced Camera YOLO Detection

This project is a Python PyQt5-based graphical interface that:
✅ Selects between IP, RTSP, or USB cameras  
✅ Runs YOLOv8 object detection (supports small, medium, large models)  
✅ Displays human, car, and ship detections with count icons  
✅ Supports GPU (CUDA) or CPU inference  
✅ Saves user settings and last-used camera configuration  
✅ Includes a splash screen with configurable app logo  

---

## ✨ Features

- ✅ Real-time video stream from USB, IP, or RTSP cameras  
- ✅ Model switcher (YOLOv8s, YOLOv8m, YOLOv8l) for performance vs. accuracy  
- ✅ Sidebar with toggleable object detections: human, car, ship  
- ✅ Right-click on video window: toggle detection lines, change resolution (HD, FHD, 4K)  
- ✅ Menu bar: switch processing mode (CPU/GPU), change logo, save settings  
- ✅ Configuration saved in `config.json` for session persistence  

---

## 📦 Requirements

| Package            | Version (recommended) |
|---------------------|-----------------------|
| Python             | 3.9 or 3.10           |
| PyQt5             | ≥ 5.15                |
| OpenCV (`opencv-python`) | ≥ 4.x          |
| Ultralytics (YOLOv8) | ≥ 8.x               |
| Torch (PyTorch)    | GPU or CPU version (matching your hardware + CUDA if GPU) |

---

### 🔧 Install dependencies

```bash
# Clone the repo
git clone https://github.com/your-username/advanced-camera-yolo-gui.git
cd advanced-camera-yolo-gui

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate

# Install required packages
pip install -r requirements.txt
