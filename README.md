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
git clone https://github.com/m7mds91/YOLO-Detection.git
cd YOLO-Detection

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate

# Install required packages
pip install -r requirements.txt
```

🛠 Example requirements.txt
```bash
PyQt5==5.15.9
opencv-python>=4.8.0
ultralytics>=8.0.0
torch>=2.0.0
```

🚀 Run the app
```bash
python gui.py
```

⚙️ Configuration
```bash
Logo icon: Set/change via the menu (Settings → Change Logo)
```

Model: 
```bash
Switch via menu (Performance → YOLOv8s / YOLOv8m / YOLOv8l)
Processing device: CPU or GPU (if available)
Saved settings: Stored in config.json in the project folder
```
📸 Supported Cameras
| Type            | Example Entry |
|---------------------|-----------------------|
| USB Camera             | 0 (default), 1, 2…           |
| IP Camera             | (http://192.168.x.x:port/path)    |
| RTSP Stream             |rtsp://192.168.x.x:port/path    |

💻 Notes
```bash
For GPU support, ensure PyTorch with CUDA is installed matching your system and GPU drivers.
Check: PyTorch Install Guide
YOLOv8 models (.pt files) should be placed in the project directory or adjust the script paths.
```

🌟 Credits

YOLOv8: [Ultralytics](https://github.com/ultralytics/ultralytics)
PyQt5 GUI:  [Riverbank Computing](https://riverbankcomputing.com/software/pyqt/intro)
