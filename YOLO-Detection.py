import sys
import os
import cv2
import json
import concurrent.futures
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QComboBox, QSplashScreen,
    QMainWindow, QAction, QFileDialog, QMessageBox, QMenu, QCheckBox, QFrame
)
from PyQt5.QtCore import QTimer, Qt, QPropertyAnimation, QSize
from PyQt5.QtGui import QImage, QPixmap, QIcon
from ultralytics import YOLO
import torch

CONFIG_FILE = 'config.json'
DEFAULT_LOGO = 'logo.png'

def load_config():
    return json.load(open(CONFIG_FILE)) if os.path.exists(CONFIG_FILE) else {}

def save_config(config):
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=4)

def list_usb_cameras(max_devices=3):
    available = []
    for i in range(max_devices):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap and cap.read()[0]:
            available.append(str(i))
        cap.release()
    # Ensure index 0 is always checked and added if working
    if '0' not in available:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if cap and cap.read()[0]:
            available.insert(0, '0')
        cap.release()
    return available

class VideoStreamApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.show_detections = True
        self.available_models = {
            "YOLOv8s": "yolov8s.pt",
            "YOLOv8m": "yolov8m.pt",
            "YOLOv8l": "yolov8l.pt"
        }

        config = load_config()
        self.current_model_name = config.get('model', 'YOLOv8s')
        self.processing_mode = config.get('processing_mode', 'cpu')
        self.model = self.load_model(self.available_models[self.current_model_name], self.processing_mode)

        logo_path = config.get('logo_path', DEFAULT_LOGO)
        self.setWindowIcon(QIcon(logo_path if os.path.exists(logo_path) else DEFAULT_LOGO))
        self.setWindowTitle('Advanced Camera Selector + YOLO Detection')
        self.setGeometry(100, 100, 1200, 800)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.video_label = QLabel('Video feed window')
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet('background-color: black; color: white; font-size: 24px;')

        self.sidebar = QFrame()
        self.sidebar.setFrameShape(QFrame.StyledPanel)
        self.sidebar.setFixedWidth(200)

        sidebar_layout = QVBoxLayout()
        sidebar_layout.setAlignment(Qt.AlignTop)

        self.human_checkbox = QCheckBox("Human Detection")
        self.car_checkbox = QCheckBox("Car Detection")
        self.ship_checkbox = QCheckBox("Ship Detection")
        self.human_count, self.car_count, self.ship_count = QLabel("0"), QLabel("0"), QLabel("0")

        sidebar_layout.addWidget(self.human_checkbox)
        sidebar_layout.addWidget(self.make_icon_row("👤", self.human_count))
        sidebar_layout.addWidget(self.car_checkbox)
        sidebar_layout.addWidget(self.make_icon_row("🚗", self.car_count))
        sidebar_layout.addWidget(self.ship_checkbox)
        sidebar_layout.addWidget(self.make_icon_row("🚢", self.ship_count))
        self.sidebar.setLayout(sidebar_layout)

        self.camera_selector = QComboBox()
        self.camera_selector.addItems(['IP Camera', 'RTSP Camera', 'USB Camera'])
        self.camera_selector.currentTextChanged.connect(self.update_input_field)

        self.input_field = QComboBox()
        self.update_input_field('IP Camera')

        self.start_button = QPushButton('Start Stream')
        self.start_button.clicked.connect(self.start_stream)

        video_layout = QVBoxLayout()
        video_layout.addWidget(self.video_label)
        control_layout = QHBoxLayout()
        control_layout.addWidget(self.camera_selector)
        control_layout.addWidget(self.input_field)
        control_layout.addWidget(self.start_button)
        video_layout.addLayout(control_layout)

        main_layout = QHBoxLayout()
        main_layout.addLayout(video_layout)
        main_layout.addWidget(self.sidebar)
        self.central_widget.setLayout(main_layout)

        menubar = self.menuBar()

        settings_menu = menubar.addMenu('Settings')
        change_logo = QAction('Change Logo', self)
        change_logo.triggered.connect(self.change_logo)
        settings_menu.addAction(change_logo)

        cpu_action = QAction('CPU', self, checkable=True)
        gpu_action = QAction('GPU', self, checkable=True)
        cpu_action.triggered.connect(lambda: self.set_processing_mode('cpu'))
        gpu_action.triggered.connect(lambda: self.set_processing_mode('cuda'))
        settings_menu.addAction(cpu_action)
        settings_menu.addAction(gpu_action)

        save_action = QAction('Save Settings', self)
        save_action.triggered.connect(self.save_current_settings)
        settings_menu.addAction(save_action)

        model_menu = menubar.addMenu("Performance")
        for name in self.available_models.keys():
            action = QAction(name, self)
            action.triggered.connect(lambda _, x=name: self.switch_model(x))
            model_menu.addAction(action)

        self.set_processing_mode(self.processing_mode)
        self.camera_selector.setCurrentText(config.get('camera_type', 'IP Camera'))
        self.input_field.addItem(config.get('camera_input', 'Enter stream URL here...'))
        self.input_field.setCurrentText(config.get('camera_input', 'Enter stream URL here...'))
        self.human_checkbox.setChecked(config.get('detection', {}).get('human', False))
        self.car_checkbox.setChecked(config.get('detection', {}).get('car', False))
        self.ship_checkbox.setChecked(config.get('detection', {}).get('ship', False))

    def make_icon_row(self, text, count_label):
        row = QHBoxLayout()
        icon = QLabel(text)
        row.addWidget(icon)
        row.addWidget(count_label)
        container = QWidget()
        container.setLayout(row)
        return container

    def load_model(self, path, device):
        if device == 'cuda' and not torch.cuda.is_available():
            QMessageBox.warning(self, "CUDA Not Available", "No compatible CUDA device found. Falling back to CPU.")
            device = 'cpu'
        self.processing_mode = device
        return YOLO(path)

    def set_processing_mode(self, mode):
        self.processing_mode = 'cuda' if mode == 'cuda' and torch.cuda.is_available() else 'cpu'

    def switch_model(self, name):
        self.model = self.load_model(self.available_models[name], self.processing_mode)
        self.current_model_name = name
        QMessageBox.information(self, "Model Switched", f"Now using {name}")

    def update_input_field(self, cam_type):
        self.input_field.clear()
        if cam_type == 'USB Camera':
            self.input_field.addItem('Scanning USB cameras...')
            QTimer.singleShot(100, self.populate_usb_cameras)
        else:
            self.input_field.setEditable(True)
            self.input_field.addItem('Enter stream URL here...')

    def populate_usb_cameras(self):
        self.input_field.clear()
        cams = list_usb_cameras()
        if cams:
            self.input_field.addItems(cams)
        else:
            self.input_field.addItem('No USB cameras found')

    def start_stream(self):
        cam_type = self.camera_selector.currentText()
        input_value = self.input_field.currentText().strip()
        if cam_type == 'USB Camera' and input_value.isdigit():
            self.capture = cv2.VideoCapture(int(input_value), cv2.CAP_DSHOW)
        elif cam_type != 'USB Camera' and input_value and 'Enter stream' not in input_value:
            self.capture = cv2.VideoCapture(input_value)
        else:
            self.video_label.setText('Invalid camera input.')
            return
        if self.capture.isOpened():
            self.timer.start(30)
        else:
            self.video_label.setText('Failed to open stream.')

    def update_frame(self):
        ret, frame = self.capture.read()
        if not ret:
            self.video_label.setText('Stream ended or failed.')
            return
        results = self.model.predict(frame, device=self.processing_mode, verbose=False)
        detections = results[0].boxes.cls.cpu().numpy()
        boxes = results[0].boxes.xyxy.cpu().numpy()
        labels = [self.model.names[int(c)] for c in detections]

        human_count = car_count = ship_count = 0
        for box, label in zip(boxes, labels):
            x1, y1, x2, y2 = map(int, box)
            if label == 'person' and self.human_checkbox.isChecked():
                human_count += 1
                if self.show_detections:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                    cv2.putText(frame, 'Person', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
            elif label == 'car' and self.car_checkbox.isChecked():
                car_count += 1
                if self.show_detections:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255,0,0), 2)
                    cv2.putText(frame, 'Car', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2)
            elif label == 'boat' and self.ship_checkbox.isChecked():
                ship_count += 1
                if self.show_detections:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,0,255), 2)
                    cv2.putText(frame, 'Ship', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)

        self.human_count.setText(str(human_count))
        self.car_count.setText(str(car_count))
        self.ship_count.setText(str(ship_count))

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = QImage(rgb_frame, rgb_frame.shape[1], rgb_frame.shape[0], QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(image)
        self.video_label.setPixmap(pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def contextMenuEvent(self, event):
        if self.video_label.underMouse():
            menu = QMenu(self)
            action_4k = menu.addAction("Set 4K (3840x2160)")
            action_fhd = menu.addAction("Set Full HD (1920x1080)")
            action_hd = menu.addAction("Set HD (1280x720)")
            toggle_lines = menu.addAction(f"Toggle Detection Lines ({'ON' if self.show_detections else 'OFF'})")
            action = menu.exec_(self.mapToGlobal(event.pos()))
            if action == action_4k:
                self.resize_video_window(3840, 2160)
            elif action == action_fhd:
                self.resize_video_window(1920, 1080)
            elif action == action_hd:
                self.resize_video_window(1280, 720)
            elif action == toggle_lines:
                self.show_detections = not self.show_detections

    def resize_video_window(self, width, height):
        self.video_label.setFixedSize(QSize(width, height))
        self.adjustSize()

    def save_current_settings(self):
        config = {
            'logo_path': load_config().get('logo_path', DEFAULT_LOGO),
            'camera_type': self.camera_selector.currentText(),
            'camera_input': self.input_field.currentText(),
            'processing_mode': self.processing_mode,
            'model': self.current_model_name,
            'detection': {
                'human': self.human_checkbox.isChecked(),
                'car': self.car_checkbox.isChecked(),
                'ship': self.ship_checkbox.isChecked()
            }
        }
        save_config(config)
        QMessageBox.information(self, "Settings Saved", "Your settings have been saved.")

    def change_logo(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select New Logo", "", "Images (*.png *.jpg *.jpeg)")
        if file_path:
            config = load_config()
            config['logo_path'] = file_path
            save_config(config)
            QMessageBox.information(self, "Logo Changed", "Restart app to apply icon/logo change.")

    def closeEvent(self, event):
        if hasattr(self, 'capture'):
            self.capture.release()
        event.accept()

def show_splash_and_start():
    app = QApplication(sys.argv)
    config = load_config()
    logo_path = config.get('logo_path', DEFAULT_LOGO)
    splash_pix = QPixmap(logo_path if os.path.exists(logo_path) else DEFAULT_LOGO)
    splash = QSplashScreen(splash_pix, Qt.WindowStaysOnTopHint)
    splash.setMask(splash_pix.mask())
    splash.show()
    splash.setWindowOpacity(0.0)
    anim = QPropertyAnimation(splash, b"windowOpacity")
    anim.setDuration(5000)
    anim.setStartValue(0.0)
    anim.setEndValue(1.0)
    anim.start()
    QTimer.singleShot(5000, splash.close)
    window = VideoStreamApp()
    QTimer.singleShot(5000, window.show)
    sys.exit(app.exec_())

if __name__ == '__main__':
    show_splash_and_start()
