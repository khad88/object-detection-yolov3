import os
import sys
import subprocess
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QLabel, 
                             QVBoxLayout, QHBoxLayout, QWidget, QGroupBox, 
                             QLineEdit, QFileDialog, QMessageBox, QTabWidget,
                             QComboBox, QSlider, QSpinBox, QDoubleSpinBox)
from PyQt5.QtGui import QPixmap, QImage, QFont, QIcon
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import os
os.environ["QT_LOGGING_RULES"] = "qt.qpa.fonts.warning=false"

class RequirementsChecker:
    @staticmethod
    def check_requirements():
        try:
            import cv2
            import numpy as np
            import imutils
        except ImportError as e:
            error_msg = f"Thiếu thư viện: {e}\n\n"
            error_msg += "Vui lòng cài đặt các thư viện cần thiết bằng lệnh:\n"
            error_msg += "pip install opencv-python numpy imutils PyQt5"
            QMessageBox.critical(None, "Lỗi Thư Viện", error_msg)
            return False

        # Kiểm tra file model YOLOv3
        yolo_files = [
            os.path.join("yolo-coco", "coco.names"),
            os.path.join("yolo-coco", "yolov3.cfg"),
            os.path.join("yolo-coco", "yolov3.weights")
        ]
        
        missing_files = []
        for file_path in yolo_files:
            if not os.path.exists(file_path):
                missing_files.append(file_path)
        
        if missing_files:
            error_msg = "Không tìm thấy các file sau:\n"
            for file in missing_files:
                error_msg += f"• {file}\n"
            error_msg += "\nVui lòng tải YOLOv3 và giải nén vào thư mục yolo-coco:\n"
            error_msg += "1. Tải file weights: https://pjreddie.com/media/files/yolov3.weights\n"
            error_msg += "2. Tải file cfg: https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg\n"
            error_msg += "3. Tải file coco.names: https://github.com/pjreddie/darknet/blob/master/data/coco.names"
            QMessageBox.critical(None, "Lỗi File YOLO", error_msg)
            return False
            
        return True


class ImageDetectionTab(QWidget):
    """Tab nhận diện đối tượng trong ảnh"""
    def __init__(self):
        super().__init__()
        self.init_ui()
        
    def init_ui(self):
        # Thiết lập layout
        main_layout = QVBoxLayout()
        
        # Nhóm input
        input_group = QGroupBox("Ảnh Đầu Vào")
        input_layout = QHBoxLayout()
        
        self.input_path = QLineEdit()
        self.input_path.setPlaceholderText("Đường dẫn đến file ảnh...")
        
        browse_btn = QPushButton("Chọn File")
        browse_btn.clicked.connect(self.browse_image)
        
        input_layout.addWidget(self.input_path)
        input_layout.addWidget(browse_btn)
        input_group.setLayout(input_layout)
        
        # Nhóm output
        output_group = QGroupBox("Ảnh Đầu Ra")
        output_layout = QHBoxLayout()
        
        self.output_path = QLineEdit()
        self.output_path.setPlaceholderText("Đường dẫn lưu kết quả (để trống nếu không muốn lưu)...")
        
        browse_output_btn = QPushButton("Chọn Vị Trí")
        browse_output_btn.clicked.connect(self.browse_output)
        
        output_layout.addWidget(self.output_path)
        output_layout.addWidget(browse_output_btn)
        output_group.setLayout(output_layout)
        
        # Nhóm tham số
        params_group = QGroupBox("Tham Số")
        params_layout = QVBoxLayout()
        
        conf_layout = QHBoxLayout()
        conf_label = QLabel("Ngưỡng Tin Cậy:")
        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setRange(1, 100)
        self.conf_slider.setValue(50)  # Mặc định 0.5
        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.01, 1.0)
        self.conf_spin.setSingleStep(0.01)
        self.conf_spin.setValue(0.5)
        
        # Kết nối slider và spinbox
        self.conf_slider.valueChanged.connect(lambda val: self.conf_spin.setValue(val/100))
        self.conf_spin.valueChanged.connect(lambda val: self.conf_slider.setValue(int(val*100)))
        
        conf_layout.addWidget(conf_label)
        conf_layout.addWidget(self.conf_slider)
        conf_layout.addWidget(self.conf_spin)
        
        thresh_layout = QHBoxLayout()
        thresh_label = QLabel("Ngưỡng NMS:")
        self.thresh_slider = QSlider(Qt.Horizontal)
        self.thresh_slider.setRange(1, 100)
        self.thresh_slider.setValue(30)  # Mặc định 0.3
        self.thresh_spin = QDoubleSpinBox()
        self.thresh_spin.setRange(0.01, 1.0)
        self.thresh_spin.setSingleStep(0.01)
        self.thresh_spin.setValue(0.3)
        
        # Kết nối slider và spinbox
        self.thresh_slider.valueChanged.connect(lambda val: self.thresh_spin.setValue(val/100))
        self.thresh_spin.valueChanged.connect(lambda val: self.thresh_slider.setValue(int(val*100)))
        
        thresh_layout.addWidget(thresh_label)
        thresh_layout.addWidget(self.thresh_slider)
        thresh_layout.addWidget(self.thresh_spin)
        
        params_layout.addLayout(conf_layout)
        params_layout.addLayout(thresh_layout)
        params_group.setLayout(params_layout)
        
        # Nút thực hiện
        run_btn = QPushButton("Thực Hiện Nhận Diện")
        run_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        run_btn.setMinimumHeight(40)
        run_btn.clicked.connect(self.run_detection)
        
        # Thêm các layout vào layout chính
        main_layout.addWidget(input_group)
        main_layout.addWidget(output_group)
        main_layout.addWidget(params_group)
        main_layout.addWidget(run_btn)
        main_layout.addStretch()
        
        self.setLayout(main_layout)
    
    def browse_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Chọn File Ảnh", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")
        if file_path:
            self.input_path.setText(file_path)
    
    def browse_output(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Chọn Vị Trí Lưu", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")
        if file_path:
            self.output_path.setText(file_path)
    
    def run_detection(self):
        input_path = self.input_path.text().strip()
        if not input_path:
            QMessageBox.warning(self, "Thiếu Thông Tin", "Vui lòng chọn file ảnh đầu vào.")
            return
            
        if not os.path.exists(input_path):
            QMessageBox.critical(self, "Lỗi", f"Không tìm thấy file {input_path}")
            return
            
        confidence = self.conf_spin.value()
        threshold = self.thresh_spin.value()
        output_path = self.output_path.text().strip()
        
        output_arg = f"-o {output_path}" if output_path else ""
        
        command = f"python image_detection.py -i {input_path} -c {confidence} -t {threshold} {output_arg}"
        try:
            subprocess.run(command, shell=True, check=True)
            QMessageBox.information(self, "Thành Công", "Nhận diện đối tượng đã hoàn tất!")
        except subprocess.CalledProcessError:
            QMessageBox.critical(self, "Lỗi", "Đã xảy ra lỗi trong quá trình nhận diện đối tượng.")


class VideoDetectionTab(QWidget):
    """Tab nhận diện đối tượng trong video"""
    def __init__(self):
        super().__init__()
        self.init_ui()
        
    def init_ui(self):
        # Thiết lập layout
        main_layout = QVBoxLayout()
        
        # Nhóm input
        input_group = QGroupBox("Video Đầu Vào")
        input_layout = QHBoxLayout()
        
        self.input_path = QLineEdit()
        self.input_path.setPlaceholderText("Đường dẫn đến file video...")
        
        browse_btn = QPushButton("Chọn File")
        browse_btn.clicked.connect(self.browse_video)
        
        input_layout.addWidget(self.input_path)
        input_layout.addWidget(browse_btn)
        input_group.setLayout(input_layout)
        
        # Nhóm output
        output_group = QGroupBox("Video Đầu Ra")
        output_layout = QHBoxLayout()
        
        self.output_path = QLineEdit()
        self.output_path.setText("output.mp4")
        
        browse_output_btn = QPushButton("Chọn Vị Trí")
        browse_output_btn.clicked.connect(self.browse_output)
        
        output_layout.addWidget(self.output_path)
        output_layout.addWidget(browse_output_btn)
        output_group.setLayout(output_layout)
        
        # Nhóm tham số nhận diện
        params_group = QGroupBox("Tham Số Nhận Diện")
        params_layout = QVBoxLayout()
        
        conf_layout = QHBoxLayout()
        conf_label = QLabel("Ngưỡng Tin Cậy:")
        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setRange(1, 100)
        self.conf_slider.setValue(50)  # Mặc định 0.5
        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.01, 1.0)
        self.conf_spin.setSingleStep(0.01)
        self.conf_spin.setValue(0.5)
        
        # Kết nối slider và spinbox
        self.conf_slider.valueChanged.connect(lambda val: self.conf_spin.setValue(val/100))
        self.conf_spin.valueChanged.connect(lambda val: self.conf_slider.setValue(int(val*100)))
        
        conf_layout.addWidget(conf_label)
        conf_layout.addWidget(self.conf_slider)
        conf_layout.addWidget(self.conf_spin)
        
        thresh_layout = QHBoxLayout()
        thresh_label = QLabel("Ngưỡng NMS:")
        self.thresh_slider = QSlider(Qt.Horizontal)
        self.thresh_slider.setRange(1, 100)
        self.thresh_slider.setValue(30)  # Mặc định 0.3
        self.thresh_spin = QDoubleSpinBox()
        self.thresh_spin.setRange(0.01, 1.0)
        self.thresh_spin.setSingleStep(0.01)
        self.thresh_spin.setValue(0.3)
        
        # Kết nối slider và spinbox
        self.thresh_slider.valueChanged.connect(lambda val: self.thresh_spin.setValue(val/100))
        self.thresh_spin.valueChanged.connect(lambda val: self.thresh_slider.setValue(int(val*100)))
        
        thresh_layout.addWidget(thresh_label)
        thresh_layout.addWidget(self.thresh_slider)
        thresh_layout.addWidget(self.thresh_spin)
        
        params_layout.addLayout(conf_layout)
        params_layout.addLayout(thresh_layout)
        params_group.setLayout(params_layout)
        
        # Nhóm tham số video
        video_params_group = QGroupBox("Tham Số Video")
        video_params_layout = QVBoxLayout()
        
        fps_layout = QHBoxLayout()
        fps_label = QLabel("FPS Đầu Ra:")
        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(1, 60)
        self.fps_spin.setValue(30)  # Mặc định 30 fps
        
        fps_layout.addWidget(fps_label)
        fps_layout.addWidget(self.fps_spin)
        
        skip_layout = QHBoxLayout()
        skip_label = QLabel("Bỏ Qua Frame:")
        self.skip_spin = QSpinBox()
        self.skip_spin.setRange(0, 30)
        self.skip_spin.setValue(0)
        skip_info = QLabel("(Để tăng tốc độ, mỗi n frame sẽ nhận diện 1 lần)")
        
        skip_layout.addWidget(skip_label)
        skip_layout.addWidget(self.skip_spin)
        skip_layout.addWidget(skip_info)
        
        video_params_layout.addLayout(fps_layout)
        video_params_layout.addLayout(skip_layout)
        video_params_group.setLayout(video_params_layout)
        
        # Nút thực hiện
        run_btn = QPushButton("Thực Hiện Nhận Diện")
        run_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        run_btn.setMinimumHeight(40)
        run_btn.clicked.connect(self.run_detection)
        
        # Thêm các layout vào layout chính
        main_layout.addWidget(input_group)
        main_layout.addWidget(output_group)
        main_layout.addWidget(params_group)
        main_layout.addWidget(video_params_group)
        main_layout.addWidget(run_btn)
        main_layout.addStretch()
        
        self.setLayout(main_layout)
    
    def browse_video(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Chọn File Video", "", "Video Files (*.mp4 *.avi *.mov *.mkv)")
        if file_path:
            self.input_path.setText(file_path)
    
    def browse_output(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Chọn Vị Trí Lưu", "", "Video Files (*.mp4)")
        if file_path:
            self.output_path.setText(file_path)
    
    def run_detection(self):
        input_path = self.input_path.text().strip()
        if not input_path:
            QMessageBox.warning(self, "Thiếu Thông Tin", "Vui lòng chọn file video đầu vào.")
            return
            
        if not os.path.exists(input_path):
            QMessageBox.critical(self, "Lỗi", f"Không tìm thấy file {input_path}")
            return
            
        output_path = self.output_path.text().strip()
        if not output_path:
            QMessageBox.warning(self, "Thiếu Thông Tin", "Vui lòng chọn đường dẫn lưu kết quả.")
            return
            
        confidence = self.conf_spin.value()
        threshold = self.thresh_spin.value()
        fps = self.fps_spin.value()
        skip_frames = self.skip_spin.value()
        
        QMessageBox.information(self, "Bắt Đầu Xử Lý", 
            "Quá trình nhận diện đối tượng trong video sẽ bắt đầu. "
            "Điều này có thể mất nhiều thời gian tùy thuộc vào độ dài của video.\n\n"
            "Bạn sẽ được thông báo khi quá trình hoàn tất.")
        
        command = f"python video_detection.py -i {input_path} -o {output_path} -c {confidence} -t {threshold} -f {fps} -s {skip_frames}"
        try:
            subprocess.run(command, shell=True, check=True)
            QMessageBox.information(self, "Thành Công", f"Nhận diện đối tượng đã hoàn tất!\n\nKết quả đã được lưu tại: {output_path}")
        except subprocess.CalledProcessError:
            QMessageBox.critical(self, "Lỗi", "Đã xảy ra lỗi trong quá trình nhận diện đối tượng.")


class RealtimeDetectionTab(QWidget):
    """Tab nhận diện đối tượng trực tiếp từ webcam"""
    def __init__(self):
        super().__init__()
        self.init_ui()
        
    def init_ui(self):
        # Thiết lập layout
        main_layout = QVBoxLayout()
        
        # Nhóm camera
        camera_group = QGroupBox("Cài Đặt Camera")
        camera_layout = QHBoxLayout()
        
        camera_label = QLabel("Camera Source:")
        self.camera_combo = QComboBox()
        self.camera_combo.addItem("Camera 0 (Mặc định)", 0)
        self.camera_combo.addItem("Camera 1", 1)
        self.camera_combo.addItem("Camera 2", 2)
        self.camera_combo.addItem("Camera 3", 3)
        
        camera_layout.addWidget(camera_label)
        camera_layout.addWidget(self.camera_combo)
        camera_group.setLayout(camera_layout)
        
        # Nhóm hiển thị
        display_group = QGroupBox("Tùy Chỉnh Hiển Thị")
        display_layout = QHBoxLayout()
        
        width_label = QLabel("Chiều Rộng Hiển Thị:")
        self.width_spin = QSpinBox()
        self.width_spin.setRange(200, 1920)
        self.width_spin.setValue(800)  # Mặc định 400 pixels
        self.width_spin.setSingleStep(50)
        
        display_layout.addWidget(width_label)
        display_layout.addWidget(self.width_spin)
        display_group.setLayout(display_layout)
        
        # Nhóm tham số nhận diện
        params_group = QGroupBox("Tham Số Nhận Diện")
        params_layout = QVBoxLayout()
        
        conf_layout = QHBoxLayout()
        conf_label = QLabel("Ngưỡng Tin Cậy:")
        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setRange(1, 100)
        self.conf_slider.setValue(50)  # Mặc định 0.5
        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.01, 1.0)
        self.conf_spin.setSingleStep(0.01)
        self.conf_spin.setValue(0.5)
        
        # Kết nối slider và spinbox
        self.conf_slider.valueChanged.connect(lambda val: self.conf_spin.setValue(val/100))
        self.conf_spin.valueChanged.connect(lambda val: self.conf_slider.setValue(int(val*100)))
        
        conf_layout.addWidget(conf_label)
        conf_layout.addWidget(self.conf_slider)
        conf_layout.addWidget(self.conf_spin)
        
        thresh_layout = QHBoxLayout()
        thresh_label = QLabel("Ngưỡng NMS:")
        self.thresh_slider = QSlider(Qt.Horizontal)
        self.thresh_slider.setRange(1, 100)
        self.thresh_slider.setValue(30)  # Mặc định 0.3
        self.thresh_spin = QDoubleSpinBox()
        self.thresh_spin.setRange(0.01, 1.0)
        self.thresh_spin.setSingleStep(0.01)
        self.thresh_spin.setValue(0.3)
        
        # Kết nối slider và spinbox
        self.thresh_slider.valueChanged.connect(lambda val: self.thresh_spin.setValue(val/100))
        self.thresh_spin.valueChanged.connect(lambda val: self.thresh_slider.setValue(int(val*100)))
        
        thresh_layout.addWidget(thresh_label)
        thresh_layout.addWidget(self.thresh_slider)
        thresh_layout.addWidget(self.thresh_spin)
        
        params_layout.addLayout(conf_layout)
        params_layout.addLayout(thresh_layout)
        params_group.setLayout(params_layout)
        
        # Nút thực hiện
        run_btn = QPushButton("Bắt Đầu Nhận Diện")
        run_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        run_btn.setMinimumHeight(40)
        run_btn.clicked.connect(self.run_detection)
        
        # Thêm các layout vào layout chính
        main_layout.addWidget(camera_group)
        main_layout.addWidget(display_group)
        main_layout.addWidget(params_group)
        main_layout.addWidget(run_btn)
        main_layout.addStretch()
        
        self.setLayout(main_layout)
    
    def run_detection(self):
        source = self.camera_combo.currentData()
        confidence = self.conf_spin.value()
        threshold = self.thresh_spin.value()
        width = self.width_spin.value()
        
        QMessageBox.information(self, "Bắt Đầu Nhận Diện", 
            "Cửa sổ nhận diện trực tiếp sẽ được mở.\n"
            "Nhấn phím 'q' để đóng cửa sổ và dừng nhận diện.")
        
        command = f"python real_time_detection.py -s {source} -c {confidence} -t {threshold} -w {width}"
        subprocess.Popen(command, shell=True)


class MainWindow(QMainWindow):
    """Cửa sổ chính của ứng dụng"""
    def __init__(self):
        super().__init__()
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle("Hệ Thống Nhận Diện Đối Tượng")
        self.setMinimumSize(700, 500)
        
        # Tạo tabs
        self.tabs = QTabWidget()
        self.image_tab = ImageDetectionTab()
        self.video_tab = VideoDetectionTab()
        self.realtime_tab = RealtimeDetectionTab()
        
        self.tabs.addTab(self.image_tab, "Nhận Diện Trong Ảnh")
        self.tabs.addTab(self.video_tab, "Nhận Diện Trong Video")
        self.tabs.addTab(self.realtime_tab, "Nhận Diện Trực Tiếp")
        
        # Thiết lập widget trung tâm
        self.setCentralWidget(self.tabs)
        
        # Thiết lập thanh trạng thái
        self.statusBar().showMessage("Sẵn sàng")


def main():
    """Hàm chính của chương trình"""
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # Thiết lập style 'Fusion' cho giao diện đẹp và đồng nhất
    
    # Kiểm tra các yêu cầu
    if not RequirementsChecker.check_requirements():
        sys.exit(1)
    
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()