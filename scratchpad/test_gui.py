import sys
import os
import cv2
import numpy as np
from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QHBoxLayout, QWidget, QPushButton
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap

# Add the root directory to the path so we can import carla_utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from carla_utils.hud import ADASHUD

class ModernHUDWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # 1. HIDE THE OS WINDOW BORDERS
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.resize(1280, 760)

        # Main Layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # 2. CREATE OUR CUSTOM GLASS TASKBAR
        self.taskbar = QWidget()
        self.taskbar.setFixedHeight(40)
        self.taskbar.setStyleSheet("""
            background-color: rgba(20, 20, 20, 200); 
            border-bottom: 1px solid rgba(255, 255, 255, 40);
        """)
        
        taskbar_layout = QHBoxLayout(self.taskbar)
        taskbar_layout.setContentsMargins(15, 0, 15, 0)

        # Title Text
        title = QLabel("VirtualVahana - Telemetry System")
        title.setStyleSheet("color: white; font-weight: bold; font-family: sans-serif; font-size: 14px;")
        taskbar_layout.addWidget(title)
        
        # This pushes everything after it to the right side
        taskbar_layout.addStretch() 

        # --- NEW: Custom Toolbar Options ---
        # --- RECREATING THE FULL OPENCV TOOLBAR ---
        def create_tool_btn(icon_text, tooltip):
            btn = QPushButton(icon_text)
            btn.setToolTip(tooltip)
            btn.setFixedSize(30, 30)
            btn.setStyleSheet("""
                QPushButton { background-color: transparent; color: white; border: none; border-radius: 5px; font-size: 16px;}
                QPushButton:hover { background-color: rgba(255, 255, 255, 40); }
                QPushButton:pressed { background-color: rgba(255, 255, 255, 20); }
            """)
            return btn

        # 1. Panning Tools
        taskbar_layout.addWidget(create_tool_btn("⬅", "Pan Left"))
        taskbar_layout.addWidget(create_tool_btn("➡", "Pan Right"))
        taskbar_layout.addWidget(create_tool_btn("⬆", "Pan Up"))
        taskbar_layout.addWidget(create_tool_btn("⬇", "Pan Down"))
        
        taskbar_layout.addSpacing(10) # Divider gap
        
        # 2. Zooming Tools
        taskbar_layout.addWidget(create_tool_btn("➕", "Zoom In"))
        taskbar_layout.addWidget(create_tool_btn("➖", "Zoom Out"))
        taskbar_layout.addWidget(create_tool_btn("🔲", "Zoom Fit"))
        taskbar_layout.addWidget(create_tool_btn("🏠", "Reset View (Home)"))
        
        taskbar_layout.addSpacing(10) # Divider gap

        # 3. Utility Tools
        self.btn_save = create_tool_btn("💾", "Save Image")
        self.btn_copy = create_tool_btn("📋", "Copy to Clipboard")
        self.btn_settings = create_tool_btn("⚙", "Properties")
        
        taskbar_layout.addWidget(self.btn_save)
        taskbar_layout.addWidget(self.btn_copy)
        taskbar_layout.addWidget(self.btn_settings)
        
        # Add a tiny gap before the close button
        taskbar_layout.addSpacing(15)

        # Custom Close Button
        close_btn = QPushButton("✕")
        close_btn.setFixedSize(30, 30)
        close_btn.setStyleSheet("""
            QPushButton { background-color: rgba(255, 50, 50, 150); color: white; border: none; border-radius: 15px; font-weight: bold;}
            QPushButton:hover { background-color: rgba(255, 0, 0, 255); }
        """)
        close_btn.clicked.connect(self.close)
        taskbar_layout.addWidget(close_btn)
        
        layout.addWidget(self.taskbar)

        # 3. CREATE THE VIDEO DISPLAY AREA
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet("background-color: black;")
        layout.addWidget(self.video_label)

        self.old_pos = None

        # --- HUD POSITIONS & DRAG STATE ---
        self.hud_pos = {
            "telemetry": [10, 10],      # [x, y]
            "warnings": [10, 160],
            "trust": [970, 10]          # Placed on the right
        }
        self.hud_sizes = {
            "telemetry": (325, 135),    # (width, height)
            "warnings": (325, 135),
            "trust": (300, 90)
        }
        self.dragging_element = None
        self.drag_offset = (0, 0)


        # 4. INITIALIZE HUD AND START VIDEO LOOP
        self.hud = ADASHUD()
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    # --- Mouse Events to make the custom taskbar draggable ---
    def mousePressEvent(self, event):
        # 1. Check if dragging the main window taskbar
        if event.button() == Qt.MouseButton.LeftButton and event.pos().y() < 40:
            self.old_pos = event.globalPosition().toPoint()
            return
            
        # 2. Check if clicking inside a HUD element
        if event.button() == Qt.MouseButton.LeftButton and event.pos().y() >= 40:
            # Adjust mouse Y because the OpenCV frame starts 40px below the window top
            mx = event.pos().x()
            my = event.pos().y() - 40 
            
            # Hit-test all panels
            for name, pos in self.hud_pos.items():
                w, h = self.hud_sizes[name]
                if pos[0] <= mx <= pos[0] + w and pos[1] <= my <= pos[1] + h:
                    self.dragging_element = name
                    self.drag_offset = (mx - pos[0], my - pos[1])
                    break

    def mouseMoveEvent(self, event):
        # Dragging the OS window
        if self.old_pos is not None:
            delta = event.globalPosition().toPoint() - self.old_pos
            self.move(self.pos() + delta)
            self.old_pos = event.globalPosition().toPoint()
            
        # Dragging a HUD Panel
        elif self.dragging_element is not None:
            mx = event.pos().x()
            my = event.pos().y() - 40
            
            # Update the panel's coordinates based on mouse movement
            new_x = mx - self.drag_offset[0]
            new_y = my - self.drag_offset[1]
            self.hud_pos[self.dragging_element] = [new_x, new_y]

    def mouseReleaseEvent(self, event):
        self.old_pos = None
        self.dragging_element = None


    # --- The Main Loop ---
    def update_frame(self):
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        cv2.rectangle(frame, (0, 0), (1280, 360), (235, 206, 135), -1)
        cv2.rectangle(frame, (0, 360), (1280, 720), (50, 100, 50), -1)
        cv2.circle(frame, (100, 80), 90, (0, 255, 255), -1) 

        # --- 1. GRAB THE DYNAMIC COORDINATES ---
        tx, ty = self.hud_pos["telemetry"]
        wx, wy = self.hud_pos["warnings"]
        trx, try_y = self.hud_pos["trust"]

        # --- 2. PASS X AND Y INTO THE DRAW FUNCTIONS ---
        self.hud.draw_telemetry(frame, tx, ty, speed=65.2, lka_active=True, acc_active=False)
        
        # Note: I swapped the text colors here to the dark ones we used for the white glass!
        self.hud.draw_warnings(frame, wx, wy, "NONE", (0, 150, 0), "VEHICLE AHEAD", (0, 100, 255), "RED LIGHT", (0, 0, 200))
        
        self.hud.draw_confidence_meter(frame, trx, try_y, trust_score=92.0)
        
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        
        self.video_label.setPixmap(QPixmap.fromImage(qt_image))


        
def main():
    app = QApplication(sys.argv)
    window = ModernHUDWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()