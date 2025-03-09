import sys
from PyQt5.QtWidgets import QVBoxLayout, QHBoxLayout, QGridLayout, \
    QWidget, QGraphicsDropShadowEffect
from PyQt5.QtGui import QColor, QLinearGradient
from matplotlib.figure import Figure
import datetime
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel
from PyQt5.QtGui import QPainter, QPen
from PyQt5.QtCore import Qt

class ProgressBarArea(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("background-color: #1E1E1E; border: 3px solid black;")
        self.setFixedSize(80, 700)
        self.current_value = 0

    def update_bar(self, new_value):
        self.current_value = new_value
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        self.draw_grid(painter)
        self.draw_bar(painter)

    def draw_grid(self, painter):
        # Set up pen for grid lines
        pen = QPen(QColor("whitesmoke"))
        pen.setWidth(1)
        painter.setPen(pen)

        total_steps = 7  # From 0 to 7
        step_height = int(self.height() / total_steps)

        # --- Fixed positions for 0 and 7 ---
        y_pos_0 = 695
        y_pos_1 = 620
        y_pos_2 = 520
        y_pos_3 = 420
        y_pos_4 = 320
        y_pos_5 = 220
        y_pos_6 = 120
        y_pos_7 = 20

        painter.drawText(5, y_pos_0, "0")  # Fixed bottom position
        painter.drawText(5, y_pos_1, "1")  # Fixed bottom position
        painter.drawLine(25, y_pos_1, self.width(), y_pos_1)  # Draw horizontal line
        painter.drawText(5, y_pos_2, "2")  # Fixed bottom position
        painter.drawLine(25, y_pos_2, self.width(), y_pos_2)  # Draw horizontal line
        painter.drawText(5, y_pos_3, "3")  # Fixed bottom position
        painter.drawLine(25, y_pos_3, self.width(), y_pos_3)  # Draw horizontal line
        painter.drawText(5, y_pos_4, "4")  # Fixed bottom position
        painter.drawLine(25, y_pos_4, self.width(), y_pos_4)  # Draw horizontal line
        painter.drawText(5, y_pos_5, "5")  # Fixed bottom position
        painter.drawLine(25, y_pos_5, self.width(), y_pos_5)  # Draw horizontal line
        painter.drawText(5, y_pos_6, "6")  # Fixed bottom position
        painter.drawLine(25, y_pos_6, self.width(), y_pos_6)  # Draw horizontal line
        painter.drawText(5, y_pos_7, "7")  # Fixed top position
        painter.drawLine(25, y_pos_7, self.width(), y_pos_7)  # Draw horizontal line


    def draw_bar(self, painter):
        total_steps = 7  # Maximum range (0 to 7)
        stepHeight = int(self.height() / total_steps)  # Divide widget height into 7 equal steps

        # Ensure valid bar height
        bar_height = (self.current_value * stepHeight) - 25

        # Create a linear gradient (bottom to top)
        gradient = QLinearGradient(0, self.height() - bar_height, 0, self.height())
        gradient.setColorAt(0.0, QColor("#26eb4a"))  # Start color (bottom)
        gradient.setColorAt(1.0, QColor("#044710"))  # End color (top)

        # Set the brush to the gradient
        painter.setBrush(gradient)
        painter.setPen(Qt.NoPen)

        # Draw the bar from the bottom up
        painter.drawRect(25, (self.height()-5) - bar_height, 30, bar_height)
