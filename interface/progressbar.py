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
        self.setStyleSheet("background-color: #141104; border: 3px solid black;")
        self.setFixedSize(80, 700)

        # S_SESSION.int_positive_counter = 0
        # S_SESSION.int_negative_counter = 5

        #  Demo to try out
        int_positive_counter = 0
        int_negative_counter = 7
        self.current_positive_value = int_positive_counter
        self.current_nagative_value = int_negative_counter

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        self.draw_grid(painter)
        self.draw_bar(painter)

    def draw_grid(self, painter):
        # Set up pen for grid lines
        pen = QPen(QColor("whitesmoke"))
        pen.setWidth(2)
        painter.setPen(pen)

        # Grid lines for positive, zero, and negative
        height = self.height()
        y_center = height // 2

        # Draw center (zero) line
        painter.drawLine(0, y_center, self.width(), y_center)

        # Add labels
        painter.drawText(5, 20, "Positive")  # Top label
        painter.drawText(5, y_center - 5, "0")  # Center label
        painter.drawText(5, height - 10, "Negative")  # Bottom label


    def draw_bar(self, painter):
        total_steps = 10  # Scale range from 0 to 9
        height = self.height() // 2  # Half-height for up/down bar
        stepHeight = int(height/total_steps)
        centerLine = total_steps * stepHeight

        # Draw positive bar if value > 0
        if self.current_positive_value > 0:
            bar_height = self.current_positive_value * stepHeight

            # Create a linear gradient
            gradient = QLinearGradient(0, centerLine - bar_height, 0, centerLine)
            gradient.setColorAt(0.0, QColor("#26eb4a"))  # Start color
            gradient.setColorAt(1.0, QColor("#044710"))  # End color

            # Set the brush to the gradient
            painter.setBrush(gradient)
            painter.setPen(Qt.NoPen)

            # Draw the positive bar
            painter.drawRect(25, centerLine - bar_height, 30, bar_height)
        else:
            bar_height = (self.current_nagative_value * stepHeight)
            # Create a linear gradient
            gradient = QLinearGradient(0, centerLine, 0, centerLine + bar_height)
            gradient.setColorAt(1.0, QColor("#fc5353"))  # Start color
            gradient.setColorAt(0.0, QColor("#4f0616"))  # End color

            # Set the brush to the gradient
            painter.setBrush(gradient)
            painter.setPen(Qt.NoPen)
            painter.drawRect(25, centerLine, 30, bar_height)