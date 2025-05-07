from PyQt5.QtGui import QColor, QLinearGradient
import datetime
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel
from PyQt5.QtGui import QPainter, QPen
from PyQt5.QtCore import Qt



class MACDArea(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("background-color: #1E1E1E; border: 3px solid black;")
        self.setFixedSize(980, 200)

        self.candle_data = [['2025-03-06T00:38:41.625', 0.1, 0.1],]
        self.start_time = datetime.datetime.fromisoformat(self.candle_data[0][0])

        # Determine Y-axis scale from the candle data
        self.macd_line_max = max(candle[1] for candle in self.candle_data)  # Highest MACD-line
        self.macd_line_min = min(candle[1] for candle in self.candle_data)  # Lowest MACD-line
        self.signal_line_max = max(candle[2] for candle in self.candle_data)  # Highest signal-line
        self.signal_line_min = min(candle[2] for candle in self.candle_data)  # Lowest signal-line

        # Calculate histogram dynamically
        self.histogram_max = max(candle[1] - candle[2] for candle in self.candle_data)  # MACD - Signal
        self.histogram_min = min(candle[1] - candle[2] for candle in self.candle_data)  # MACD - Signal

        # Use the widest range for consistency
        self.y_axis_min = min(self.macd_line_min, self.signal_line_min, self.histogram_min)
        self.y_axis_max = max(self.macd_line_max, self.signal_line_max, self.histogram_max)

        # Determine horizontal grid spacing
        self.grid_spacing_x = int(self.width() / 35)  # 35 divisions

    def paintEvent(self, event):
        """Draw the grid for MACD area."""
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Draw vertical and horizontal grid lines
        self.draw_grid(painter)

        self.draw_macd_values(painter)

    def draw_grid(self, painter):
        """Draw horizontal and vertical grid lines."""
        # Set the pen for grid lines
        grid_pen = QPen(Qt.darkGray, 1, Qt.SolidLine)
        painter.setPen(grid_pen)

        # Horizontal grid (Y-axis values)
        step_count = 5  # Number of horizontal grid lines above and below zero
        y_center = self.height() // 2  # Y-coordinate of the center (zero line)
        value_step = (self.y_axis_max - self.y_axis_min) / (step_count * 2)  # Adjust for symmetrical grid
        y_step = self.height() / (step_count * 2)  # Vertical distance between lines

        # Draw lines and labels
        for i in range(-step_count, step_count + 1):  # Symmetric range around zero
            y = int(y_center - i * y_step)
            value = i * value_step  # Value corresponding to this grid line
            painter.drawLine(0, y, self.width(), y)

            # Draw labels, skipping the zero line to avoid clutter
            if i != 0:
                text = f"{value:.1f}"
                text_width = painter.fontMetrics().width(text)
                painter.drawText(self.width() - text_width - 5, y - 5, text)

        # Highlight the center (zero line)
        zero_pen = QPen(Qt.white, 2, Qt.SolidLine)
        painter.setPen(zero_pen)
        painter.drawLine(0, y_center, self.width(), y_center)

        # X-axis (time)
        self.start_time = datetime.datetime.fromisoformat(self.candle_data[0][0])
        # print(self.start_time)
        for index, x in enumerate(range(self.rect().width() - self.grid_spacing_x - 15, 0, -self.grid_spacing_x)):
            painter.drawLine(x, 0, x, self.height())  # Vertical line

            # # Only draw time labels for every other line
            # if index % 2 == 0:
            #     time_label = self.start_time.strftime("%H:%M")
            #     painter.drawText(x + 5, self.height() - 5, time_label)  # Label slightly above the bottom
            #
            # self.start_time -= datetime.timedelta(minutes=5)  # Subtract 5 minutes for the next label

    def draw_macd_values(self, painter: object) -> object:
        step_x = self.grid_spacing_x  # Horizontal spacing
        y_range = self.y_axis_max - self.y_axis_min

        def value_to_y(value):
            # Map value proportionally to the Y-axis
            if y_range == 0:
                return self.height() // 2
            return int(self.height() - ((value - self.y_axis_min) / y_range * self.height()))

        y_center = self.height() // 2  # Y-coordinate of the center (zero line)
        macd_points = [(self.grid_spacing_x-step_x, 100)]
        signal_points = [(self.grid_spacing_x-step_x, 100)]
        bar_width = int(step_x * 0.6)

        for i, candle in enumerate(self.candle_data):
            time, macd_value, signal_value = candle  # Only unpack the available values
            histogram_value = macd_value - signal_value  # Calculate Histogram dynamically
            x = self.grid_spacing_x + step_x * i  # Compute X-coordinate

            if histogram_value >= 0:
                bar_height = (histogram_value / y_range) * self.height()
                bar_top = int(y_center - bar_height)  # Top is above the centerline
                bar_bottom = y_center  # Base is the centerline

                # Create gradient for positive bars
                gradient = QLinearGradient(0, bar_top, 0, bar_bottom)
                gradient.setColorAt(0.0, QColor("#26eb4a"))  # Start color
                gradient.setColorAt(1.0, QColor("#044710"))  # End color

                # Set brush to gradient
                painter.setBrush(gradient)
                painter.setPen(Qt.NoPen)

                # Draw positive bar
                painter.drawRect(x - bar_width // 2, bar_top, bar_width, bar_bottom - bar_top)

            # Negative bar: grows downward from the centerline
            else:
                bar_height = (abs(histogram_value) / y_range) * self.height()
                bar_top = y_center  # Top starts at centerline
                bar_bottom = int(y_center + bar_height)  # Bottom is below the centerline

                # Create gradient for negative bars
                gradient = QLinearGradient(0, bar_top, 0, bar_bottom)
                gradient.setColorAt(1.0, QColor("#fc5353"))  # Start color
                gradient.setColorAt(0.0, QColor("#4f0616"))  # End color

                # Set brush to gradient
                painter.setBrush(gradient)
                painter.setPen(Qt.NoPen)

                # Draw negative bar
                painter.drawRect(x - bar_width // 2, bar_top, bar_width, bar_bottom - bar_top)

            # Convert values to Y-coordinates (adjust 0.0 to center)
            y_macd = value_to_y(macd_value) if macd_value != 0.0 else y_center
            y_signal = value_to_y(signal_value) if signal_value != 0.0 else y_center

            x = self.grid_spacing_x + step_x * i  # X-coordinate

            # Clamp X values to prevent overflow
            if 0 <= x <= self.width():
                macd_points.append((x, y_macd))
                signal_points.append((x, y_signal))

        # Draw the MACD line
        painter.setPen(QPen(QColor(50, 50, 200), 2))  # Blue for MACD line
        for j in range(1, len(macd_points)):
            painter.drawLine(macd_points[j - 1][0], macd_points[j - 1][1], macd_points[j][0], macd_points[j][1])

        # Draw the Signal line
        painter.setPen(QPen(QColor(200, 50, 50), 2))  # Red for Signal line
        for j in range(1, len(signal_points)):
            painter.drawLine(signal_points[j - 1][0], signal_points[j - 1][1], signal_points[j][0], signal_points[j][1])

    def update_macd(self, new_macd_values):
        self.candle_data.append(new_macd_values)  # Add to the list
        #print(self.candle_data)
        # Ignore first 32 candles (MACD isn't valid before this)
        if len(self.candle_data) < 35:
            return  # Wait until enough data is collected

        # Ensure we keep only the last 30 MACD values **after** we have at least 32
        self.candle_data = self.candle_data[-35:]

        # Determine Y-axis scale from valid MACD data
        self.macd_line_max = max(candle[1] for candle in self.candle_data)
        self.macd_line_min = min(candle[1] for candle in self.candle_data)
        self.signal_line_max = max(candle[2] for candle in self.candle_data)
        self.signal_line_min = min(candle[2] for candle in self.candle_data)

        # Set Y-axis range using the widest values
        self.y_axis_min = min(candle[2] for candle in self.candle_data) - 10
        self.y_axis_max = max(candle[1] for candle in self.candle_data) + 10

        # Determine horizontal grid spacing
        self.grid_spacing_x = int(self.width() / 34)  # 35 divisions