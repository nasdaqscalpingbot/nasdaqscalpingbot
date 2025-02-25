import sys
from PyQt5.QtWidgets import QVBoxLayout, QHBoxLayout, QGridLayout, \
    QWidget, QGraphicsDropShadowEffect
from PyQt5.QtGui import QColor, QLinearGradient
from matplotlib.figure import Figure
import datetime
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel
from PyQt5.QtGui import QPainter, QPen
from PyQt5.QtCore import Qt

class ChartArea(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("background-color: #141104; border: 3px solid black;")
        self.setFixedSize(1000, 700)

        self.candle_data = [
            [20552, 20560, 20576, 20539, "2024-11-19T04:15:44"],
        ]

        self.candle_data.reverse()  # Reverse to make the oldest first

        # Determine the Y-axis scale from the candle data
        self.price_max = max(candle[2] for candle in self.candle_data)+400  # Highest high price
        self.price_min = min(candle[3] for candle in self.candle_data)-400  # Lowest low price
        price_range = self.price_max - self.price_min
        self.price_step = round(price_range / 15)

        # Determine the start time from the first candle
        self.start_time = datetime.datetime.fromisoformat(self.candle_data[0][4])


        # Independent grid spacings
        self.grid_spacing_x = int(self.width() / 35)  # 35 divisions # Space between vertical lines in pixels

        # Candle width and space
        self.candle_width = int(self.grid_spacing_x * 0.8)  # 80% of grid spacing
        self.candle_space = int(self.grid_spacing_x * 0.2)  # Remaining 20% is spacing


    def paintEvent(self, event):
        """Draw the grid inside the chart area."""
        super().paintEvent(event)

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Draw grid
        self.draw_grid(painter)

        # Draw candles
        is_last_candle = False
        x_position = int(self.candle_width - (self.candle_width * 0.5)) # Start drawing from the left
        int_number_of_candles = len(self.candle_data)
        int_counter = 0
        for candle in self.candle_data:
            if int_counter == int_number_of_candles-1:
                is_last_candle = True
            else:
                int_counter += 1
            self.draw_candle(painter, candle, x_position, is_last_candle)
            x_position += self.grid_spacing_x

    def draw_grid(self, painter):
        """Draw grid lines and labels for X and Y axes."""
        # Set pen for grid lines
        pen = QPen(Qt.darkGray, 1, Qt.SolidLine)
        painter.setPen(pen)

        # X-axis (time)
        current_time = self.start_time
        for index, x in enumerate(range(10, self.width(), self.grid_spacing_x)):
            painter.drawLine(x, 0, x, self.height())  # Vertical line
            # Only draw time labels for every other line
            if index % 2 == 0:
                time_label = current_time.strftime("%H:%M")
                painter.drawText(x + 5, self.height() - 5, time_label)  # Label slightly above the bottom
            current_time += datetime.timedelta(minutes=5)  # Increment by 5 minutes

        # Y-axis (prices)
        price = self.price_max
        for y in range(0, self.height(), self.price_step):
            painter.drawLine(0, y, self.width(), y)  # Horizontal line
            text = f"{price:.0f}"

            # Calculate text position for the right side
            text_width = painter.fontMetrics().width(text)
            painter.drawText(self.width() - text_width - 5, y - 5, text)  # Right-aligned label

            price -= self.price_step

    def draw_candle(self, painter, candle, x_position, is_last_candle):
        open_price, close_price, high_price, low_price, time = candle

        # Map prices to Y-coordinates
        y_open = self.price_to_y(open_price)
        y_close = self.price_to_y(close_price)
        y_high = self.price_to_y(high_price)
        y_low = self.price_to_y(low_price)

        # Determine candle color
        color_up = QColor(50, 200, 50)  # Bright green for bullish candles
        color_down = QColor(200, 50, 50)  # Bright red for bearish candles
        candle_color = color_up if close_price >= open_price else color_down
        body_top = min(y_open, y_close)
        body_bottom = max(y_open, y_close)
        body_height = abs(y_open - y_close)


        # Gradient for candle body
        gradient = QLinearGradient(x_position, body_top, x_position, body_bottom)
        if close_price >= open_price:
            gradient.setColorAt(0.0, QColor("#26eb4a"))  # Start color
            gradient.setColorAt(1.0, QColor("#044710"))  # End color
        else:
            gradient.setColorAt(1.0, QColor("#fc5353"))  # Start color
            gradient.setColorAt(0.0, QColor("#4f0616"))  # End color

        painter.setBrush(gradient)
        painter.setPen(QPen(candle_color))

        # Draw the wick (high to low)
        painter.drawLine(x_position + self.candle_width // 2, y_high, x_position + self.candle_width // 2, y_low)


        painter.drawRoundedRect(x_position, body_top, self.candle_width, body_height, 1, 1)

        # Highlight the last candle
        if is_last_candle:
            glow_pen = QPen(Qt.yellow, 2, Qt.SolidLine)  # Yellow outline for the last candle
            painter.setPen(glow_pen)
            painter.drawRoundedRect(
                x_position - 2, body_top - 2, self.candle_width + 4, body_height + 4, 1, 1
            )

            # Draw a dotted line for the last candle's open price
            dotted_pen = QPen(Qt.darkYellow, 1, Qt.DotLine)
            painter.setPen(dotted_pen)
            painter.drawLine(x_position + self.candle_width, y_open, self.width(), y_open)

            # Draw the open price as a label
            font = painter.font()
            font.setPointSize(10)
            font.setBold(True)
            painter.setFont(font)
            painter.setPen(Qt.white)  # White text for the label
            painter.drawText(self.width() - 60, y_open - 5, f"{open_price:.0f}")

    def price_to_y(self, price):
        """Convert a price value to a Y-coordinate."""
        # Prevent division by zero
        price_range = self.price_max - self.price_min
        if price_range == 0:
            return self.height() // 2

        # Map price to Y-coordinate
        y = self.height() - ((price - self.price_min) / price_range * self.height())
        return int(max(0, min(y, self.height())))  # Clamp Y within bounds

    def update_candles(self, new_candle_data):
        """Update the candle data and refresh the chart."""
        print("Updating candles...")

        if not new_candle_data or len(new_candle_data[0]) < 7:
            print("⚠️ Incomplete or empty candle data.")
            return

        # ✅ Replace hardcoded data with live data
        self.candle_data = new_candle_data[::-1]  # Reverse if oldest-first is required

        # ✅ Recalculate Y-axis scaling based on new data
        try:
            self.price_max = max(candle[2] for candle in self.candle_data) + 400  # High price
            self.price_min = min(candle[3] for candle in self.candle_data) - 400  # Low price
            price_range = self.price_max - self.price_min
            self.price_step = round(price_range / 15)

            # ✅ Set start time for X-axis
            self.start_time = datetime.datetime.fromisoformat(self.candle_data[0][6])

        except Exception as e:
            print(f"⚠️ Error during candle update: {e}")
            return

        # ✅ Trigger repaint safely
        self.update()
