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
        self.setStyleSheet("background-color: #1E1E1E; border: 3px solid black;")
        self.setFixedSize(980, 700)

        self.candle_data = [
            [20555.2, 20557.0, '2025-03-06T00:38:41.625', 20561.0, 20550],
        ]

        self.candle_data.reverse()  # Reverse to make the oldest first

        # Determine the Y-axis scale from the candle data
        self.price_max = max(candle[3] for candle in self.candle_data)+200  # Highest high price
        self.price_min = min(candle[4] for candle in self.candle_data)-200  # Lowest low price
        price_range = self.price_max - self.price_min
        self.price_step = round(price_range / 25)

        # Determine the start time from the first candle
        self.start_time = datetime.datetime.fromisoformat(self.candle_data[0][2])


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
        max_labels = 20  # Target number of visible labels
        step_size = max(1, self.height() // max_labels)  # Dynamically adjust step size

        price_range = self.price_max - self.price_min  # Total price range displayed
        price_step_value = price_range / max_labels  # Price difference per step

        price = self.price_max
        for i in range(max_labels + 1):  # Ensure all labels fit
            y = i * step_size  # Calculate Y position

            painter.drawLine(0, y, self.width(), y)  # Draw horizontal grid line
            text = f"{price:.0f}"  # Format price

            # Ensure proper alignment on the right side
            text_width = painter.fontMetrics().width(text)
            painter.drawText(self.width() - text_width - 5, y - 5, text)  # Right-aligned label

            price -= price_step_value  # Decrease by correct price step

    def draw_candle(self, painter, candle, x_position, is_last_candle):
        open_price, close_price, time, high, low = candle

        # Map prices to Y-coordinates
        y_open = self.price_to_y(open_price)
        y_close = self.price_to_y(close_price)
        y_high = self.price_to_y(high)  # Highest point of wick
        y_low = self.price_to_y(low)  # Lowest point of wick

        # Determine candle color
        if close_price > open_price:  # Bullish candle (green)
            candle_color = QColor(50, 200, 50)  # Bright green
            body_top = y_close  # Close is higher
            body_bottom = y_open  # Open is lower
        else:  # Bearish candle (red)
            candle_color = QColor(200, 50, 50)  # Bright red
            body_top = y_open  # Open is higher
            body_bottom = y_close  # Close is lower

        body_height = abs(body_bottom - body_top)

        # Gradient for candle body
        gradient = QLinearGradient(x_position, body_top, x_position, body_bottom)
        if close_price > open_price:  # Bullish
            gradient.setColorAt(0.0, QColor("#26eb4a"))  # Start color (top)
            gradient.setColorAt(1.0, QColor("#044710"))  # End color (bottom)
        else:  # Bearish
            gradient.setColorAt(0.0, QColor("#fc5353"))  # Start color (top)
            gradient.setColorAt(1.0, QColor("#4f0616"))  # End color (bottom)

        painter.setBrush(gradient)
        painter.setPen(QPen(candle_color))

        # **Draw Wick (Shadow)**
        wick_pen = QPen(candle_color, 2)  # Thin wick line
        painter.setPen(wick_pen)
        painter.drawLine(
            x_position + self.candle_width // 2, y_high,  # Wick top (high)
            x_position + self.candle_width // 2, y_low  # Wick bottom (low)
        )

        # **Draw Candle Body**
        painter.drawRoundedRect(x_position, body_top, self.candle_width, body_height, 1, 1)

        # **Highlight the last candle**
        if is_last_candle:
            glow_pen = QPen(Qt.yellow, 2, Qt.SolidLine)  # Yellow outline for last candle
            painter.setPen(glow_pen)
            painter.drawRoundedRect(
                x_position - 2, body_top - 2, self.candle_width + 4, body_height + 4, 1, 1
            )

            # **Draw a dotted line for the last candle's open price**
            dotted_pen = QPen(Qt.darkYellow, 1, Qt.DotLine)
            painter.setPen(dotted_pen)
            painter.drawLine(x_position + self.candle_width, y_open, self.width(), y_open)

            # **Draw the open price as a label**
            font = painter.font()
            font.setPointSize(10)
            font.setBold(True)
            painter.setFont(font)
            painter.setPen(Qt.white)  # White text for the label
            painter.drawText(self.width() - 60, y_open - 5, f"{open_price:.0f}")

    #
    def price_to_y(self, price):
        """Convert a price value to a Y-coordinate."""

        # Ensure we use the latest price_max and price_min
        price_range = self.price_max - self.price_min

        # Prevent division by zero
        if price_range == 0:
            return self.height() // 2

        # Map price to Y-coordinate dynamically
        y = ((self.price_max - price) / price_range) * self.height()
        return int(max(0, min(y, self.height())))  # Clamp Y within bounds

    def update_candles(self, new_candle_data):
        if not new_candle_data or len(new_candle_data) < 3:
            return
        try:
            new_candle_data.reverse()
            self.candle_data = [
                [candle[0], candle[1], candle[2], candle[3], candle[4]] for candle in new_candle_data
            ]
            # Adjust max/min calculations
            self.price_max = max(candle[3] for candle in self.candle_data) + 200
            self.price_min = min(candle[4] for candle in self.candle_data) - 200
            price_range = self.price_max - self.price_min
            self.price_step = round(price_range / 25)
            self.start_time = datetime.datetime.fromisoformat(self.candle_data[1][2])
        except Exception as e:
            print(f"⚠️ Error during candle update: {e}")
