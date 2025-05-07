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
        self.setFixedSize(980, 900)

        self.candle_data = [
            [20555.2, 20557.0, '2025-03-06T00:38:41.625', 20561.0, 20550.0],
        ]

        self.candle_data.reverse()  # Reverse to make the oldest first
        self.macd_data = [
            ['2025-03-06T00:00:00', 0.15, 0.135],
            ['2025-03-06T00:05:00', -0.12, -0.108],
            ['2025-03-06T00:10:00', 0.18, 0.162],
        ]

        # Determine the Y-axis scale from the candle data
        self.price_max = max(candle[3] for candle in self.candle_data)+200  # Highest high price
        self.price_min = min(candle[4] for candle in self.candle_data)-200  # Lowest low price

        self.macd_max = max(entry[1] for entry in self.macd_data) + 0.1  # Highest MACD value + buffer
        self.macd_min = min(entry[1] for entry in self.macd_data) - 0.1  # Lowest MACD value - buffer

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
        """Draw the grid, candles, and MACD."""
        super().paintEvent(event)

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Define areas
        chart_height = int(self.height() * 0.75)  # Top 75% for candlestick chart
        macd_height = self.height() - chart_height  # Bottom 25% for MACD

        # Draw candlestick chart
        painter.setViewport(0, 0, self.width(), chart_height)
        painter.setWindow(0, 0, self.width(), chart_height)
        self.draw_grid(painter)
        self.draw_candles(painter)
        self.draw_macd(painter)


    def draw_grid(self, painter):
        """Draw grid lines and labels for X and Y axes."""
        # Set pen for grid lines
        pen = QPen(Qt.darkGray, 1, Qt.SolidLine)
        painter.setPen(pen)

        # Define the upper 75% area for the candlestick chart
        chart_height = int(self.height() * 0.75)

        # --- X-axis (time) ---
        # Draw vertical grid lines from top to bottom
        current_time = self.start_time
        for index, x in enumerate(range(10, self.width(), self.grid_spacing_x)):
            painter.drawLine(x, 0, x, self.height())  # Vertical line from top to bottom

            # Only draw time labels for every other line
            if index % 2 == 0:
                time_label = current_time.strftime("%H:%M")
                painter.drawText(x + 5, chart_height - 5, time_label)  # Label at the bottom of the upper 75%
            current_time += datetime.timedelta(minutes=5)  # Increment by 5 minutes

        # --- Y-axis (prices) ---
        # Draw horizontal grid lines and labels in the upper 75%
        max_labels = 10  # Target number of visible labels
        step_size = chart_height // max_labels  # Dynamically adjust step size

        price_range = self.price_max - self.price_min  # Total price range displayed
        price_step_value = price_range / max_labels  # Price difference per step

        price = self.price_max
        for i in range(max_labels + 1):  # Ensure all labels fit
            y = i * step_size  # Calculate Y position
            painter.drawLine(0, y, self.width(), y)  # Draw horizontal grid line within the upper 75%
            text = f"{price:.0f}"  # Format price
            # Ensure proper alignment on the right side
            text_width = painter.fontMetrics().width(text)
            painter.drawText(self.width() - text_width - 5, y - 5, text)  # Right-aligned label
            price -= price_step_value  # Decrease by correct price step

        # --- MACD Y-axis (values) ---
        # Draw horizontal grid lines in the bottom 25% (MACD area)
        macd_height = self.height() // 4  # MACD area is the bottom 25%
        macd_top = self.height() - macd_height  # Top of the MACD area
        macd_bottom = self.height()  # Bottom of the MACD area

        step_count = 5  # Number of grid lines in the MACD area
        value_step = (self.macd_max - self.macd_min) / step_count

        for i in range(step_count + 1):
            value = self.macd_min + i * value_step
            y = int(macd_top + ((self.macd_max - value) / (self.macd_max - self.macd_min)) * macd_height)
            painter.drawLine(0, y, self.width(), y)  # Draw horizontal grid line in the MACD area


    def draw_candles(self, painter):
        """Draw candlestick chart in the upper 75% of the widget."""
        num_candles = len(self.candle_data)
        if num_candles == 0:
            return

        for i, candle in enumerate(self.candle_data):
            x = self.grid_spacing_x * i
            if x < 0 or x > self.width():
                continue  # Skip if out of bounds
            self.draw_candle(painter, candle, x, i == num_candles - 1)


    def price_to_y(self, price):
        """Convert a price value to a Y-coordinate within the upper 75% of the widget."""
        chart_height = int(self.height() * 0.75)  # Upper 75% for candlestick chart

        # Ensure we use the latest price_max and price_min
        price_range = self.price_max - self.price_min

        # Prevent division by zero
        if price_range == 0:
            return chart_height // 2

        # Map price to Y-coordinate dynamically
        y = ((self.price_max - price) / price_range) * chart_height
        return int(max(0, min(y, chart_height)))  # Clamp Y within the upper 75%

    def draw_candle(self, painter, candle, x_position, is_last_candle):
        open_price, close_price, time, high, low = candle[:5]

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
        chart_height = int(self.height() * 0.75)  # Top 75% for candlestick chart

        # Ensure we use the latest price_max and price_min
        price_range = self.price_max - self.price_min

        # Prevent division by zero
        if price_range == 0:
            return chart_height // 2

        # Map price to Y-coordinate dynamically
        y = ((self.price_max - price) / price_range) * chart_height
        return int(max(0, min(y, chart_height)))  # Clamp Y within bounds

    def draw_macd(self, painter):
        # Map MACD values to Y-coordinates within the MACD area
        def macd_to_y(value):
            """Map MACD values to Y-coordinates within the MACD area (bottom 25%)."""
            macd_height = self.height() // 4  # MACD area is the bottom 25%
            macd_top = self.height() - macd_height  # Top of the MACD area
            macd_bottom = self.height()  # Bottom of the MACD area

            if self.macd_max == self.macd_min:
                print("⚠️ MACD Max and Min are equal! Using default Y-coordinate.")  # Debug print
                return macd_top + (macd_height // 2)  # Default to the middle of the MACD area

            # Map MACD values to Y-coordinates within the MACD area
            y = int(macd_bottom - ((value - self.macd_min) / (self.macd_max - self.macd_min) * macd_height))
            return y

        """Draw MACD line, signal line, and histogram aligned with candlestick candles."""
        if not self.macd_data:
            return

        # Calculate MACD area dimensions
        macd_height = self.height() // 4  # MACD area is the bottom 25%
        macd_top = self.height() - macd_height  # Top of the MACD area
        macd_bottom = self.height()  # Bottom of the MACD area

        # Draw MACD line and signal line
        macd_points = []
        signal_points = []
        for i, macd_entry in enumerate(self.macd_data):
            if len(macd_entry) < 3:
                continue  # Skip invalid entries

            # Use the same X positions as the candlestick candles
            x = self.grid_spacing_x * i
            macd_points.append((x, macd_to_y(macd_entry[1])))  # MACD line
            signal_points.append((x, macd_to_y(macd_entry[2])))  # Signal line


        # Draw MACD line
        painter.setPen(QPen(QColor(50, 50, 200), 2))  # Blue for MACD line
        for j in range(1, len(macd_points)):
            painter.drawLine(macd_points[j - 1][0], macd_points[j - 1][1], macd_points[j][0], macd_points[j][1])

        # Draw Signal line
        painter.setPen(QPen(QColor(200, 50, 50), 2))  # Red for Signal line
        for j in range(1, len(signal_points)):
            painter.drawLine(signal_points[j - 1][0], signal_points[j - 1][1], signal_points[j][0], signal_points[j][1])

        # Draw MACD histogram
        bar_width = int(self.grid_spacing_x * 0.6)
        for i, macd_entry in enumerate(self.macd_data):
            if len(macd_entry) < 3:
                continue  # Skip invalid entries

            # Use the same X positions as the candlestick candles
            x = self.grid_spacing_x * i
            histogram_value = macd_entry[1] - macd_entry[2]  # MACD - Signal

            if histogram_value >= 0:
                bar_top = macd_to_y(0)
                bar_bottom = macd_to_y(histogram_value)
                gradient = QLinearGradient(0, bar_top, 0, bar_bottom)
                gradient.setColorAt(0.0, QColor("#26eb4a"))  # Green for positive
                gradient.setColorAt(1.0, QColor("#044710"))
            else:
                bar_top = macd_to_y(histogram_value)
                bar_bottom = macd_to_y(0)
                gradient = QLinearGradient(0, bar_top, 0, bar_bottom)
                gradient.setColorAt(1.0, QColor("#fc5353"))  # Red for negative
                gradient.setColorAt(0.0, QColor("#4f0616"))


            painter.setBrush(gradient)
            painter.setPen(Qt.NoPen)
            bar_height = abs(bar_bottom - bar_top)
            painter.drawRect(x - bar_width // 2, min(bar_top, bar_bottom), bar_width, bar_height)







    def update_candles(self, new_candle_data):
        if not new_candle_data or len(new_candle_data) < 5:
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

    def update_macd(self, new_macd_data):
        if not new_macd_data or len(new_macd_data) < 3:  # Ensure MACD data has time, MACD, and signal values
            return
        try:
            time, macd_value, signal_value = new_macd_data
            self.macd_data.append([time, macd_value, signal_value])

            # Adjust max/min calculations for MACD
            self.macd_max = max(self.macd_max, macd_value) + 0.1
            self.macd_min = min(self.macd_min, macd_value) - 0.1

            # Keep only the last 30 MACD values
            if len(self.macd_data) > 30:
                self.macd_data = self.macd_data[-30:]
        except Exception as e:
            print(f"⚠️ Error during MACD update: {e}")
