
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QLabel, QProgressBar, QGridLayout, \
    QWidget, QSpacerItem, QSizePolicy, QGraphicsDropShadowEffect
from PyQt5.QtGui import QColor, QLinearGradient, QPainterPath
from PyQt5.QtCore import Qt
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from mplfinance.original_flavor import candlestick_ohlc
import matplotlib.dates as mdates
import random
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
            ["2024-11-19T04:15:44", 20552, 20560, 20576, 20539],
            ["2024-11-19T04:10:44", 20553, 20600, 20630, 20500],
            ["2024-11-19T04:05:44", 20600, 20590, 20700, 20560],
            ["2024-11-19T04:00:44", 20601, 20595, 20650, 20570],
            ["2024-11-19T03:55:44", 20610, 20597, 20645, 20588],
            ["2024-11-19T03:50:44", 20605, 20606, 20640, 20587],
        ]

        self.candle_data.reverse()  # Reverse to make the oldest first

        # Determine the Y-axis scale from the candle data
        self.price_max = max(candle[3] for candle in self.candle_data)+400  # Highest high price
        self.price_min = min(candle[4] for candle in self.candle_data)-400  # Lowest low price
        price_range = self.price_max - self.price_min
        self.price_step = round(price_range / 15)

        # Determine the start time from the first candle
        self.start_time = datetime.datetime.fromisoformat(self.candle_data[0][0])


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
        time, open_price, close_price, high_price, low_price = candle

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

class MACDArea(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("background-color: #141104; border: 3px solid black;")
        self.setFixedSize(1000, 200)

        self.candle_data = [
            ["2024-11-19T04:15:44", 13.8, 10.4, 3.4],
            ["2024-11-19T04:10:44", 13.5, 10.2, -3.3],
            ["2024-11-19T04:05:44", 13.2, 10.0, -3.2],
            ["2024-11-19T04:00:44", 0.0, 0.0, 0.0],
            ["2024-11-19T03:55:44", 15.2, 12.4, -2.8],
            ["2024-11-19T03:50:44", 15.0, 12.4, 2.6],
        ]

        self.candle_data.reverse()  # Reverse to make the oldest first

        # Determine the Y-axis scale from the candle data
        self.macd_line_max = max(candle[1] for candle in self.candle_data)  # Highest MACD-line
        self.macd_line_min = min(candle[1] for candle in self.candle_data)  # Lowest MACD-line
        self.signal_line_max = max(candle[2] for candle in self.candle_data)  # Highest signal-line
        self.signal_line_min = min(candle[2] for candle in self.candle_data)  # Lowest signal-line
        self.histogram_max = max(candle[3] for candle in self.candle_data)  # Highest histogram
        self.histogram_min = min(candle[3] for candle in self.candle_data)  # Lowest histogram

        # Determine the start time from the first candle
        self.start_time = datetime.datetime.fromisoformat(self.candle_data[0][0])

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
        current_time = self.start_time
        for index, x in enumerate(range(10, self.width(), self.grid_spacing_x)):
            painter.drawLine(x, 0, x, self.height())  # Vertical line
            # Only draw time labels for every other line
            if index % 2 == 0:
                time_label = current_time.strftime("%H:%M")
                painter.drawText(x + 5, self.height() - 5, time_label)  # Label slightly above the bottom
            current_time += datetime.timedelta(minutes=5)  # Increment by 5 minutes

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
            time, macd_value, signal_value, histogram_value = candle
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


class ConditionsArea(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("background-color: #141104; border: 3px solid black;")
        self.setFixedSize(300, 300)

        self.candle_data = [
            ["2024-11-19T04:15:44", 20552, 20560, 20576, 20539],
            ["2024-11-19T04:10:44", 20553, 20600, 20630, 20500],
            ["2024-11-19T04:05:44", 20600, 20590, 20700, 20560],
            ["2024-11-19T04:00:44", 20601, 20595, 20650, 20570],
            ["2024-11-19T03:55:44", 20610, 20597, 20645, 20588],
            ["2024-11-19T03:50:44", 20605, 20606, 20640, 20587],
        ]

        self.candle_data.reverse()  # Reverse to make the oldest first

        # Initialize the layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)

        # Add a title at the top
        title = QLabel("Conditions Overview")
        title.setStyleSheet("font-size: 16px; font-weight: bold; text-align: center; color: #FFD700;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # Create a grid layout for the conditions and results
        self.condition_grid = QGridLayout()
        layout.addLayout(self.condition_grid)

        from contracts.contracts import buy_candle_check, sell_candle_check, buy_macd_evaluation, sell_macd_evaluation, \
            quick_buy_check, quick_sell_check

        # Evaluate the conditions based on the candle data
        boo_buy_check = buy_candle_check(
            float(self.candle_data[0][1]), float(self.candle_data[1][1]), float(self.candle_data[2][1])
        )
        boo_sell_check = sell_candle_check(
            float(self.candle_data[0][1]), float(self.candle_data[1][1]), float(self.candle_data[2][1])
        )
        boo_quick_buy_check = quick_buy_check(float(self.candle_data[0][1]), float(self.candle_data[1][1]))
        boo_quick_sell_check = quick_sell_check(float(self.candle_data[0][1]), float(self.candle_data[1][1]))
        boo_buy_macd_check = buy_macd_evaluation()
        boo_sell_macd_check = sell_macd_evaluation()

        # Pair conditions with their results
        conditions = [
            ("BUY Candle Check", boo_buy_check),
            ("SELL Candle Check", boo_sell_check),
            ("BUY MACD Evaluation", boo_buy_macd_check),
            ("SELL MACD Evaluation", boo_sell_macd_check),
            ("Quick BUY Check", boo_quick_buy_check),
            ("Quick SELL Check", boo_quick_sell_check),
        ]

        # Populate the grid layout
        for i, (condition_name, result) in enumerate(conditions):
            # Condition label
            condition_label = QLabel(condition_name)
            condition_label.setStyleSheet("font-size: 12px; color: white;")
            self.condition_grid.addWidget(condition_label, i, 0)

            # Result label
            result_text = "✅" if result else "❌"
            result_label = QLabel(result_text)
            result_label.setAlignment(Qt.AlignCenter)
            result_label.setStyleSheet("font-size: 12px; font-weight: bold; color: white;")
            self.condition_grid.addWidget(result_label, i, 1)





class ScalpingbotView(QMainWindow):
    def create_shadow_effect(self):
        """Reusable method to create a shadow effect."""
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(20)  # Strength of the blur
        shadow.setColor(QColor(0, 0, 0, 160))  # Shadow color (semi-transparent black)
        shadow.setOffset(10, 10)  # Offset of the shadow (x, y)
        return shadow

    def __init__(self, parent=None, width=8, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.ax = fig.add_subplot(111)  # Add a subplot for the candlestick chart
        super().__init__()

        self.setWindowTitle("Scalping AI - Layout Test")
        self.setGeometry(100, 100, 1600, 1024)  # Location on the screen at startup (100 100) and the screen size (1600x1024)
        self.setStyleSheet("background-color: whitesmoke;")

        # Time update placeholder
        self.time_update = QLabel("Last update:", self)
        self.time_update.setStyleSheet(
            "font-size: 14px; color: whitesmoke; background-color: #141104; border: 3px solid black;"
        )
        self.time_update.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.time_update.setFixedSize(200, 50)
        self.time_update.move(25, 15)  # (x, y) position relative to the window
        self.time_update.setGraphicsEffect(self.create_shadow_effect())

        # Chart placeholder
        self.chart_area = ChartArea(self)
        self.chart_area.move(25, 80)  # (x, y) position relative to the window
        self.chart_area.setGraphicsEffect(self.create_shadow_effect())

        # MACD Placeholder
        self.macd_area = MACDArea(self)
        self.macd_area.move(25, 800)  # (x, y) position relative to the window
        self.macd_area.setGraphicsEffect(self.create_shadow_effect())


        # Profit details
        self.profit_details = QWidget(self)
        self.profit_details.setStyleSheet("background-color: #141104; border: 3px solid black;")
        self.profit_details.setFixedSize(300, 50)  # Adjust size as needed
        self.profit_details.move(724, 15)  # Position at the top-right corner
        self.profit_details.setGraphicsEffect(self.create_shadow_effect())
        # Layout for the profit details
        profit_layout = QHBoxLayout(self.profit_details)
        profit_layout.setContentsMargins(10, 10, 10, 10)  # Padding inside the container
        profit_layout.setSpacing(10)  # Space between elements

        # Profit details labels and values
        self.current_pl_label = QLabel("P/L: €")
        self.current_pl_value = QLabel("1.4")
        self.total_profit_label = QLabel("Profit: €")
        self.total_profit_value = QLabel("10.5")

        for label in [self.current_pl_label, self.total_profit_label]:
            label.setStyleSheet("font-size: 14px; font-weight: bold; color: whitesmoke;")
        for value in [self.current_pl_value, self.total_profit_value]:
            value.setStyleSheet("font-size: 14px; font-weight: bold; color: whitesmoke;")

        profit_layout.addWidget(self.current_pl_label)
        profit_layout.addWidget(self.current_pl_value)
        profit_layout.addSpacing(20)  # Optional spacing between groups
        profit_layout.addWidget(self.total_profit_label)
        profit_layout.addWidget(self.total_profit_value)

        profit_layout.addStretch()  # Push everything to the top


        # Progress bar placeholder
        self.progressbar = ProgressBarArea(self)
        self.progressbar.move(1100, 80)  # (x, y) position relative to the window
        self.progressbar.setGraphicsEffect(self.create_shadow_effect())


        # Contract details
        self.contract_details = QWidget(self)
        self.contract_details.setStyleSheet("background-color: #141104; border: 3px solid black;")
        self.contract_details.setFixedSize(300, 300)  # Adjust size as needed
        self.contract_details.move(self.width()-365, 80)  # Position at the top-right corner
        self.contract_details.setGraphicsEffect(self.create_shadow_effect())


        # Layout for the profit details
        contract_layout = QVBoxLayout(self.contract_details)
        contract_layout.setContentsMargins(10, 10, 10, 10)  # Padding inside the container
        contract_layout.setSpacing(10)  # Space between elements

        # Profit details labels and values
        self.current_contract_label = QLabel("Current contract:")
        self.current_contract_value = QLabel("HOLD")

        self.contract_size_label = QLabel("Contract size:")
        self.contract_size_value = QLabel("1.0")

        self.take_profit_label = QLabel("Take profit:")
        self.take_profit_value = QLabel("$" + str(8))

        self.stop_loss_label = QLabel("Stop loss:")
        self.stop_loss_value = QLabel("$" + str(4))

        # Add styles to labels and values
        for label in [self.current_contract_label, self.contract_size_label, self.take_profit_label,
                      self.stop_loss_label]:
            label.setStyleSheet("font-size: 14px; font-weight: bold; color: whitesmoke;")

        for value in [self.current_contract_value, self.contract_size_value, self.take_profit_value,
                      self.stop_loss_value]:
            value.setStyleSheet("font-size: 14px; font-weight: bold; color: white;")

        # Function to add label-value pair in a horizontal layout
        def add_label_value_pair(label, value):
            label.setFixedWidth(150)  # Adjust this width as needed
            row_layout = QHBoxLayout()  # Create a horizontal layout for each pair
            row_layout.addWidget(label)
            row_layout.addWidget(value)
            row_layout.addStretch()  # Pushes the content to the left
            contract_layout.addLayout(row_layout)  # Add the horizontal layout to the main vertical layout

        # Add each label-value pair to the contract layout
        add_label_value_pair(self.current_contract_label, self.current_contract_value)
        add_label_value_pair(self.contract_size_label, self.contract_size_value)
        add_label_value_pair(self.take_profit_label, self.take_profit_value)
        add_label_value_pair(self.stop_loss_label, self.stop_loss_value)

        # Conditions overview placeholder
        self.conditions_area = ConditionsArea(self)
        self.conditions_area.move(self.width()-365, 500)  # (x, y) position relative to the window
        self.conditions_area.setGraphicsEffect(self.create_shadow_effect())

        # # Right: Indicator and condition overview
        # indicator_layout = QVBoxLayout()
        # lower_layout.addLayout(indicator_layout)
        #
        # advice_label = QLabel("Current Advice: BUY")
        # advice_label.setStyleSheet("font-size: 16px; font-weight: bold; color: green;")
        # advice_label.setAlignment(Qt.AlignCenter)
        # indicator_layout.addWidget(advice_label)
        #
        # condition_grid = QGridLayout()
        # indicator_layout.addLayout(condition_grid)
        #
        # conditions = ["BUY Candle Check", "SELL Candle Check", "MACD Evaluation", "Quick BUY Check", "Quick SELL Check"]
        # results = ["✅", "❌", "✅", "✅", "❌"]
        # for i, (condition, result) in enumerate(zip(conditions, results)):
        #     label_condition = QLabel(condition)
        #     label_result = QLabel(result)
        #     label_result.setAlignment(Qt.AlignCenter)
        #     condition_grid.addWidget(label_condition, i, 0)
        #     condition_grid.addWidget(label_result, i, 1)
        #



if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ScalpingbotView()
    window.show()
    sys.exit(app.exec_())

