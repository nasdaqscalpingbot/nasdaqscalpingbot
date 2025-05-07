from PyQt5.QtWidgets import QVBoxLayout, QHBoxLayout, QWidget, QGraphicsDropShadowEffect
from PyQt5.QtGui import QColor
from matplotlib.figure import Figure
from PyQt5.QtWidgets import QMainWindow, QLabel
from PyQt5.QtCore import Qt, pyqtSignal

from datetime import datetime

from connection import account_details
import interface.chart
import interface.conditions
import interface.macd
import interface.progressbar

class ScalpingbotView(QMainWindow):
    update_ui_signal = pyqtSignal(
         list, float, float, str, float, str, int, str, float, int, float, int
    )
    def create_shadow_effect(self):
        """Reusable method to create a shadow effect."""
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(20)  # Strength of the blur
        shadow.setColor(QColor(0, 0, 0, 160))  # Shadow color (semi-transparent black)
        shadow.setOffset(10, 10)  # Offset of the shadow (x, y)
        return shadow

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Scalping AI")
        self.setGeometry(100, 100, 1600, 1024)

        # ✅ Connect the signal to the update method
        self.update_ui_signal.connect(self.update_ui)

        self.setWindowTitle("Scalping AI Version 2.2.1")
        self.setGeometry(100, 100, 1600, 1024)  # Location on the screen at startup (100 100) and the screen size (1600x1024)
        margin_left = 50
        margin_right = 50
        self.setStyleSheet(
            "background-color: #141414;"  # Dark background for the window
            "font-family: 'Arial';"
            "font-size: 16px;"
        )

        # Common widget style
        widget_style = "background-color: #1E1E1E; color: whitesmoke; border: 2px solid black;"

        self.top_bar = QWidget(self)
        self.top_bar.setFixedSize(self.width(), 50)  # Full-width container
        self.top_bar.move(0, 15)  # Position at the top

        # Create a horizontal layout
        top_layout = QHBoxLayout(self.top_bar)
        top_layout.setContentsMargins(margin_left, 0, margin_right, 0)  # Margins for spacing

        # Define label texts
        label_texts = ["Start:", "Last update:", "Candle #:", "Current P&L:", "Day profit:", "Balance:"]

        # Get the available width (total width minus margins)
        available_width = self.top_bar.width() - 100  # Subtract left & right margins (50px each)

        # Define a fixed label width (you can adjust this)
        label_width = 230
        num_labels = len(label_texts)

        # Calculate dynamic spacing
        if num_labels > 1:
            spacing = (available_width - (num_labels * label_width)) // (num_labels - 1)
        else:
            spacing = 0  # No spacing needed if only one label

        top_layout.setSpacing(spacing)  # Apply calculated spacing

        # Create and add labels dynamically
        self.labels = []
        for text in label_texts:
            label = QLabel(text, self.top_bar)
            label.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
            label.setFixedSize(label_width, 50)
            label.setStyleSheet(widget_style)
            label.setGraphicsEffect(self.create_shadow_effect())

            self.labels.append(label)
            top_layout.addWidget(label)

        # Add a stretch to push the balance label to the right
        top_layout.addStretch()


        # Chart placeholder
        self.chart_area = interface.chart.ChartArea(self)
        self.chart_area.move(margin_left, 80)  # (x, y) position relative to the window
        self.chart_area.setGraphicsEffect(self.create_shadow_effect())


        # # MACD Placeholder
        # self.macd_area = interface.macd.MACDArea(self)
        # self.macd_area.move(margin_left, 800)  # (x, y) position relative to the window
        # self.macd_area.setGraphicsEffect(self.create_shadow_effect())


        # Progress bar placeholder
        self.progressbar = interface.progressbar.ProgressBarArea(self)
        self.progressbar.move(1090, 80)  # (x, y) position relative to the window
        self.progressbar.setGraphicsEffect(self.create_shadow_effect())


        # Contract details
        self.contract_details = QWidget(self)
        self.contract_details.setFixedSize(300, 200)  # Adjust size as needed
        self.contract_details.move(self.width()-350, 80)  # Position at the top-right corner
        self.contract_details.setStyleSheet(widget_style)
        self.contract_details.setGraphicsEffect(self.create_shadow_effect())


        # Layout for the profit details
        contract_layout = QVBoxLayout(self.contract_details)
        contract_layout.setContentsMargins(10, 10, 10, 10)  # Padding inside the container
        contract_layout.setSpacing(10)  # Space between elements

        # Profit details labels and values
        self.current_contract_label = QLabel("Current contract:")
        self.current_contract_value = QLabel()

        self.contract_size_label = QLabel("Contract size:")
        self.contract_size_value = QLabel()

        self.stop_loss_label = QLabel("Stop loss distance:")
        self.stop_loss_value = QLabel()

        value_style = "color: #FFD700; font-weight: bold;"  # Soft yellowish color
        self.current_contract_value.setStyleSheet(value_style)
        self.contract_size_value.setStyleSheet(value_style)
        self.stop_loss_value.setStyleSheet(value_style)

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
        add_label_value_pair(self.stop_loss_label, self.stop_loss_value)

        # Conditions overview placeholder
        self.conditions_area = interface.conditions.ConditionsArea(self)
        self.conditions_area.move(self.width()-350, 340)  # (x, y) position relative to the window
        self.conditions_area.setStyleSheet(widget_style)
        self.conditions_area.setGraphicsEffect(self.create_shadow_effect())

        # # Market details
        self.market_details = QWidget(self)
        self.market_details.setFixedSize(300, 200)  # Adjust size as needed
        self.market_details.setStyleSheet(widget_style)
        self.market_details.move(self.width()-350, 800)  # Position at the top-right corner
        self.market_details.setGraphicsEffect(self.create_shadow_effect())
        # Layout for the profit details
        market_layout = QVBoxLayout(self.market_details)
        market_layout.setContentsMargins(5, 5, 5, 5)  # Padding inside the container
        market_layout.setSpacing(5)  # Space between elements

        # Market details labels and values
        self.current_candle_label = QLabel("Current open:")
        self.current_candle_value = QLabel()
        self.previous_candle_label = QLabel("Previous open:")
        self.previous_candle_value = QLabel()
        self.oldest_candle_label = QLabel("Oldest open:")
        self.oldest_candle_value = QLabel()
        self.macd_label = QLabel("MACD-line:")
        self.macd_value = QLabel()
        self.signal_label = QLabel("Signal-line:")
        self.signal_value = QLabel()

        value_style = "color: #FFD700; font-weight: bold;"  # Soft yellowish color
        self.current_candle_value.setStyleSheet(value_style)
        self.previous_candle_value.setStyleSheet(value_style)
        self.oldest_candle_value.setStyleSheet(value_style)
        self.macd_value.setStyleSheet(value_style)
        self.signal_value.setStyleSheet(value_style)


        for label, value in zip(
                [self.current_candle_label, self.previous_candle_label, self.oldest_candle_label, self.macd_label, self.signal_label],
                [self.current_candle_value, self.previous_candle_value, self.oldest_candle_value, self.macd_value, self.signal_value]):
            row_layout = QHBoxLayout()
            row_layout.addWidget(label)
            row_layout.addWidget(value)
            market_layout.addLayout(row_layout)

    def update_ui(self, candles, macd, signal, advice, start_balance, start_datetime,
                  int_number_of_candles, market_condition, flo_contract_size,
                  int_stop_loss_distance, int_this_contract_profit, int_positive_counter):

        current_time = datetime.now().strftime("%H:%M:%S")

        # print(candles, macd, signal, advice, start_balance, start_datetime,
        # int_number_of_candles, market_condition, flo_contract_size,
        # int_stop_loss_distance, int_this_contract_profit, int_positive_counter)


        account_information = account_details()
        flo_current_balance = round(account_information['accounts'][0]['balance']['balance'], 2)
        total_profit = round(flo_current_balance - start_balance, 2)

        label_color = "whitesmoke"
        value_color = "#D4AF37"

        self.label_map = {
            "Start:": self.labels[0],
            "Last update:": self.labels[1],
            "Candle #:": self.labels[2],
            "Current P&L:": self.labels[3],
            "Day profit:": self.labels[4],
            "Balance:": self.labels[5]
        }

        # Enable rich text formatting for QLabel
        for label in self.label_map.values():
            label.setTextFormat(Qt.RichText)

        # Function to update labels dynamically with different colors
        def update_label(label_key, value):
            self.label_map[label_key].setText(
                f'<span style="color: {label_color};">{label_key}</span> '
                f'<span style="color: {value_color};">{value}</span>'
            )

        # Updating values
        update_label("Start:", start_datetime)
        update_label("Last update:", current_time)
        update_label("Candle #:", f"{int_number_of_candles}")
        update_label("Current P&L:", f"€{str(int_this_contract_profit)}")
        update_label("Day profit:", f"€{str(total_profit)}")
        update_label("Balance:", f"€{flo_current_balance}")

        self.chart_area.update_candles(candles)
        self.chart_area.update_macd([candles[-1][2], macd, signal])
        self.progressbar.update_bar(int_positive_counter)
        self.current_contract_value.setText(advice)
        self.contract_size_value.setText(str(flo_contract_size))
        self.stop_loss_value.setText(str(int_stop_loss_distance))
        self.current_candle_value.setText(str(candles[0][0]))
        if len(candles) > 1:
            self.previous_candle_value.setText(str(candles[1][0]))
        if len(candles) > 2:
            self.oldest_candle_value.setText(str(candles[2][0]))
        self.macd_value.setText(str(macd))
        self.signal_value.setText(str(signal))
        self.conditions_area.update_conditions(candles, macd, signal, advice, market_condition)


        self.update()