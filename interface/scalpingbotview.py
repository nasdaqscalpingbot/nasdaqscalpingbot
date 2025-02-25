from PyQt5.QtWidgets import QVBoxLayout, QHBoxLayout, QWidget, QGraphicsDropShadowEffect
from PyQt5.QtGui import QColor
from matplotlib.figure import Figure
from PyQt5.QtWidgets import QMainWindow, QLabel
from PyQt5.QtCore import Qt, pyqtSignal

import interface.chart
import interface.conditions
import interface.macd
import interface.progressbar

class ScalpingbotView(QMainWindow):

    update_ui_signal = pyqtSignal(list, float, float, str, str, float)
    def create_shadow_effect(self):
        """Reusable method to create a shadow effect."""
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(20)  # Strength of the blur
        shadow.setColor(QColor(0, 0, 0, 160))  # Shadow color (semi-transparent black)
        shadow.setOffset(10, 10)  # Offset of the shadow (x, y)
        return shadow

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Scalping AI - Layout Test")
        self.setGeometry(100, 100, 1600, 1024)

        # ✅ Connect the signal to the update method
        self.update_ui_signal.connect(self.update_ui)

        # Example UI components
        self.label_balance = QLabel("Balance: ", self)
        self.label_balance.move(50, 50)

        self.label_advice = QLabel("Advice: ", self)
        self.label_advice.move(50, 80)

        self.label_time = QLabel("Time: ", self)
        self.label_time.move(50, 110)

        self.label_macd = QLabel("MACD: ", self)
        self.label_macd.move(50, 140)

        self.label_signal = QLabel("Signal: ", self)
        self.label_signal.move(50, 170)

        self.setWindowTitle("Scalping AI")
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
        self.chart_area = interface.chart.ChartArea(self)
        self.chart_area.move(25, 80)  # (x, y) position relative to the window
        self.chart_area.setGraphicsEffect(self.create_shadow_effect())

        # MACD Placeholder
        self.macd_area = interface.macd.MACDArea(self)
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
        self.progressbar = interface.progressbar.ProgressBarArea(self)
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
        self.conditions_area = interface.conditions.ConditionsArea(self)
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

    def update_ui(self, candles, macd, signal, advice, current_time, balance):
        """Update the UI elements with new data."""
        print("Updating UI...")
        print(f"Candles: {candles}")
        print(f"MACD: {macd}")
        print(f"Signal: {signal}")
        print(f"Advice: {advice}")
        print(f"Current Time: {current_time}")
        print(f"Balance: {balance}")

        self.chart_area.update_candles(candles)

        self.label_balance.setText(f"Balance: {balance}")
        self.label_advice.setText(f"Advice: {advice}")
        self.label_time.setText(f"Time: {current_time}")
        self.label_macd.setText(f"MACD: {macd}")
        self.label_signal.setText(f"Signal: {signal}")