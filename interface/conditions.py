import sys
from PyQt5.QtWidgets import QVBoxLayout, QHBoxLayout, QGridLayout, \
    QWidget, QGraphicsDropShadowEffect
from PyQt5.QtGui import QColor, QLinearGradient
from matplotlib.figure import Figure
import datetime
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel
from PyQt5.QtGui import QPainter, QPen
from PyQt5.QtCore import Qt

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

        # from strategy import strategycheck

        # macd_advice, macd, signal = strategycheck()

        # # Evaluate the conditions based on the candle data
        # boo_buy_check = buy_candle_check(
        #     float(self.candle_data[0][1]), float(self.candle_data[1][1]), float(self.candle_data[2][1])
        # )
        # boo_sell_check = sell_candle_check(
        #     float(self.candle_data[0][1]), float(self.candle_data[1][1]), float(self.candle_data[2][1])
        # )
        # boo_quick_buy_check = quick_buy_check(float(self.candle_data[0][1]), float(self.candle_data[1][1]))
        # boo_quick_sell_check = quick_sell_check(float(self.candle_data[0][1]), float(self.candle_data[1][1]))
        # boo_buy_macd_check = buy_macd_evaluation()
        # boo_sell_macd_check = sell_macd_evaluation()
        #
        # # Pair conditions with their results
        # conditions = [
        #     ("BUY Candle Check", boo_buy_check),
        #     ("SELL Candle Check", boo_sell_check),
        #     ("BUY MACD Evaluation", boo_buy_macd_check),
        #     ("SELL MACD Evaluation", boo_sell_macd_check),
        #     ("Quick BUY Check", boo_quick_buy_check),
        #     ("Quick SELL Check", boo_quick_sell_check),
        # ]
        #
        # # Populate the grid layout
        # for i, (condition_name, result) in enumerate(conditions):
        #     # Condition label
        #     condition_label = QLabel(condition_name)
        #     condition_label.setStyleSheet("font-size: 12px; color: white;")
        #     self.condition_grid.addWidget(condition_label, i, 0)
        #
        #     # Result label
        #     result_text = "✅" if result else "❌"
        #     result_label = QLabel(result_text)
        #     result_label.setAlignment(Qt.AlignCenter)
        #     result_label.setStyleSheet("font-size: 12px; font-weight: bold; color: white;")
        #     self.condition_grid.addWidget(result_label, i, 1)