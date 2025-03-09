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
        self.setStyleSheet("background-color: #1E1E1E; border: 3px solid black;")
        self.setFixedSize(300, 400)  # Adjust size as needed

        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(10, 10, 10, 10)

        # Advice Label
        self.advice_label = QLabel("Current Advice: HOLD")
        self.advice_label.setStyleSheet("font-size: 16px; font-weight: bold; color: whitesmoke;")
        self.advice_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.advice_label)

        # Conditions Grid
        self.condition_grid = QGridLayout()
        self.layout.addLayout(self.condition_grid)

        # Store condition labels for updates
        self.condition_labels = {}
        self.result_labels = {}

        # Condition list (initialized as False)
        self.conditions = {
            "Current > previous": False,
            "Previous > oldest": False,
            "MACD > Signal": False,
            "BUY Candle Check": False,
            "Current < previous": False,
            "Previous < oldest": False,
            "Signal > MACD": False,
            "SELL Candle Check": False,
        }

        buy_criteria_added = False
        sell_criteria_added = False
        row_index = 0  # Track row index for proper placement

        for condition in self.conditions.keys():
            # Add "Buy Criteria" label before the first buy condition
            if condition == "Current > previous" and not buy_criteria_added:
                label_buy_criteria = QLabel("Buy Criteria")
                label_buy_criteria.setStyleSheet("font-size: 16px; font-weight: bold; color: white;")
                self.condition_grid.addWidget(label_buy_criteria, row_index, 0, 1, 2)  # Span two columns
                buy_criteria_added = True
                row_index += 1  # Move to the next row

            # Add "Sell Criteria" label before the first sell condition
            if condition == "Current < previous" and not sell_criteria_added:
                label_sell_criteria = QLabel("Sell Criteria")
                label_sell_criteria.setStyleSheet("font-size: 16px; font-weight: bold; color: white;")
                self.condition_grid.addWidget(label_sell_criteria, row_index, 0, 1, 2)  # Span two columns
                sell_criteria_added = True
                row_index += 1  # Move to the next row

            # Create condition label
            label_condition = QLabel(condition)
            label_condition.setStyleSheet("font-size: 14px; color: whitesmoke;")

            # Create result label (default ❌)
            label_result = QLabel("❌")
            label_result.setAlignment(Qt.AlignCenter)
            label_result.setStyleSheet("font-size: 14px; color: red;")

            # Add to grid
            self.condition_grid.addWidget(label_condition, row_index, 0)
            self.condition_grid.addWidget(label_result, row_index, 1)

            # Store references for updates
            self.condition_labels[condition] = label_condition
            self.result_labels[condition] = label_result
            row_index += 1  # Move to the next row

    def update_conditions(self, lst_fetched_candles, macd, signal, advice):
        """Update the advice and condition results dynamically."""
        if not lst_fetched_candles or len(lst_fetched_candles) < 3:
            return
        # Reset all conditions before checking new values
        for key in self.conditions.keys():
            self.conditions[key] = False

        # Get relevant candle values
        current_candle_open = lst_fetched_candles[0][0]
        previous_candle_open = lst_fetched_candles[1][0]
        oldest_candle_open = lst_fetched_candles[2][0]

        # Condition Checks
        if current_candle_open > (previous_candle_open + 5):
            self.conditions["Current > previous"] = True

        if previous_candle_open > (oldest_candle_open + 5):
            self.conditions["Previous > oldest"] = True

        if macd > (signal + 2):
            self.conditions["MACD > Signal"] = True

        if current_candle_open < (previous_candle_open - 5):
            self.conditions["Current < previous"] = True

        if previous_candle_open < (oldest_candle_open - 5):  # Fixed condition logic
            self.conditions["Previous < oldest"] = True

        if signal > (macd + 2):
            self.conditions["Signal > MACD"] = True

        # Update Advice Label
        self.advice_label.setText(f"Current Advice: {advice}")
        self.advice_label.setStyleSheet(
            f"font-size: 16px; font-weight: bold; color: whitesmoke; "
            f"background-color: {'green' if advice == 'BUY' else 'red' if advice == 'SELL' else 'transparent'}; "
            "border-radius: 5px; padding: 5px;"
        )

        # Update Condition Labels
        for condition, passed in self.conditions.items():  # Fixed `self.conditions.items()`
            if condition in self.result_labels:
                if passed:
                    self.result_labels[condition].setText("✅")
                    self.result_labels[condition].setStyleSheet("font-size: 14px; color: green;")
                else:
                    self.result_labels[condition].setText("❌")
                    self.result_labels[condition].setStyleSheet("font-size: 14px; color: red;")
