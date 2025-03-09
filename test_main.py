# startloop.py
import sys
import threading


from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtCore import QTimer

import interface.GridVisualization
from interface.scalpingbotview import ScalpingbotView


def main():
    import sys
    app = QApplication(sys.argv)

    window = QMainWindow()
    window.setGeometry(100, 100, 1000, 600)

    # Create grid
    window.grid = interface.GridVisualization.GridVisualization(window)
    window.grid.move(50, 50)
    window.show()

    # Use QTimer correctly
    timer = QTimer(window)  # Attach the timer to the window
    timer.setSingleShot(True)
    timer.timeout.connect(lambda: window.grid.add_line([(150, 250), (400, 50)]))
    timer.start(3000)

    sys.exit(app.exec_())


# ========================================== Main start ================================================================
if __name__ == "__main__":
    main()
