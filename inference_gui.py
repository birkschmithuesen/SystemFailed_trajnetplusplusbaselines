from collections import deque

from PyQt5 import QtWidgets, uic
import sys

from starting_inference_helpers import start_inference_server

FPS_AVERAGING_WINDOW = 10

class Ui(QtWidgets.QMainWindow):
    def __init__(self):
        super(Ui, self).__init__() # Call the inherited classes __init__ method
        uic.loadUi('inference_server.ui', self) # Load the .ui file

        self.button = self.findChild(QtWidgets.QPushButton, 'start_button')
        self.ml_fps = self.findChild(QtWidgets.QLCDNumber, 'ml_fps')
        self.pharus_fps = self.findChild(QtWidgets.QLCDNumber, 'pharus_fps')
        self.button.clicked.connect(self.start_inference_server)

        self.pharus_fps_deque = deque(maxlen=FPS_AVERAGING_WINDOW)
        self.ml_fps_deque = deque(maxlen=FPS_AVERAGING_WINDOW)

        self.show() # Show the GUI

    def start_inference_server(self):
        start_inference_server(pharus_receiver_ip="192.168.0.3", fps_callback=self.fps_callback, pharus_fps_callback=self.pharus_fps_callback)
        self.ml_fps.setStyleSheet("background-color: rgb(78, 154, 6);")
        self.ml_fps.display(200)
        self.pharus_fps.setStyleSheet("background-color: rgb(78, 154, 6);")
        self.pharus_fps.display(200)

    def fps_callback(self, fps):
        self.ml_fps_deque.append(fps)
        avg_fps = sum(list(self.ml_fps_deque))//FPS_AVERAGING_WINDOW
        self.ml_fps.display(avg_fps)

    def pharus_fps_callback(self, fps):
        self.pharus_fps_deque.append(fps)
        avg_fps = sum(list(self.pharus_fps_deque))//FPS_AVERAGING_WINDOW
        self.pharus_fps.display(avg_fps)

app = QtWidgets.QApplication(sys.argv) # Create an instance of QtWidgets.QApplication
window = Ui() # Create an instance of our class
app.exec_() # Start the application
