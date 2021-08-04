from collections import deque
import sys

from PyQt5 import QtWidgets, uic
from PyQt5.QtCore import QTimer
from pyqtgraph import PlotWidget, plot
import pyqtgraph as pg

from starting_inference_helpers import start_inference_server
from evaluator.server_udp import PHARUS_FIELD_SIZE_X, PHARUS_FIELD_SIZE_Y

FPS_AVERAGING_WINDOW = 10

PRED_LENGTH = 9

class Ui(QtWidgets.QMainWindow):
    def __init__(self):
        # Call the inherited classes __init__ method
        super(Ui, self).__init__()
        uic.loadUi('inference_server.ui', self)  # Load the .ui file

        self.plot_view_pharus = pg.PlotWidget()
        self.plot_view_ml = pg.PlotWidget()

        self.scatter_plot_item_pharus = pg.ScatterPlotItem(pen=pg.mkPen(width=5, color='r'), symbol='o', size=1)
        self.scatter_plot_item_pharus_obs = pg.ScatterPlotItem(pen=pg.mkPen(width=5, color='r'), symbol='o', size=1)
        self.scatter_plot_item_ml = pg.ScatterPlotItem(pen=pg.mkPen(width=5, color='g'), symbol='o', size=1)
        self.plot_view_pharus.addItem(self.scatter_plot_item_pharus)
        self.plot_view_pharus.setXRange(0, PHARUS_FIELD_SIZE_X)
        self.plot_view_pharus.setYRange(0,PHARUS_FIELD_SIZE_Y)
        self.plot_view_ml.setXRange(0, PHARUS_FIELD_SIZE_X)
        self.plot_view_ml.setYRange(0, PHARUS_FIELD_SIZE_Y)
        self.plot_view_ml.addItem(self.scatter_plot_item_ml)
        self.plot_view_ml.addItem(self.scatter_plot_item_pharus_obs)

        self.visualizer_tab_widget = self.findChild(QtWidgets.QTabWidget, 'visualizer')
        self.visualizer_tab_widget.addTab(self.plot_view_ml, "ml")
        self.visualizer_tab_widget.addTab(self.plot_view_pharus, "pharus")

        self.pharus_data = {}
        self.pharus_obs_data = []
        self.ml_data = []

        self.button = self.findChild(QtWidgets.QPushButton, 'start_button')
        self.ml_fps = self.findChild(QtWidgets.QLCDNumber, 'ml_fps')
        self.pharus_fps = self.findChild(QtWidgets.QLCDNumber, 'pharus_fps')
        self.button.clicked.connect(self.start_inference)

        self.pharus_fps_deque = deque(maxlen=FPS_AVERAGING_WINDOW)
        self.ml_fps_deque = deque(maxlen=FPS_AVERAGING_WINDOW)

        self.threads = []

        self.show()  # Show the GUI

    def start_inference(self):
        if self.button.text() == "stop":
            return
        self.button.setText("stop")
        self.threads = start_inference_server(pharus_receiver_ip="localhost",
                                             touch_designer_ip="192.168.0.1",
                                             fps_callback=self.fps_callback,
                                             pharus_fps_callback=self.pharus_fps_callback)
        self.ml_fps.setStyleSheet("background-color: rgb(78, 154, 6);")
        self.pharus_fps.setStyleSheet("background-color: rgb(78, 154, 6);")
        self.button.clicked.connect(self.stop_inference_server)


    def stop_inference_server(self):
        for thread in self.threads:
            thread.terminate()

    def fps_callback(self, fps, obs_paths, paths):
        self.ml_fps_deque.append(fps)
        avg_fps = sum(list(self.ml_fps_deque))//FPS_AVERAGING_WINDOW
        self.ml_fps.display(avg_fps)

        self.ml_data.clear()
        for row in paths:
            self.ml_data.append({"pos": [row.x, row.y]})

        self.pharus_obs_data.clear()
        for person_paths in obs_paths:
            for row in person_paths:
                self.pharus_obs_data.append({"pos": [row.x, row.y]})


    def pharus_fps_callback(self, fps, paths):
        pass
        """
        self.pharus_fps_deque.append(fps)
        avg_fps = sum(list(self.pharus_fps_deque))//FPS_AVERAGING_WINDOW
        self.pharus_fps.display(avg_fps)

        if not paths:
            return

        for person_paths in paths:
            ped_id = person_paths[0].pedestrian
            if not ped_id in self.pharus_data:
                self.pharus_data[ped_id] = deque(maxlen=PRED_LENGTH)
            for row in person_paths:
                self.pharus_data[row.pedestrian].append({"pos": [row.x, row.y]})

        del_ped_ids = []
        for ped_id in self.pharus_data:
            match = False
            for person_paths in paths:
                if ped_id == person_paths[0].pedestrian:
                    match = True
            if not match:
                del_ped_ids.append(ped_id)

        for ped_id in del_ped_ids:
            del self.pharus_data[ped_id]
        """

    def plot_graph(self):
        """
        pharus_data = []
        for entries in [list(ped_deque) for _, ped_deque in self.pharus_data.items()]:
            pharus_data.extend(entries)
        self.scatter_plot_item_pharus.setData(pharus_data)
        """
        self.scatter_plot_item_ml.setData(self.ml_data)
        self.scatter_plot_item_pharus_obs.setData(self.pharus_obs_data)

# Create an instance of QtWidgets.QApplication
app = QtWidgets.QApplication(sys.argv)
window = Ui()  # Create an instance of our class

timer = QTimer()
timer.timeout.connect(window.plot_graph)
timer.start(25)
app.exec_()  # Start the
