from collections import deque
import sys, os
import matplotlib.pyplot as plt


from PyQt5 import QtWidgets, QtCore, QtGui, uic
from PyQt5.QtCore import QTimer, QSettings
from pyqtgraph import PlotWidget, plot
import pyqtgraph as pg

import pandas as pd

from data_conversions_helpers import pharus_convert
from starting_inference_helpers import start_inference_server, start_udp_splitter
from starting_training_helpers import start_training_thread, get_training_data, training_folder_is_valid, pharus_recording_is_valid,get_training_df_positions
from evaluator.server_udp import PHARUS_FIELD_SIZE_X, PHARUS_FIELD_SIZE_Y

FPS_AVERAGING_WINDOW = 10

PRED_LENGTH = 4

cmap = plt.cm.get_cmap('rainbow')

def get_rgb_val(total_n, index):
    if index == 0:
        rgba = cmap(0)
        return (rgba[0]*255, rgba[1]*255, rgba[2]*255)
    rgba = cmap(index/total_n)
    return (rgba[0]*255, rgba[1]*255, rgba[2]*255)


class Ui(QtWidgets.QMainWindow):
    def __init__(self):
        # Call the inherited classes __init__ method
        super(Ui, self).__init__()
        uic.loadUi('gui/gui.ui', self)  # Load the .ui file
        self.settings = QSettings("General", "SystemFailed Trajectory Inference")
        self.load_settings()

        self.plot_view_pharus = pg.PlotWidget()
        self.plot_view_ml = pg.PlotWidget()

        self.scatter_plot_item_pharus = pg.ScatterPlotItem(
            pen=pg.mkPen(width=5, color='r'), symbol='o', size=1)
        self.scatter_plot_item_pharus_obs = pg.ScatterPlotItem(
            pen=pg.mkPen(width=5, color='r'), symbol='o', size=1)
        self.scatter_plot_item_ml = pg.ScatterPlotItem(
            pen=pg.mkPen(width=5, color='g'), symbol='o', size=1)
        self.plot_view_pharus.addItem(self.scatter_plot_item_pharus)
        self.plot_view_pharus.setXRange(0, PHARUS_FIELD_SIZE_X)
        self.plot_view_pharus.setYRange(0, PHARUS_FIELD_SIZE_Y)
        self.plot_view_ml.setXRange(0, PHARUS_FIELD_SIZE_X)
        self.plot_view_ml.setYRange(0, PHARUS_FIELD_SIZE_Y)
        self.plot_view_ml.addItem(self.scatter_plot_item_ml)
        self.plot_view_ml.addItem(self.scatter_plot_item_pharus_obs)

        self.visualizer_tab_widget = self.findChild(
            QtWidgets.QTabWidget, 'visualizer')
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

        # training tab
        self.plot_view_training = pg.PlotWidget()
        self.plot_view_training.setXRange(0, PHARUS_FIELD_SIZE_X)
        self.plot_view_training.setYRange(0, PHARUS_FIELD_SIZE_Y)
        self.plot_items_training = []

        self.visualizer_training_tab_widget = self.findChild(
            QtWidgets.QTabWidget, 'training_visualizer')
        self.visualizer_training_tab_widget.addTab(self.plot_view_training, "training")

        self.training_data_select_button = self.findChild(QtWidgets.QPushButton, 'training_data_select_button')
        self.training_data_select_button.clicked.connect(self.select_training_data)
        self.training_data_select_label = self.findChild(QtWidgets.QLabel, 'training_data_select_label')
        self.training_data_visualize_button = self.findChild(QtWidgets.QPushButton, 'training_visualize_data_button')
        self.training_data_visualize_button.clicked.connect(self.visualize_training_data)

        self.pharus_data_select_button = self.findChild(QtWidgets.QPushButton, 'pharus_data_select_button')
        self.pharus_data_select_button.clicked.connect(self.select_pharus_data)
        self.pharus_data_select_label = self.findChild(QtWidgets.QLabel, 'pharus_data_select_label')
        self.training_convert_pharus_button = self.findChild(QtWidgets.QPushButton, 'training_convert_pharus_button')
        self.training_convert_pharus_button.clicked.connect(self.convert_pharus_data)

        self.training_start_button = self.findChild(QtWidgets.QPushButton, 'training_start_button')
        self.training_start_button.clicked.connect(self.start_training)

        self.training_start_button = self.findChild(QtWidgets.QPushButton, 'continue_training_button')
        self.training_start_button.clicked.connect(self.continue_training)

        # variables used for non UI functionality
        self.training_data_path = ""


        self.threads = []
        self.training_threads = []
        self.udp_splitter_thread = None

        self.show()  # Show the GUI

    def load_settings(self):
        try:
            self.findChild(QtWidgets.QPlainTextEdit, 'pharus_listener_ip').setPlainText(self.settings.value('pharus_listener_ip'))
            self.findChild(QtWidgets.QPlainTextEdit, 'touch_designer_pc_ip').setPlainText(self.settings.value('touch_designer_pc_ip'))
            self.findChild(QtWidgets.QSpinBox, 'inference_pred_length').setValue(int(self.settings.value('inference_pred_length')))
            self.findChild(QtWidgets.QSpinBox, 'sliding_window_frames').setValue(int(self.settings.value('sliding_window_frames')))

            self.findChild(QtWidgets.QSpinBox, 'epochs').setValue(int(self.settings.value('epochs')))
            self.findChild(QtWidgets.QSpinBox, 'pred_length').setValue(int(self.settings.value('pred_length')))
            self.findChild(QtWidgets.QSpinBox, 'obs_length').setValue(int(self.settings.value('obs_length')))
            self.findChild(QtWidgets.QSpinBox, 'batch_size').setValue(int(self.settings.value('batch_size')))
            self.findChild(QtWidgets.QDoubleSpinBox, 'learning_rate').setValue(float(self.settings.value('learning_rate')))
            self.findChild(QtWidgets.QSpinBox, 'step_size_lr_scheduler').setValue(int(self.settings.value('step_size_lr_scheduler')))
            self.findChild(QtWidgets.QSpinBox, 'save_every_n_epochs').setValue(int(self.settings.value('save_every_n_epochs')))
        except:
            pass

    def save_settings(self):
        pharus_listener_ip = self.findChild(QtWidgets.QPlainTextEdit, 'pharus_listener_ip').toPlainText()
        touch_designer_pc_ip = self.findChild(QtWidgets.QPlainTextEdit, 'touch_designer_pc_ip').toPlainText()
        inference_pred_length = self.findChild(QtWidgets.QSpinBox, 'inference_pred_length').value()
        inference_obs_length = self.findChild(QtWidgets.QSpinBox, 'inference_obs_length').value()
        sliding_window_frames  = self.findChild(QtWidgets.QSpinBox, 'sliding_window_frames').value()

        epochs = self.findChild(QtWidgets.QSpinBox, 'epochs').value()
        pred_length = self.findChild(QtWidgets.QSpinBox, 'pred_length').value()
        obs_length = self.findChild(QtWidgets.QSpinBox, 'obs_length').value()
        batch_size = self.findChild(QtWidgets.QSpinBox, 'batch_size').value()
        learning_rate = self.findChild(QtWidgets.QDoubleSpinBox, 'learning_rate').value()
        step_size_lr_scheduler = self.findChild(QtWidgets.QSpinBox, 'step_size_lr_scheduler').value()
        save_every_n_epochs = self.findChild(QtWidgets.QSpinBox, 'save_every_n_epochs').value()

        self.settings.setValue('pharus_listener_ip', pharus_listener_ip)
        self.settings.setValue('touch_designer_pc_ip', touch_designer_pc_ip)
        self.settings.setValue('inference_pred_length', inference_pred_length)
        self.settings.setValue('inference_obs_length', inference_obs_length)
        self.settings.setValue('sliding_window_frames', sliding_window_frames)

        self.settings.setValue('epochs', epochs)
        self.settings.setValue('pred_length', pred_length)
        self.settings.setValue('obs_length', obs_length)
        self.settings.setValue('batch_size', batch_size)
        self.settings.setValue('learning_rate', learning_rate)
        self.settings.setValue('step_size_lr_scheduler', step_size_lr_scheduler)
        self.settings.setValue('obs_length', obs_length)
        self.settings.setValue('save_every_n_epochs', save_every_n_epochs)

    def start_inference(self):
        if self.button.text() == "stop":
            return
        self.button.setText("stop")
        listener_ip = self.findChild(QtWidgets.QPlainTextEdit, 'pharus_listener_ip').toPlainText()
        touch_designer_pc_ip = self.findChild(QtWidgets.QPlainTextEdit, 'touch_designer_pc_ip').toPlainText()
        pred_length = self.findChild(QtWidgets.QSpinBox, 'inference_pred_length').value()
        obs_length = self.findChild(QtWidgets.QSpinBox, 'inference_obs_length').value()
        fileselection = QtWidgets.QFileDialog.getOpenFileName(self, "Select Model (e.g., model.pkl.epoch30)")
        path = fileselection[0]
        client_and_threads = start_inference_server(model_path=path,
                                                    pharus_receiver_ip=listener_ip,
                                                    touch_designer_ip=touch_designer_pc_ip,
                                                    fps_callback=self.fps_callback,
                                                    pharus_fps_callback=self.pharus_fps_callback,
                                                    pred_length=pred_length,
                                                    obs_length=obs_length)

        client = client_and_threads[0]
        t1 = client_and_threads[1]
        t2 = client_and_threads[2]

        self.threads.extend([(client, t1), t2])

        if not self.udp_splitter_thread:
            self.udp_splitter_thread = start_udp_splitter(listener_ip, touch_designer_pc_ip)
        self.findChild(QtWidgets.QSpinBox, 'inference_pred_length').valueChanged.connect(client._listener[0].update_pred_length)
        self.findChild(QtWidgets.QSpinBox, 'inference_obs_length').valueChanged.connect(client._listener[0].update_obs_length)
        self.findChild(QtWidgets.QSpinBox, 'sliding_window_frames').valueChanged.connect(client._listener[0].update_sliding_window_size)
        self.ml_fps.setStyleSheet("background-color: rgb(78, 154, 6);")
        self.pharus_fps.setStyleSheet("background-color: rgb(78, 154, 6);")
        self.button.clicked.disconnect()
        self.button.clicked.connect(self.stop_inference_server)

    def stop_inference_server(self):
        if self.button.text() == "start":
            return
        self.button.setText("start")
        for thread in self.threads:
            if type(thread) is tuple:
                thread[0].shutdown()
                thread[1].join()
            else:
                thread.join()

        self.threads.clear()
        self.findChild(QtWidgets.QSpinBox, 'sliding_window_frames').valueChanged.disconnect()
        self.button.clicked.disconnect()
        self.button.clicked.connect(self.start_inference)

    def closeEvent(self, event):
        self.save_settings()
        print("Close event received")

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
        self.pharus_fps_deque.append(fps)
        avg_fps = sum(list(self.pharus_fps_deque))//FPS_AVERAGING_WINDOW
        self.pharus_fps.display(avg_fps)

        """
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

    def loop(self):
        """
        pharus_data = []
        for entries in [list(ped_deque) for _, ped_deque in self.pharus_data.items()]:
            pharus_data.extend(entries)
        self.scatter_plot_item_pharus.setData(pharus_data)
        """
        self.scatter_plot_item_ml.setData(self.ml_data)
        self.scatter_plot_item_pharus_obs.setData(self.pharus_obs_data)

    def loop_slow(self):
        text_window = self.findChild(QtWidgets.QTextBrowser, 'training_output')
        if self.training_threads:
            text_window.append(self.training_threads[0].stdout.readline().decode("utf-8") + "\n")

    def select_training_data(self):
        fileselection = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Directory")
        path = str(fileselection)
        dirname = os.path.basename(path)
        is_valid, msg = training_folder_is_valid(path)
        if not is_valid:
            self.show_error(msg)
            return
        self.training_data_select_label.setText(dirname)
        self.training_data_select_label.setStyleSheet("background-color: rgb(78, 154, 6);")
        self.training_data_path = path

    def visualize_training_data(self):
        training_data_df = get_training_data(self.training_data_path)
        if type(training_data_df) is str:
            self.show_error(training_data_df)
            return

        self.reset_training_plot()

        person_paths = get_training_df_positions(training_data_df)
        for index, person_path in enumerate(person_paths):
            start_marker = pg.ScatterPlotItem(
                pen=pg.mkPen(width=5, color='r'), symbol='o', size=1)
            start_marker.setData([{"pos": [person_path[0][0], person_path[1][0]]}])
            curve = pg.PlotCurveItem(
                pen=pg.mkPen(width=1, color=get_rgb_val(len(person_paths), index),
                             style=QtCore.Qt.DotLine), symbol='o', size=1)
            self.plot_view_training.addItem(curve)
            self.plot_view_training.addItem(start_marker)
            curve.setData(person_path[0], person_path[1])
            self.plot_items_training.append(curve)
            self.plot_items_training.append(start_marker)

    def reset_training_plot(self):
        for plot_item in self.plot_items_training:
            self.plot_view_training.removeItem(plot_item)

    def show_error(self, msg):
        error_dialog = QtWidgets.QErrorMessage()
        error_dialog.showMessage(msg)
        error_dialog.exec_()

    def select_pharus_data(self):
        fileselection = QtWidgets.QFileDialog.getOpenFileName(self, "Select Pharus Recording (e.g., recording.trk)")
        path = fileselection[0]
        is_valid, msg = pharus_recording_is_valid(path)
        if not is_valid:
            self.show_error(msg)
            return
        self.pharus_data_path = path
        self.pharus_data_select_label.setText(os.path.basename(path))
        self.pharus_data_select_label.setStyleSheet("background-color: rgb(78, 154, 6);")

    def convert_pharus_data(self):
        is_valid, msg = pharus_recording_is_valid(self.pharus_data_path)
        if not is_valid:
            self.show_error(msg)
            return
        self.show_error("Starting conversion")
        pharus_convert(self.pharus_data_path, "DATA_BLOCK/")
        self.show_error("Conversion completed.")

    def start_training(self):
        is_valid, msg = training_folder_is_valid(self.training_data_path)
        if not is_valid:
            self.show_error(msg)
            return
        basefolder = os.path.basename(self.training_data_path)
        epochs = self.findChild(QtWidgets.QSpinBox, 'epochs').value()
        pred_length = self.findChild(QtWidgets.QSpinBox, 'pred_length').value()
        obs_length = self.findChild(QtWidgets.QSpinBox, 'obs_length').value()
        batch_size = self.findChild(QtWidgets.QSpinBox, 'batch_size').value()
        save_every_n_epochs = self.findChild(QtWidgets.QSpinBox, 'save_every_n_epochs').value()
        learning_rate = self.findChild(QtWidgets.QDoubleSpinBox, 'learning_rate').value()
        step_size_lr_scheduler = self.findChild(QtWidgets.QSpinBox, 'step_size_lr_scheduler').value()

        self.training_threads.append(start_training_thread(basefolder, str(epochs), str(pred_length), str(obs_length),
                                                           str(batch_size), str(learning_rate),
                                                           str(step_size_lr_scheduler), str(save_every_n_epochs)))

    def continue_training(self):
        is_valid, msg = training_folder_is_valid(self.training_data_path)
        if not is_valid:
            self.show_error(msg)
            return
        basefolder = os.path.basename(self.training_data_path)
        epochs = self.findChild(QtWidgets.QSpinBox, 'epochs').value()
        pred_length = self.findChild(QtWidgets.QSpinBox, 'pred_length').value()
        obs_length = self.findChild(QtWidgets.QSpinBox, 'obs_length').value()
        batch_size = self.findChild(QtWidgets.QSpinBox, 'batch_size').value()
        save_every_n_epochs = self.findChild(QtWidgets.QSpinBox, 'save_every_n_epochs').value()
        learning_rate = self.findChild(QtWidgets.QDoubleSpinBox, 'learning_rate').value()
        step_size_lr_scheduler = self.findChild(QtWidgets.QSpinBox, 'step_size_lr_scheduler').value()

        fileselection = QtWidgets.QFileDialog.getOpenFileName(self, "Select Model (e.g., model.pkl.epoch30)")
        path = fileselection[0]

        self.training_threads.append(start_training_thread(basefolder, str(epochs), str(pred_length), str(obs_length),
                                                           str(batch_size), str(learning_rate),
                                                           str(step_size_lr_scheduler), str(save_every_n_epochs),
                                                           prev_model_path=path))



# Create an instance of QtWidgets.QApplication
app = QtWidgets.QApplication(sys.argv)
window = Ui()  # Create an instance of our class
window.show()

timer = QTimer()
timer.timeout.connect(window.loop)
timer.start(25)
timer2 = QTimer()
timer2.timeout.connect(window.loop_slow)
timer2.start(25)
sys.exit(app.exec_())  # Start the
