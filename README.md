SystemFailed trajectory prediction
================================================

This repository implements a GUI for data conversion from a Pharus laser tracker recording format to trajnetplusplus format. It also implements training a social lstm and running an inference server listening for live Pharus laser tracker data. The code is heavily based on [vita-epfl/trajnetplusplusbaselines](https://github.com/vita-epfl/trajnetplusplusbaselines).

## Usage

### Install

### Start softwares
#### Pharus
- pharus is here: ``cd /home/ml/pharus/bin/```
- start pharus-replay: add recorded file as argument ``./pharus /home/ml/Documents/SystemFailed_trajnetplusplusdataset/data/raw/pharus/publikum_lindenfels.trk``
- Tracker -> Track Gen mode: unconfirmed
- Link: add TUIO Sender 127.0.0.1
#### ML Software
- ML software is here: ``/home/ml/Documents/SystemFailed_trajnetplusplusbaselines``
- start ML GUI with ``python3.7 gui/gui.py``

### Usage
#### sliding window frames
- smoothes the input data 
#### Pharus Listener IP
- when Pharus on same computer: 127.0.0.1
#### TD PC IP
- a.k.a where the prediciton is sent to


### Command line

- ``cd SystemFailed_trajnetplusplusbaselines``

- Start server for live trajectory inference
    - ``python3.7 main_control_server.py``
    
- Train model (preloading weights from an existing model)
    - ``python3.7 -m trajnetbaselines.lstm.trainer --type social --path pharus_saturday --load-state OUTPUT_BLOCK/trajdata/lstm_social_None_uni.pkl.epoch25 --epochs 50``

- Visualize predictions:
   - ``python3.7 -m  evaluator.visualize_predictions ../DATA_BLOCK/pharus_saturday/test/5personen.ndjson ../DATA_BLOCK/pharus_saturday/test_pred/lstm_social_None_crowd.epoch25_modes1/5personen.ndjson -o 5pers_crowd_epoch25``

- Start prediction server listing for Tuio
   - ``python3.7 -m evaluator.server_udp --output OUTPUT_BLOCK/trajdata/lstm_social_None.pkl.epoch25 --gpu True``

## Development

### Repository structure

 - `OUTPUTBLOCK/trajdata` contains the trained models.
 - `evaluator/server_udp.py` is the inference server.
 - `gui` contains the PyQT GUI app including helper functions.
   - `gui.py` is the PyQT GUI.
   - `gui.ui` is the PyQT UI design file containing the GUI layout.
   - the helper files contain helper functions related to the file name.
     - `starting_inference_helpers.py` starts the inference server and udp splitter that splits pharus udp live stream into two streams to localost and a given external IP.
     - `starting_training_helpers.py` executes the training command, validates file names and more.
