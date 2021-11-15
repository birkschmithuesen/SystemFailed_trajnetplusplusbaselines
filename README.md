SystemFailed trajectory prediction
================================================

This repository implements a GUI for data conversion from a Pharus laser tracker recording format to trajnetplusplus format. It also implements training a social lstm and running an inference server listening for live Pharus laser tracker data. The code is heavily based on [vita-epfl/trajnetplusplusbaselines](https://github.com/vita-epfl/trajnetplusplusbaselines).

## Usage

### Install

### Start softwares
#### Pharus
- pharus is here: `cd /home/ml/pharus/bin/`
- start pharus-replay: add recorded file as argument `./pharus /home/ml/Documents/SystemFailed_trajnetplusplusdataset/data/raw/pharus/publikum_lindenfels.trk`
- Tracker -> Track Gen mode: unconfirmed
- Link: add TUIO Sender 127.0.0.1
#### ML Software

- ML software is here: `/home/ml/Documents/SystemFailed_trajnetplusplusbaselines`
- start ML GUI with `python3.7 gui/gui.py` and to prevent networking and other trouble you can run `killall udp-splitter python3.7 && python3.7 gui/gui.py`

### GUI Usage
#### Inference Tab
- Load Model by clicking "start inference" button
    - models are currently stored in `OUTPUT_BLOCK`
    - best model right now is `publikum_lindenfels`
    - choose file with ending `.epoch` not with `.state`
 - sliding window frames
    - smoothes the input data
 - observation length (not longer then choosen in the training)
 - Pharus Listener IP 
    - when Pharus on same computer: 127.0.0.1
    - IP needs to be changed, before "start" is pressed to take effect
 - TD PC IP
    - a.k.a where the prediciton is sent to
    - IP needs to be changed, before "start" is pressed to take effect
 - start/stop button
    - stop doesn't work. Just restart the whole program
#### Training Tab
- train 50 - 80 epochs at least
- 1. convert pharus data to scene format
    - select pharus data. Currently stored in: ``/home/ml/Documents/SystemFailed_trajnetplusplusdataset/data/raw``
    - destination folder is: ``DATA_BLOCK``
- 2. visualize szene data
    - select dataset from ``DATA_BLOCK`` to visualize
    - select whole folder
    - red dot marks the start of the path
- 3. train network
    - to combine several training data:
        - copy a mix from *.json files in ``train`` folder in the respective training dataset
    - to use all data for training, a.k.a skip testing and validation data
        - copy all files from ``val`` and ``test`` and ``test_private``in train
    - all fields under batch size are currently not in use
    - observation length: leave deafult 9
    - prediction length: leave default 6
    - terminal in GUI may freeze
    - when ready, there comes a PopUp
    - destination folder is: ``OUTPUT_BLOCK``

### hard coded config
- field size: evaluator/server_udp.py PHARUS_FIELD_SIZE

### Command line

- ``cd SystemFailed_trajnetplusplusbaselines``

- Start server for live trajectory inference
    - ``python3.7 main_control_server.py``
    
- Train model (preloading weights from an existing model)
    - ``python3.7 -m trajnetbaselines.lstm.trainer --type social --path pharus_saturday --load-state OUTPUT_BLOCK/trajdata/lstm_social_None_uni.pkl.epoch25 --epochs 50``
    - --path -> select training data set
    - --load-state -> select model to continue
    - --epochs -> number of epochs to train (the number is not how many epochs will get trained, but till what number. When The model is already trained with 25 epochs, the argument ``--epochs 50``will train for 25 epochs more

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
