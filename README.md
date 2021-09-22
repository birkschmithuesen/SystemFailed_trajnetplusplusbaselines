SystemFailed trajectory prediction
================================================

This repository implements a GUI for data conversion from a Pharus laser tracker recording format to trajnetplusplus format. It also implements training a social lstm and running an inference server listening for live Pharus laser tracker data.

- ``cd SystemFailed_trajnetplusplusbaselines``

- Start server for live trajectory inference
    - ``python3.7 main_control_server.py``


- Train model (preloading weights from an existing model)
    - ``python3.7 -m trajnetbaselines.lstm.trainer --type social --path pharus_saturday --load-state OUTPUT_BLOCK/trajdata/lstm_social_None_uni.pkl.epoch25 --epochs 50``

- Visualize predictions:
   - ``python3.7 -m  evaluator.visualize_predictions ../DATA_BLOCK/pharus_saturday/test/5personen.ndjson ../DATA_BLOCK/pharus_saturday/test_pred/lstm_social_None_crowd.epoch25_modes1/5personen.ndjson -o 5pers_crowd_epoch25``

- Start prediction server listing for Tuio
   - ``python3.7 -m evaluator.server_udp --output OUTPUT_BLOCK/trajdata/lstm_social_None.pkl.epoch25 --gpu True``
