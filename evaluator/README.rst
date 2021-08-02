SystemFailed Prediction Server
=======================

 - Train model (preloading weights from an existing model): `python3.7 -m trajnetbaselines.lstm.trainer --type social --path pharus_saturday --load-state OUTPUT_BLOCK/trajdata/lstm_social_None_uni.pkl.epoch25 --epochs 50`

 - Visualize predictions:
   `python3.7 -m  evaluator.visualize_predictions ../DATA_BLOCK/pharus_saturday/test/5personen.ndjson ../DATA_BLOCK/pharus_saturday/test_pred/lstm_social_None_crowd.epoch25_modes1/5personen.ndjson -o 5pers_crowd_epoch25`

 - Start prediction server listing for Tuio: `python3.7 -m evaluator.server_udp --output OUTPUT_BLOCK/trajdata/lstm_social_None.pkl.epoch25 --gpu True`

 Evaluation on TrajNet++
 =======================

Description of the full evaluation procedure:
-------------------------------------------------------------------

The evaluation script and its help menu: ``python -m evaluator.trajnet_evaluator --help``

1.  Make sure that (a) test scenes (only containing the observations) are present in the 'test' folder (b) groundtruth scenes (containing the observations as well as the groundtruth predictions) are present in the 'test_private.' These conditions are true if the dataset was generated using trajnetplusplusdataset.

2. The full evaluation procedure generates a 'test_pred' folder containing the predictions of your model for the files in the test folder. This process is carried out using 'write.py' (See write.py for more details.) NOTE: If Model predictions already exist in 'test_pred' from a previous run of the same model, the evaluator command SKIPS WRITING the new model predictions. In other words, already existing model predictions are not overwritten.

3. Once the predictions are written in 'test_pred', our trajnet_evaluator compares the model predictions in 'test_pred' with groundtruth predictions in 'test_private' providing a complete table of evaluation metrics as Results.png.

Eg: ``python -m evaluator.trajnet_evaluator --path trajdata --output OUTPUT_BLOCK/trajdata/occupancy.pkl``


Description of the FAST evaluation procedure (No categorization-specific evaluation):
-------------------------------------------------------------------------------------

The evaluation script and its help menu: ``python -m evaluator.fast_evaluator --help``

1.  This procedure does not generate a test_pred file. It quickly provides the requested metrics in the terminal.

Eg: ``python -m evaluator.fast_evaluator --path trajdata --output OUTPUT_BLOCK/trajdata/occupancy.pkl --unimodal``
