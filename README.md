# Project Canopy - DRC slash-and-burn detection model documentation

This repository contains the code used to train a new model detecting the presence of slash-and-burn agriculture in Sentinel-2 satellite images of the Congo Basin.

The model was based on the one developed in [CB Feature Detection](https://github.com/Project-Canopy/cb_feature_detection). Each directory contains an "old_notebooks" or "old_files"
folder, within which you can find the notebooks and files relevant to that project that the current notebooks were based on.

This ReadMe contains general information on each directory in this repository, in rough order of when the code contained within those directories should be run.

Please contact David Nagy (davidanagy@gmail.com) with any questions.

<br />

## <u>s2cloudless</u>
 <br />
Description: Downloading cloudfree imagery from Google Earth Engine using the S2Cloudless algorithm
<br />
<br />

Requirements:

* Registered a Google Earth Engine account
* Linked that account to a Google Cloud Services account

Assets:

* <b>polygons_101320.csv</b>: A CSV file containing information on the "Misha polygons"; polygons that Misha chose as containing forest disturbances for training
* <b>DRC_squares_3.geojson</b>: A GeoJSON file containing a polygon that's "gridded out" into 10km x 10km squares. Gridding is required because Google Earth Engine has a size limit for individual downloads.
* <b>reuse_training_data/labels_and_boundaries_sab.csv</b>: A CSV file containing labels and coordinates for the training data used in our pervious model

<br />

Notebooks:

* <b>s2cloudless_polygon_export</b> - Example code of using the s2cloudless algorithm to download a satellite image from within a certani area of interest
* <b>s2cloudless_DRC_export</b> - Uses the s2cloudless algorithm to download cloudfree satellite images in 2021 and 2019 within our area of interest (mainly the forests in the Democratic Republic of the Congo).
* <b>reuse_training_data/Filter_old_polygons</b> - Re-downloads the training data used in our previous model with the s2cloudless algorithm, keeping the original labels. This code was largely written by <a href="https://github.com/wwymak">Wendy Mak</a>.

<br />

Scripts:

* <b>reuse_training_data/s2cloudless_pipeline.py</b> - A pipeline for downloading images in Google Earth Engine using the s2cloudless algorithm. Mostly written by <a href="https://github.com/wwymak">Wendy Mak</a>.

<br />

Suggested future directions:

* Download images from 2017 as well. When I tried to do this, the search was unable to find any satellite images from that year, and I lacked the time to find out why.

<br />

## <u>data-prep</u>
 <br />
Description: Additional data preparation before training the model, mainly creating label files and adding a "Normalized Burn Ratio" band
<br />
<br />

Requirements:

* Access to the Project Canopy AWS account, specifically filenames in S3

<br />

Assets:

* All CSV files in this folder are label files for training, validation, and test data. The "v2" train, val, and test files were used to train and evaluate the final model.

<br />

Notebooks:

* <b>making_label_files</b> - Makes a "base" (raw, unbalanced) label file from the training data stored in S3
* <b>rebalance_csvs</b> - Rebalances the label CSV to eliminate the huge class imbalance in the raw data
* <b>add_nbr</b> - Per Lloyd Hughes's suggestion, adds a Normalized Burn Ratio band to geotiff files

<br />

Scripts:

* none

<br />

Suggested future directions:

* Maybe try out additional "extra bands" to see if they improve the model?

<br />

## <u>sagemaker-staging</u>
 <br />
Description: Code used both to train the model and run inference on the full dataset in SageMaker
<br />
<br />

Requirements:

* Access to AWS S3 (for training data) and SageMaker (for running the training)

<br />

Assets:

* Training chips can be found here: s3://canopy-production-ml/chips/model2_s2cloudless/training_v2/null/ (currently in Glacier)
* Full dataset can be found here: s3://canopy-production-ml/full_drc/ (currently in Glacier)
* Pre-trained models can be found here: s3://canopy-production-ml/pretrained_models/ (currently in Glacier)
* <b>resnet50.pth</b>: A ResNet backbone pre-trained on Sentinel-2 data; can also be found <a href="https://github.com/HSG-AIML/SSLTransformerRS">here</a>
* <b>weights_resnet.onnx</b>: The above model converted from PyTorch to Onnx by <a href="https://github.com/system123">Lloyd Hughes</a>
* <b>sentinel_resnet_tf</b>: The above model converrted from Onnx to Tensorflow using <a href="https://github.com/gmalivenko/onnx2keras">onnx2keras</a>
* Current best versions of the model can be found here: s3://canopy-production-ml/inference/model_files/ (currently in Glacier)
* <b>model-best.h5</b> and <b>model_weights_best.h5</b> are the ISL (logging roads) model
* <b>best_SAB_model.h5</b> and <b>best_SAB_weights.h5</b> are the SAB (slash-and-burn) model

<br />

Notebooks:

* <b>training/training_notebook</b> - Used to train the model in SageMaker using the scripts in `docker_test_folder`
* <b>inference/inference_pipeline</b> - Used to run inference on the full dataset in SageMaker using the best model and best weights

<br />

Scripts:

* <b>training/docker_test_folder/Dockerfile</b> - Dockerfile used when training the model on SageMaker
* <b>training/docker_test_folder/training.py</b> - The version of the model code I was working on until the deadline
* <b>training/docker_test_folder/training_used_for_current_model.py</b> - The version of the model code used to train the "best SAB model," using the "sentinel_resnet" pretrained model found on line 500

<br />

Suggested future directions:

* Adjust the "resnet_sentinel" code in <b>training.py</b> (starting on line 581) so it provides good results (unfreeze more layers?)
* Try out adding a Dropout() layer and retrain the model (suggestion by Daniel Firebanks-Quevado)
* Depending on the results, try out playing around with the threshold if needed, or Monte Carlo Dropout (suggestion by Daniel Firebanks-Quevado)

<br />

## <u>model-development</u>
 <br />
Description: Notebooks relevant to both developing model code and testing trained models for accuracy, recall, etc.
<br />
<br />

Assets:

* <b>DRC_labels_SAB_train/val/test.csv</b>: Labels for training, eval, and test data
* <b>SAB_labels.json</b>: Necessary json file for testing trained SAB models

<br />

Notebooks:

* <b>Canopy_RGB_Train</b> - A model trained purely on RGB bands. This notebook was written by <a href="https://github.com/ShaileshSridhar2403">Shailesh Sridhar</a> and <a href="https://github.com/midsterx">Midhush</a>
* <b>Canopy_Additional_Bands_model</b> - A model using the ResNet50 architecture that first separates out the non-RGB bands, then adds them back in later. We were unable to figure out how to get this code to run on SageMaker. This notebook was written by <a href="https://github.com/ShaileshSridhar2403">Shailesh Sridhar</a> and <a href="https://github.com/midsterx">Midhush</a>
* <b>evaluation_master</b> - Code used to evaluate trained models

<br />

Scripts:

* <b>test_generator.py</b> - Builds a generator for testing trained models. Used with `evaluation_master.ipynb`

<br />

Suggested future directions:

* Integrate the code found in <b>Canopy_Additional_Bands_model.ipynb</b> into the model code found in <b>sagemaker-staging/training/docker_test_folder</b>
* Find both ISL and SAB models that improve on our current metrics: ~80% accuracy and recall for ISL (old model); 69% accuracy and 64% recall for SAB (new model)

<br />

## <u>inference</u>
 <br />
Description: Local code run relevant to making predictions on the full DRC dataset
<br />
<br />

Assets:

* <b>raw_predictions.zip</b>: The results of the inference code found in <b>sagemaker-staging/inference/inference_pipeline.ipynb</b>
* <b>predictions/ISL_2019_preds.geojson</b>, etc: Model predictions in the correct geojson format

<br />

Notebooks:

* <b>json_to_geojson</b> - Code used to translate the raw predictions into the correct geojson format; results in the files in the <b>predictions</b> folder

<br />

Scripts:

* <b>cortex_drc</b> - My attempts to use Cortex to run inference on the full dataset. I couldn't get it to work but I kept the code here for your possible interest.

<br />

Suggested future directions:

* Figure out Cortex

<br />

## <u>analytics</u>
 <br />
Description: Post-inference improvement on predictions
<br />
<br />

Assets:

* none

<br />

Notebooks:

* <b>remove_orphans</b> - Code used to remove "orphan predictions"; i.e., single or double predictions with no other predictions nearby

<br />

Scripts:

* none

<br />

Suggested future directions:

* Integrate the Open Street Map filtering found in <b>old_notebooks/osm_filter</b>

<br />

## <u>display</u>
 <br />
Description: Code used to display results online (no new code compared to the previous model so everything is in old_files)
