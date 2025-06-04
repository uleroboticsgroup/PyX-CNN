# PyX-CNN: Pyramidal eXplainability for Convolutional Neural Networks


# Classification Example of Explainability

## Dataset

[Pascal VOC part](http://roozbehm.info/pascal-parts/pascal-parts.html) is the based dataset that is gone to be filtered and analyse for generate the final dataset.

Dataset has been processed
[dataset.ipynb](https://github.com/uleroboticsgroup/PyX-CNN/blob/main/dataset.ipynb)

Study of animals category with only one detection by image, displays similarity with cow and horse categories.
| Cat   | Imgs | Train | Val | %Occupation          |
|-------|------|-------|-----|----------------------|
| sheep | 96   | 51    | 45  | 0.2886039996430471  |
| dog   | 701  | 351   | 350 | 0.3137457986858771  |
| cow   | 106  | 58    | 48  | 0.26646911903663134 |
| cat   | 717  | 353   | 364 | 0.37630319767611536 |
| horse | 148  | 69    | 79  | 0.28260008694909383 |

Then a dataset analysis of the final dataset has been done
[dataset_analysis.ipynb](https://github.com/uleroboticsgroup/PyX-CNN/blob/main/dataset_analysis.ipynb)

File [dataset_info_parts.csv](https://github.com/uleroboticsgroup/PyX-CNN/blob/main/dataset_info_parts.csv) has been generated with the columns:
- file: file_name
- split: train or val
- per_occ: percentage of occupation of detected object in image [0-1]
- cat: category name [cow or horse]
- img_w: width image (original image)
- img_h: height image (orginal image)
- aspect_ratio: apect ratio (original image)
- per_occ_torso: percentage of occupation of torso mask in image [0-1]
- per_occ_head: percentage of occupation of head mask in image [0-1]
- per_occ_leg: percentage of occupation of leg mask in image [0-1]

Images generated to explain dataset distribution of the data:

![dataset_ratios](https://github.com/uleroboticsgroup/PyX-CNN/blob/main/images/dataset_ratios.png)
![dataset_splits](https://github.com/uleroboticsgroup/PyX-CNN/blob/main/images/dataset_splits.png)
![parts_occupation](https://github.com/uleroboticsgroup/PyX-CNN/blob/main/images/parts_occupation.png)
![parts_sample_cow](https://github.com/uleroboticsgroup/PyX-CNN/blob/main/images/parts_sample_cow.png)
![parts_sample_horse](https://github.com/uleroboticsgroup/PyX-CNN/blob/main/images/parts_sample_horse.png)


## Classifier (VGG16 architechture)

### With Transfer Learning and Fine Tuning

[vgg16_model.ipynb](https://github.com/uleroboticsgroup/PyX-CNN/blob/main/vgg16_model.ipynb)

It generates the following files:
- vgg16_model.h5 with the trained model
- [vgg16_train.csv](https://github.com/uleroboticsgroup/PyX-CNN/blob/main/vgg16_train.csv) with the training metrics

### Without Transfer Learning

[vgg16_model_without_tf.ipynb](https://github.com/uleroboticsgroup/PyX-CNN/blob/main/vgg16_model_without_tf.ipynb)

It generates the following files:
- vgg16_without_model.h5 with the trained model
- [vgg16_without_train.csv](https://github.com/uleroboticsgroup/PyX-CNN/blob/main/vgg16_without_train.csv) with the training metrics

### Classifier results

[generate_visual_explanation.ipynb](https://github.com/uleroboticsgroup/PyX-CNN/blob/main/generate_visual_explanation.ipynb) and [calculate_MIS_metric.ipynb](https://github.com/uleroboticsgroup/PyX-CNN/blob/main/calculate_MIS_metric.ipynb) generate results files from trained models.

File [dataset_info_parts_metrics_methods.csv](https://github.com/uleroboticsgroup/PyX-CNN/blob/main/dataset_info_parts_metrics_methods.csv) (model with transfer learning) and [dataset_info_parts_metrics_methods_without.csv](https://github.com/uleroboticsgroup/PyX-CNN/blob/main/dataset_info_parts_metrics_methods_without.csv) (model without transfer learning) results with the columns:
- file: file_name
- split: train or val which corresponds with training and validation data
- per_occ: percentage of occupation of detected object in image [0-1]
- cat: category name
- img_w: width image (original image)
- img_h: height image (orginal image)
- aspect_ratio: apect ratio (original image)
- per_occ_torso: percentage of occupation of torso mask in image [0-1]
- per_occ_head: percentage of occupation of head mask in image [0-1]
- per_occ_leg: percentage of occupation of leg mask in image [0-1]
- file_path: complete path to the file
- pred_id: category predicted id
- pred_ok: correct prediction [0-1]
- cat_id: category id
- mean_act_obj: mean activation value of object region
- mean_act_back: mean activation value of background region
- perc_act_obj: percentage of activation greather than 0.5 of object region
- perc_act_back: percentage of activation greather than 0.5 of background region
- method: visual activation method [gradcam, gradcamplus, scorecam]
- oc_part: oclussion part [NaN, object, head, torso, leg]
- oc_part_perc: oclussion part percentage
- pred_prob: probability of prediction
- hact_1_part: higher activation part (First) [background, object, head, torso, leg]
- hact_1_perc: percentage of occupation of the higher activation part (First)
- hact_2_part: higher activation part (Second) [background, object, head, torso, leg]
- hact_2_perc: percentage of occupation of the higher activation part (Second)
- hact_3_part: higher activation part (Third) [background, object, head, torso, leg]
- hact_3_perc: percentage of occupation of the higher activation part (Third)


File [mis_stats_0.csv](https://github.com/uleroboticsgroup/PyX-CNN/blob/main/mis_stats_0.csv) (model with transfer learning) and [mis_stats_1.csv](https://github.com/uleroboticsgroup/PyX-CNN/blob/main/mis_stats_1.csv) (model without transfer learning) results with the columns:
- layer: layer name
- mean: mean of the MIS value in that layer
- std: standar desviation of the MIS values in the layer
- max: maximum of the MIS values in the layer
- min: minimum of the MIS values in the layer
- median: median of the MIS values in the layer

## Metrics

[all_metrics.ipynb](https://github.com/uleroboticsgroup/PyX-CNN/blob/main/all_metrics.ipynb) generate metrics charts for each model.

# Python environment (Python 3.10)

> conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0

> pip install --upgrade pip

> pip install "tensorflow<2.11"

> python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

Si da error numpy
> pip uninstall numpy

> pip install "numpy<2.0"

> pip install tf-keras-vis

> pip install notebook

> pip install opencv-python


# Acknowledgments

<img src="https://user-images.githubusercontent.com/3810011/192087445-9aa45366-1fec-41f5-a7c9-fa612901ecd9.png" alt="DMARCE_logo drawio" width="200"/>

DMARCE (EDMAR+CASCAR) Project: EDMAR PID2021-126592OB-C21 -- CASCAR PID2021-126592OB-C22 funded by MCIN/AEI/10.13039/501100011033 and by ERDF A way of making Europe

<img src="https://raw.githubusercontent.com/DMARCE-PROJECT/DMARCE-PROJECT.github.io/main/logos/micin-uefeder-aei.png" alt="DMARCE_EU eu_logo" width="200"/>
