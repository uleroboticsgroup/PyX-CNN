# PyX-CNN: Pyramidal eXplainability for Convolutional Neural Networks


# Classification Example of Explainability

## Dataset

[Pascal VOC part](http://roozbehm.info/pascal-parts/pascal-parts.html)

Dataset has been processed
[dataset.ipynb](https://github.com/uleroboticsgroup/xai/blob/main/dataset.ipynb)

Study of animals category with only one detection by image, displays similarity with cow and horse categories.
| Cat   | Imgs | Train | Val | %Occupation          |
|-------|------|-------|-----|----------------------|
| sheep | 96   | 51    | 45  | 0.2886039996430471  |
| dog   | 701  | 351   | 350 | 0.3137457986858771  |
| cow   | 106  | 58    | 48  | 0.26646911903663134 |
| cat   | 717  | 353   | 364 | 0.37630319767611536 |
| horse | 148  | 69    | 79  | 0.28260008694909383 |

Then a dataset analysis of the final dataset has been done
[dataset_analysis.ipynb](https://github.com/uleroboticsgroup/xai/blob/main/dataset_analysis.ipynb)

File [voc2010_info_parts.csv](https://github.com/uleroboticsgroup/xai/blob/main/voc2010_info.csv) has been generated with the columns:
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

![dataset_ratios](https://github.com/uleroboticsgroup/xai/blob/main/images/dataset_ratios.png)
![dataset_splits](https://github.com/uleroboticsgroup/xai/blob/main/images/dataset_splits.png)
![parts_occupation](https://github.com/uleroboticsgroup/xai/blob/main/images/parts_occupation.png)
![parts_sample_cow](https://github.com/uleroboticsgroup/xai/blob/main/images/parts_sample_cow.png)
![parts_sample_horse](https://github.com/uleroboticsgroup/xai/blob/main/images/parts_sample_horse.png)