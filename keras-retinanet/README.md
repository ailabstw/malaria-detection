# Keras RetinaNet [![Build Status](https://travis-ci.org/fizyr/keras-retinanet.svg?branch=master)](https://travis-ci.org/fizyr/keras-retinanet)
Keras implementation of RetinaNet object detection as described in [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)
by Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He and Piotr Dollár.

## Installation

1) Clone this repository.
2) In the repository, execute `pip install . --user`.
   Note that due to inconsistencies with how `tensorflow` should be installed,
   this package does not define a dependency on `tensorflow` as it will try to install that (which at least on Arch Linux results in an incorrect installation).
   Please make sure `tensorflow` is installed as per your systems requirements.
   Also, make sure Keras 2.1.3 is installed.
3) As of writing, this repository requires the master branch of `keras-resnet` (run `pip install --user --upgrade git+https://github.com/broadinstitute/keras-resnet`).
4) Optionally, install `pycocotools` if you want to train / test on the MS COCO dataset by running `pip install --user git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI`.

## Training
`keras-retinanet` can be trained using [this](https://github.com/fizyr/keras-retinanet/blob/master/keras_retinanet/bin/train.py) script.
Note that the train script uses relative imports since it is inside the `keras_retinanet` package.
If you want to adjust the script for your own use outside of this repository,
you will need to switch it to use absolute imports.

If you installed `keras-retinanet` correctly, the train script will be installed as `retinanet-train`.
However, if you make local modifications to the `keras-retinanet` repository, you should run the script directly from the repository.
That will ensure that your local changes will be used by the train script.

The default backbone is 'resnet50'. You can change this using the '--backbone=xxx' argument in the running script.
xxx can be one of the backbones in resnet models (resnet50, resnet101, resnet152) or mobilenet models 
(mobilenet128_1.0, mobilenet128_0.75, mobilenet160_1.0, etc). The different options are defined by each model in 
their corresponding python scripts (resnet.py, mobilenet.py, etc).

### Usage
For training on [Pascal VOC](http://host.robots.ox.ac.uk/pascal/VOC/), run:
```shell
# Running directly from the repository:
keras_retinanet/bin/train.py pascal /path/to/VOCdevkit/VOC2007

# Using the installed script:
retinanet-train pascal /path/to/VOCdevkit/VOC2007
```

For training on [MS COCO](http://cocodataset.org/#home), run:
```shell
# Running directly from the repository:
keras_retinanet/bin/train.py coco /path/to/MS/COCO

# Using the installed script:
retinanet-train coco /path/to/MS/COCO
```

The pretrained MS COCO model can be downloaded [here](https://github.com/fizyr/keras-retinanet/releases/download/0.1/resnet50_coco_best_v1.2.2.h5). Results using the `cocoapi` are shown below (note: according to the paper, this configuration should achieve a mAP of 0.343).

```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.345
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.533
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.368
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.189
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.380
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.465
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.301
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.482
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.529
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.364
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.565
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.666
```

For training on [OID](https://github.com/openimages/dataset), run:
```shell
# Running directly from the repository:
keras_retinanet/bin/train.py oid /path/to/OID

# Using the installed script:
retinanet-train oid /path/to/OID

# You can also specify a list of labels if you want to train on a subset
# by adding the argument 'labels_filter':
keras_retinanet/bin/train.py oid /path/to/OID --labels_filter=Helmet,Tree
```

For training on a custom dataset, a CSV file can be used as a way to pass the data.
See below for more details on the format of these CSV files.
To train using your CSV, run:
```shell
# Running directly from the repository:
keras_retinanet/bin/train.py csv /path/to/csv/file/containing/annotations /path/to/csv/file/containing/classes

# Using the installed script:
retinanet-train csv /path/to/csv/file/containing/annotations /path/to/csv/file/containing/classes
```

In general, the steps to train on your own datasets are:
1) Create a model by calling for instance `keras_retinanet.models.resnet50_retinanet` and compile it.
   Empirically, the following compile arguments have been found to work well:
```python
model.compile(
    loss={
        'regression'    : keras_retinanet.losses.smooth_l1(),
        'classification': keras_retinanet.losses.focal()
    },
    optimizer=keras.optimizers.adam(lr=1e-5, clipnorm=0.001)
)
```
2) Create generators for training and testing data (an example is show in [`keras_retinanet.preprocessing.PascalVocGenerator`](https://github.com/fizyr/keras-retinanet/blob/master/keras_retinanet/preprocessing/pascal_voc.py)).
3) Use `model.fit_generator` to start training.

## Testing
An example of testing the network can be seen in [this Notebook](https://github.com/delftrobotics/keras-retinanet/blob/master/examples/ResNet50RetinaNet.ipynb).
In general, output can be retrieved from the network as follows:
```python
_, _, detections = model.predict_on_batch(inputs)
```

Where `detections` are the resulting detections, shaped `(None, None, 4 + num_classes)` (for `(x1, y1, x2, y2, cls1, cls2, ...)`).

Loading models can be done in the following manner:
```python
from keras_retinanet.models.resnet import custom_objects
model = keras.models.load_model('/path/to/model.h5', custom_objects=custom_objects)
```

Execution time on NVIDIA Pascal Titan X is roughly 75msec for an image of shape `1000x800x3`.

## CSV datasets
The `CSVGenerator` provides an easy way to define your own datasets.
It uses two CSV files: one file containing annotations and one file containing a class name to ID mapping.

### Annotations format
The CSV file with annotations should contain one annotation per line.
Images with multiple bounding boxes should use one row per bounding box.
Note that indexing for pixel values starts at 0.
The expected format of each line is:
```
path/to/image.jpg,x1,y1,x2,y2,class_name
```

Some images may not contain any labeled objects.
To add these images to the dataset as negative examples,
add an annotation where `x1`, `y1`, `x2`, `y2` and `class_name` are all empty:
```
path/to/image.jpg,,,,,
```

A full example:
```
/data/imgs/img_001.jpg,837,346,981,456,cow
/data/imgs/img_002.jpg,215,312,279,391,cat
/data/imgs/img_002.jpg,22,5,89,84,bird
/data/imgs/img_003.jpg,,,,,
```

This defines a dataset with 3 images.
`img_001.jpg` contains a cow.
`img_002.jpg` contains a cat and a bird.
`img_003.jpg` contains no interesting objects/animals.


### Class mapping format
The class name to ID mapping file should contain one mapping per line.
Each line should use the following format:
```
class_name,id
```

Indexing for classes starts at 0.
Do not include a background class as it is implicit.

For example:
```
cow,0
cat,1
bird,2
```

## Debugging
Creating your own dataset does not always work out of the box. There is a [`debug.py`](https://github.com/fizyr/keras-retinanet/blob/master/keras_retinanet/bin/debug.py) tool to help find the most common mistakes.

Particularly helpful is the `--annotations` flag which displays your annotations on the images from your dataset. Annotations are colored in green when there are anchors available and colored in red when there are no anchors available. If an annotation doesn't have anchors available, it means it won't contribute to training. It is normal for a small amount of annotations to show up in red, but if most or all annotations are red there is cause for concern. The most common issues are that the annotations are too small or too oddly shaped (stretched out).

## Results

### MS COCO


## Status
Example output images using `keras-retinanet` are shown below.

<p align="center">
  <img src="https://github.com/delftrobotics/keras-retinanet/blob/master/images/coco1.png" alt="Example result of RetinaNet on MS COCO"/>
  <img src="https://github.com/delftrobotics/keras-retinanet/blob/master/images/coco2.png" alt="Example result of RetinaNet on MS COCO"/>
  <img src="https://github.com/delftrobotics/keras-retinanet/blob/master/images/coco3.png" alt="Example result of RetinaNet on MS COCO"/>
</p>

### Notes
* This repository requires Keras 2.1.3.
* This repository is [tested](https://github.com/fizyr/keras-retinanet/blob/master/.travis.yml) using OpenCV 3.4.
* This repository is [tested](https://github.com/fizyr/keras-retinanet/blob/master/.travis.yml) using Python 2.7 and 3.6.
* Warnings such as `UserWarning: Output "non_maximum_suppression_1" missing from loss dictionary.` can safely be ignored. These warnings indicate no loss is connected to these outputs, but they are intended to be outputs of the network for the user (ie. resulting network detections) and not loss outputs.

Contributions to this project are welcome.

### Discussions
Feel free to join the `#keras-retinanet` [Keras Slack](https://keras-slack-autojoin.herokuapp.com/) channel for discussions and questions.
