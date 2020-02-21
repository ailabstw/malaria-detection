# Assessment of Expert-Level Automated Detection of Plasmodium falciparum in Digitized Thin Blood Smear Images
The malaria detection algorithm was developed based on [Retinanet](https://arxiv.org/abs/1708.02002), a 1-stage object detection neural network. We referenced to the keras implementation of Retinanet ([fizyr/keras-retinanet](https://github.com/fizyr/keras-retinanet)).

We offered the source code to train the malaria-detection algorithm, and the inference code to analyse testing blood smear images. The pre-trained weight of the model after trained on TIME (Taiwan Images for Malaria Eradication, see Data Availability section) is available upon request.


## Data Availability
The publicly-available clinically-validated malaria image data sets, the Taiwan Images for Malaria Eradication (TIME), as described in our paper, are available at https://ai.cdc.gov.tw/datasets/


## Installation
### Prerequisite
keras
tensorflow


## Training
After downloading TIME and processing the file paths of images and annotations according to the format outline [here](https://github.com/fizyr/keras-retinanet#csv-datasets), the malaria detection algorithm could be trained by:
```
python ./keras-retinanet/keras-retinanet/bin/train.py /path/to/
```


## Inference
An example script `inference.py` of testing the network on one blood smear image is included in the repository. For usage, run:
```
# Running directly from the repository:
inference.py /path/to/image /path/to/predicted_image /path/to/model/weight
```

## Authors


## License


## Acknowledgement


