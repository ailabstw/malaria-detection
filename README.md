# Assessment of Expert-Level Automated Detection of Plasmodium falciparum in Digitized Thin Blood Smear Images
The malaria detection algorithm was developed based on [Retinanet](https://arxiv.org/abs/1708.02002), a 1-stage object detection neural network. We referenced to the keras implementation of Retinanet ([fizyr/keras-retinanet](https://github.com/fizyr/keras-retinanet)).

We offered the source code to train the malaria-detection neural network and the inference code to analyse testing blood smear images. The pre-trained weight of the model after trained on TIME (Taiwan Images for Malaria Eradication, see Data Availability section) is available upon request.


## Data Availability
The publicly-available clinically-validated malaria image data sets, the Taiwan Images for Malaria Eradication (TIME), as described in our paper, are available at https://ai.cdc.gov.tw/datasets/


## Installation
1. Clone this repository
2. Prerequisite
   - keras
   - tensorflow


## Inference
An example script `inference.py` for testing the malaria detection network on one blood smear image is included in the repository. For usage, run:
```
# Running directly from the repository:
inference.py /path/to/image /path/to/output/image /path/to/model/weight
```


## Training
After downloading TIME and processing the file paths of images and annotation files according to the format outlined [here](https://github.com/fizyr/keras-retinanet#csv-datasets), the malaria detection algorithm could be trained by:
```
python ./keras-retinanet/keras-retinanet/bin/train.py csv /path/to/csv/containing/annotation /path/to/csv/containing/classes
```


## Authors
Po-Chen Kuo, MD; Hao-Yuan Cheng, MD, MSc; Pi-Fang Chen, MSc; Yu-Lun Liu, MD, MSc; Martin Kang, MD; Min-Chu Kuo, AS; Shih-Fen Hsu, BSc; Hsin-Jung Lu, BSc; Stefan Hong, MSc; Chan-Hung Su, MSc; Ding-Ping Liu, PhD; Yi-Chin Tu, MSc; Jen-Hsiang Chuang, MD, PhD


## License
GNU GPL v3.0


## Acknowledgement
We would like to thank Chi-Cheng Jou, PhD, Hsiao-Ju Chang, MSc, Cheng-Hsien Shen, MSc (Taiwan AI Labs), and Yu-Chiang Wang, PhD (National Taiwan University), for providing helpful discussion on algorithm development

