# for prediction on one blood smear image

import os.path, sys

import numpy as np
import cv2
import keras
import tensorflow as tf

from keras_retinanet.models.resnet import custom_objects
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image



# use this environment flag to change which GPU to use
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

MODEL_PATH = sys.argv[3]

# load labels to names mapping for visualization
LABELS2NAMES = {
    0: 'P. falciparum_ring',
    1: 'P. falciparum_trophozoite',
    2: 'P. falciparum_gametocyte',
    3: 'P. falciparum_schizont'}   



def get_session():
    '''get tensorflow session'''

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    return tf.Session(config=config)


def preprocess(image):
    '''
    preprocess and resize the image for retinanet input,
    return the processed image and the resize scale
    '''
    image = preprocess_image(image)
    image, scale = resize_image(image)
    
    return image, scale


def predict(model, image, scale):
    '''
    run model inference on image, resized with scale,
    return a list of predictions of infected cells, 
    sorted according to prediction score,
    and processed with non-maximal suppression.
    Each prediction is a list [bounding box coordinates, label, score]
    '''
    _, _, detections = model.predict_on_batch(np.expand_dims(image, axis=0))

    # compute predicted labels and scores
    predicted_labels = np.argmax(detections[0, :, 4:], axis=1)
    scores = detections[0, np.arange(detections.shape[1]), 4 + predicted_labels]

    # correct for image scale
    detections[0, :, :4] /= scale

    preds_img = []
    for idx, (label, score) in enumerate(zip(predicted_labels, scores)):
        if score <= 0:
            continue
        pred = []
        b_pred = detections[0, idx, :4].astype(int)
        pred.append(b_pred)
        pred.append(LABELS2NAMES[label])
        pred.append(score)
        preds_img.append(pred)

    # sorted by score
    def getKey(item):
        return item[2]
    preds_img_sort = sorted(preds_img, key=getKey, reverse=True)

    # non-maximal suppression
    if preds_img_sort:
        boxes = tf.convert_to_tensor(
            np.array([[preds[0][1],
                      preds[0][0],
                      preds[0][3],
                      preds[0][2]] for preds in preds_img_sort]),
            np.float32
        )
        scores = tf.convert_to_tensor(
            np.array([preds[2] for preds in preds_img_sort]),
            np.float32
        )
        preds_img_select = tf.image.non_max_suppression(
            boxes, scores, scores.shape[0], 0.3
        )
        with tf.Session():
            preds_img_select = preds_img_select.eval()
            preds_img_sort = [
                item for ind, item in enumerate(preds_img_sort)
                     if ind in preds_img_select
            ]
    
    return preds_img_sort

    
# visualization
def draw_pred(draw, bb_pred, label, score):
    '''
    draw one predicted bounding box 
    '''
    
    caption = "{} {:.3f}".format(label, score)
    cv2.rectangle(
        draw, 
        (int(bb_pred[0]), int(bb_pred[1])), 
        (int(bb_pred[2]), int(bb_pred[3])), 
        (0, 0, 255), 3
    )   
    cv2.putText(
        draw, 
        caption, 
        (int(bb_pred[0]), int(bb_pred[1]) - 10), 
        cv2.FONT_HERSHEY_PLAIN, 
        1.5, (0, 0, 255), 3
    )
    
    return draw



if __name__ == '__main__':
    
    ## load image, default image type=jpg
    img_fp = sys.argv[1]
    img_out_fp = sys.argv[2]
    
    img = read_image_bgr(img_fp)
    # copy for visualization
    draw = img.copy()
    draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)
    
    ## load retinanet model
    # set the modified tf session as backend in keras
    keras.backend.tensorflow_backend.set_session(get_session())
    model = keras.models.load_model(MODEL_PATH, custom_objects=custom_objects)
    print('successfully reading model weights from: ', MODEL_PATH)
    
    ## detect malaria infection
    img, scale = preprocess(img)
    preds_img_sort = predict(img, scale) 
    
    ## visualize
    if preds_img_sort:
        for i, pred in enumerate(preds_img_sort):
            draw = draw_pred(draw, pred[0], pred[1], pred[2])
    # save the result to img_out_fp
    cv2.imwrite(img_out_fp, draw)
    print('successfully save drawn image into: ', img_out_fp)
