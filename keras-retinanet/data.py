# !/usr/bin/env python3
#
# Created by PoChen on 2018/07/23
# Loading data from cdc-annotated malaria images directories, 
# generating data statistics and annotation csv file for retina net.

# desired output
# data statistics: slide, image, annotations
# annotation csv files for retina net input

### TO DO
# ann_img, draw: ring_trophozoite was ruled out

import collections
import os
import glob
import xml.etree.cElementTree as ET
import pickle

import numpy as np
import cv2
import pandas as pd
import csv

import keras
import tensorflow as tf

from keras_retinanet.models.resnet import custom_objects
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
# set tf backend to allow memory to grow
# instead of claiming everything





PREDEFINED_LABEL_TXT = '/volume/workspace/fredpckuo/malaria/data/cdc/annotation/20180703/DO_NOT_MODIFIED_0703/predefined_classes.txt'

LABELS2NAMES = {
    0: 'P. falciparum_ring',
    1: 'P. falciparum_trophozoite',
    2: 'P. falciparum_gametocyte',
    3: 'P. falciparum_schizont'
}
# load label to names mapping for visualization purposes


# get predefined labels
def get_predefined_labels(classes_filepath=PREDEFINED_LABEL_TXT):
	"""
    return a list of predefined labels from an input txt file
    """
	l_classes = []
	with open (classes_filepath, 'r') as fh:
		for line in fh:
			l_classes.append(line.replace('\n', '').replace('\r', ''))
	return l_classes
predefined_labels = get_predefined_labels(PREDEFINED_LABEL_TXT)
# for combined analysis of ring and trophozoite stage
predefined_labels += ['P. falciparum_ring_trophozoite', 
    'No_concensus', 'malaria_infected'
]
# labels to calculate statistics
labels4stats = ['P. falciparum_ring_trophozoite',
                'P. falciparum_ring',
                'P. falciparum_trophozoite', 
                'P. falciparum_schizont',
                'P. falciparum_gametocyte',
                'malaria_infected'
]



## base class to store malaria bounding boxes
Bndbox = collections.namedtuple(
    'Bndbox', 
    ['label', 'coords', 'annotator', 'confidence'] 
) 
# coords in the format of (x1, y1, x2, y2)

## base class to store a match pair
Match_Pair = collections.namedtuple(
    'Match_Pair',
    ['annotator_pair', 'bndbox_pair', 'iou', 'intersection', 'union']
)

## base class to store a key to match pair lists
Key2Match_Pairs = collections.namedtuple(
    'Key_Match_Pairs_Stats',
    ['label', 'iou_threshold', 'confidence_threshold']
)

Box_Level_Stats = collections.namedtuple(
    'Box_Level_Stats',
    ['tp', 'fp', 'fn', 'precision', 'recall', 'f1_score']
)

Img_Level_Stats = collections.namedtuple(
    'Img_Level_Stats',
    ['tp', 'fp', 'fn', 'tn', 'precision', 'recall', 'f1_score', 'spe']
)

class Imgwise_Match_Pairs_Stats:
    """
    base class to store match pairs statistics, given a match_pairs_key(annotator_pair, iou_threshold, label)
    """

    def __init__(self, annotator1, annotator2, csv_out_dir):
        self.annotator_pair = (annotator1, annotator2)
        self.csv_out_dir = csv_out_dir
        
        self.match_pairs = {}
        # key = key2match_pairs, value = list of match pairs after matching
        self.no_match_1 = {}
        self.no_match_2 = {}

        self.box_level_contingency = {}
        # key = key2match_pairs, value = table of box level inter-rater contingency
        self.box_level_contingency_by_label = {}
        # key = key2match_pairs, value = table of label-wise box level inter-rater contingency

        self.img_level_truth_value = {}
        # key = key2match_pairs, value = table of img level inter-rater contingency

    def creat_contingency_table(self):
        columns = [label_name for label_name in predefined_labels]
        df = pd.DataFrame(index=columns, columns=columns)
        df = df.fillna(0)
        return df

    def get_box_level_contingency(self, key2match_pairs):
        """
        """
        
        ## initialize
        # csv file path to store imgwise box_level contingency table 
        csv_out_path = self.csv_out_dir + 'box_level_contingency_iou_' + str(int(key2match_pairs.iou_threshold * 100)) + '_conf_' +  str(int(key2match_pairs.confidence_threshold * 100)) + '.csv'
        # contingency table
        self.box_level_contingency[key2match_pairs] = self.creat_contingency_table()

        ## fill in contingency table from match_pairs and no_match lists
        for match_pair in self.match_pairs[key2match_pairs]:
            col_name = match_pair.bndbox_pair[0].label
            row_name = match_pair.bndbox_pair[1].label
            self.box_level_contingency[key2match_pairs][col_name][row_name] += 1
        for miss in self.no_match_1[key2match_pairs]:
            col_name = miss.label
            row_name = 'Negative'
            self.box_level_contingency[key2match_pairs][col_name][row_name] += 1
        for miss in self.no_match_2[key2match_pairs]:
            row_name = miss.label
            col_name = 'Negative'
            self.box_level_contingency[key2match_pairs][col_name][row_name] += 1

        ## save csv file
        self.box_level_contingency[key2match_pairs].to_csv(csv_out_path)

    def get_img_level_truth_value(self):
        
        def create_contingency_table_by_label(label):
            # initialize table
            columns = [label, 'Others']
            df = pd.DataFrame(index=columns, columns=columns)
            df = df.fillna(0)

            return df

        def _get_box_level_contingency_by_label(self, label, key2match_pairs):
            ## initialize 
            # output csv file path 
            labelwise_contingency_dir = self.csv_out_dir + 'labelwise/'
            if not os.path.exists(labelwise_contingency_dir):
                os.makedirs(labelwise_contingency_dir)
            csv_out_path = labelwise_contingency_dir + label + '_contingency_iou_' + str(int(key2match_pairs.iou_threshold * 100)) + '_conf_' + str(int(key2match_pairs.confidence_threshold * 100)) + '.csv'
            # imgwise inter rater contingency by label
            df = create_contingency_table_by_label(label)
            # key 
            _key2match_pairs = Key2Match_Pairs(label, key2match_pairs.iou_threshold, key2match_pairs.confidence_threshold)

            ## fill in table
            # true positive
            df[label][label] = self.box_level_contingency[key2match_pairs][label][label]
            # false positive and fals negative
            for label_ in list(self.box_level_contingency[key2match_pairs].columns.values):
                if label_ is not label and '_sum' not in label_:
                    df[label]['Others'] += self.box_level_contingency[key2match_pairs][label][label_]
            for label_ in list(self.box_level_contingency[key2match_pairs].index):
                if label_ is not label and '_sum' not in label_:
                    df['Others'][label] += self.box_level_contingency[key2match_pairs][label_][label]
            
            ## assgin to box level contingency
            self.box_level_contingency_by_label[_key2match_pairs] = df
            
            ## save csv file
            df.to_csv(csv_out_path)

        def _get_img_level_truth_value(self, label, key2match_pairs):
            
            img_level_truth_value = 'TN'

            _key2match_pairs = Key2Match_Pairs(label, key2match_pairs.iou_threshold, key2match_pairs.confidence_threshold)

            
            #print(self.box_level_contingency_by_label[_key2match_pairs])
            
            count_label_by_ann1 = self.box_level_contingency_by_label[_key2match_pairs][label].sum()
            if count_label_by_ann1 > 0:
                if self.box_level_contingency_by_label[_key2match_pairs][label][label] > 0:
                    img_level_truth_value = 'TP'
                else:
                    img_level_truth_value = 'FN'
            else:
                if self.box_level_contingency_by_label[_key2match_pairs]['Others'][label] > 0:
                    img_level_truth_value = 'FP'
            #print(img_level_truth_value)

            self.img_level_truth_value[_key2match_pairs] = img_level_truth_value

        ## get labelwise contingency and statistics
        for key, stats in self.box_level_contingency.items():
            #print('------ img_level_truth_value: {}'.format(key))
            for label in list(stats.columns.values):
                if label in labels4stats:
                    _get_box_level_contingency_by_label(self, label, key)
                    _get_img_level_truth_value(self, label, key)



class AnnImg:
    """
    Base class to store malaria image and annotation
    """

    def __init__(self, img_name, img_path):
        self.img_name = img_name # img file name
        self.img_path = img_path # image file path
        self.slide_name = ''

        self.bndboxes = {} 
        # key = annotator, value = list of Bodboxes
        self.differential_counts_by_annotator = {} 
        # key = annotator, value = dictionary of label and counts
        
        self.match_pairs_stats = {}
        # key = annotator pair, value = imgwise_match_pairs_stats

        self.expert_concensus = None

        self.retina_ann_csv_lists = {} 
        # key = annotator, value = corresponding list for output csv file
    
    def get_slide_name(self):
        if '_Pf_thin' in self.img_name:
            self.slide_name = self.img_name[ : self.img_name.find('_Pf_thin')]
        elif '_N_thin' in self.img_name:
            self.slide_name = self.img_name[ : self.img_name.find('_N_thin')]
        else:
            self.slide_name = 'unexpected_slide_name'

    def add_bndbox(self, label, coords, annotator, confidence): 
        # coords in the format of (x1, y1, x2, y2)
        bndbox = Bndbox(label, coords, annotator, confidence)
        if bndbox.annotator not in self.bndboxes.keys():
            self.bndboxes[annotator] = [bndbox]
        else:
            self.bndboxes[annotator].append(bndbox)
    
    def get_counts_by_annotator(self, annotator):
        """
        for a given annotator, counting number of boxes for each label, storing in differential_counts_by_annotator
        """
        counts = {label : 0 for label in predefined_labels} 
        # recording unexpected class aside from pre-defined labels
        wrong_labels = [] 

        if annotator in self.bndboxes.keys():
            for bndbox in self.bndboxes[annotator]:
                if bndbox.label in predefined_labels:
                    counts[bndbox.label] += 1
                else:
                    wrong_labels.append(bndbox.label)
        else:
            print('-- WARNING: no known annotator as {}'.format(annotator))
        
        if wrong_labels:
            counts['Wrong_labels'] = wrong_labels
        
        self.differential_counts_by_annotator[annotator] = counts
    
    def draw_bndboxes(self, annotator_list, img_out_dir):
        """
        draw bndboxes from annotators in annotator_list on the smear img, 
        save it for visualization
        """
        def drawing(img, bndbox, annotator, color):
            """
            draw one bounding box
            """
            cv2.rectangle(img, tuple(bndbox.coords[:2]), tuple(bndbox.coords[2:]), color, 2)
            if annotator == 'model':
                caption = "{} {:.3f}".format(bndbox.label, bndbox.confidence)
                cv2.putText(img, caption, tuple(bndbox.coords[:2]), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color)    
            else:
                cv2.putText(img, bndbox.label, tuple(bndbox.coords[:2]), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color)
        
        img_out_path = img_out_dir + 'draw_' + self.img_name + '.jpg'
        
        # generate color list
        color_list = [(0, 0, 0), (0,0,255), (0,255,0), (255,0,0), (0,255,255),
                        (255,0,255), (255,255,0), (255,255,255)]
        
        if len(annotator_list) > len(color_list):
            print('-- WARNING: annotator list is too long!')
        else:
            img = cv2.imread(self.img_path)
            img_copy = img.copy()
            for i, annotator in enumerate(annotator_list):
                if annotator in self.bndboxes.keys():
                    for bndbox in self.bndboxes[annotator]:
                        #do not draw ring_trophozoite
                        if bndbox.label != 'P. falciparum_ring_trophozoite':
                            drawing(img_copy, bndbox, annotator, color_list[i])
            cv2.imwrite(img_out_path, img_copy)    
   
    def matching(self, annotator1, annotator2,  bndboxes1, bndboxes2, iou_threshold, confidence_threshold): 
        """
        given 2 lists of bndboxes,
        calculate iou for all pairs of bndboxes,
        sort the matrix by iou value,
        return a list of bndbox match pairs
        """
        
        def get_iou(box1, box2):
            
            def get_area(box):
                return float((box[2] - box[0] + 1) * (box[3] - box[1] + 1))
            area1 = get_area(box1)
            area2 = get_area(box2)
            # get the coordinates of intersection
            x0 = max(box1[0], box2[0])
            y0 = max(box1[1], box2[1])
            x1 = min(box1[2], box2[2])
            y1 = min(box1[3], box2[3])
            box_inter = (x0, y0, x1, y1)
            if x1 >= x0 and y1 >= y0:
                area_inter = get_area(box_inter)
                iou = area_inter / (area1 + area2 - area_inter)
            else:
                iou = 0
            return iou

        def get_intersection_coord(box1, box2):
            x1 = max(box1[0], box2[0])
            y1 = max(box1[1], box2[1])
            x2 = min(box1[2], box2[2])
            y2 = min(box1[3], box2[3])
            
            return (x1, y1, x2, y2)

        def get_union_coord(box1, box2):
            x1 = min(box1[0], box2[0])
            y1 = min(box1[1], box2[1])
            x2 = max(box1[2], box2[2])
            y2 = max(box1[3], box1[3])

            return (x1, y1, x2, y2)

        iou_matrix = []
        match_pairs = []
        no_match_1, no_match_2 = [], []

        # calculating iou to generate iou matrix
        for i, bndbox1 in enumerate(bndboxes1):
            iou_list = []
            for j, bndbox2 in enumerate(bndboxes2):
                iou = get_iou(bndbox1.coords, bndbox2.coords)
                iou_list.append(iou)
            iou_matrix.append(iou_list)
        iou_matrix = np.array(iou_matrix)
        
        # sort the iou matrix by iou
        sorted_flat_index = np.argsort(iou_matrix.ravel())[::-1]
        sorted_2d_index_stack = np.dstack(np.unravel_index(sorted_flat_index, iou_matrix.shape))

        # select bndbox match pairs based on iou and confidence
        picked_i, picked_j = [], []
        for i, j in sorted_2d_index_stack[0]:
            if (i not in picked_i and 
                j not in picked_j and 
                iou_matrix[i][j] > iou_threshold):
                
                if (bndboxes1[i].confidence >= confidence_threshold and 
                    bndboxes2[j].confidence >= confidence_threshold):
                    
                    intersection = get_intersection_coord(
                        bndboxes1[i].coords,
                        bndboxes2[j].coords
                    )
                    union = get_union_coord(
                        bndboxes1[i].coords,
                        bndboxes2[j].coords
                    )
                    match_pair = Match_Pair(
                        (annotator1, annotator2), 
                        (bndboxes1[i], bndboxes2[j]), 
                        iou_matrix[i][j],
                        intersection,
                        union
                    )
                    match_pairs.append(match_pair)
                    picked_i.append(i)
                    picked_j.append(j)
        
        # if not picked up by the above selection, it's a no match
        for _i in range(len(bndboxes1)):
            if _i not in picked_i:
                no_match_1.append(bndboxes1[_i])
        for _j in range(len(bndboxes2)):
            if _j not in picked_j:
                if bndboxes2[_j].confidence >= confidence_threshold:
                    no_match_2.append(bndboxes2[_j])
        
        #print('-- input number of bndboxes, respectively: {} {}'.format(len(bndboxes1), len(bndboxes2)))
        #print('-- output number of match pairs, no match 1, no match 2: {} {} {}'.format(len(match_pairs), len(no_match_1), len(no_match_2)))
        
        return match_pairs, no_match_1, no_match_2

    def get_match_pairs_with_answers(self, annotator1, annotator2, csv_out_dir):
        """
        
        """
        

        def matching(bndboxes1, bndboxes2, BNDBOX_MATCH_THRESH=0.3): # list of Bndboxes
            
            def get_iou(box1, box2):
                
                def get_area(box):
                    return float((box[2] - box[0] + 1) * (box[3] - box[1] + 1))
                area1 = get_area(box1)
                area2 = get_area(box2)
                # get the coordinates of intersection
                x0 = max(box1[0], box2[0])
                y0 = max(box1[1], box2[1])
                x1 = min(box1[2], box2[2])
                y1 = min(box1[3], box2[3])
                box_inter = (x0, y0, x1, y1)
                if x1 >= x0 and y1 >= y0:
                    area_inter = get_area(box_inter)
                    iou = area_inter / (area1 + area2 - area_inter)
                else:
                    iou = 0
                return iou

            match_pairs = []
            ann1_pos_ann2_neg = []

            for bndbox1 in bndboxes1:
                iou_temp_list = []
                bndboxes2_ = bndboxes2[:]
                #print('-- matching for {}'.format(bndbox1))
                for bndbox2 in bndboxes2_:
                    iou = get_iou(bndbox1.coords, bndbox2.coords)
                    #if iou > 0:
                    #    bndbox_pair = (bndbox1, bndbox2, iou)
                    #    pairs.append(Match_Pair(annotator_pair, bndbox_pair, iou))
                    if iou > iou_temp:
                        iou_temp = iou
                        bndbox_pair = (bndbox1, bndbox2)
                if iou_temp > BNDBOX_MATCH_THRESH:
                    #print('-- bndbox to remove: {}'.format(bndbox_pair[1]))
                    #print('-- bndboxes2_: {}'.format(bndboxes2_))
                    bndboxes2_ = [bndbox_ for bndbox_ in bndboxes2_ 
                                            if not (np.array_equal(bndbox_.coords, bndbox_pair[1].coords))]
                    match_pairs.append(Match_Pair(annotator_pair, bndbox_pair, iou_temp))
                else:
                    ann1_pos_ann2_neg.append(bndbox1)
            return match_pairs, ann1_pos_ann2_neg
        
        csv_out_path = csv_out_dir + annotator1 + '_' + annotator2 + '_' + self.img_name + 'contingency.csv' 

        #print('-- getting match pairs of {} and {}:'.format(annotator1, annotator2))
        annotator_pair = (annotator1, annotator2)
        
        create_inter_rater_contingency(self, annotator_pair)

        match_pairs_by_ann1, miss_by_ann2 = [], []
        match_pairs_by_ann2, miss_by_ann1 = [], []

        # match by annotator1
        if annotator1 in self.bndboxes.keys():
            if annotator2 in self.bndboxes.keys():
                match_pairs_by_ann1, miss_by_ann2 = matching(self.bndboxes[annotator1], self.bndboxes[annotator2])
            else:
                print('-- WARNING: given {}, missing annotations from {}'.format(annotator1, annotator2))
        else:
            print('-- WARNING: No annotations from {}'.format(annotator1))
        #print('-- match pairs by ann1: {}'.format(match_pairs_by_ann1))

        # match by annotator2
        if annotator2 in self.bndboxes.keys():
            if annotator1 in self.bndboxes.keys():
                match_pairs_by_ann2, miss_by_ann1 = matching(self.bndboxes[annotator2], self.bndboxes[annotator1])
            else:
                print('-- given {}, missing annotations from {}'.format(annotator2, annotator1))
        else:
            print('-- No annotations from {}'.format(annotator2))
        #print('-- match pairs by ann2: {}'.format(match_pairs_by_ann2))

        # fill in contingency table
        if len(match_pairs_by_ann1) == len(match_pairs_by_ann2):
            self.match_pairs[annotator_pair] = match_pairs_by_ann1
            for match_pair in match_pairs_by_ann1:
                col_name = match_pair.bndbox_pair[0].label
                row_name = match_pair.bndbox_pair[1].label
                self.inter_rater_contingency[annotator_pair][col_name][row_name] += 1
        else:
            print(self.img_name)
            print('-- WARNING: match pairs by {} are different!'.format(annotator_pair))
            print('-- match pairs by ann1: {}'.format(len(match_pairs_by_ann1)))
            print('-- match pairs by ann2: {}'.format(len(match_pairs_by_ann2)))
            count = 0
            for match_pair in match_pairs_by_ann1:
                col_name = match_pair.bndbox_pair[0].label
                row_name = match_pair.bndbox_pair[1].label
                if col_name == row_name:
                    count += 1
            print(count)
            count2 = 0
            for match_pair in match_pairs_by_ann2:
                col_name = match_pair.bndbox_pair[0].label
                row_name = match_pair.bndbox_pair[1].label
                if col_name == row_name:
                    count2 += 1
            print(count2)
            
        
        for miss in miss_by_ann2:
            col_name = miss.label
            row_name = 'Negative'
            self.inter_rater_contingency[annotator_pair][col_name][row_name] += 1
        for miss in miss_by_ann1:
            row_name = miss.label
            col_name = 'Negative'
            self.inter_rater_contingency[annotator_pair][col_name][row_name] += 1
        
        self.inter_rater_contingency[annotator_pair].to_csv(csv_out_path)

        #print('match pairs by {} \n {}'.format(annotator1, match_pairs_by_ann1))
        #print('match pairs by {} \n {}'.format(annotator2, match_pairs_by_ann2))
        #print('ann1_pos_ann2_neg {}'.format(miss_by_ann2))
        #print('ann2_pos_ann1_neg {}'.format(miss_by_ann1))
        #print('{}'.format(self.inter_rater_contingency[annotator_pair]))

    def gen_expert_concensus(
        self, expert_names, iou_threshold, confidence_threshold
        ):
        """
        get match pairs for 2 of the experts,
        get match_pairs, no_match_1, no_match_2,
        then match expert 3's annotations to the above 3 lists,

        """
        match_pairs_12, pos_1_neg_2, pos_2_neg_1 = self.matching(
            expert_names[0],
            expert_names[1],
            self.bndboxes[expert_names[0]],
            self.bndboxes[expert_names[1]],
            iou_threshold,
            confidence_threshold
        )
        match_boxes_12 = [
            Bndbox(
                (match_pair.bndbox_pair[0].label, match_pair.bndbox_pair[1].label),
                match_pair.union,
                (match_pair.annotator_pair[0], match_pair.annotator_pair[1]),
                1
            ) for match_pair in match_pairs_12
        ]
        match_pairs_123, pos_12_neg_3, pos_3_neg_12 = self.matching(
            expert_names[0] + '_' + expert_names[1],
            expert_names[2],
            match_boxes_12,
            self.bndboxes[expert_names[2]],
            iou_threshold,
            confidence_threshold
        )
        pos_13_neg_2, pos_1_neg_23, pos_3_neg_12_1 = self.matching(
            'POS_' + expert_names[0] + '_NEG_' + expert_names[1],
            expert_names[2],
            pos_1_neg_2,
            pos_3_neg_12,
            iou_threshold,
            confidence_threshold
        )
        pos_23_neg_1, pos_2_neg_13, pos_3_neg_12_2 = self.matching(
            'POS_' + expert_names[1] + '_NEG_' + expert_names[0],
            expert_names[2],
            pos_2_neg_1,
            pos_3_neg_12_1,
            iou_threshold,
            confidence_threshold
        )

        df = pd.DataFrame(
            columns=['Img_name', 'Box', 'befun','cd6397',
                'tmumt4009', 'Source', 'Concensus'
            ]
        )

        def voting(labels):
            votes = {}
            for label in labels:
                if label not in votes.keys():
                    votes[label] = 1
                else:
                    votes[label] += 1
            result = 'No_concensus'
            for label, count in votes.items():
                if count >= 2:
                    result = label
            return result

        for match_pair in match_pairs_123:
            img_name = self.img_name
            box = match_pair.union
            _befun = match_pair.bndbox_pair[0].label[0]
            _cd6397 = match_pair.bndbox_pair[0].label[1]
            _tmumt4009 = match_pair.bndbox_pair[1].label
            source = 'pos_123'
            concensus = voting([_befun, _cd6397, _tmumt4009])

            row = {
                'Img_name': img_name,
                'Box': box,
                'befun': _befun,
                'cd6397': _cd6397,
                'tmumt4009': _tmumt4009,
                'Source': source,
                'Concensus': concensus
            }
            df = df.append(row, ignore_index=True)

        for match_pair in pos_13_neg_2:
            img_name = self.img_name
            box = match_pair.union
            _befun = match_pair.bndbox_pair[0].label
            _cd6397 = 'Negative'
            _tmumt4009 = match_pair.bndbox_pair[1].label
            source = 'pos_13_neg_2'
            concensus = voting([_befun, _cd6397, _tmumt4009])

            row = {
                'Img_name': img_name,
                'Box': box,
                'befun': _befun,
                'cd6397': _cd6397,
                'tmumt4009': _tmumt4009,
                'Source': source,
                'Concensus': concensus
            }
            df = df.append(row, ignore_index=True)

        for match_pair in pos_23_neg_1:
            img_name = self.img_name
            box = match_pair.union
            _befun = 'Negative'
            _cd6397 = match_pair.bndbox_pair[0].label
            _tmumt4009 = match_pair.bndbox_pair[1].label
            source = 'pos_23_neg_1'
            concensus = voting([_befun, _cd6397, _tmumt4009])
            
            row = {
                'Img_name': img_name,
                'Box': box,
                'befun': _befun,
                'cd6397': _cd6397,
                'tmumt4009': _tmumt4009,
                'Source': source,
                'Concensus': concensus
            }
            df = df.append(row, ignore_index=True)

        for bndbox in pos_12_neg_3:
            img_name = self.img_name
            box = bndbox.coords
            _befun = bndbox.label[0]
            _cd6397 = bndbox.label[1]
            _tmumt4009 = 'Negative'
            source = 'pos_12_neg_3'
            concensus = voting([_befun, _cd6397, _tmumt4009])

            row = {
                'Img_name': img_name,
                'Box': box,
                'befun': _befun,
                'cd6397': _cd6397,
                'tmumt4009': _tmumt4009,
                'Source': source,
                'Concensus': concensus
            }
            df = df.append(row, ignore_index=True)
        
        for bndbox in pos_3_neg_12_2:
            img_name = self.img_name
            box = bndbox.coords
            _befun = 'Negative'
            _cd6397 = 'Negative'
            _tmumt4009 = bndbox.label
            source = 'pos_3_neg_12_2'
            concensus = voting([_befun, _cd6397, _tmumt4009])

            row = {
                'Img_name': img_name,
                'Box': box,
                'befun': _befun,
                'cd6397': _cd6397,
                'tmumt4009': _tmumt4009,
                'Source': source,
                'Concensus': concensus
            }
            df = df.append(row, ignore_index=True)

        for bndbox in pos_1_neg_23:
            img_name = self.img_name
            box = bndbox.coords
            _befun = bndbox.label
            _cd6397 = 'Negative'
            _tmumt4009 = 'Negative'
            source = 'pos_1_neg_23'
            concensus = voting([_befun, _cd6397, _tmumt4009])

            row = {
                'Img_name': img_name,
                'Box': box,
                'befun': _befun,
                'cd6397': _cd6397,
                'tmumt4009': _tmumt4009,
                'Source': source,
                'Concensus': concensus
            }
            df = df.append(row, ignore_index=True)

        for bndbox in pos_2_neg_13:
            img_name = self.img_name
            box = bndbox.coords
            _befun = 'Negative'
            _cd6397 = bndbox.label
            _tmumt4009 = 'Negative'
            source = 'pos_2_neg_13'
            concensus = voting([_befun, _cd6397, _tmumt4009])

            row = {
                'Img_name': img_name,
                'Box': box,
                'befun': _befun,
                'cd6397': _cd6397,
                'tmumt4009': _tmumt4009,
                'Source': source,
                'Concensus': concensus
            }
            df = df.append(row, ignore_index=True)
        
        self.expert_concensus = df

        ## generate expert concensus boxes
        self.bndboxes['Concensus'] = []
        for _, row in df.iterrows():
            label = row['Concensus']
            box = row['Box']
            annotator = 'Concensus'
            confidence = 1
            self.bndboxes['Concensus'].append(
                Bndbox(label, box, annotator, confidence)
            )

    def get_match_pairs(self, annotator1, annotator2, iou_threshold, confidence_threshold, csv_out_dir):
        """
        given annotator pairs, iou_threshold, confidence_threshold,
        matching bounding boxes for the 2 annotators,
        return the list of match pairs,
        fill in irr table
        """
        
        ## initialize
        #print('-- getting match pairs of {} and {}:'.format(annotator1, annotator2))
        # output dir for imgwise contingency
        imgwise_csv_out_dir = csv_out_dir + self.img_name + '/'
        if not os.path.exists(imgwise_csv_out_dir):
            os.makedirs(imgwise_csv_out_dir)
        # imgwise_match_pairs_stats
        if (annotator1, annotator2) not in self.match_pairs_stats.keys():
            self.match_pairs_stats[(annotator1, annotator2)] = Imgwise_Match_Pairs_Stats(annotator1, annotator2, imgwise_csv_out_dir)
        # key2match_pairs
        key2match_pairs = Key2Match_Pairs('All', iou_threshold, confidence_threshold)

        ## match
        if annotator1 in self.bndboxes.keys():
            if annotator2 in self.bndboxes.keys():
                match_pairs, no_match_1, no_match_2 = self.matching(
                    annotator1, annotator2,
                    self.bndboxes[annotator1],
                    self.bndboxes[annotator2],
                    iou_threshold,
                    confidence_threshold
                )
            else:
                print('-- WARNING: given {}, no annotations from {}'.format(annotator1, annotator2))
        else:
            print('-- WARNING: No annotations from {}'.format(annotator1))

        ## fill in box_level contingency table
        self.match_pairs_stats[(annotator1, annotator2)].match_pairs[key2match_pairs] = match_pairs
        self.match_pairs_stats[(annotator1, annotator2)].no_match_1[key2match_pairs] = no_match_1
        self.match_pairs_stats[(annotator1, annotator2)].no_match_2[key2match_pairs] = no_match_2
        self.match_pairs_stats[(annotator1, annotator2)].get_box_level_contingency(key2match_pairs)

        
    def get_img_level_matching(self, annotator1, annotator2, iou_threshold, confidence_threshold):
        for label in labels4stats:
            key2match_pairs = Key2Match_Pairs(label, iou_threshold, confidence_threshold)
            img_level_truth_value = 'TN'
            count_label_ann1, count_label_ann2 = 0, 0
            for bndbox in self.bndboxes[annotator1]:
                if bndbox.label == label:
                    count_label_ann1 += 1
            for bndbox in self.bndboxes[annotator2]:
                if bndbox.label == label and bndbox.confidence >= confidence_threshold:
                    count_label_ann2 += 1
            if count_label_ann1 > 0:
                if count_label_ann2 > 0:
                    img_level_truth_value = 'TP'
                else:
                    img_level_truth_value = 'FN'
            else:
                if count_label_ann2 > 0:
                    img_level_truth_value = 'FP'
            self.match_pairs_stats[(annotator1, annotator2)].img_level_truth_value[key2match_pairs] = img_level_truth_value


    def gen_retina_ann_csv_list(self, annotator, label_for_indeterminate):
        if annotator not in self.retina_ann_csv_lists.keys():
            self.retina_ann_csv_lists[annotator] = []
        if annotator not in self.bndboxes.keys():
            self.retina_ann_csv_lists[annotator].append([self.img_path, '', '', '', '', ''])
        else:
            for _, bndbox in enumerate(self.bndboxes[annotator]):
            	if bndbox[0] != 'Negative': # drop out Negative cases
                    if bndbox[0] == 'Indeterminate': # include all indeterminate cases in Pf ring
                        self.retina_ann_csv_lists[annotator].append([
                            self.img_path,
                            bndbox[1][0], bndbox[1][1], bndbox[1][2], bndbox[1][3],
                            label_for_indeterminate
                            ])
                    else:
                        self.retina_ann_csv_lists[annotator].append([
                            self.img_path, 
                            bndbox[1][0], bndbox[1][1], bndbox[1][2], bndbox[1][3], bndbox[0]
                            ])
    
    def inference(self, model):

        def preprocess(image):
            image = preprocess_image(image)
            image, scale = resize_image(image)
            return image, scale
        
        def predict(model, image, scale):
            
            print(image.shape)
            print(np.expand_dims(image, axis=0).shape)

            _, _, detections = model.predict_on_batch(
                np.expand_dims(image, axis=0)
            )

            # compute predicted labels and scores
            predicted_labels = np.argmax(
                detections[0, :, 4:], axis=1
            )
            scores = detections[0, 
                                np.arange(detections.shape[1]), 
                                4 + predicted_labels]
            # correct for image scale
            detections[0, :, :4] /= scale

            preds_img = []
            for idx, (label, score) in enumerate(
                zip(predicted_labels, scores)
                ):
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
            preds_img_sort = sorted(
                preds_img, key=getKey, reverse=True)
            print('-- preds_img_sort: {}'.format(preds_img_sort))

            return preds_img_sort

        img = read_image_bgr(self.img_path)
        img, scale = preprocess(img)
        preds_img_sort = predict(model, img, scale) 
        # preds_img[0] = pred_bb, 
        # [1] = pred_label, 
        # [2] = pred_score
        print(preds_img_sort)
       
        if preds_img_sort:    
            boxes = tf.convert_to_tensor(
                np.array([[
                    preds[0][1], 
                    preds[0][0],
                    preds[0][3],
                    preds[0][2]]
                    for preds in preds_img_sort]),
                np.float32
            )
            scores = tf.convert_to_tensor(
                np.array(
                    [preds[2]
                    for preds in preds_img_sort]),
                np.float32
            )
            preds_img_select = tf.image.non_max_suppression(
                boxes,
                scores,
                scores.shape[0],
                0.3
            )
            with tf.Session():
                preds_img_select = preds_img_select.eval()
                preds_img_sort = [
                    item 
                    for ind, item in enumerate(preds_img_sort)
                    if ind in preds_img_select
                    ]
                print('  --preds: {}'.format(preds_img_sort))
        
        for pred in preds_img_sort:
            self.add_bndbox(
                pred[1], #label
                pred[0], #coord
                'model', #annotator
                pred[2]  #confidence
            )



class Datasetwise_Match_Pairs_Stats:
    """

    """
    
    def __init__(self, annotator1, annotator2, csv_out_dir):
        self.annotator_pair = (annotator1, annotator2)
        self.csv_out_dir = csv_out_dir
        self.contingency_dir = ''
        self.stat_dir = ''

        self.box_level_contingency = {}
        # key = key2match_pairs, value = table of box level inter-rater contingency
        self.box_level_stats = None
        self.box_level_contingency_by_label = {}
        # key = key2match_pairs, value = table of label-wise box level inter-rater contingency
        self.box_level_stats_by_label = {}

        self.img_level_contingency = {}
        # key = key2match_pairs, value = table of img level inter-rater contingency
        self.img_level_stats = None
        self.img_level_contingency_by_label = {}
        # key = key2match_pairs, value = talbe of label-wise img level inter-rater contingency
        self.img_level_stats_by_label = {}

    def create_contingency_dir(self):
        contingency_dir = self.csv_out_dir + self.annotator_pair[0] + '_' + self.annotator_pair[1] + '/contingency/'
        if not os.path.exists(contingency_dir):
            os.makedirs(contingency_dir)
            print('-- successfully created contingency_dir for {} and {} at: {}'.format(self.annotator_pair[0], self.annotator_pair[1], contingency_dir))
        self.contingency_dir = contingency_dir
    def create_stat_dir(self):
        stat_dir = self.csv_out_dir + self.annotator_pair[0] + '_' + self.annotator_pair[1] + '/stat/'
        if not os.path.exists(stat_dir):
            os.makedirs(stat_dir)
        print('-- successfully created stat_dir for {} and {} at: {}'.format(self.annotator_pair[0], self.annotator_pair[1], stat_dir))

        self.stat_dir = stat_dir

    def creat_contingency_table(self):
        # initialize table
        columns = [label_name for label_name in predefined_labels]
        df = pd.DataFrame(index=columns+[self.annotator_pair[0]+'_sum'], columns=columns+[self.annotator_pair[1]+'_sum'])
        df = df.fillna(0)
        
        return df

    def get_box_level_statistics_by_label(self, key2match_pairs):
        """
        """
        
        def create_contingency_table_by_label(label):
            # initialize table
            columns = [label, 'Others']
            df = pd.DataFrame(index=columns, columns=columns)
            df = df.fillna(0)

            return df

        def _get_box_level_contingency_by_label(self, label, key2match_pairs):
            
            ## initialize 
            # output csv file path 
            labelwise_contingency_dir = self.contingency_dir + 'labelwise/'
            if not os.path.exists(labelwise_contingency_dir):
                os.makedirs(labelwise_contingency_dir)
            csv_out_path = labelwise_contingency_dir + label + '_contingency_iou_' + str(int(key2match_pairs.iou_threshold * 100)) + '_conf_' + str(int(key2match_pairs.confidence_threshold * 100)) + '.csv'
            # datasetwise inter rater contingency by label
            df = create_contingency_table_by_label(label)
            # key 
            _key2match_pairs = Key2Match_Pairs(label, key2match_pairs.iou_threshold, key2match_pairs.confidence_threshold)

            ## fill in table
            # true positive
            df[label][label] = self.box_level_contingency[key2match_pairs][label][label]
            # false positive and fals negative
            for label_ in list(self.box_level_contingency[key2match_pairs].columns.values):
                if label_ is not label and '_sum' not in label_:
                    df[label]['Others'] += self.box_level_contingency[key2match_pairs][label][label_]
            for label_ in list(self.box_level_contingency[key2match_pairs].index):
                if label_ is not label and '_sum' not in label_:
                    df['Others'][label] += self.box_level_contingency[key2match_pairs][label_][label]
            
            ## assgin to box level contingency
            self.box_level_contingency_by_label[_key2match_pairs] = df
            
            ## save csv file
            df.to_csv(csv_out_path)

        def _get_box_level_statistics_by_label(self, label, key2match_pairs):
            
            #
            _key2match_pairs = Key2Match_Pairs(label, key2match_pairs.iou_threshold, key2match_pairs.confidence_threshold)

            print('getting box_level_statistics_by_label: {} {}'.format(self.annotator_pair, _key2match_pairs))

            tp = self.box_level_contingency_by_label[_key2match_pairs][label][label]
            fn = self.box_level_contingency_by_label[_key2match_pairs][label]['Others']
            fp = self.box_level_contingency_by_label[_key2match_pairs]['Others'][label]
            #print('{} {} {} {}'.format(label, tp, fp, fn))
            
            # calculate precision = TP / (TP + FP)
            precision = tp / float('{0:.3f}'.format(tp + fp))
            # calculate recall = TP/ (TP + FN) = Sensitivity
            recall = tp / float('{0:.3f}'.format(tp + fn))
            # calculate F1_score = 2 * TP / (2 * TP + FN + FP)
            # the order of FN and FP is interchangeable
            f1_score = 2 * tp / float('{0:.3f}'.format(2 * tp + fn + fp))
            # calculate specificity = TN / (TN + FP) 
            
            # 
            self.box_level_stats_by_label[_key2match_pairs] = Box_Level_Stats(tp, fp, fn, precision, recall, f1_score)

        ## get labelwise contingency and statistics
        for label in list(self.box_level_contingency[key2match_pairs].columns.values):
            if label in labels4stats:
                _get_box_level_contingency_by_label(self, label, key2match_pairs)
                _get_box_level_statistics_by_label(self, label, key2match_pairs)
        
        print('{}'.format(self.box_level_stats_by_label))

    def get_box_level_statistics(self):
        
        self.create_stat_dir()
        csv_out_path = self.stat_dir + 'box_level_stats_' + self.annotator_pair[0] + '_' + self.annotator_pair[1] + '.csv'
        
        df = pd.DataFrame(columns = ['Annotator_pair', 'Label', 'IoU_threshold', 'Confidence_threshold', 'TP', 'FP', 'FN', 'Precision', 'Recall','F1_score'])

        for key, stats in self.box_level_stats_by_label.items():

            if key.label in labels4stats:
                row = {'Annotator_pair': self.annotator_pair, 
                'Label': key.label, 
                'IoU_threshold': key.iou_threshold, 
                'Confidence_threshold': key.confidence_threshold, 
                'TP': stats.tp, 
                'FP': stats.fp, 
                'FN': stats.fn, 
                'Precision': stats.precision, 
                'Recall': stats.recall, 
                'F1_score': stats.f1_score
                }
                df = df.append(row, ignore_index=True)
        
        self.box_level_stats = df

        df.to_csv(csv_out_path)



class AnnDataset:
    """construct input annotated data structure """
    def __init__(self, out_dir):
        self.out_dir = out_dir
        self.draw_out_dir = ''
        self.datasetwise_csv_out_dir = ''
        self.slidewise_csv_out_dir = ''
        self.imgwise_csv_out_dir = ''

        self.ann_images = {} 
        # key = ann_img_name, value = ann_img
        self.image_wise_counts_table_by_annotator = {} 
        # key = annotator, value = counts table
	
        self.slides = {} 
        # key = slide_name, value = list of ann_img_name
        self.slide_wise_counts_table_by_annotator = {}
        # key = annotator, value = counts table
        self.total_counts_table = None
        
        self.match_pairs_stats = {}
        # key = annotator_pair, value = datasetwise_match_pairs_stats
        self.img_level_stats = None

        def get_draw_out_dir(self):
            draw_out_dir = self.out_dir + 'draw/'
            if not os.path.exists(draw_out_dir):
                os.makedirs(draw_out_dir)
            print('successfully created draw_out_dir at: {}'.format(draw_out_dir))
            self.draw_out_dir = draw_out_dir

        def get_csv_out_dir(self):
            datasetwise_csv_out_dir = self.out_dir + 'stats/datasetwise/'
            slidewise_csv_out_dir = self.out_dir + 'stats/slidewise/'
            imgwise_csv_out_dir = self.out_dir + 'stats/imgwise/'

            if not os.path.exists(datasetwise_csv_out_dir):
                os.makedirs(datasetwise_csv_out_dir)
            if not os.path.exists(imgwise_csv_out_dir):
                os.makedirs(imgwise_csv_out_dir)
            if not os.path.exists(slidewise_csv_out_dir):
                os.makedirs(slidewise_csv_out_dir)
            print('successfully created datasetwise_csv_out_dir at: {}'.format(datasetwise_csv_out_dir))
            print('successfully created imgwise_csv_out_dir at: {}'.format(imgwise_csv_out_dir)) 
            print('successfully created slidewise_csv_out_dir at: {}'.format(slidewise_csv_out_dir)) 
            
            self.datasetwise_csv_out_dir = datasetwise_csv_out_dir
            self.imgwise_csv_out_dir = imgwise_csv_out_dir
            self.slidewise_csv_out_dir = slidewise_csv_out_dir

        get_draw_out_dir(self)
        get_csv_out_dir(self)


    def __str__(self):
        return ''
    def __setitem__(self, img_name, ann_img): # ann_img is an instance of AnnImg
        self.ann_images[img_name] = ann_img
    def __getitem__(self, img_name):
        return self.ann_images[img_name]

    def load_images(self, img_fps):
        """
        given the list of image file paths,
        initialize ann_img and ann_dataset,
        and get slide name for all images.
        """
        print('###### loading images...')
        for img_fp in img_fps:
            # process img_fp to get img_name
            img_name = img_fp[img_fp.rfind('/') + 1 : img_fp.find('.jpg')]
            #print(img_name)

            # initializing ann_img
            ann_img = AnnImg(img_name, img_fp)
            ann_img.get_slide_name()
            self.ann_images[img_name] = ann_img

        print('number of raw images loaded: {}'.format(len(self.ann_images)))
    
    def load_annotations(
        self,
        annotator,
        ann_xml_fps,
        mode
        ):
        """
        given the lists of ann_img files and annotation xml files,
        parse each xml for annotation,
        fill in to ann_img.bndboxes,
        and get counts by annotator.
        mode: 'separated', 'ring_trophozoite_combined', 'all_combined'
        """
        def parse_xml_filename(xml, imgs_root_dir):
            tree  = ET.parse(xml)
            root = tree.getroot()
            for filename in root.iterfind('filename'):
                img_filename = filename.text
            img_fp = imgs_root_dir + img_filename[img_filename.find('Layer'):img_filename.find('_Z')] + '/' + img_filename
            return img_fp, img_filename

        def parse_xml_bndbox(xml):
            l_bndbox = []
            tree  = ET.parse(xml)
            root = tree.getroot()
            for obj in root.iterfind('object'):
                label, coords = '', []
                for box in obj.iterfind('bndbox'):
                    for coord in box:
                        coords.append(int(coord.text))
                for name in obj.iterfind('name'):
                    label = name.text
                l_bndbox.append((label, coords))
            return l_bndbox

        print('###### loading annotations from {} ...'.format(annotator))
        for ann_img_name, ann_img in self.ann_images.items():
            #print('image name: {}'.format(ann_img_name))
            
            # retrieve corresponding annotation xml files,
            # parse bndboxes
            ann_by_annotator = False
            for ann_xml in ann_xml_fps:
                if ann_img_name + '.xml' in ann_xml:
                    bndboxes = parse_xml_bndbox(ann_xml)
                    ann_by_annotator = True
            
            ## load annotations to ann_img
            
            # all_combined
            if mode == 'all_combined':
                if ann_by_annotator == True and bndboxes:
                    for bndbox in bndboxes:
                        # 'indeterminate' regarded as infection as well
                        if (bndbox[0] != 'Negative' and
                            bndbox[0] != 'Unknown'):
                            ann_img.add_bndbox(
                                'malaria_infected',
                                bndbox[1],
                                annotator,
                                float(1)
                            )
                        else:
                            ann_img.add_bndbox(
                            bndbox[0], 
                            bndbox[1], 
                            annotator, 
                            float(1)
                        )
                else:
                    ann_img.bndboxes[annotator] = []

            # ring_trophozoite_combined
            elif mode == 'ring_trophozoite_combined':
                if ann_by_annotator == True and bndboxes:
                    for bndbox in bndboxes:
                        # generate ring_trophozoite spectrum
                        if (bndbox[0] == 'P. falciparum_ring' or
                            bndbox[0] == 'P. falciparum_trophozoite'):
                            ann_img.add_bndbox(
                                'P. falciparum_ring_trophozoite',
                                bndbox[1],
                                annotator,
                                float(1)
                            )
                        else:
                            ann_img.add_bndbox(
                            bndbox[0], 
                            bndbox[1], 
                            annotator, 
                            float(1)
                        )
                else:
                    ann_img.bndboxes[annotator] = []
            
            # ring, trophozoite separated
            elif mode == 'separated':
                if ann_by_annotator == True and bndboxes:
                    for bndbox in bndboxes:
                        ann_img.add_bndbox(
                            bndbox[0], 
                            bndbox[1], 
                            annotator, 
                            float(1)
                        )
                else:
                    ann_img.bndboxes[annotator] = []
            
            else:
                print('please specify a mode to load annotation')
            
            # get counts
            ann_img.get_counts_by_annotator(annotator)

    def group_by_slide(self):
        """
        group all ann_imgs based on their slide name
        """
        for ann_img_name, ann_img in self.ann_images.items():
            if ann_img.slide_name not in self.slides.keys():
                self.slides[ann_img.slide_name] = [ann_img_name]
            else:
                self.slides[ann_img.slide_name].append(ann_img_name)
    
    def visualize(self, annotator_list):
        """
        draw bndboxes from annotator in annotator list on the image,
        and save the results.
        """
        # make proper dir to store images
        img_out_dir = self.draw_out_dir
        for i, annotator in enumerate(annotator_list):
            if i < len(annotator_list)-1:
                img_out_dir = img_out_dir + annotator + '_'
            else:
                img_out_dir += annotator
        img_out_dir += '/'
        if not os.path.exists(img_out_dir):
            os.makedirs(img_out_dir)
        print('successfully created draw_out_dir for annotator list {} at: {}'.format(annotator_list, img_out_dir))
        
        # draw bndboxes for all ann_images
        for ann_img_name, ann_img in self.ann_images.items():
            ann_img.draw_bndboxes(annotator_list, img_out_dir)
    
    def get_total_counts(self, annotator_list):
        
        def get_image_wise_counts_table_by_annotator(self, annotator):
            print('###### getting image wise counts by {}'.format(annotator))
            csv_out_path = self.out_dir + 'image_wise_counts_by_' + annotator + '.csv'
            
            df = pd.DataFrame(
                index=self.ann_images.keys(), 
                columns=predefined_labels
            )
            df = df.fillna(0)

            for ann_img_name, ann_img in self.ann_images.items():
                # get differential counts
                if annotator not in ann_img.differential_counts_by_annotator.keys():
                    if annotator in ann_img.bndboxes.keys():
                        ann_img.get_counts_by_annotator(annotator)
                    else:
                        print('-- WARNING: annotator {} not present!'.format(
                            annotator
                            )
                        )
                # get image wise counts
                if annotator in ann_img.differential_counts_by_annotator.keys():
                    for label, count in ann_img.differential_counts_by_annotator[annotator].items():
                        if label == 'Wrong_labels':
                            print('-- wrong_labels: {}'.format(ann_img.img_name))
                            print('--- {}'.format(ann_img.bndboxes))
                        else:
                            df[label][ann_img_name] += count

            df.to_csv(csv_out_path)
            self.image_wise_counts_table_by_annotator[annotator] = df

        def get_slide_wise_counts_table_by_annotator(self, annotator):
            print('###### getting slide wise counts by {}'.format(annotator))
            csv_out_path = self.out_dir + 'slide_wise_counts_by_' + annotator + '.csv'
            
            _predefined_labels = predefined_labels + ['Wrong_labels']
            df = pd.DataFrame(
                index=self.slides.keys(), 
                columns=_predefined_labels
            )
            df = df.fillna(0)

            for slide_name, ann_img_names in self.slides.items():
                for ann_img_name in ann_img_names:
                    if annotator in self.ann_images[ann_img_name].differential_counts_by_annotator.keys():
                        for label, count in self.ann_images[ann_img_name].differential_counts_by_annotator[annotator].items():
                            if label == 'Wrong_labels':
                                df['Wrong_labels'][self.ann_images[ann_img_name].slide_name] += len(count)
                            else:
                                df[label][self.ann_images[ann_img_name].slide_name] += count
                    else:
                        print('-- annotator {} not included in differential counts'.format(annotator))
            #print('slide wise counts by annotator: befun\n{}'.format(df))
            
            df.to_csv(csv_out_path)
            self.slide_wise_counts_table_by_annotator[annotator] = df

        csv_out_path = self.out_dir + 'total_counts.csv'
        _predefined_labels = predefined_labels + ['Wrong_labels']
        df = pd.DataFrame(
            columns=['Annotator'] + _predefined_labels
        )
        for annotator in annotator_list:
            get_image_wise_counts_table_by_annotator(self, annotator)
            get_slide_wise_counts_table_by_annotator(self, annotator)
            _df = self.slide_wise_counts_table_by_annotator[annotator]
            series = _df.sum(axis=0)
            series['Annotator'] = annotator
            print('{}'.format(series))
            df = df.append(series, ignore_index=True)
        
        df.to_csv(csv_out_path)
        self.total_counts_table = df
        print('successfully created total counts table at: {}'.format(
            csv_out_path
            )
        )

    def gen_retina_ann_csv(self, annotator, label_for_indeterminate):
        csv_out_path = self.out_dir + 'retina_ann_' + annotator + '.csv'
        csv_list = []
        count = 0
        for ann_img_name, ann_img in self.ann_images.items():
            ann_img.gen_retina_ann_csv_list(annotator, label_for_indeterminate)
            if annotator in ann_img.retina_ann_csv_lists.keys():
                for ann_csv_list in ann_img.retina_ann_csv_lists[annotator]:
                    csv_list.append(ann_csv_list)
                count += 1
            else:
                print('no annotator in csv lists: {}'.format(ann_img_name))
        df = pd.DataFrame(csv_list)
        df.to_csv(csv_out_path, index=False)
        print('total counts for retina ann csv list: {}'.format(count))


    def inference(self):
        '''
        inference raw images in the dataset
        '''
        def get_session():
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            return tf.Session(config=config)

        
        MODEL_PATH = '/volume/workspace/fredpckuo/malaria/result/20181002/snapshots/resnet50_csv_50.h5'
        
        # set the modified tf session as backend in keras
        keras.backend.tensorflow_backend.set_session(get_session())
        # load retinanet model
        model = keras.models.load_model(
            MODEL_PATH,
            custom_objects=custom_objects
        )
        print('successfully reading model weights from: {}'.format(
            MODEL_PATH))
        
        for ann_img_name, ann_img in self.ann_images.items():
            ann_img.inference(model)


    def get_expert_concensus(
        self, expert_names, iou_threshold, confidence_threshold
        ):
        
        csv_out_path = self.datasetwise_csv_out_dir + 'expert_concensus.csv'
        df = pd.DataFrame(
            columns=[
                'Img_name', 'Box', 'befun','cd6397',
                'tmumt4009', 'Source', 'Concensus'
            ]
        )
        for ann_img_name, ann_img in self.ann_images.items():
            ann_img.gen_expert_concensus(
                expert_names, iou_threshold, confidence_threshold
            )
            df = df.append(ann_img.expert_concensus, ignore_index=True)
        df.to_csv(csv_out_path)

    def get_box_level_contingency(self, annotator1, annotator2, iou_threshold, confidence_threshold):
        """
        given annotator pair and key2match_pairs,
        get match pairs for each image, save imgwise box_level_contingency,
        get datasetwise box level contingency,
        get datasetwise box_level statistics by label.
        """

        def initialize_datasetwise_match_pairs_stats(self, annotator1, annotator2):
        
            ## initialize
            # datasetwise match pairs stats
            datasetwise_match_pairs_stats = Datasetwise_Match_Pairs_Stats(annotator1, annotator2, self.datasetwise_csv_out_dir)
            datasetwise_match_pairs_stats.create_contingency_dir()
            
            self.match_pairs_stats[(annotator1, annotator2)] = datasetwise_match_pairs_stats

        ## initialize
        # imgwise csv out dir
        imgwise_csv_out_dir = self.imgwise_csv_out_dir + annotator1 + '_' + annotator2 + '/'
        if not os.path.exists(imgwise_csv_out_dir):
            os.makedirs(imgwise_csv_out_dir)
        # key2match_pairs
        key2match_pairs = Key2Match_Pairs('All', iou_threshold, confidence_threshold)
        # datasetwise_match_pairs_stats
        if (annotator1, annotator2) not in self.match_pairs_stats.keys():
            initialize_datasetwise_match_pairs_stats(self, annotator1, annotator2)
        # output csv file path for datasetwise contingency table
        datasetwise_contingency_path = self.match_pairs_stats[(annotator1, annotator2)].contingency_dir + 'iou_'+ str(int(iou_threshold*100)) + '_conf_' + str(int(confidence_threshold*100)) + '.csv'
        # datasetwise inter rater contingency table
        df = self.match_pairs_stats[(annotator1, annotator2)].creat_contingency_table()

        ## get match pairs for each image
        for ann_img_name, ann_img in self.ann_images.items():
            ann_img.get_match_pairs(annotator1, annotator2, iou_threshold, confidence_threshold, imgwise_csv_out_dir)
            ann_img.get_img_level_matching(annotator1, annotator2, iou_threshold, confidence_threshold)
            df = df.add(ann_img.match_pairs_stats[(annotator1, annotator2)].box_level_contingency[key2match_pairs], fill_value=0)
        
        ## summarize columns and rows for datasetwise contingency table
        # sum over columns to get annotator1 sum
        for label_name in predefined_labels:
            for _label_name in predefined_labels:
                df[label_name][annotator1+'_sum'] += df[label_name][_label_name]
        # sum over rows to get annotator2 sum
        for label_name in predefined_labels:
            for _label_name in predefined_labels:
                df[annotator2+'_sum'][label_name] += df[_label_name][label_name]

        ## assign to match pairs stats
        self.match_pairs_stats[(annotator1, annotator2)].box_level_contingency[key2match_pairs] = df
        
        ## save datasetwise inter rater contingency table
        df.to_csv(datasetwise_contingency_path)

        ## get datasetwise box level contingency by label
        #print('-- keys to cal box level statistics by label: {}'.format(key2match_pairs))
        print('got box_level_contingency for {} {} under threshold {}'.format(annotator1, annotator2, key2match_pairs))

    def get_box_level_statistics_by_label(self):

        for annotator_pair, datasetwise_stats in self.match_pairs_stats.items():
            for key2match_pairs in datasetwise_stats.box_level_contingency.keys():
                datasetwise_stats.get_box_level_statistics_by_label(key2match_pairs)

    def get_box_level_statistics(self):
        
        for annotator_pair, datasetwise_stats in self.match_pairs_stats.items():
            datasetwise_stats.get_box_level_statistics()

    def get_img_level_contingency_by_label(self):
        
        csv_out_path = self.datasetwise_csv_out_dir + 'imgwise_img_level_truth_value.csv'
        df = pd.DataFrame(columns = ['Img_name', 'Annotator_pair', 'Label', 'IoU_threshold', 'Confidence_threshold', 'Truth_value'])

        # get imagewise img_level truth value
        for ann_img_name, ann_img in self.ann_images.items():  
            for annotator_pair, imgwise_stats in ann_img.match_pairs_stats.items():
                for key2match_pairs, truth_value in imgwise_stats.img_level_truth_value.items():
                    # initialize the dictionary for datasetwise img_level_contingency_by_label
                    #print('---- {}'.format(key2match_pairs))
                    if key2match_pairs not in self.match_pairs_stats[annotator_pair].img_level_contingency_by_label.keys():
                        self.match_pairs_stats[annotator_pair].img_level_contingency_by_label[key2match_pairs] = {
                            'TP': 0,
                            'FP': 0,
                            'FN': 0,
                            'TN': 0
                        }
                    # fill in imgwise img_level truth value in datasetwise img_level_contingency_by_label
                    self.match_pairs_stats[annotator_pair].img_level_contingency_by_label[key2match_pairs][truth_value] += 1
        
                    row = {'Img_name': ann_img_name,
                            'Annotator_pair': annotator_pair, 
                            'Label': key2match_pairs.label, 
                            'IoU_threshold': key2match_pairs.iou_threshold, 
                            'Confidence_threshold': key2match_pairs.confidence_threshold, 
                            'Truth_value':truth_value
                            }
                    df = df.append(row, ignore_index=True)
        df.to_csv(csv_out_path)

    def get_img_level_statistics_by_label(self):

        def _get_division(top, bottom):
            return float(top)/float('{0:.3f}'.format(bottom)) if bottom else 0

        csv_out_path = self.datasetwise_csv_out_dir + 'img_level_statistics.csv'
        df = pd.DataFrame(columns = ['Annotator_pair', 'Label', 'IoU_threshold', 'Confidence_threshold', 'TP', 'FP', 'FN', 'Precision', 'Recall','F1_score', 'Specificity'])
        
        for annotator_pair, datasetwise_stats in self.match_pairs_stats.items():
            #print(annotator_pair)
            for key2match_pairs, img_level_contingency_by_label in datasetwise_stats.img_level_contingency_by_label.items():
                
                tp = img_level_contingency_by_label['TP']
                fp = img_level_contingency_by_label['FP']
                fn = img_level_contingency_by_label['FN']
                tn = img_level_contingency_by_label['TN']
                
                precision = _get_division(tp, tp+fp)
                recall = _get_division(tp, tp+fn)
                f1_score = _get_division(2*tp, 2*tp+fn+fp)
                spe = _get_division(tn, tn+fp)
                print('-- {}'.format(key2match_pairs))
                print('---- tp {}, fp {}, fn {}, tn {}'.format(tp, fp, fn, tn))
                print('---- precision {}, recall {}, f1 score {}, spe {}'.format(precision, recall, f1_score, spe))

                datasetwise_stats.img_level_stats_by_label[key2match_pairs] = Img_Level_Stats(tp, fp, fn, tn, precision, recall, f1_score, spe)
                
                row = {'Annotator_pair': annotator_pair, 
                        'Label': key2match_pairs.label, 
                        'IoU_threshold': key2match_pairs.iou_threshold, 
                        'Confidence_threshold': key2match_pairs.confidence_threshold, 
                        'TP': tp, 
                        'FP': fp, 
                        'FN': fn, 
                        'TN': tn,
                        'Precision': precision, 
                        'Recall': recall, 
                        'F1_score': f1_score,
                        'Specificity': spe
                        }
                df = df.append(row, ignore_index=True)
                
        self.img_level_stats = df
        df.to_csv(csv_out_path)

    #def get_img_level_inter_rater_statistics(self, ):

    #def get_pooling_statistics(self, ):


def convert_to_rt_combined(input_dataset, output_dataset):
    for img_name, ann_img in input_dataset.ann_images.items():
        output_dataset.ann_images[img_name] = AnnImg(
            img_name, ann_img.img_path
        )
        output_dataset.ann_images[img_name].slide_name = ann_img.slide_name
        
        for annotator, _bndboxes in ann_img.bndboxes.items():
            if _bndboxes:
                for bndbox in _bndboxes:
                    if (bndbox.label == 'P. falciparum_ring' or
                        bndbox.label == 'P. falciparum_trophozoite'):
                        output_dataset.ann_images[img_name].add_bndbox(
                            'P. falciparum_ring_trophozoite',
                            bndbox.coords,
                            annotator,
                            bndbox.confidence
                        )
                    else:
                        output_dataset.ann_images[img_name].add_bndbox(
                            bndbox.label, 
                            bndbox.coords, 
                            annotator, 
                            bndbox.confidence
                        )
            else:
                output_dataset.ann_images[img_name].bndboxes[annotator] = []

def convert_to_all_combined(input_dataset, output_dataset):
    for img_name, ann_img in input_dataset.ann_images.items():
        output_dataset.ann_images[img_name] = AnnImg(
            img_name, ann_img.img_path
        )
        output_dataset.ann_images[img_name].slide_name = ann_img.slide_name
        
        for annotator, _bndboxes in ann_img.bndboxes.items():
            if _bndboxes:
                for bndbox in _bndboxes:
                    # 'indeterminate' regarded as infection as well
                    if (bndbox.label != 'Negative' and
                        bndbox.label != 'Unknown'):
                        output_dataset.ann_images[img_name].add_bndbox(
                            'malaria_infected',
                            bndbox.coords,
                            annotator,
                            bndbox.confidence
                        )
                    else:
                        output_dataset.ann_images[img_name].add_bndbox(
                            bndbox.label, 
                            bndbox.coords, 
                            annotator, 
                            bndbox.confidence
                        )
            else:
                output_dataset.ann_images[img_name].bndboxes[annotator] = []


