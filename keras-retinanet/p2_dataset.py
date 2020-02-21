import data

import glob
import dill
import os.path



## INPUT ##
# input folder paths
IMG_DIR_ROOT = '/volume/workspace/fredpckuo/malaria/data/cdc/phase_2/'
ANN_DIR_ROOT = '/volume/workspace/fredpckuo/malaria/data/cdc/phase_2/DO_NOT_MODIFIED/'
IMG_DIR_LIST = [
    '20181005/', '20190315/', '20190329/'
    ]


## PARAMETERs ##
IOU_THRESHs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
CONFI_THRESH = 0.5


## OUTPUT ##
PHASE2_RESULT_DIR = '/volume/workspace/fredpckuo/malaria/result/phase_2/'
PHASE2_DATASET_FP = PHASE2_RESULT_DIR + 'phase_2_dataset'



def choose_ann_fps(img_fps, ann_fps):
    '''
    getting correct ann_fp for each img_fp,
    return a list of ann_fps that match,
    and a list of no match img_fps
    '''

    ## separate befun ann files from expert ann files
    ann_fps_befun = [ann_fp for ann_fp in ann_fps 
        if 'befun' in ann_fp
    ]
    ann_fps_expert = [ann_fp for ann_fp in ann_fps
        if 'expert' in ann_fp
    ]
    
    chosen_ann_fps = []
    no_match_img_fps = []
    print('### getting ann files for imgs...')
    for img_fp in img_fps:
        # process img file path to get img name
        img_name = img_fp[img_fp.rfind('/') + 1 : img_fp.find('.jpg')]
        #print(' - {}'.format(img_name))
        
        annotated_by_expert = False
        annotated_by_befun = False
        for ann_fp in ann_fps_expert:
            if img_name+'.xml' in ann_fp:
                annotated_by_expert = True
                chosen_ann_fps.append(ann_fp)
                break
        if not annotated_by_expert:
            for ann_fp in ann_fps_befun:
                if img_name+'.xml' in ann_fp:
                    annotated_by_befun = True
                    chosen_ann_fps.append(ann_fp)
                    break
        if (not annotated_by_befun and
            not annotated_by_expert):
            no_match_img_fps.append(img_fp)
        
    return chosen_ann_fps, no_match_img_fps



if __name__ == "__main__":
    
    if os.path.isfile(PHASE2_DATASET_FP):
        
        print('loading existing p2_dataset...')
        with open(PHASE2_DATASET_FP, 'rb') as fi:
            p2_dataset = dill.load(fi)
        
        p2_dataset.inference()
        p2_dataset.visualize(['model', 'befun'])

        with open(PHASE2_DATASET_FP, 'wb') as fo:
            dill.dump(p2_dataset, fo)
    
    
    else:
        ## get predifined labels
        predefined_labels = data.get_predefined_labels()
        
        ## p2_dataset development ##
        total_img_fps = []
        total_no_match = []
        total_chosen_ann_fps = []

        # load total p2 images and annotations
        for img_dir in IMG_DIR_LIST:

            img_fps = glob.glob(IMG_DIR_ROOT + img_dir + '*/*.jpg')
            for img_fp in img_fps:
                total_img_fps.append(img_fp)

            ann_fps = []
            for annotator in ['befun/', 'expert/']:
                ann_fps += glob.glob(ANN_DIR_ROOT + annotator
                                    + img_dir + '*/*.xml'
                                    )
            
            chosen_ann_fps, no_match_img_fps = choose_ann_fps(
                img_fps, ann_fps
                )
            print('for {}, {} img files have no match ann files'.format(
                img_dir, len(no_match_img_fps)
                ))
            
            for chosen_ann_fp in chosen_ann_fps:
                total_chosen_ann_fps.append(chosen_ann_fp)
            total_no_match.append(no_match_img_fps)

        print('size of total input images: {}'.format(len(total_img_fps)))
        print('total no match annotations: {}'.format(total_no_match))


        # load to separated mode
        p2_dataset = data.AnnDataset(PHASE2_RESULT_DIR)
        p2_dataset.load_images(total_img_fps)
        p2_dataset.load_annotations(
            'befun', total_chosen_ann_fps, mode='separated'
            )
        p2_dataset.group_by_slide()
        p2_dataset.get_total_counts(['befun'])
        print('size of ann dataset: {}'.format(
            len(p2_dataset.ann_images)
            ))

        with open(PHASE2_DATASET_FP, 'wb') as fo:
            dill.dump(p2_dataset, fo)

        p2_dataset.inference()
        p2_dataset.visualize(['model', 'befun'])

        with open(PHASE2_DATASET_FP, 'wb') as fo:
            dill.dump(p2_dataset, fo)
        
