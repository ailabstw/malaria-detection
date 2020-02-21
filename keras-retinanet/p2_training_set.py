# 
import data

import glob
import dill



## input 
ROOT_DIR = '/volume/workspace/fredpckuo/'

P1_IMG_DIR_ROOT = ROOT_DIR + 'malaria/data/cdc/annotation/'
P1_IMG_DIR_LIST = [
    '20180703/', '20180816/', '20180820/', '20180827/', 
    '20180903/', '20180905/', '20180912/', '20180927/', 
    '20181001/'
]

TURING_IMG_DIR_ROOT = ROOT_DIR + 'malaria/data/cdc/annotation/turing_pic/'
TURING_ANN_DIR_ROOT = {
    'befun': ROOT_DIR + 'malaria/data/cdc/annotation/turing_expert/',
    'cd6397': ROOT_DIR + 'malaria/data/cdc/annotation/DO_NOT_MODIFIED_turing_cd6397/',
    'tmumt4009': ROOT_DIR + 'malaria/data/cdc/annotation/DO_NOT_MODIFIED_turing_tmumt4009/'
}
TURING_CONCENSUS = ROOT_DIR + 'malaria/result/turing/expert/stats/datasetwise/expert_concensus.csv'

P2_IMG_DIR_ROOT = ROOT_DIR + 'malaria/data/cdc/phase_2/'
P2_ANN_DIR_ROOT = ROOT_DIR + 'malaria/data/cdc/phase_2/DO_NOT_MODIFIED/'
P2_IMG_DIR_LIST = [
    '20181005/', '20190315/', '20190329/'
]
P2_VAL_SLIDES = [
    'T2018101602L_N_thin', 'T2019021401G_N_thin', 'T2019060302_N_thin',
    '2018080416G_Pf_thin', '940821PFRTG_Pf_thin', 'CN10305U20180803_Pf_thin',
    'CN10313RT20180904_Pf_thin', 'CN09124RASG20190218_Pv_thin', 
    'CN09515TSG20190123_Pv_thin', 'CN09524RASG20190319_Pv_thin',
    'CN09716TSG20181212_Pv_thin', 'CN09920ASG20190211_Pv_thin',
    'CN10101AG20190411_Pv_thin', 'CN10211RASG20181212_Pv_thin',
    'CN10502RASG20190128_Pv_thin', 'CN10511ASG20181212_Pv_thin',
    'CN10801RTSG20190123L_Pv_thin', 'T2018100303_Pv_thin',
    'T2019012301_Pv_thin', 'T2019031804_Pv_thin',
    'T2019032002_Pv_thin', 'T2019070402_Pv_thin'
]


NON_BEFUN_EXPERT_DIR_ROOTs = {
    'cd6397': ROOT_DIR + 'malaria/data/cdc/annotation/reliability/DO_NOT_MODIFIED_interrater_cd6397/',
    'tmumt4009': ROOT_DIR + 'malaria/data/cdc/annotation/reliability/DO_NOT_MODIFIED_interrater_tmumt4009/'
}


## parameters
IOU_THRESHs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
CONFI_THRESH = 0.5


## output
# p2 train dataset & p2 val dataset
P2_TRAIN_DIR = ROOT_DIR + 'malaria/result/phase_2_train/'
P2_TRAIN_SET = ROOT_DIR + 'malaria/result/phase_2_train/p2_train_set'
P2_VAL_DIR = ROOT_DIR + 'malaria/result/phase_2_val/'
P2_VAL_SET = ROOT_DIR + 'malaria/result/phase_2_val/p2_val_set'

#CSV_OUT_DIR = '/home/fredpckuo/malaria/result/20181002/'
CSV_PATH = ''



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
    
    ## get predifined labels
    predefined_labels = data.get_predefined_labels()
    
    p2_train_img_fps = []
    p2_train_chosen_ann_fps = []
    p2_train_no_match = []
    
    p2_val_img_fps = []
    p2_val_chosen_ann_fps = []
    p2_val_no_match = []
    

    # get p2 training images and annotations
    # from p1 training set
    for img_dir in P1_IMG_DIR_LIST:
        img_fps = glob.glob(P1_IMG_DIR_ROOT + img_dir + '*/*/*.jpg')
        for img_fp in img_fps:
            p2_train_img_fps.append(img_fp)
        ann_fps = glob.glob(P1_IMG_DIR_ROOT + img_dir + '*/*/*/*/*.xml')
        chosen_ann_fps, no_match_img_fps = choose_ann_fps(
            img_fps, ann_fps
        )
        print('for p1 training set, {}, {} img files have no match ann files'.format(
            img_dir, len(no_match_img_fps)
        ))
        for chosen_ann_fp in chosen_ann_fps:
            p2_train_chosen_ann_fps.append(chosen_ann_fp)
        for no_match_img_fp in no_match_img_fps:
            p2_train_no_match.append(no_match_img_fp)
    
    # get p2 training images and annotations
    # from p2 data sets
    for img_dir in P2_IMG_DIR_LIST:
        img_fps = glob.glob(P2_IMG_DIR_ROOT + img_dir + '*/*.jpg')
        for img_fp in img_fps:
            if img_fp[img_fp.find(img_dir)+len(img_dir):img_fp.find('thin')+4] not in P2_VAL_SLIDES:
                p2_train_img_fps.append(img_fp)
            else:
                p2_val_img_fps.append(img_fp)

        ann_fps = []
        for annotator in ['befun/', 'expert/']:
            ann_fps += glob.glob(
                P2_ANN_DIR_ROOT + annotator + img_dir + '*/*.xml'
            )
        chosen_ann_fps, no_match_img_fps = choose_ann_fps(
            img_fps, ann_fps
        )
        print('for p2 set, {}, {} img files have no match ann files'.format(
            img_dir, len(no_match_img_fps)
            ))
        
        for chosen_ann_fp in chosen_ann_fps:
            if chosen_ann_fp[chosen_ann_fp.find(img_dir)+len(img_dir):chosen_ann_fp.find('thin')+4] not in P2_VAL_SLIDES:
                p2_train_chosen_ann_fps.append(chosen_ann_fp)
            else:
                p2_val_chosen_ann_fps.append(chosen_ann_fp)

        for no_match_img_fp in no_match_img_fps:
            if no_match_img_fp[no_match_img_fp.find(img_dir)+len(img_dir):no_match_img_fp.find('thin')+4] not in P2_VAL_SLIDES:
                p2_train_no_match.append(no_match_img_fp)
            else:
                p2_val_no_match.append(no_match_img_fp)

    print('size of p2 train images: {}'.format(len(p2_train_img_fps)))
    print('size of p2 train anns: {}'.format(len(p2_train_chosen_ann_fps)))
    print('total no annotation match for p2 train images: {}'.format(p2_train_no_match))
    print('size of p2 val images: {}'.format(len(p2_val_img_fps)))
    print('size of p2 val anns: {}'.format(len(p2_val_chosen_ann_fps)))
    print('total no annotation match for p2 val images: {}'.format(p2_val_no_match))


    # load to separated mode
    # p2 train dataset
    p2_train_dataset = data.AnnDataset(P2_TRAIN_DIR)
    p2_train_dataset.load_images(p2_train_img_fps)
    p2_train_dataset.load_annotations(
        'befun', p2_train_chosen_ann_fps, mode='separated'
    )
    p2_train_dataset.group_by_slide()
    p2_train_dataset.get_total_counts(['befun'])
    print('size of p2 train dataset: {}'.format(len(p2_train_dataset.ann_images)))

    # load to separated mode
    # p2 val dataset
    p2_val_dataset = data.AnnDataset(P2_VAL_DIR)
    p2_val_dataset.load_images(p2_val_img_fps)
    p2_val_dataset.load_annotations(
        'befun', p2_val_chosen_ann_fps, mode='separated'
    )
    p2_val_dataset.group_by_slide()
    p2_val_dataset.get_total_counts(['befun'])
    print('size of p2 val dataset: {}'.format(len(p2_val_dataset.ann_images)))


    # get retina ann csv
    p2_train_dataset.gen_retina_ann_csv('befun', 'P. falciparum_ring')

    with open(P2_TRAIN_SET, 'wb') as fo:
        dill.dump(p2_train_dataset, fo)
    with open(P2_VAL_SET, 'wb') as fo:
        dill.dump(p2_val_dataset, fo)



    '''
    ## construct dataset for inter rater agreement
    
    #with open(TRAINING_DATASET, 'rb') as fi:
    #    training_dataset = dill.load(fi)

    irr_dataset = data.AnnDataset(IRR_RT_COMBINED_DATASET_DIR)
    irr_dataset.get_csv_out_dir()

    # get the list of images for training_set inter-expert agreement
    ann_fps = glob.glob(NON_BEFUN_EXPERT_DIR_ROOTs['cd6397'] + '*/*/*/*.xml')
    list4inter_rater_agreement = []
    for ann_fp in ann_fps:
        img_name = ann_fp[ann_fp.rfind('/') + 1 : ann_fp.find('.xml')]
        list4inter_rater_agreement.append(ann_fp)
        # copy from turing_dataset
        irr_dataset.ann_images[img_name] = training_dataset.ann_images[img_name]
    
    with open(LIST4IRR, 'wb') as fo:
        dill.dump(list4inter_rater_agreement, fo)


     # load annotations from other experts
    for expert, dir_root in NON_BEFUN_EXPERT_DIR_ROOTs.items():
        ann_fps = glob.glob(dir_root + '*/*/*/*.xml')
        irr_dataset.load_annotations(expert, ann_fps, ring_trophozoite_combined=True)
    
    print('number of images for inter rater agreement: {}'.format(
        len(irr_dataset.ann_images.keys())
        )
    )
    
    experts = ['befun', 'cd6397', 'tmumt4009']
    irr_dataset.get_total_counts(experts)

    with open(IRR_RT_COMBINED_DATASET, 'wb') as fo:
        dill.dump(irr_dataset, fo)
    

    

    with open(IRR_RT_COMBINED_DATASET, 'rb') as fi:
        irr_dataset = dill.load(fi)

    with open(LIST4IRR, 'rb') as fi:
        list4inter_rater_agreement = dill.load(fi)

    experts = ['befun', 'cd6397', 'tmumt4009']

    ## get inter rater reliability
    # generate expert_pairs
    expert_pairs = []
    for expert1 in experts:
        experts.remove(expert1)
        for expert2 in experts:
            expert_pairs.append([expert1, expert2])
    print('check expert_pairs: {}'.format(expert_pairs))
    # get inter rater
    for expert_pair in expert_pairs:
        for iou_thresh in IOU_THRESHs:
            irr_dataset.get_box_level_contingency(
                expert_pair[0], expert_pair[1], iou_thresh, CONFI_THRESH
            )
    irr_dataset.get_box_level_statistics_by_label()
    irr_dataset.get_box_level_statistics()
    irr_dataset.get_img_level_contingency_by_label()
    irr_dataset.get_img_level_statistics_by_label()

    with open(IRR_RT_COMBINED_DATASET, 'wb') as fo:
        dill.dump(irr_dataset, fo)

    with open(IRR_DATASET, 'rb') as fi:
        irr_dataset = dill.load(fi)
    irr_dataset.get_draw_out_dir()
    irr_dataset.visualize(['befun', 'cd6397', 'tmumt4009'])
    '''
