# script to conver h5 data to a format that acceptable to deepreg

import os
import random
import shutil

import h5py
import numpy as np


DATA_PATH = os.path.join(os.getenv("HOME"),'Scratch/data/mrusv2/normalised')
SAVE_PATH = os.path.join(DATA_PATH,"deepreg")
NUM_FOLDS = 12

h5_image_us = h5py.File(os.path.join(DATA_PATH,'us_images_resampled800.h5'),'r')
h5_image_mr = h5py.File(os.path.join(DATA_PATH,'mr_images_resampled800.h5'),'r')
h5_label_us = h5py.File(os.path.join(DATA_PATH,'us_labels_resampled800_post3.h5'),'r')
h5_label_mr = h5py.File(os.path.join(DATA_PATH,'mr_labels_resampled800_post3.h5'),'r')

num_pat = len(h5_image_us)
print(num_pat)

num_labels = h5_label_us['/num_labels'][0]
if any(num_labels != h5_label_mr['/num_labels'][0]) | any(num_labels != h5_label_us['/num_important'][0]):
    raise("numbers of labels are not compatible.")

#commit to write
if os.path.exists(SAVE_PATH):
    shutil.rmtree(SAVE_PATH)

filenames = {
    "img_mov": "moving_images.h5", 
    "lab_mov": "moving_labels.h5", 
    "img_fix": "fixed_images.h5", 
    "lab_fix": "fixed_labels.h5" 
    }

random.seed(1)
pat_indices = [i for i in range(num_pat)]
random.shuffle(pat_indices)
for ii, idx in enumerate(pat_indices):
    # partition
    idx_fold = ii % NUM_FOLDS  # indexed by remainder (instead of mode)
    fold_path = os.path.join(SAVE_PATH, "fold%02d" % idx_fold)
    h5_paths = {k:os.path.join(fold_path,fn) for (k,fn) in filenames.items()}
    if not os.path.exists(fold_path):
        os.makedirs(fold_path)
        h5_id = {k:h5py.File(path,'x') for (k,path) in h5_paths.items()}
    h5_id = {k:h5py.File(path,'a') for (k,path) in h5_paths.items()}
    # load images and labels
    image_name = '/case%06d' % idx
    label_names = ['/case%06d_bin%03d' % (idx, j) for j in range(num_labels[idx])]
    data = {
        "img_mov": h5_image_mr[image_name], 
        "img_fix": h5_image_us[image_name], 
        "lab_mov": np.stack([h5_label_mr[n] for n in label_names],axis=3), 
        "lab_fix": np.stack([h5_label_us[n] for n in label_names],axis=3) 
        }
    # check all shapes
    if ii==0:
        shape_us, shape_mr = data["img_fix"].shape, data["img_mov"].shape
        print(shape_us, shape_mr)
    if (data["img_fix"].shape!=shape_us) | (data["lab_fix"].shape[0:3]!=shape_us):
        raise('us shapes not consistent')
    if (data["img_mov"].shape!=shape_mr) | (data["lab_mov"].shape[0:3]!=shape_mr):
        raise('mr shapes not consistent')
    # write
    for (tn,fid) in h5_id.items():
        fid.create_dataset('/obs%03d' % idx, data[tn].shape, dtype=data[tn].dtype, data=data[tn])
        fid.flush()
        fid.close()
    # print(ii,fold_path,idx)

print('Done.')
