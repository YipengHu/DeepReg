# script to conver h5 data to a format that acceptable to deepreg

import os

import h5py
import numpy as np

DATA_FOLDER = os.path.join(os.getenv("HOME"),'Scratch/data/mrusv2/normalised')
SAVE_FOLDER = os.path.join(DATA_FOLDER,"deepreg")

h5_image_us = h5py.File(os.path.join(DATA_FOLDER,'us_images_resampled800.h5'),'r')
h5_image_mr = h5py.File(os.path.join(DATA_FOLDER,'mr_images_resampled800.h5'),'r')
h5_label_us = h5py.File(os.path.join(DATA_FOLDER,'us_labels_resampled800_post3.h5'),'r')
h5_label_mr = h5py.File(os.path.join(DATA_FOLDER,'mr_labels_resampled800_post3.h5'),'r')


num_pat = len(h5_image_us)
print(num_pat)

num_labels = h5_label_us['/num_labels'][0]
if any(num_labels != h5_label_mr['/num_labels'][0]) | any(num_labels != h5_label_us['/num_important'][0]) | any(num_labels != h5_label_mr['/num_important'][0]):
    raise("numbers of labels are not compatible.")

for idx in range(num_pat):

    # load images and labels
    image_name = '/case%06d' % idx
    image_us = h5_image_us[image_name]
    image_mr = h5_image_mr[image_name]
    label_names = ['/case%06d_bin%03d' % (idx, j) for j in range(num_labels[idx])]
    label_us = np.stack([h5_label_us[n] for n in label_names],axis=3)
    label_mr = np.stack([h5_label_mr[n] for n in label_names],axis=3)

    # check all shapes
    if idx==0:
        shape_us, shape_mr = image_us.shape, image_mr.shape
        print(shape_us, shape_mr)
    if (image_us.shape!=shape_us) | (label_us.shape[0:3]!=shape_us):
        raise('us shapes not consistent')
    if (image_mr.shape!=shape_mr) | (label_mr.shape[0:3]!=shape_mr):
        raise('mr shapes not consistent')

    # write into 12 folds