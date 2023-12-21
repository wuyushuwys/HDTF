import math
import cv2
import os
import h5py

from typing import Dict
from pathlib import Path
from glob import glob
from PIL import Image

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image


class ImageFolder(Dataset):

    def __init__(self, folder, ext='jpg', output_mode='tensor') -> None:
        super().__init__()
        self.img_lists = sorted(glob(f"{folder}/*.{ext}"))
        self.output_mode = output_mode

    def __len__(self):
        return len(self.img_lists)

    def __getitem__(self, index: int) -> Dict:
        fname = self.img_lists[index]
        if self.output_mode == 'tensor':
            img = read_image(fname)
        elif self.output_mode == 'cv2':
            img = cv2.imread(fname)
        elif self.output_mode == 'PIL':
            img = Image.open(fname)
        else:
            raise NotImplementedError
        return dict(name=Path(fname).stem, img=img)
    
    def max_bsz_sfd(self, gpu_idx):
        return 2**int(math.log2(int(torch.cuda.mem_get_info(gpu_idx)[1] / 1024**2 - 361) / (0.000696875 * read_image(self.img_lists[0]).numel())))

    def max_bsz_retinaface(self, gpu_idx):
        return 2**int(math.log2(int(torch.cuda.mem_get_info(gpu_idx)[1] / 1024**2 - 620) / (0.00015 * read_image(self.img_lists[0]).numel())))
     
    def pop(self) -> Dict:
        fname = self.img_lists.pop()
        return Path(fname).stem, read_image(fname)


def get_largest_face(det_faces, h, w):

    def get_location(val, length):
        if val < 0:
            return 0
        elif val > length:
            return length
        else:
            return val

    face_areas = []
    for det_face in det_faces:
        left = get_location(det_face[0], w)
        right = get_location(det_face[2], w)
        top = get_location(det_face[1], h)
        bottom = get_location(det_face[3], h)
        face_area = (right - left) * (bottom - top)
        face_areas.append(face_area)
    largest_idx = face_areas.index(max(face_areas))
    return det_faces[largest_idx], largest_idx


def get_center_face(det_faces, h=0, w=0, center=None):
    if center is not None:
        center = np.array(center)
    else:
        center = np.array([w / 2, h / 2])
    center_dist = []
    for det_face in det_faces:
        face_center = np.array([(det_face[0] + det_face[2]) / 2, (det_face[1] + det_face[3]) / 2])
        dist = np.linalg.norm(face_center - center)
        center_dist.append(dist)
    center_idx = center_dist.index(min(center_dist))
    return det_faces[center_idx], center_idx


class Hdf5:

    def __init__(self, fname, lib='h5py', overwrite=False):
        self.fname = fname
        self.lib = lib
        self.file = None
        if overwrite and os.path.exists(fname):
            os.remove(fname)

    def add(self, key, value):
        with h5py.File(self.fname, 'a', libver='latest') as f:
            if key in f.keys():
                print(f"{key} already existed in {self.fname}, skipping...")
            else:
                f.create_dataset(
                    key,
                    data=value,
                    maxshape=value.shape,
                    compression='gzip',
                    compression_opts=9,
                    shuffle=True,
                    track_times=False,
                    # track_order=False,
                )

    def get(self, key):
        if not self.file:
            self.file = h5py.File(self.fname, 'r', libver='latest')
        if '/' in key:
            value = self.file
            for k in key.split('/'):
                value = value[k]
        else:
            value = self.file[key]
        return value

    @property
    def keys(self):
        with h5py.File(self.fname, mode='r', libver='latest') as f:
            return list(f.keys())
