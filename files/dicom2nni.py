#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author       : ChenjieXu
Date         : 2022-10-05 18:11:26
LastEditors  : ChenjieXu
LastEditTime : 2022-10-05 18:11:28
FilePath     : /projects/cervical/dicom2nni.py
Description  : 
'''
import os
import dicom2nifti
from tqdm import tqdm

output_dir = ''
os.makedirs(output_dir, exist_ok=True)

input_dir = ''
patient_list = os.listdir(input_dir)
for patient in tqdm(patient_list):
    dicom2nifti.convert_directory(input_dir + patient, output_dir, compression=True)
    print(f"{patient} convert done!")