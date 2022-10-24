import os
import SimpleITK as sitk
from ttictoc import tic,toc

import tensorflow as tf
import keras
import segmentation_models_3D
import nibabel as nib
from patchify import patchify, unpatchify
import numpy as np
from matplotlib import pyplot as plt
from keras import backend
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Path Dataset
NFBS_Dataset_path = 'F:/Desktop/Universidad/Semestres/NovenoSemestre/Procesamiento_Imagenes/3D-U-Net/NFBS_Dataset'

# Path Imágenes corregidas
corrected_path = 'F:/Desktop/Universidad/Semestres/NovenoSemestre/Procesamiento_Imagenes/3D-U-Net/Corrected'

tic()

i = 0
# Preprocesamiento de la imágenes
# N4 Bias Field Correction
for patient in os.listdir(NFBS_Dataset_path):
    # MRI T1 path
    mri_name = os.listdir(os.path.join(NFBS_Dataset_path,patient))[0]
    mri_path = os.path.join(NFBS_Dataset_path,patient,mri_name)
    
    # Lectura de MRI T1 formato nifti
    inputImage = sitk.ReadImage(mri_path, sitk.sitkFloat32)
    image = inputImage
    
    # N4 Bias Field Correction
    maskImage = sitk.OtsuThreshold(inputImage, 0, 1, 200)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrected_image = corrector.Execute(image, maskImage)
    log_bias_field = corrector.GetLogBiasFieldAsImage(inputImage)
    corrected_image_full_resolution = inputImage / sitk.Exp(log_bias_field)
    
    # Guardado de imagen filtrada
    sitk.WriteImage(corrected_image_full_resolution, os.path.join(corrected_path,patient)+'.nii')

    i = i+1
    print(f'Listo {i}', flush=True)
    

    
# print(tf.__version__)
# print(keras.__version__)

fin = toc()

print(fin)