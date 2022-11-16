import os
import numpy as np
import nibabel as nib
import segmentation_models_3D as sm
from keras import backend as K
from keras.models import load_model
from matplotlib import pyplot as plt
from patchify import patchify, unpatchify

# Métricas de desempeño
def dice_coefficient(y_true, y_pred):
    smoothing_factor = 1
    flat_y_true = K.flatten(y_true)
    flat_y_pred = K.flatten(y_pred)
    return (2. * K.sum(flat_y_true * flat_y_pred) + smoothing_factor) / (K.sum(flat_y_true) + K.sum(flat_y_pred) + smoothing_factor)

# Cargar modelo preentrenado
model = load_model('modelosV2/3d_Unet_6000.h5', custom_objects={'dice_coefficient':dice_coefficient, 'iou_score':sm.metrics.IOUScore(threshold=0.5)})

# Importación imágenes de prueba (X)
for i, test_patch in enumerate(os.listdir('test/X_test')):
    if i==0:
        X_test = np.load('test/X_test/'+ test_patch)
    else:
        X_test = np.concatenate((X_test, np.load('test/X_test/'+ test_patch)), axis=0)
        
# Importación máscaras
for j, test_patch in enumerate(os.listdir('test/y_test')):
    if j==0:
        y_test = np.load('test/y_test/'+ test_patch)
    else:
        y_test = np.concatenate((y_test, np.load('test/y_test/'+ test_patch)), axis=0)
   
# Evaluación
results = model.evaluate(X_test, y_test, batch_size=1)
