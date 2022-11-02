import os
import numpy as np
import nibabel as nib
import SimpleITK as sitk
from ttictoc import tic,toc
from patchify import patchify, unpatchify
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

# Path Dataset
NFBS_Dataset_path = 'F:/Desktop/Universidad/Semestres/NovenoSemestre/Procesamiento_Imagenes/3D-U-Net/NFBS_Dataset'

# Path Imágenes corregidas
corrected_path = 'F:/Desktop/Universidad/Semestres/NovenoSemestre/Procesamiento_Imagenes/3D-U-Net/Dataset_Corrected'

# Inicio timer pre procesamiento N4
# tic()

i = 0
'''---------------------Preprocesamiento de la imágenes---------------------'''
# N4 Bias Field Correction
for patient in os.listdir(NFBS_Dataset_path):
    # MRI T1 path
    mri_file = os.listdir(os.path.join(NFBS_Dataset_path,patient))[0]
    mri_path = os.path.join(NFBS_Dataset_path,patient,mri_file)
    
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

# Tiempo n4: 27114 s = 7.5 h 

tic()
  
# Listas para imágenes 
mri_images = np.zeros((125,256,256,192),np.float32) # MRI
mask_images = np.zeros((125,256,256,192),np.uint8) # Brain Masks

# Importado de MRI corregido
for i,mri in enumerate(os.listdir(corrected_path)):
    # Corrected MRI T1
    corr_mri = nib.load(os.path.join(corrected_path,mri)).get_fdata()
    
    # Normalización 
    corr_mri = (corr_mri-np.min(corr_mri))/(np.max(corr_mri)-np.min(corr_mri))  
    
    # Agregar a lista 
    mri_images[i,:,:,:] = corr_mri.astype(np.float32)

# Importado de Máscaras
for i,patient in enumerate(os.listdir(NFBS_Dataset_path)):
    # Mask path
    mask_file = os.listdir(os.path.join(NFBS_Dataset_path,patient))[2]
    mask_path = os.path.join(NFBS_Dataset_path,patient,mask_file)
    
    # Importación
    mask = nib.load(mask_path).get_fdata()
    mask = mask.astype(np.uint8)
    
    # Agregar a lista
    mask_images[i,:,:,:] = mask

print(toc())

# Tiempo Normalización: 2265 s = 37.7 min

# Volume sampling 
X = np.zeros((6000,64,64,64,3),np.float32)
y = np.zeros((6000,64,64,64,1),np.uint8)

for i in range(len(mask_images)):
    # volume sampling
    mri_patches = patchify(mri_images[i], (64, 64, 64), step=64)
    mask_patches = patchify(mask_images[i], (64, 64, 64), step=64)
    
    mri_patches = mri_patches.reshape(-1, mri_patches.shape[-3], mri_patches.shape[-2], mri_patches.shape[-1])
    mask_patches = mask_patches.reshape(-1, mask_patches.shape[-3], mask_patches.shape[-2], mask_patches.shape[-1])
    
    # Convertir imágen a 3 canales
    mri = np.stack((mri_patches,)*3, axis=-1)
    mask = np.expand_dims(mask_patches, axis=4)
    
    X[i*48:(i*48)+48,:,:,:,:] = mri
    y[i*48:(i*48)+48,:,:,:,:] = mask

# Datos procesados    
# np.save('Processed/X.npy', X)
np.save('Processed/y.npy', y)

# Conjuntos de train y test 
# 6 conjutos de 1000 

X = np.load('Processed/X.npy')
y = np.load('Processed/y.npy')

for i in range(6):
    # Segmentación conjuntos de entrenamiento (80%) y prueba (20%)
    X_train, X_test, y_train, y_test = train_test_split(X[i*1000:i*1000 + 1000], y[i*1000:i*1000 + 1000], train_size = 0.80, test_size = 0.20, random_state = 1)
    
    # Guardado de archivos
    np.save(f'train/X_train/X_train{i}.npy', X_train)
    np.save(f'train/y_train/y_train{i}.npy', y_train)
    np.save(f'test/X_test/X_test{i}.npy', X_test)
    np.save(f'test/y_test/y_test{i}.npy', y_test)
