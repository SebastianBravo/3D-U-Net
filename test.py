import numpy as np
import nibabel as nib
import segmentation_models_3D as sm
from keras import backend as K
from keras.models import load_model
from matplotlib import pyplot as plt
from patchify import patchify, unpatchify
from sklearn.preprocessing import MinMaxScaler

# Métricas de desempeño
def dice_coefficient(y_true, y_pred):
    smoothing_factor = 1
    flat_y_true = K.flatten(y_true)
    flat_y_pred = K.flatten(y_pred)
    return (2. * K.sum(flat_y_true * flat_y_pred) + smoothing_factor) / (K.sum(flat_y_true) + K.sum(flat_y_pred) + smoothing_factor)

# Cargar modelo preentrenado
model = load_model('modelos/3d_Unet_2000.h5', custom_objects={'dice_coefficient':dice_coefficient, 'iou_score':sm.metrics.IOUScore(threshold=0.5)})

"""Segmentación imagen completa"""
# MinMaxScaler()
scaler = MinMaxScaler()

# Importación imagen
# mri_image = nib.load('Dataset_Corrected/A00028185.nii').get_fdata()
mri_image_no_corrected = nib.load('F:/Desktop/Universidad/Semestres/NovenoSemestre/Procesamiento_Imagenes/Proyecto/mri/NIFTI/I130146.nii.gz').get_fdata()
mri_image = nib.load('F:/Desktop/Universidad/Semestres/NovenoSemestre/Procesamiento_Imagenes/3D-U-Net/Dataset_Corrected/A00060516.nii').get_fdata()
# mri_image = np.append(mri_image, np.zeros((192-mri_image.shape[0],256,256)), axis=0)

# Normalización 
mri_image = (mri_image-np.min(mri_image))/(np.max(mri_image)-np.min(mri_image))
mri_image = mri_image.astype(np.float32)

# Volume sampling
mri_patches = patchify(mri_image, (64, 64, 64), step=64)

# Máscara de cerebro para cada volúmen
predicted_patches = []
for i in range(mri_patches.shape[0]):
  for j in range(mri_patches.shape[1]):
    for k in range(mri_patches.shape[2]):
      single_patch = mri_patches[i,j,k,:,:,:]
      single_patch_3ch = np.stack((single_patch,)*3, axis=-1)
      single_patch_3ch_input = np.expand_dims(single_patch_3ch, axis=0)
      single_patch_prediction = model.predict(single_patch_3ch_input)
      single_patch_prediction_th = (single_patch_prediction[0,:,:,:,0] > 0.5).astype(np.uint8)
      predicted_patches.append(single_patch_prediction_th)

#Convert list to numpy array
predicted_patches = np.array(predicted_patches)
print(predicted_patches.shape)

#Reshape to the shape we had after patchifying
predicted_patches_reshaped = np.reshape(predicted_patches, 
                                        (mri_patches.shape[0], mri_patches.shape[1], mri_patches.shape[2],
                                         mri_patches.shape[3], mri_patches.shape[4], mri_patches.shape[5]) )
print(predicted_patches_reshaped.shape)

#Repach individual patches into the orginal volume shape
reconstructed_image = unpatchify(predicted_patches_reshaped, mri_image.shape)

q = 80

# Plot de las imágenes
fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
fig.suptitle('Máscara de cerebro obtenida')
ax1.set_title('MRI Original')
ax1.imshow(mri_image[:,:,q],cmap='gray')
ax2.set_title('MRI corregida')
ax2.imshow(mri_image[:,:,q],cmap='gray')
ax3.set_title('Máscara cerebro')
ax3.imshow(reconstructed_image[:,:,q],cmap='gray')

