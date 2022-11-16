import os
import numpy as np
import nibabel as nib
import segmentation_models_3D as sm
from keras import backend as K
from keras.models import load_model
from matplotlib import pyplot as plt
from patchify import patchify, unpatchify
from matplotlib.widgets import Slider
from scipy import ndimage as ndi
from skimage import morphology
import cv2

# Métricas de desempeño
def dice_coefficient(y_true, y_pred):
    smoothing_factor = 1
    flat_y_true = K.flatten(y_true)
    flat_y_pred = K.flatten(y_pred)
    return (2. * K.sum(flat_y_true * flat_y_pred) + smoothing_factor) / (K.sum(flat_y_true) + K.sum(flat_y_pred) + smoothing_factor)

# Cargar modelo preentrenado
model = load_model('modelosV2/3d_Unet_6000.h5', custom_objects={'dice_coefficient':dice_coefficient, 'iou_score':sm.metrics.IOUScore(threshold=0.5)})

"""Segmentación imagen completa"""
# Importación imagen random
mri_path = 'F:/Desktop/Universidad/Semestres/NovenoSemestre/Procesamiento_Imagenes/3D-U-Net/Dataset_Corrected'
mask_path = 'F:/Desktop/Universidad/Semestres/NovenoSemestre/Procesamiento_Imagenes/3D-U-Net/NFBS_Dataset'
img_idx = np.random.randint(0,125)

patient = os.listdir(mask_path)[img_idx]
patient_mask = os.listdir(os.path.join(mask_path,patient))[2]

mri_image = nib.load(os.path.join(mri_path,patient)+'.nii').get_fdata()
mask_image = nib.load(os.path.join(mask_path,patient,patient_mask)).get_fdata()
mri_image_no_corrected = nib.load('F:/Desktop/Universidad/Semestres/NovenoSemestre/Procesamiento_Imagenes/Proyecto/mri/Corrected/I130146.nii.gz').get_fdata()
# mri_image = nib.load('F:/Desktop/Universidad/Semestres/NovenoSemestre/Procesamiento_Imagenes/Proyecto/mri/Corrected/I130146.nii.gz').get_fdata()
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

# Conversión a numpy array
predicted_patches = np.array(predicted_patches)

# Reshape para proceso de reconstrucción
predicted_patches_reshaped = np.reshape(predicted_patches, 
                                        (mri_patches.shape[0], mri_patches.shape[1], mri_patches.shape[2],
                                         mri_patches.shape[3], mri_patches.shape[4], mri_patches.shape[5]) )

# Reconstrucción 
reconstructed_mask = unpatchify(predicted_patches_reshaped, mri_image.shape)

# Dilatación de máscaras
dilated_mask = ndi.binary_dilation(reconstructed_mask, morphology.ball(radius=4))


# Aplicar máscaras a imagen mri
reconstructed_mri = np.multiply(mri_image,reconstructed_mask)
dilated_mri = np.multiply(mri_image,dilated_mask.astype(np.uint8))

# Visualización resultados 
mri_slice = 100

# Plot Comparación máscaras
fig, axs = plt.subplots(2, 2)
fig.subplots_adjust(bottom=0.15)
fig.suptitle('Comparación Máscaras Obtenidas')

axs[0,0].set_title('MRI Original (x)')
axs[0,0].imshow(mri_image[:,:,mri_slice].T,cmap='gray', vmin=0, vmax=np.max(mri_image))

axs[0,1].set_title('MRI aplicando máscara 3D U-Net')
axs[0,1].imshow(reconstructed_mri[:,:,mri_slice].T,cmap='gray') #, vmin=0, vmax=np.max(mri_image))

# axs[0,2].set_title('MRI aplicando máscara 3D U-Net dilatada')
# axs[0,2].imshow(dilated_mri[:,:,mri_slice].T,cmap='gray') #, vmin=0, vmax=np.max(mri_image))

axs[1,0].set_title('Máscara original (y)')
axs[1,0].imshow(mask_image[:,:,mri_slice].T,cmap='gray', vmin=0, vmax=np.max(mri_image))

axs[1,1].set_title('Máscara con 3D U-Net')
axs[1,1].imshow(reconstructed_mask[:,:,mri_slice].T,cmap='gray', vmin=0, vmax=np.max(mri_image))

# axs[1,2].set_title('Máscara con 3D U-Net dilatada')
# axs[1,2].imshow(dilated_mask[:,:,mri_slice].T,cmap='gray', vmin=0, vmax=np.max(mri_image))

# Slider para cambiar slice
ax_slider = plt.axes([0.15, 0.05, 0.75, 0.03])
mri_slice_slider = Slider(ax_slider, 'Slice', 0, 192, 100, valstep=1)

# Plot comparación de contornos
fig2, axs2 = plt.subplots()
fig2.subplots_adjust(bottom=0.15)
axs2.set_title('Comparación Contornos de Máscaras')

# Mri
axs2.imshow(mri_image[:,:,mri_slice].T,cmap='gray', vmin=0, vmax=np.max(mri_image))

# Contornos
mask_contour = axs2.contour(mask_image[:,:,mri_slice].T, levels=np.logspace(-4.7, -3., 10), colors='#0016fe', linewidths=2)
# out_mask_contour = axs2.contour(reconstructed_mask[:,:,mri_slice].T, levels=np.logspace(-4.7, -3., 10), colors='#fe7500', linestyles='dotted', linewidths=2)
out_mask_contour = axs2.contour(reconstructed_mask[:,:,mri_slice].T, levels=np.logspace(-4.7, -3., 10), colors='#2aff00', linestyles='dotted', linewidths=2)
# dilated_mask_contour = axs2.contour(dilated_mask[:,:,mri_slice].T, levels=np.logspace(-4.7, -3., 10), colors='#2aff00', linestyles='dotted', linewidths=2)

# Etiquetas
h1,_ = mask_contour.legend_elements()
h2,_ = out_mask_contour.legend_elements()
# h3,_ = dilated_mask_contour.legend_elements()

# axs2.legend([h1[0], h2[0], h3[0]], ['Original', '3D U-Net', '3D U-Net Dilatada'],facecolor="white")
axs2.legend([h1[0], h2[0]], ['Original', '3D U-Net'],facecolor="white")

# Slider para cambiar slice
ax2_slider = plt.axes([0.15, 0.05, 0.75, 0.03])
mri_slice_slider2 = Slider(ax2_slider, 'Slice', 0, 192, 100, valstep=1)

def update(val):
    mri_slice = mri_slice_slider.val
    axs[0,0].imshow(mri_image[:,:,mri_slice].T,cmap='gray', vmin=0, vmax=np.max(mri_image))
    axs[0,1].imshow(reconstructed_mri[:,:,mri_slice].T,cmap='gray') #, vmin=0, vmax=np.max(mri_image))
    # axs[0,2].imshow(dilated_mri[:,:,mri_slice].T,cmap='gray') #, vmin=0, vmax=np.max(mri_image))
    axs[1,0].imshow(mask_image[:,:,mri_slice].T,cmap='gray')
    axs[1,1].imshow(reconstructed_mask[:,:,mri_slice].T,cmap='gray')
    # axs[1,2].imshow(dilated_mask[:,:,mri_slice].T,cmap='gray')

def update2(val):
    axs2.cla()
    axs2.set_title('Comparación Contornos de Máscaras')
    
    # Update slice
    mri_slice = mri_slice_slider2.val
    
    # Update Mri
    axs2.imshow(mri_image[:,:,mri_slice].T,cmap='gray', vmin=0, vmax=np.max(mri_image))
    
    # Update contornos
    mask_contour = axs2.contour(mask_image[:,:,mri_slice].T, levels=np.logspace(-4.7, -3., 10), colors='#0016fe', linewidths=2)
    # out_mask_contour = axs2.contour(reconstructed_mask[:,:,mri_slice].T, levels=np.logspace(-4.7, -3., 10), colors='#fe7500', linestyles='dotted', linewidths=2)
    out_mask_contour = axs2.contour(reconstructed_mask[:,:,mri_slice].T, levels=np.logspace(-4.7, -3., 10), colors='#2aff00', linestyles='dotted', linewidths=2)
    # dilated_mask_contour = axs2.contour(dilated_mask[:,:,mri_slice].T, levels=np.logspace(-4.7, -3., 10), colors='#2aff00', linestyles='dotted', linewidths=2)
    
    # Etiquetas
    h1,_ = mask_contour.legend_elements()
    h2,_ = out_mask_contour.legend_elements()
    # h3,_ = dilated_mask_contour.legend_elements()

    # axs2.legend([h1[0], h2[0], h3[0]], ['Original', '3D U-Net', '3D U-Net Dilatada'],facecolor="white")
    axs2.legend([h1[0], h2[0]], ['Original', '3D U-Net'],facecolor="white")

    
# Actualizar plot comparación máscaras
mri_slice_slider.on_changed(update)

# Actualizar plot comparación de contornos
mri_slice_slider2.on_changed(update2)