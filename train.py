import keras
import numpy as np
import tensorflow as tf
import segmentation_models_3D as sm
from keras import backend as K
from keras.models import load_model
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import Sequence
from matplotlib import pyplot as plt

# GPU para entrenamiento
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))

# Parametros modelo
encoder_weights = 'imagenet'
BACKBONE = 'vgg16'  #Try vgg16, efficientnetb7, inceptionv3, resnet50
activation = 'sigmoid'
input_shape = (64,64,64,3)
n_classes = 1

# Learning rate
LR = 0.0001
optim = tf.keras.optimizers.Adam(LR)

# Función de perdida
loss = sm.losses.BinaryCELoss()

# Métricas de desempeño
def dice_coefficient(y_true, y_pred):
    smoothing_factor = 1
    flat_y_true = K.flatten(y_true)
    flat_y_pred = K.flatten(y_pred)
    return (2. * K.sum(flat_y_true * flat_y_pred) + smoothing_factor) / (K.sum(flat_y_true) + K.sum(flat_y_pred) + smoothing_factor)

metrics = [dice_coefficient, 'accuracy', sm.metrics.IOUScore(threshold=0.5)]

#Define the model. Here we use Unet but we can also use other model architectures from the library.
model = sm.Unet(BACKBONE, classes=n_classes, 
                input_shape = input_shape, 
                encoder_weights = encoder_weights,
                activation = activation)

model.compile(optimizer = optim, loss=loss, metrics=metrics)
print(model.summary())

class DataGenerator(Sequence):
    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        return int(len(self.x)/self.batch_size)

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        return batch_x, batch_y

for i in range(1,3):
    if (i>0):
        # Cargar modelo preentrenado
        model = load_model(f'modelosV2/3d_Unet_{i}000.h5', custom_objects={'dice_coefficient': dice_coefficient, 'iou_score':sm.metrics.IOUScore(threshold=0.5)})
    
    # Importación de conjuntos
    X_train, y_train = np.load(f'train/X_train/X_train{i}.npy'), np.load(f'train/y_train/y_train{i}.npy').astype(np.float32)
    
    # Conjunto entrenamiento y validación
    X_train, X_val, y_train, y_val= train_test_split(X_train, y_train, train_size = 0.75, test_size = 0.25, random_state = 1)
    
    train_gen = DataGenerator(X_train, y_train, 8)
    val_gen = DataGenerator(X_val, y_val, 8)
    
    # Fit the model
    history = model.fit(train_gen,
                        epochs=100,
                        workers=8, 
                        verbose=1,
                        validation_data = val_gen)
    
    # Modelo parcialmente entrenado
    model.save(f'modelosV2/3d_Unet_{i+1}000.h5')
    
    # Gráficas de los scores y losses
    acc = history.history['iou_score']
    val_acc = history.history['val_iou_score']
    epochs = range(1, len(acc) + 1)
    
    plt.figure(1)
    plt.plot(epochs, acc, 'y', label='Training IOU')
    plt.plot(epochs, val_acc, 'r', label='Validation IOU')
    plt.title('Training and validation IOU')
    plt.xlabel('Epochs')
    plt.ylabel('IOU')
    plt.ylim([0,1])
    plt.legend()
    plt.savefig(f'Graphs/iou/3d_unet_{i+1}000_iou.svg')
    
    plt.figure(2)
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    plt.plot(epochs, loss, 'y', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.ylim([0,1])
    plt.legend()
    plt.savefig(f'Graphs/loss/3d_unet_{i+1}000_loss.svg')