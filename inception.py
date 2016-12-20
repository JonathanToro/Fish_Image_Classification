import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from keras.applications.inception_v3 import InceptionV3
import os
from keras.layers import Flatten, Dense, AveragePooling2D
from keras.models import Model
from keras.optimizers import RMSprop, SGD
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import History

from keras.layers import merge
from keras.layers.core import Lambda
from keras.models import Model
import tensorflow as tf

import itertools
import numpy as np
import scipy
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from sklearn.metrics import classification_report, confusion_matrix

import make_parallel

learning_rate = 0.0001
img_width = 299
img_height = 299
nbr_train_samples = 3016
nbr_validation_samples = 756
nbr_epochs = 70
batch_size = 80

train_data_dir = '/home/ubuntu/jupyter/Fish/Final_Project/train2/train_split'
val_data_dir = '/home/ubuntu/jupyter/Fish/Final_Project/train2/val_split'

FishNames = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']

print('Loading InceptionV3 Weights ...')
InceptionV3_notop = InceptionV3(include_top=False, weights='imagenet', input_tensor=None, input_shape = (3, 299,299))
# Note that the preprocessing of InceptionV3 is:
# (x / 255 - 0.5) x 2

print('Adding Average Pooling Layer and Softmax Output Layer ...')
output = InceptionV3_notop.get_layer(index = -1).output  # Shape: (8, 8, 2048)
output = AveragePooling2D((8, 8), strides=(8, 8), name='avg_pool')(output)
output = Flatten(name='flatten')(output)
output = Dense(8, activation='softmax', name='predictions')(output)

InceptionV3_model = Model(InceptionV3_notop.input, output)
#InceptionV3_model.summary()
history = History()

InceptionV3_model = make_parallel(InceptionV3_model,4)
optimizer = SGD(lr = learning_rate, momentum = 0.9, decay = 0.0, nesterov = True)
InceptionV3_model.compile(loss='categorical_crossentropy', optimizer = optimizer, metrics = ['accuracy','precision','recall'])

# autosave best Model
best_model_file = "./inception_weights.h5"
best_model = ModelCheckpoint(best_model_file, monitor='val_acc', verbose = 1, save_best_only = True)

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.1,
        zoom_range=0.1,
        rotation_range=10.,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True)

# this is the augmentation configuration we will use for validation:
# only rescaling
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size = (img_width, img_height),
        batch_size = batch_size,
        shuffle = True,
        # save_to_dir = 
        # save_prefix = 'aug',
        classes = FishNames,
        class_mode = 'categorical')

validation_generator = val_datagen.flow_from_directory(
        val_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        shuffle = True,
        #save_to_dir = ,
        #save_prefix = 'aug',
        classes = FishNames,
        class_mode = 'categorical')

InceptionV3_model.fit_generator(
        train_generator,
        samples_per_epoch = nbr_train_samples,
        nb_epoch = nbr_epochs,
        validation_data = validation_generator,
        nb_val_samples = nbr_validation_samples,
        nb_worker = 4,
        callbacks = [best_model, history])

fig = plt.figure()
axes = fig.add_axes([.3, 0.3, 1, 1])
axes.plot(range(0,70),history.history["val_acc"])
axes.plot(range(0,70), history.history["acc"])
axes.set_title("Validation Accuracy vs Accuracy")
axes.set_xlabel("Epoch")
axes.set_ylabel("Accuracy")
axes.legend(['Validation Accuracy', 'Accuracy'], loc = 'best')
plt.savefig("/home/ubuntu/jupyter/Fish/Final_Project/inception_graphs/accuracy")


fig = plt.figure()
axes = fig.add_axes([.3, 0.3, 1, 1])
axes.plot(range(0,70),history.history["val_loss"])
axes.plot(range(0,70), history.history["loss"])
axes.set_title("Validation Loss vs. Loss")
axes.set_xlabel("Epoch")
axes.set_ylabel("Loss")
axes.legend(['Validation Loss', 'Loss'], loc = 'best')
plt.savefig("/home/ubuntu/jupyter/Fish/Final_Project/inception_graphs/loss")

fig = plt.figure()
axes = fig.add_axes([.3, 0.3, 1, 1])
axes.plot(range(0,70),history.history["val_precision"])
axes.plot(range(0,70), history.history["precision"])
axes.set_title("Validation Precision vs. Precision")
axes.set_xlabel("Epoch")
axes.set_ylabel("Precision")
axes.legend(['Validation Precision', 'Precision'], loc = 'best')
plt.savefig("/home/ubuntu/jupyter/Fish/Final_Project/inception_graphs/precision")

fig = plt.figure()
axes = fig.add_axes([.3, 0.3, 1, 1])
axes.plot(range(0,70),history.history["val_recall"])
axes.plot(range(0,70), history.history["recall"])
axes.set_xlabel("Epoch")
axes.set_ylabel("Recall")
axes.set_title("Validation Recall vs. Recall")
axes.legend(['Validation Recall', 'Recall'], loc = 'best')
plt.savefig("/home/ubuntu/jupyter/Fish/Final_Project/inception_graphs/recall")

fig = plt.figure()
axes = fig.add_axes([.3, 0.3, 1, 1])
axes.plot(range(0,70),history.history["precision"])
axes.plot(range(0,70), history.history["recall"])
axes.set_xlabel("Epoch")
axes.set_title("Precision vs. Recall")
axes.legend(['Precision', 'Recall'], loc = 'best')
plt.savefig("/home/ubuntu/jupyter/Fish/Final_Project/inception_graphs/precision_recall")

TEST_DIR = val_data_dir+'/'
FISH_CLASSES = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']
ROWS = 299
COLS = 299
CHANNELS = 3

def get_images(fish):
    """Load files from train folder"""
    fish_dir = TEST_DIR+'{}'.format(fish)
    images = [fish+'/'+im for im in os.listdir(fish_dir)]
    return images

def read_image(src):
    """Read and resize individual images"""
    im = scipy.ndimage.imread(src)
    im = scipy.misc.imresize(im, (ROWS, COLS))
    im = np.reshape(im,(3,ROWS,COLS))
    return im

files = []
y_all = []

for fish in FISH_CLASSES:
    fish_files = get_images(fish)
    files.extend(fish_files)
    
    y_fish = np.tile(fish, len(fish_files))
    y_all.extend(y_fish)
    print("{0} photos of {1}".format(len(fish_files), fish))
    
y_all = np.array(y_all)


X_all = np.ndarray((len(files), ROWS, COLS, CHANNELS), dtype=np.uint8)
X_all = np.reshape(X_all,(756,3,COLS,ROWS))

for i, im in enumerate(files): 
    X_all[i] = read_image(TEST_DIR+im)
    if i%1000 == 0: print('Processed {} of {}'.format(i, len(files)))


y_all = LabelEncoder().fit_transform(y_all)
y_all = np_utils.to_categorical(y_all)
y_all.shape


pred = InceptionV3_model.predict(X_all)
y_pred = np.argmax(pred, axis = 1)

text_file = open("/home/ubuntu/jupyter/Fish/Final_Project/inception_graphs/classification_report.txt", "w")
text_file.write(classification_report(np.argmax(y_all,axis = 1),y_pred,target_names = FISH_CLASSES))
text_file.close()
