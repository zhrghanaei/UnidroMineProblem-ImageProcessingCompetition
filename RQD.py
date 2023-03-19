#@title RQD: Image Preprocessing
from PIL import Image
import pandas as pd
import numpy as np
import shutil
import os
from tqdm import tqdm

if os.path.exists('train_image'):
    shutil.rmtree('train_image')
    shutil.rmtree('train_label')
os.makedirs('train_image')
os.makedirs('train_label')

label_df = pd.read_excel('label.xlsx')

for img_name in tqdm(os.listdir('train')):
  labels = label_df[label_df['image_name'] == img_name]

  img = np.array(Image.open('train/' + img_name))
  h, w, _ = img.shape
  h = h // 2         # crop top half of image
  img = img[h:,:,:]
  lbl = np.zeros((h, w, 3), dtype = np.uint8)
  lbl[:,:,2] = 255
  for i in range(len(labels)):
    xmin = labels.iloc[i]['xmin']
    ymin = labels.iloc[i]['ymin']
    width = labels.iloc[i]['width']
    height = labels.iloc[i]['height']

    lbl[ymin - h: ymin - h + height, xmin:xmin + width, 2] = 0
    if labels.iloc[i]['label_name'] == '+10cm rock':
      lbl[ymin - h: ymin - h + height, xmin:xmin + width, 0] = 255
    else:
      lbl[ymin - h: ymin - h + height, xmin:xmin + width, 1] = 255

  # extract rows
  part_len = h // 5
  for i in range(5):
    Image.fromarray(img[i*part_len: (i+1)*part_len:, :, :]).save('train_image/' + img_name.split('.')[0] + '_' + str(i) + '.jpg')
    Image.fromarray(lbl[i*part_len: (i+1)*part_len:, :, :]).save('train_label/' + img_name.split('.')[0] + '_' + str(i) + '.png')

#@title RQD: Data Generator
from tensorflow.keras.utils import Sequence
from PIL import Image
import numpy as np
import os

class DataGenerator(Sequence):
    'Generates data for Keras'
    def __init__(self, img_list, target_size = (256, 2048), batch_size = 4, shuffle = True, random_flip = True):
        'Initialization'
        self.target_size = target_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.random_flip = random_flip
        self.n = 0
        self.img_list = img_list
        self.max = self.__len__()
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return len(self.img_list) // self.batch_size

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        path_list_temp = [self.img_list[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(path_list_temp)

        return X, y

    def __next__(self):
        if self.n >= self.max:
           self.n = 0
        result = self.__getitem__(self.n)
        self.n += 1
        return result

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.img_list))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, batch_paths):
        'Generates data containing batch_size samples'
        # Initialization
        batch_x = np.empty((self.batch_size, *self.target_size, 3))
        batch_y = np.empty((self.batch_size, *self.target_size, 3))
        # Generate data
        for i, img_name in enumerate(batch_paths):
          imgPath = 'train_image/' + img_name
          lblPath = 'train_label/' + img_name.split('.')[0] + '.png'

          img = Image.open(imgPath)
          lbl = Image.open(lblPath)

          w, h = img.size

          if h <= self.target_size[0]:
            img = img.resize((w, self.target_size[0]+1))
            lbl = lbl.resize((w, self.target_size[0]+1))
            w, h = img.size

          img = np.array(img)
          lbl = np.array(lbl)#[:,:,:2]

          x_pos = np.random.randint(h - self.target_size[0])
          y_pos = np.random.randint(w - self.target_size[1])
          img = img[x_pos:x_pos+self.target_size[0], y_pos:y_pos+self.target_size[1], :]
          lbl = lbl[x_pos:x_pos+self.target_size[0], y_pos:y_pos+self.target_size[1], :]

          if self.random_flip and np.random.rand() > 0.5:
            img = np.fliplr(img)
            lbl = np.fliplr(lbl)
          if self.random_flip and np.random.rand() > 0.5:
            img = np.flipud(img)
            lbl = np.flipud(lbl)

          batch_x[i] = img
          batch_y[i] = lbl

        batch_x = np.array(batch_x).astype(np.float32)
        batch_y = np.array(batch_y).astype(np.float32)
        batch_x = batch_x / 255
        # batch_x = batch_x * 2 - 1
        batch_y = batch_y / 255

        return (batch_x, batch_y)

#@title RQD: Model Class
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Conv2D, BatchNormalization, Conv2DTranspose, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
import tensorflow.keras.backend as K
import tensorflow as tf
from PIL import Image
import pandas as pd
import os
import shutil
from tqdm import tqdm

class SegmentationNet():
  def __init__(self, input_size, learning_rate):
    self.input_size = input_size
    self.learning_rate = learning_rate
    self.model = self.unet()

  def dice_loss(self, y_true, y_pred):
      smooth = 1e-6
      intersection = K.sum(y_true * y_pred, axis = [1,2])
      union = K.sum(y_true, axis = [1,2]) + K.sum(y_pred, axis = [1,2])
      dice = (2 * intersection) / (union + smooth)
      return 1 - dice

  def dice(self, y_true, y_pred):
      smooth = 1e-6
      y_true_f = K.flatten(y_true)
      y_pred_f = K.flatten(y_pred)
      intersection = K.sum(y_true_f * y_pred_f)
      dice = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
      return dice

  def tp(self, y_true, y_pred):
      smooth = 1e-6
      y_pred_pos = K.round(K.clip(y_pred, 0, 1))
      y_pos = K.round(K.clip(y_true, 0, 1))
      tp = (K.sum(y_pos * y_pred_pos) + smooth)/ (K.sum(y_pos) + smooth)
      return tp

  def tn(self, y_true, y_pred):
      smooth = 1e-6
      y_pred_pos = K.round(K.clip(y_pred, 0, 1))
      y_pred_neg = 1 - y_pred_pos
      y_pos = K.round(K.clip(y_true, 0, 1))
      y_neg = 1 - y_pos
      tn = (K.sum(y_neg * y_pred_neg) + smooth) / (K.sum(y_neg) + smooth )
      return tn

  def unet(self):
    self.filters = [32, 64, 128, 128, 512]
    inputs = Input(shape = self.input_size)
    # encoder block 1
    n_filters = self.filters[0]
    self.droprate = 0.05
    conv1 = Conv2D(n_filters, (3, 3), activation='relu', padding="same")(inputs)
    conv1 = Conv2D(n_filters, (3, 3), activation='relu', padding="same")(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    pool1 = Dropout(self.droprate)(pool1)
    pool1 = BatchNormalization()(pool1)

    # encoder block 2
    n_filters = self.filters[1]
    conv2 = Conv2D(n_filters, (3, 3), activation='relu', padding="same")(pool1)
    conv2 = Conv2D(n_filters, (3, 3), activation='relu', padding="same")(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    pool2 = Dropout(self.droprate)(pool2)
    pool2 = BatchNormalization()(pool2)

    # encoder block 3
    n_filters = self.filters[2]
    conv3 = Conv2D(n_filters, (3, 3), activation='relu', padding="same")(pool2)
    conv3 = Conv2D(n_filters, (3, 3), activation='relu', padding="same")(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    pool3 = Dropout(self.droprate)(pool3)
    pool3 = BatchNormalization()(pool3)

    # encoder block 4
    n_filters = self.filters[3]
    conv4 = Conv2D(n_filters, (3, 3), activation='relu', padding="same")(pool3)
    conv4 = Conv2D(n_filters, (3, 3), activation='relu', padding="same")(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    pool4 = Dropout(self.droprate)(pool4)
    pool4 = BatchNormalization()(pool4)

    # bottlneck
    n_filters = self.filters[4]
    conv5 = Conv2D(n_filters, (3, 3), activation='relu', padding="same")(pool4)
    conv5 = Conv2D(n_filters, (3, 3), activation='relu', padding="same")(conv5)

    # decoder block 1
    n_filters = self.filters[3]
    up6 = Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding='same')(conv5)
    up6 = concatenate([up6, conv4], axis=3)
    conv6 = Conv2D(n_filters, (3, 3), activation='relu', padding="same")(up6)
    conv6 = Conv2D(n_filters, (3, 3), activation='relu', padding="same")(conv6)
    conv6 = Dropout(self.droprate)(conv6)
    conv6 = BatchNormalization()(conv6)

    # decoder block 2
    n_filters = self.filters[2]
    up7 = Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding='same')(conv6)
    up7 = concatenate([up7, conv3], axis=3)
    conv7 = Conv2D(n_filters, (3, 3), activation='relu', padding="same")(up7)
    conv7 = Conv2D(n_filters, (3, 3), activation='relu', padding="same")(conv7)
    conv7 = Dropout(self.droprate)(conv7)
    conv7 = BatchNormalization()(conv7)

    # decoder block 3
    n_filters = self.filters[1]
    up8 = Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding='same')(conv7)
    up8 = concatenate([up8, conv2], axis=3)
    conv8 = Conv2D(n_filters, (3, 3), activation='relu', padding="same")(up8)
    conv8 = Conv2D(n_filters, (3, 3), activation='relu', padding="same")(conv8)
    conv8 = Dropout(self.droprate)(conv8)
    conv8 = BatchNormalization()(conv8)

    # decoder block 4
    n_filters = self.filters[0]
    up9 = Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding='same')(conv8)
    up9 = concatenate([up9, conv1], axis=3)
    conv9 = Conv2D(n_filters, (3, 3), activation='relu', padding="same")(up9)
    conv9 = Conv2D(n_filters, (3, 3), activation='relu', padding="same")(conv9)
    conv9 = Dropout(self.droprate)(conv9)
    conv9 = BatchNormalization()(conv9)

    # output
    conv10 = Conv2D(3, (1, 1), activation='softmax')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])
    model.compile(optimizer=Adam(self.learning_rate), loss=self.dice_loss, metrics=['accuracy', self.dice, self.tp, self.tn])
    return model

  def train_model(self, train_list, valid_list, batch_size, epoch, save_path):
    self.batch_size = batch_size
    self.epoch = epoch
    self.use_pretrained = False
    self.initial_epoch = 0

    print('Load data...')
    train_data_gen = DataGenerator(train_list, target_size = (self.input_size[0], self.input_size[1]), batch_size = self.batch_size)
    valid_data_gen = DataGenerator(valid_list, target_size = (self.input_size[0], self.input_size[1]), batch_size = self.batch_size)

    csv_logger  = CSVLogger(save_path + 'history.csv')
    checkpoint = ModelCheckpoint(save_path + 'weights_best.hdf5',
                                 monitor = 'val_dice',
                                 verbose = 1,
                                 save_best_only = True,
                                 save_weights_only = True,
                                 mode = 'max')
    callback_list = [checkpoint, csv_logger]

    print('Start training...')
    hist = self.model.fit(train_data_gen,
                          validation_data = valid_data_gen,
                          epochs = self.epoch,
                          callbacks = callback_list)

    print('Save final weights...')
    self.model.save_weights(save_path + '_weights_final.hdf5')

    print('All training process is done!')

  def predict_image(self, img):
    img = np.array(img)
    h, w, _ = img.shape
    h = h // 2         # crop top half of image
    img = img[h:,:,:]

    sp_h = 100
    sp_w = 500
    part_h = h // 5
    part_w = w

    label = []
    for i in range(5):
      data = img[i*part_h: (i+1)*part_h:, :, :]
      data = np.array(data).astype(np.float32)
      data = data / 255
      out = np.zeros([part_h, part_w, 3], dtype = np.float)
      out_count = np.zeros([part_h, part_w], dtype = np.float)
      hh = 0
      while hh + INPUT_SIZE[0] < part_h:
        ww = 0
        while ww + INPUT_SIZE[1] < part_w:
          _img = data[hh:hh + INPUT_SIZE[0], ww:ww + INPUT_SIZE[1], :]
          _img = np.expand_dims(_img, axis = 0)
          out[hh:hh + INPUT_SIZE[0], ww:ww + INPUT_SIZE[1], :] += np.squeeze(self.model.predict(_img))
          out_count[hh:hh + INPUT_SIZE[0], ww:ww + INPUT_SIZE[1]] += 1
          ww += sp_w

        ### last part of the row
        _img = data[hh:hh + INPUT_SIZE[0], part_w - INPUT_SIZE[1]:part_w, :]
        _img = np.expand_dims(_img, axis = 0)
        out[hh:hh + INPUT_SIZE[0], part_w - INPUT_SIZE[1]:part_w, :] += np.squeeze(self.model.predict(_img))
        out_count[hh:hh + INPUT_SIZE[0], part_w - INPUT_SIZE[1]:part_w] += 1

        hh += sp_h

      ### last row
      ww = 0
      while ww + INPUT_SIZE[1] < part_w:
        _img = data[part_h - INPUT_SIZE[0]:part_h, ww:ww + INPUT_SIZE[1], :]
        _img = np.expand_dims(_img, axis = 0)
        out[part_h - INPUT_SIZE[0]:part_h, ww:ww + INPUT_SIZE[1], :] += np.squeeze(self.model.predict(_img))
        out_count[part_h - INPUT_SIZE[0]:part_h, ww:ww + INPUT_SIZE[1]] += 1
        ww += sp_w

      ### last part of the row
      _img = data[part_h - INPUT_SIZE[0]:part_h, part_w - INPUT_SIZE[1]:part_w, :]
      _img = np.expand_dims(_img, axis = 0)
      out[part_h - INPUT_SIZE[0]:part_h, part_w - INPUT_SIZE[1]:part_w, :] += np.squeeze(self.model.predict(_img))
      out_count[part_h - INPUT_SIZE[0]:part_h, part_w - INPUT_SIZE[1]:part_w] += 1

      out /= np.repeat(out_count[:, :, np.newaxis], 3, axis=-1)
      label.append(np.argmax(out, axis=-1))

    return np.array(label)

#@title RQD: Train Model
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import plot_model
import pandas as pd
import time
import sys
import os

INPUT_SIZE = (256, 2048, 3)
EPOCH = 20
BATCH_SIZE = 4
LEARNING_RATE = 1e-4

date_time = time.strftime('%Y_%m_%d-%H_%M_%S', time.localtime())
model_path = 'drive/MyDrive/Unidro/Models/{}/{}/'.format(date_time, LEARNING_RATE)

if not os.path.exists(model_path):
  os.makedirs(model_path)

data_list = os.listdir('train_image')
np.random.shuffle(data_list)
train_list = data_list[:700].copy()
valid_list = data_list[700:].copy()

unet = SegmentationNet(INPUT_SIZE, LEARNING_RATE)

unet.train_model(train_list,
                 valid_list,
                 batch_size = BATCH_SIZE,
                 epoch = EPOCH,
                 save_path = None)

# print(unet.model.summary())

# plot_model(unet_model, 'model.png', show_layer_names=False, rankdir='TB', show_shapes=True)

#@title RQD: RQD Calculation
import numpy as np
import pandas as pd
import cv2
import os
import matplotlib
from google.colab.patches import cv2_imshow

def remove_parts(img, type, thresh):
  res = np.zeros_like(img)
  nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity = 8)
  for i in range(1, len(stats)):
    if stats[i][type] > thresh:
        res[output == i] = 255
  return res

INPUT_SIZE = (256, 2048, 3)

### uncomment to load saved model
# model_path = 'drive/MyDrive/Unidro/Models/2021_08_22-14_20_42/0.0001/weights_best.hdf5'

# unet = SegmentationNet(INPUT_SIZE, 1)
# unet.model.load_weights(model_path)

ft_df = pd.read_excel('from-to-rqd.xlsx')
ft_df['Percent'] = ''
ft_df['Prediction'] = ''

for img_name in os.listdir('test-rqd'):
  print(img_name)
  img = Image.open('test-rqd/' + img_name)
  scale = 1.1 / img.size[0]
  pred = unet.predict_image(img)
  pred_all = np.hstack([pr for pr in pred])
  rock = (pred_all == 0).astype('uint8')
  wood = (pred_all == 1).astype('uint8')
  wood = remove_parts(wood, cv2.CC_STAT_AREA, 5000)
  wood_nb_components, wood_output, wood_stats, _ = cv2.connectedComponentsWithStats(wood, connectivity = 8)

  rock = cv2.morphologyEx(rock, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (int(0.02/scale), 1)))
  rock = remove_parts(rock, cv2.CC_STAT_WIDTH, 0.08 / scale)
  rock = remove_parts(rock, cv2.CC_STAT_HEIGHT, 100)
  rock_nb_components, rock_output, rock_stats, _ = cv2.connectedComponentsWithStats(rock, connectivity = 8)

  sorted_wood = sorted(wood_stats[1:], key=lambda x: x[cv2.CC_STAT_LEFT])
  sorted_rock = sorted(rock_stats[1:], key=lambda x: x[cv2.CC_STAT_LEFT])

  run_rock_length = np.zeros(len(sorted_wood) + 1)
  rock_id = 0
  for run_id in range(len(sorted_wood)):
    while rock_id < len(sorted_rock) and sorted_rock[rock_id][cv2.CC_STAT_LEFT] < sorted_wood[run_id][cv2.CC_STAT_LEFT]:
      run_rock_length[run_id] += sorted_rock[rock_id][cv2.CC_STAT_WIDTH]
      rock_id += 1
      if rock_id > len(sorted_rock):
        break

  run_id += 1
  while rock_id < len(sorted_rock) and sorted_rock[rock_id][cv2.CC_STAT_LEFT] < rock.shape[1]:
    run_rock_length[run_id] += sorted_rock[rock_id][cv2.CC_STAT_WIDTH]
    rock_id += 1
    if rock_id >= len(sorted_rock):
      break

  for i in range(len(run_rock_length)):
    run_name = img_name.split('.')[0] + '-' + str(i+1)
    if ft_df['RunId'].str.contains(run_name).sum() > 0:
      l = ft_df.loc[ft_df['RunId'] == run_name, 'to'].values[0] - ft_df.loc[ft_df['RunId'] == run_name, 'from'].values[0]
      if run_name == 'M3-BH3300-2-1': l = 0.6
      percent = 100 * (run_rock_length[i]*scale) / l
      ft_df.loc[ft_df['RunId'] == run_name, 'Percent']  = percent

      if percent <= 25:
        ft_df.loc[ft_df['RunId'] == run_name, 'Prediction'] = 1
      elif percent <= 50:
        ft_df.loc[ft_df['RunId'] == run_name, 'Prediction'] = 2
      elif percent <= 75:
        ft_df.loc[ft_df['RunId'] == run_name, 'Prediction'] = 3
      elif percent <= 90:
        ft_df.loc[ft_df['RunId'] == run_name, 'Prediction'] = 4
      else:
        ft_df.loc[ft_df['RunId'] == run_name, 'Prediction'] = 5

ft_df.to_csv('output.csv', mode='w', columns=['RunId', 'Prediction'], index=False)
