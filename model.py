import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout
from keras.regularizers import l2, activity_l2
from keras.layers import Lambda, Convolution2D, BatchNormalization
from keras.callbacks import ModelCheckpoint

# Program Constants
# size of resized images, left, right steering correction and training batch size
ROWS = 64
COLS = 128
CORRECTION = 0.35
batch_size = 128

def preprocess(img):
    # crop image inside take 50 from top and 20 from bottom
    # go to HSV colorspace
    # print(img.shape)
    hsvimg = cv2.cvtColor(img[50:140, 0: 320], cv2.COLOR_RGB2HSV)
    newimg = cv2.resize(hsvimg, (COLS, ROWS))
    # print(newimg.shape)
    return newimg

# load log for image filenames and variables
lines = []
with open('../data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
    log_labels = lines.pop(0)

images = []
measurements = []
augmented_images, augmented_measurements = [] ,[]

# start augmenting left-right images
for line in lines:
    source_path = line[0]
    filename = source_path.split('/')[-1]
    current_path = '../data/IMG/' + filename
    image = cv2.imread(current_path)
    img = preprocess(image)
    images.append(img)
    measurement = float(line[3])
    measurements.append(measurement)
    
    source_path = line[1]
    filename = source_path.split('/')[-1]
    current_path = '../data/IMG/' + filename
    image = cv2.imread(current_path)
    img = preprocess(image)
    images.append(img)
    # add correction to steering for left camera
    measurement = float(line[3])  + CORRECTION
    measurements.append(measurement)

    source_path = line[2]
    filename = source_path.split('/')[-1]
    current_path = '../data/IMG/' + filename
    image = cv2.imread(current_path)
    img = preprocess(image)
    images.append(img)
    # subtract correction to steering for right camera
    measurement = float(line[3]) - CORRECTION
    measurements.append(measurement ) 

for image, measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(image, 1))
    augmented_measurements.append(-1.0*measurement)

# make into numpy arrays
X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

#print(X_train.shape)


model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(ROWS, COLS, 3)))
model.add(Convolution2D(1,1,1, border_mode='valid', activation='relu'))
model.add(BatchNormalization())
model.add(Convolution2D(6,3,3, border_mode='valid', activation='relu'))
model.add(BatchNormalization())
model.add(Convolution2D(1,1,1, border_mode='valid', activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.1))
model.add(Flatten())
#model.add(Dense(1, W_regularizer=l2(0.03)))
model.add(Dense(1, init='uniform'))

print(model.summary())

# use model checkpointer to save best validation loss
model.compile(loss='mse', optimizer='adam')
checkpointer = ModelCheckpoint(filepath="model.h5", verbose=1, save_best_only=True)
model.fit(X_train, y_train, batch_size=batch_size, validation_split=0.2,  shuffle=True, nb_epoch=10, callbacks=[checkpointer])

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

