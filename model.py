import csv
import cv2
import numpy as np

path1='/home/workspace/CarND-Behavioral-Cloning-P3/data/'
path2='/opt/training/'
images = []
measurements = []
def flip(image,measurment):
    image_flipped = np.fliplr(image)
    measurement_flipped = -measurment
    images.append(image_flipped)
    measurements.append(measurement_flipped)

def load(path):
    
    lines=[]
    with open(path+'driving_log.csv') as csvfile:
        reader=csv.reader(csvfile)
        for line in reader:
            lines.append(line)
        i=0
        for line in lines:
            if i==0:
                i+=1
                continue
            center_path = line[0]
            left_path = line[1]
            right_path = line[2]
            
            filename_center=center_path.split('/')[-1]
            filename_left=left_path.split('/')[-1]
            filename_right=right_path.split('/')[-1]
            
            path_center = path + 'IMG/' + filename_center
            path_left = path + 'IMG/' + filename_left
            path_right = path + 'IMG/' + filename_right

            image_center = cv2.imread(path_center)
            image_left = cv2.imread(path_left)
            image_right = cv2.imread(path_right)
            
            measurment_center = float(line[3])
            measurment_left = float(line[3]) + 0.25
            measurment_right = float(line[3]) - 0.25
            
            images.append(image_center)
            images.append(image_left)
            images.append(image_right)
            
            measurements.append(measurment_center)
            measurements.append(measurment_left)
            measurements.append(measurment_right)
            
            # Flip the image to gain more training data
            flip(image_center,measurment_center)
            flip(image_left,measurment_left)
            flip(image_right,measurment_right)

load(path1)
load(path2)
X_train = np.array(images)
y_train = np.array(measurements)



from keras.models import Sequential
from keras.layers import Lambda, Cropping2D, ELU
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D


model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5,input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))

model.add(Conv2D(filters=24, kernel_size=(5, 5), activation='relu'))
model.add(MaxPooling2D())

model.add(Conv2D(filters=36, kernel_size=(5, 5), activation='relu'))
model.add(MaxPooling2D())

model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D())

model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D())

model.add(Flatten())
model.add(Dense(units=120, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=84, activation='relu'))
model.add(Dense(units=10, activation='relu'))
model.add(Dense(units=1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=3)

model.save('model.h5')
    