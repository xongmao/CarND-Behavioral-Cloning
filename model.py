import csv
import cv2
from scipy import ndimage
import numpy as np
import sklearn
import random

# two lap of center lane driving
lines = []
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

# one times of driving from right sides to center
lines2 = []
with open('data2/driving_log.csv') as csvfile2:
    reader2 = csv.reader(csvfile2)
    for line in reader2:
        lines2.append(line)
tmp2 = lines2.copy()
# Because of the restart the new record, delete the data at the right sides with a steering angle zero
k=0
for line in lines2:
    measurement = float(line[3])
    if measurement!=0.0:
        break
    tmp2.pop(k)
    k+=1
del lines2

# one times of driving from left sides to center
lines3 = []
with open('data3/driving_log.csv') as csvfile3:
    reader3 = csv.reader(csvfile3)
    for line in reader3:
        lines3.append(line)
tmp3 = lines3.copy()
k=0
for line in lines3:
    measurement = float(line[3])
    if measurement!=0.0:
        break
    tmp3.pop(k)
    k+=1
del lines3

# one times of driving from right sides to center at a great band
lines4 = []
with open('data4/driving_log.csv') as csvfile4:
    reader4 = csv.reader(csvfile4)
    for line in reader4:
        lines4.append(line)
tmp4 = lines4.copy()
k=0
for line in lines4:
    measurement = float(line[3])
    if measurement!=0.0:
        break
    tmp4.pop(k)
    k+=1
del lines4

# one times of driving from left sides to center at a great band
lines5 = []
with open('data5/driving_log.csv') as csvfile5:
    reader5 = csv.reader(csvfile5)
    for line in reader5:
        lines5.append(line)
tmp5 = lines5.copy()
k=0
for line in lines5:
    measurement = float(line[3])
    if measurement!=0.0:
        break
    tmp5.pop(k)
    k+=1
del lines5

# Increase the number of data from the left or right sides to the center to half the number of data of two lap of center lane driving
l1 = len(lines)
l2 = len(tmp2)
l3 = len(tmp3)
l4 = len(tmp4)
l5 = len(tmp5)

l1 = int(l1/2)
tot = int(l1/4)
x2 = 0
r = tot%l2
if r==0:
    x2 = int(tot/l2)
else:
    x2 = int(tot/l2) + 1

lines2 = []
for line in tmp2:
    for i in range(0, x2):
        lines2.append(line)
del tmp2

x3 = 0
r = tot%l3
if r==0:
    x3 = int(tot/l3)
else:
    x3 = int(tot/l3) + 1

lines3 = []
for line in tmp3:
    for i in range(0, x3):
        lines3.append(line)
del tmp3

x4 = 0
r = tot%l4
if r==0:
    x4 = int(tot/l4)
else:
    x4 = int(tot/l4) + 1

lines4 = []
for line in tmp4:
    for i in range(0, x4):
        lines4.append(line)
del tmp4

x5 = 0
r = tot%l5
if r==0:
    x5 = int(tot/l5)
else:
    x5 = int(tot/l5) + 1

lines5 = []
for line in tmp5:
    for i in range(0, x5):
        lines5.append(line)
del tmp5

lines.extend(lines2)
lines.extend(lines3)
lines.extend(lines4)
lines.extend(lines5)
del lines2, lines3, lines4, lines5

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(lines, test_size=0.2)

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1:
        random.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            measurements = []
            s = set()
            for batch_sample in batch_samples:
                measurement = float(batch_sample[3])
                s.add(measurement)
            d = {}
            s2=s.copy()
            for i in range(0, len(s)):
                d[s2.pop()]=0
            del s, s2
            for batch_sample in batch_samples:
                measurement = float(batch_sample[3])
                d[measurement]+=1
            # Count the number of steering angles zero
            m=0
            for key, value in d.items():
                if m < value:
                    m=value
            m = int(m/10)
            for batch_sample in batch_samples:
                for i in range(3):
                    source_path = batch_sample[i]
                    if i%3==0:
                        if source_path[-40]=='2':
                            filename = source_path[-34:]
                            current_path = 'data2/IMG/' + filename
                        elif source_path[-40]=='3':
                            filename = source_path[-34:]
                            current_path = 'data3/IMG/' + filename
                        elif source_path[-40]=='4':
                            filename = source_path[-34:]
                            current_path = 'data4/IMG/' + filename
                        elif source_path[-40]=='5':
                            filename = source_path[-34:]
                            current_path = 'data5/IMG/' + filename
                        else:
                            filename = source_path[-34:]
                            current_path = 'data/IMG/' + filename
                    elif i%3==1:
                        if source_path[-38]=='2':
                            filename = source_path[-32:]
                            current_path = 'data2/IMG/' + filename
                        elif source_path[-38]=='3':
                            filename = source_path[-32:]
                            current_path = 'data3/IMG/' + filename
                        elif source_path[-38]=='4':
                            filename = source_path[-32:]
                            current_path = 'data4/IMG/' + filename
                        elif source_path[-38]=='5':
                            filename = source_path[-32:]
                            current_path = 'data5/IMG/' + filename
                        else:
                            filename = source_path[-32:]
                            current_path = 'data/IMG/' + filename
                    else:
                        if source_path[-39]=='2':
                            filename = source_path[-33:]
                            current_path = 'data2/IMG/' + filename
                        elif source_path[-39]=='3':
                            filename = source_path[-33:]
                            current_path = 'data3/IMG/' + filename
                        elif source_path[-39]=='4':
                            filename = source_path[-33:]
                            current_path = 'data4/IMG/' + filename
                        elif source_path[-39]=='5':
                            filename = source_path[-33:]
                            current_path = 'data5/IMG/' + filename
                        else:
                            filename = source_path[-33:]
                            current_path = 'data/IMG/' + filename
                    
                    image = ndimage.imread(current_path)
                    measurement = float(batch_sample[3])
                    # Increase the number of steering angles data to the same order of magnitude as at least the numble of the steering angle zero
                    if d[measurement]< m:
                        x = 0
                        r = m%d[measurement]
                        if r==0:
                            x = int(m/d[measurement])
                        else:
                            x = int(m/d[measurement]) + 1
                        for j in range(0, x):
                            images.append(image)
                            if i%3==0:
                                measurements.append(measurement)
                            elif i%3==1:
                                measurements.append(measurement+0.2)
                            else:
                                measurements.append(measurement-0.2)
                    else:
                        images.append(image)
                        if i%3==0:
                            measurements.append(measurement)
                        elif i%3==1:
                            measurements.append(measurement+0.2)
                        else:
                            measurements.append(measurement-0.2)
            del d
            augmented_images, augmented_measurements = [], []
            for image, measurement in zip(images, measurements):
                augmented_images.append(image)
                augmented_measurements.append(measurement)
                augmented_images.append(cv2.flip(image, 1))
                augmented_measurements.append(measurement*-1.0)

            X_train = np.array(augmented_images)
            y_train = np.array(augmented_measurements)
            yield sklearn.utils.shuffle(X_train, y_train)

train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

from keras.models import Sequential
from keras.layers import Lambda, Flatten, Dense, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25), (0,0))))
model.add(Convolution2D(24,5,5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(36,5,5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(48,5,5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(64,3,3, activation="relu"))
model.add(Convolution2D(64,3,3, activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.5))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, samples_per_epoch= len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=5)
model.save('model.h5')
exit()