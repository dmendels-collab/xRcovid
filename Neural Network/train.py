from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard, ModelCheckpoint
from imutils import paths, rotate
import numpy as np
import argparse
import random
import cv2
import matplotlib
matplotlib.use("Agg")


class CovidNet:
	@staticmethod
	def build():
		model = Sequential()
		inputShape = (256, 256, 1)
		model.add(Conv2D(32, (3, 3), input_shape=inputShape, activation="relu"))
		model.add(Dropout(0.25))
		model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

		model.add(Conv2D(64, (3, 3), activation="relu"))
		model.add(Dropout(0.25))
		model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

		model.add(Conv2D(128, (3, 3), activation="relu"))
		model.add(Dropout(0.25))
		model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

		model.add(Flatten())
		model.add(Dense(512, activation="relu"))
		model.add(Dropout(0.5))
		model.add(Dense(2, activation="softmax"))
		return model

model = CovidNet.build()
model.summary()

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
ap.add_argument("-m", "--model", required=True, help="path to output model")
args = vars(ap.parse_args())

EPOCHS = 300
INIT_LR = 0.5 * 1e-3
BS = 16
print("[INFO] loading images...")
data = []
labels = []
imagePaths = sorted(list(paths.list_images(args["dataset"])))
random.seed(24)
random.shuffle(imagePaths)


for imagePath in imagePaths:
	image = cv2.imread(imagePath)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	image = cv2.resize(image, (256, 256))
	image = rotate(image, 180)
	image = img_to_array(image)
	data.append(image)
	if 'POSITIVE' in imagePath:
		label = 1
	elif 'NEGATIVE' in imagePath:
		label = 0
	else:
		print(' ERROR : NOT POSITIVE OR NEGATIVE. ')
	labels.append(label)

data = np.array(data, dtype="float")
labels = np.array(labels)
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.1, random_state=24)
trainY = to_categorical(trainY, num_classes=2)
testY = to_categorical(testY, num_classes=2)

aug = ImageDataGenerator(width_shift_range=0.1,
						 brightness_range=[0.9, 1.1],
						 height_shift_range=0.1,
						 zoom_range=0.4,
						 # horizontal_flip=True,
						 # rotation_range=10,
						 # shear_range=0.2,
						 fill_mode="nearest",
						 samplewise_center=True, samplewise_std_normalization=True)

std_flow = ImageDataGenerator(samplewise_center=True, samplewise_std_normalization=True)

print("[INFO] compiling model...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt,
			  metrics=["accuracy"])

# early_stop = EarlyStopping(monitor="val_loss", patience=5)
# reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=100, min_lr=0.000001)
mcp_save_best = ModelCheckpoint('best.h5', save_best_only=True, monitor='val_loss', mode='min')
mcp_save_last = ModelCheckpoint('last.h5', save_best_only=False)
tensorboard_cb = TensorBoard(log_dir='logs')

print("[INFO] training network...")
try:
	H = model.fit(x=aug.flow(trainX, trainY, batch_size=BS),
				  validation_data=std_flow.flow(testX, testY), steps_per_epoch=len(trainX) // BS,
				  epochs=EPOCHS, verbose=1, callbacks=[tensorboard_cb, mcp_save_best, mcp_save_last])
except KeyboardInterrupt:
	save_path = 'covidNet.ckpt'
	model.save(save_path)
	print('Output saved to: "{}./*"'.format(save_path))

print("[INFO] saving model...")
model.save(args["model"], save_format="h5")
print('Output saved to: "{}./*"'.format(args["model"]))
