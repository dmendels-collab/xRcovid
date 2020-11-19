from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import imutils
from imutils import paths
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to trained model model")
ap.add_argument("-i", "--images", required=True,
	help="path to input images")
args = vars(ap.parse_args())

print("[INFO] loading model...")
model = load_model(args["model"])

imagePaths = sorted(list(paths.list_images(args["images"])))

pos_count = 0
neg_count = 0
for image_path in imagePaths:
	image = cv2.imread(image_path)
	orig = image.copy()
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	image = cv2.resize(image, (256, 256))
	image = imutils.rotate(image, 180)
	image = image.astype("float") / 255.0
	mu = np.mean(image)
	sigma = np.mean(image)
	image = (image - mu) / sigma
	image = img_to_array(image)
	image = np.expand_dims(image, axis=0)

	(negative, positive) = model.predict(image)[0]
	if negative > positive:
		neg_count += 1
	elif positive > negative:
		pos_count += 1
	# label = "Positive" if positive > negative else "Negative"
	# proba = positive if positive > negative else negative
	# label = "{}: {:.2f}%".format(label, proba * 100)
	# output = imutils.resize(orig, width=400)
	# cv2.putText(output, label, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
	# cv2.imshow("Output", output)
	# cv2.waitKey(0)

print(pos_count)
print(neg_count)