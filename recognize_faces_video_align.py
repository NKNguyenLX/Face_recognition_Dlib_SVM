# USAGE
# python recognize_faces_video.py --encodings encodings.pickle
# python recognize_faces_video.py --encodings encodings.pickle --output output/jurassic_park_trailer_output.avi --display 0

# import the necessary packages
from imutils.video import VideoStream
import face_recognition
import argparse
import imutils
import pickle
import time
import cv2
import util
import sys

import numpy as np
from sklearn.mixture import GMM

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", type=str,
	help="path to output video")
ap.add_argument("-y", "--display", type=int, default=1,
	help="whether or not to display output frame to screen")
ap.add_argument("-d", "--detectionMethod", type=str, default="cnn",
	help="face detection model to use: either `hog` or `cnn`")
ap.add_argument('--imgDim', type=int,
					help="Default image dimension.", default=96)
ap.add_argument(
	'classifierModel',
	type=str,
	help='The Python pickle representing the classifier. This is NOT the Torch network model, which can be set with --networkModel.')
args = ap.parse_args()

# load the known faces and embeddings
print("[INFO] loading encodings...")

# initialize the video stream and pointer to output video file, then
# allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
# vs = VideoStream("rtsp://admin:Redbean2018@192.168.1.211/Streaming/channels/1").start()
writer = None
time.sleep(2.0)

# create align object
align = util.AlignDlib('/home/nknight/works/git/PIF_Redbean/face_recognizer/face-recognition-opencv/models/dlib/shape_predictor_68_face_landmarks.dat')

# Load model
with open(args.classifierModel, 'r') as f:
	if sys.version_info[0] < 3:
			(le, clf) = pickle.load(f)  # le - label and clf - classifer
	else:
			(le, clf) = pickle.load(f, encoding='latin1')  # le - label and clf - classifer

# loop over frames from the video file stream
while True:
	# grab the frame from the threaded video stream
	frame = vs.read()
	
	persons = []
	confidences = []

	# convert the input frame from BGR to RGB then resize it to have
	# a width of 750px (to speedup processing)
	rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	# rgb = imutils.resize(frame, width=750)
	r = frame.shape[1] / float(rgb.shape[1])

	# detect the (x, y)-coordinates of the bounding boxes
	# corresponding to each face in the input frame, then compute
	# the facial embeddings for each face
	boxes = face_recognition.face_locations(rgb,
		model="cnn")
	# encodings = face_recognition.face_encodings(rgb, boxes)
	names = []

	# loop over the facial embeddings
	for box in boxes:
		# attempt to match each face in the input image to our known
		# encodings
		alignFace = align.align(
			args.imgDim,
			rgb,
			box,
			landmarkIndices=util.AlignDlib.OUTER_EYES_AND_NOSE
			)

		encodings = face_recognition.face_encodings(alignFace)

		for encoding in encodings:

			predictions = clf.predict_proba(encoding).ravel()
			# print (predictions)
			maxI = np.argmax(predictions)
			# max2 = np.argsort(predictions)[-3:][::-1][1]
			persons.append(le.inverse_transform(maxI))
			# print (str(le.inverse_transform(max2)) + ": "+str( predictions [max2]))
			# ^ prints the second prediction
			confidences.append(predictions[maxI])
			# print("Predict {} with {:.2f} confidence.".format(person.decode('utf-8'), confidence))
			if isinstance(clf, GMM):
				dist = np.linalg.norm(encoding - clf.means_[maxI])
				print("  + Distance from the mean: {}".format(dist))
				pass
			
	print ("P: " + str(persons) + " C: " + str(confidences))
	try:
		# append with two floating point precision
		confidenceList.append('%.2f' % confidences[0])
	except:
		# If there is no face detected, confidences matrix will be empty.
		# We can simply ignore it.
		pass

	for i, c in enumerate(confidences):
		if c <= args.threshold:  # 0.5 is kept as threshold for known face.
			persons[i] = "Unknown"

	# Print the person name and conf value on the frame next to the person
	# Also print the bounding box

	# # Openface
	# for idx,person in enumerate(persons):
	#     cv2.rectangle(frame, (bbs[idx].left(), bbs[idx].top()), (bbs[idx].right(), bbs[idx].bottom()), (0, 255, 0), 2)
	#     cv2.putText(frame, "{} @{:.2f}".format(person, confidences[idx]),
	#                 (bbs[idx].left(), bbs[idx].bottom()+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

	# Face recognition
	for ((top, right, bottom, left), name) in zip(boxes, persons):
		# rescale the face coordinates
		r = 1
		top = int(top * r)
		right = int(right * r)
		bottom = int(bottom * r)
		left = int(left * r)

		# draw the predicted face name on the image
		cv2.rectangle(frame, (left, top), (right, bottom),
			(0, 255, 0), 2)
		y = top - 15 if top - 15 > 15 else top + 15
		cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
			0.75, (0, 255, 0), 2)

	cv2.imshow('', frame)



	# check to see if we are supposed to display the output frame to
	# the screen
	if args.display > 0:
		cv2.imshow("Frame", frame)
		key = cv2.waitKey(1) & 0xFF

		# if the `q` key was pressed, break from the loop
		if key == ord("q"):
			break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()

# check to see if the video writer point needs to be released
if writer is not None:
	writer.release()
