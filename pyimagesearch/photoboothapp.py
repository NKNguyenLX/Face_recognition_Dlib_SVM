# import the necessary packages
from __future__ import print_function
from PIL import Image
from PIL import ImageTk
import Tkinter as tki
import threading
import datetime
import imutils
import cv2
import os

import time

import pickle
import sys
import dlib
from gtts import gTTS

import numpy as np
np.set_printoptions(precision=2)
import face_recognition
from imutils.video import VideoStream

class PhotoBoothApp:
	def __init__(self, vs, outputPath):
		# store the video stream object and output path, then initialize
		# the most recently read frame, thread for reading frames, and
		# the thread stop event
		self.vs = vs
		self.outputPath = outputPath
		self.frame = None
		self.thread = None
		self.stopEvent = None

		# initialize the root window and image panel
		self.root = tki.Tk()
		self.panel = None
		
		

		# create a button, that when pressed, will take the current
		# frame and save it to file
		btn = tki.Button(self.root, text="Identify",font='Helvetica 12',
			command=self.takeSnapshot)
		btn.pack(side="bottom", fill="both", expand="yes", padx=10,
			pady=10)


		self.label = tki.Label(self.root, text = "Person name: ",font='Helvetica 12')
		self.label2 = tki.Label(self.root, text = "",font='Helvetica 14 bold')
		self.label2.pack(side="bottom", fill="both", expand="yes", padx=10,
			pady=2)
		self.label.pack(side="bottom", fill="both", expand="yes", padx=10,
			pady=2)
		

		# start a thread that constantly pools the video sensor for
		# the most recently read frame
		self.stopEvent = threading.Event()
		self.thread = threading.Thread(target=self.videoLoop, args=())
		self.thread.start()

		# set a callback to handle when the window is closed
		self.root.wm_title("Face recognition")
		self.root.wm_protocol("WM_DELETE_WINDOW", self.onClose)

	def infer(self,img):
		with open("./models/classifier.pkl", 'r') as f:
			if sys.version_info[0] < 3:
					(le, clf) = pickle.load(f)  # le - label and clf - classifer
			else:
					(le, clf) = pickle.load(f, encoding='latin1')  # le - label and clf - classifer

		# Face recognition
		bbs = face_recognition.face_locations(img, model="cnn")
		print("Bounding box:"+str(bbs))
		reps = face_recognition.face_encodings(img, bbs)

		persons = []
		confidences = []
		for rep in reps:
			try:
				rep = rep.reshape(1, -1)
			except:
				print ("No Face detected")
				return (None, None)
			start = time.time()
			predictions = clf.predict_proba(rep).ravel()
			# print (predictions)
			maxI = np.argmax(predictions)
			# max2 = np.argsort(predictions)[-3:][::-1][1]
			persons.append(le.inverse_transform(maxI))
			# print (str(le.inverse_transform(max2)) + ": "+str( predictions [max2]))
			# ^ prints the second prediction
			confidences.append(predictions[maxI])
			print("Prediction took {} seconds.".format(time.time() - start))

		return (persons, confidences ,bbs)

	def takeSnapshot(self):
		# Start time
		start = time.time()

		self.label2.configure(text = "")
		# grab the current timestamp and use it to construct the
		# output path
		ts = datetime.datetime.now()
		filename = "{}.jpg".format(ts.strftime("%Y-%m-%d_%H-%M-%S"))
		p = os.path.sep.join((self.outputPath, filename))

		# save the file
		frame = self.frame.copy()
		cv2.imwrite(p, frame)
		print("[INFO] saved {}".format(filename))

		frame = imutils.resize(frame, width=320)

		persons, confidences, bbs = self.infer(frame)
		print ("P: " + str(persons) + " C: " + str(confidences))
		try:
			# append with two floating point precision
			confidenceList.append('%.2f' % confidences[0])
		except:
			# If there is no face detected, confidences matrix will be empty.
			# We can simply ignore it.
			pass

		for i, c in enumerate(confidences):
			if c <= 0.5:  # 0.5 is kept as threshold for known face.
				persons[i] = "Unknown"
		
		# Update label
		self.label2.configure(text = str(persons[i]))

		# Print the person name and conf value on the frame next to the person
		# Also print the bounding box

		for ((top, right, bottom, left), name) in zip(bbs, persons):
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

		# End time
		end = time.time()

		# Calculate frames per second
		# fps  = 1 / (end - start)
		time_ = end - start
		print ("Estimated processing time: {0}".format(time_))

		# if persons != None:
			# Speech name
		tts = gTTS(text=persons[0], lang='vi')
		tts.save("name.mp3")
		os.system("mpg321 name.mp3")

	def videoLoop(self):
		# DISCLAIMER:
		# I'm not a GUI developer, nor do I even pretend to be. This
		# try/except statement is a pretty ugly hack to get around
		# a RunTime error that Tkinter throws due to threading
		try:
			# keep looping over frames until we are instructed to stop
			while not self.stopEvent.is_set():
				# grab the frame from the video stream and resize it to
				# have a maximum width of 320 pixels
				self.frame = self.vs.read()
				self.frame = imutils.resize(self.frame, width=320)

				persons, confidences, bbs = self.infer(self.frame)

				for ((top, right, bottom, left), name) in zip(bbs, persons):
					# rescale the face coordinates
					r = 1
					top = int(top * r)
					right = int(right * r)
					bottom = int(bottom * r)
					left = int(left * r)

					# draw the predicted face name on the image
					cv2.rectangle(self.frame, (left, top), (right, bottom),
						(0, 255, 0), 2)
					
				self.frame = imutils.resize(self.frame, width=640)
				# OpenCV represents images in BGR order; however PIL
				# represents images in RGB order, so we need to swap
				# the channels, then convert to PIL and ImageTk format
				image = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
				image = Image.fromarray(image)
				image = ImageTk.PhotoImage(image)

				


				# if the panel is not None, we need to initialize it
				if self.panel is None:
					self.panel = tki.Label(image=image)
					self.panel.image = image
					self.panel.pack(side="left", padx=10, pady=10)
		
				# otherwise, simply update the panel
				else:
					self.panel.configure(image=image)
					self.panel.image = image

		except RuntimeError, e:
			print("[INFO] caught a RuntimeError")

	def onClose(self):
		# set the stop event, cleanup the camera, and allow the rest of
		# the quit process to continue
		print("[INFO] closing...")
		self.stopEvent.set()
		self.vs.stop()
		self.root.quit()

	