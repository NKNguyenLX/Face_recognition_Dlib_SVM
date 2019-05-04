#!/usr/bin/env python2
#
# Example to run classifier on webcam stream.
# Brandon Amos & Vijayenthiran
# 2016/06/21
#
# Copyright 2015-2016 Carnegie Mellon University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Contrib: Vijayenthiran
# This example file shows to run a classifier on webcam stream. You need to
# run the classifier.py to generate classifier with your own dataset.
# To run this file from the openface home dir:
# ./demo/classifier_webcam.py <path-to-your-classifier>


import time

start = time.time()

import argparse
import cv2
import os
import pickle
import sys
import time
import imutils
import dlib

import numpy as np
np.set_printoptions(precision=2)
from sklearn.mixture import GMM
import face_recognition
from imutils.video import VideoStream
import util

fileDir = os.path.dirname(os.path.realpath(__file__))
modelDir = os.path.join(fileDir, '..', 'models')
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')


def infer(img, args):
    with open(args.classifierModel, 'r') as f:
        if sys.version_info[0] < 3:
                (le, clf) = pickle.load(f)  # le - label and clf - classifer
        else:
                (le, clf) = pickle.load(f, encoding='latin1')  # le - label and clf - classifer

    # Face recognition
    bbs = face_recognition.face_locations(img, model=args.detectionMethod)
    print(bbs)
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
        if args.verbose:
            print("Prediction took {} seconds.".format(time.time() - start))
            pass
        # print("Predict {} with {:.2f} confidence.".format(person.decode('utf-8'), confidence))
        if isinstance(clf, GMM):
            dist = np.linalg.norm(rep - clf.means_[maxI])
            print("  + Distance from the mean: {}".format(dist))
            pass
    return (persons, confidences ,bbs)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dlibFacePredictor',
        type=str,
        help="Path to dlib's face predictor.",
        default=os.path.join(
            dlibModelDir,
            "shape_predictor_68_face_landmarks.dat"))
    parser.add_argument(
        '--networkModel',
        type=str,
        help="Path to Torch network model.",
        default=os.path.join(
            openfaceModelDir,
            'nn4.small2.v1.t7'))
    parser.add_argument('--imgDim', type=int,
                        help="Default image dimension.", default=96)
    parser.add_argument(
        '--captureDevice',
        type=int,
        default=0,
        help='Capture device. 0 for latopinfer webcam and 1 for usb webcam')
    parser.add_argument("-d", "--detectionMethod", type=str, default="cnn",
	    help="face detection model to use: either `hog` or `cnn`")
    parser.add_argument('--width', type=int, default=320)
    parser.add_argument('--height', type=int, default=240)
    parser.add_argument('--threshold', type=float, default=0.7)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument(
        'classifierModel',
        type=str,
        help='The Python pickle representing the classifier. This is NOT the Torch network model, which can be set with --networkModel.')

    args = parser.parse_args()

    # align = openface.AlignDlib(args.dlibFacePredictor)
    net = util.TorchNeuralNet(
        args.networkModel,
        imgDim=args.imgDim,
        cuda= 1,
        # cuda=args.cuda
        )

    # Capture device. Usually 0 will be webcam and 1 will be usb cam.
    vs = VideoStream(src=0).start()
    # vs = VideoStream("rtsp://admin:Redbean2018@192.168.1.211/Streaming/channels/1").start()

    confidenceList = []
    while True:
        # Start time
        start = time.time()

        frame = vs.read()
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = imutils.resize(frame, width=args.width)

        persons, confidences, bbs = infer(frame, args)
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

        cv2.imshow('', frame)
        # quit the program on the press of key 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # End time
        end = time.time()

        # Calculate frames per second
        fps  = 1 / (end - start)
        print ("Estimated frames per second : {0}".format(fps))
 
# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()


