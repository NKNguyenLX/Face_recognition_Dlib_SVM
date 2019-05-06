import argparse
import cv2
import numpy as np
import imutils
from scipy.misc import imresize
from moviepy.editor import VideoFileClip
from IPython.display import HTML

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",required=True,
                help="path to the (optional) video file")
ap.add_argument("-n", "--className",required=True,
                help="input class name, require a folder of the class to be created")
args = vars(ap.parse_args())

camera = cv2.VideoCapture(args["video"])

save_path = str(args["className"]) + "/"
count = 0
frame_drop = 4
rotate_angle = 90

while (camera.isOpened()):
    # Grab the current frame
    (grabbed, frame) = camera.read()
    frame = imutils.rotate(frame, rotate_angle)

    # If we are viewing a video and we did not grab a frame, then we have reached the end of the video
    if count % frame_drop == 0:

        cv2.imwrite(save_path + str(int(count/frame_drop)) + ".jpg",frame)

    count += 1

    cv2.imshow("Frame", frame)
    quit_ = cv2.waitKey(1)
    if quit_ == 'q':
        break

print("[INFO] finish create dataset")
# When everything done, release the video capture object
camera.release()
 
# Closes all the frames
cv2.destroyAllWindows()
