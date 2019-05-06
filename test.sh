## face detection
# ../util/align-dlib.py ./test/ align outerEyesAndNose ./test-aligned-images/ --size 96

## Face recognition
# ../demos/classifier.py infer ./generated-embeddings/classifier.pkl ./test/unknow/2.png

# ## Video stream 
# python svm_classifier_video.py ./models/classifier.pkl
# python svm_classifier_video_align.py ./models/classifier.pkl
# python recognize_faces_video_align.py ./models/classifier.pkl

# GUI
python svm_classifier_gui.py --output output

