# Face_recognition_Dlib_SVM
This work is based on Dlib face recognition and python face_recognition package. The image goes through the CNN model to produce bouding box and 128-d vector for each face in the image. The output is then feed in an RadialSvm model for classification and label the bouding box.<br/>
<br/>
For more information about this algorithm check out these websites:<br/>
https://www.pyimagesearch.com/2018/06/18/face-recognition-with-opencv-python-and-deep-learning/<br/>
http://blog.dlib.net/2017/02/high-quality-face-recognition-with-deep.html<br/>
https://medium.com/@ageitgey/machine-learning-is-fun-part-4-modern-face-recognition-with-deep-learning-c3cffc121d78<br/>
<br/>
![alt text](https://github.com/NKNguyenLX/Face_recognition_Dlib_SVM/blob/master/dest.jpg)<br/>
## Installation
Install Dlib with GPU support
```
git clone https://github.com/davisking/dlib.git
cd dlib
mkdir build
cd build
cmake .. -DDLIB_USE_CUDA=1 -DUSE_AVX_INSTRUCTIONS=1
cmake --build .
cd ..
python setup.py install --yes USE_AVX_INSTRUCTIONS --yes DLIB_USE_CUDA
```
Install face_recognition package and imutils
```
pip install face_recognition
pip install imutils
```
## Usage
### Prepare data set
Prepare video for face recognition and put in datasetTool/rawData<br/>
Run run_dataset.sh to generate image for recognition. Copy images folder to dataset folder.
```
./datasetTool/run_dataset.sh
```
### Train the model
```
./train.sh
```
### Validate the result
```
./test.sh
```
You can view the result with a gui by uncomment *python svm_classifier_gui.py --output output* in test.sh 
