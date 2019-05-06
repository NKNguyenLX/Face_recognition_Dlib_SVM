# USAGE
# python encode_faces.py --dataset dataset --encodings encodings.pickle

# import the necessary packages
from imutils import paths
import pandas as pd
import face_recognition
import argparse
import pickle
import cv2
import os

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--dataset", required=True,
	help="path to input directory of faces + images")
ap.add_argument("-e", "--encodings", required=True,
	help="path to serialized db of facial encodings")
ap.add_argument("-d", "--detection-method", type=str, default="cnn",
	help="face detection model to use: either `hog` or `cnn`")
args = vars(ap.parse_args())


df = pd.DataFrame(columns= ['Name','0','1','2','3','4','5','6','7','8','9',
									'10','11','12','13','14','15','16','17','18','19',
									'20','21','22','23','24','25','26','27','28','29',
									'30','31','32','33','34','35','36','37','38','39',
									'40','41','42','43','44','45','46','47','48','49',
									'50','51','52','53','54','55','56','57','58','59',
									'60','61','62','63','64','65','66','67','68','69',
									'70','71','72','73','74','75','76','77','78','79',
									'80','81','82','83','84','85','86','87','88','89',
									'90','91','92','93','94','95','96','97','98','99',
									'100','101','102','103','104','105','106','107','108','109',
									'110','111','112','113','114','115','116','117','118','119',
									'120','121','122','123','124','125','126','127'])

# grab the paths to the input images in our dataset
print("[INFO] quantifying faces...")
imagePaths = list(paths.list_images(args["dataset"]))

# initialize the list of known encodings and known names
knownEncodings = []
knownNames = []

# loop over the image paths
for (i, imagePath) in enumerate(imagePaths):
	# extract the person name from the image path
	print("[INFO] processing image {}/{}".format(i + 1,
		len(imagePaths)))
	name = imagePath.split(os.path.sep)[-2]

	# load the input image and convert it from RGB (OpenCV ordering)
	# to dlib ordering (RGB)
	image = cv2.imread(imagePath)
	rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

	# detect the (x, y)-coordinates of the bounding boxes
	# corresponding to each face in the input image
	boxes = face_recognition.face_locations(rgb,
		model=args["detection_method"])

	# compute the facial embedding for the face
	encodings = face_recognition.face_encodings(rgb, boxes)

	# loop over the encodings
	for encoding in encodings:
		# add each encoding + name to our set of known names and
		# encodings
		knownEncodings.append(encoding)
		knownNames.append(name)
		# print(encoding)
		df = df.append({'Name':name,'0':encoding[0],'1':encoding[1],'2':encoding[2],'3':encoding[3],'4':encoding[4],'5':encoding[5],'6':encoding[6],'7':encoding[7],'8':encoding[8],'9':encoding[9],
								'10':encoding[10],'11':encoding[11],'12':encoding[12],'13':encoding[13],'14':encoding[14],'15':encoding[15],'16':encoding[16],'17':encoding[17],'18':encoding[18],'19':encoding[19],
								'20':encoding[20],'21':encoding[21],'22':encoding[22],'23':encoding[23],'24':encoding[24],'25':encoding[25],'26':encoding[26],'27':encoding[27],'28':encoding[28],'29':encoding[29],
								'30':encoding[30],'31':encoding[31],'32':encoding[32],'33':encoding[33],'34':encoding[34],'35':encoding[35],'36':encoding[36],'37':encoding[37],'38':encoding[38],'39':encoding[39],
								'40':encoding[40],'41':encoding[41],'42':encoding[42],'43':encoding[43],'44':encoding[44],'45':encoding[45],'46':encoding[46],'47':encoding[47],'48':encoding[48],'49':encoding[49],
								'50':encoding[50],'51':encoding[51],'52':encoding[52],'53':encoding[53],'54':encoding[54],'55':encoding[55],'56':encoding[56],'57':encoding[57],'58':encoding[58],'59':encoding[59],
								'60':encoding[60],'61':encoding[61],'62':encoding[62],'63':encoding[63],'64':encoding[64],'65':encoding[65],'66':encoding[66],'67':encoding[67],'68':encoding[68],'69':encoding[69],
								'70':encoding[70],'71':encoding[71],'72':encoding[72],'73':encoding[73],'74':encoding[74],'75':encoding[75],'76':encoding[76],'77':encoding[77],'78':encoding[78],'79':encoding[79],
								'80':encoding[80],'81':encoding[81],'82':encoding[82],'83':encoding[83],'84':encoding[84],'85':encoding[85],'86':encoding[86],'87':encoding[87],'88':encoding[88],'89':encoding[89],
								'90':encoding[90],'91':encoding[91],'92':encoding[92],'93':encoding[93],'94':encoding[94],'95':encoding[95],'96':encoding[96],'97':encoding[97],'98':encoding[98],'99':encoding[99],
								'100':encoding[100],'101':encoding[101],'102':encoding[102],'103':encoding[103],'104':encoding[104],'105':encoding[105],'106':encoding[106],'107':encoding[107],'108':encoding[108],'109':encoding[109],
								'110':encoding[110],'111':encoding[111],'112':encoding[112],'113':encoding[113],'114':encoding[114],'115':encoding[115],'116':encoding[116],'117':encoding[117],'118':encoding[118],'119':encoding[119],
								'120':encoding[120],'121':encoding[121],'122':encoding[122],'123':encoding[123],'124':encoding[124],'125':encoding[125],'126':encoding[126],'127':encoding[127]},ignore_index=True)

save = args["encodings"] + str('.csv')
print(save)
df.to_csv(save)