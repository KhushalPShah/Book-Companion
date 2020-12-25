import cv2
import argparse
from imutils import paths
import os
import glob

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, required=True,
	help="path to input video to be split into images")

ap.add_argument("-n", "--name", type=str, required=True,
	help="Name of the folder for images")

args = vars(ap.parse_args())

imageName = args["name"]

if os.path.exists("im/"+imageName):
    # Delete all the contents here
    print("Deleting older images")
    filelist = [ f for f in os.listdir("im/"+imageName) if f.endswith(".jpg") ]
    for f in filelist:
        os.remove(os.path.join("im/"+imageName, f))

else:
    # Make a new folder
    print("Making a new folder")
    path = os.path.join(os.getcwd(), "im/"+imageName)
    os.mkdir(path)

videoPath = args["input"]
vidcap = cv2.VideoCapture(videoPath)
success, image = vidcap.read()
# Use the first image. Hence make count as 0.
# When the image is blurred, then the SIFT does not work well.
count = 0
while success:
    success, image = vidcap.read()
    if count%50 == 0:
        cv2.imwrite("im/"+imageName +"/image_%d.jpg" % count, image)    
        print('Saved image ', count)
    count += 1