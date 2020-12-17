# import the necessary packages
from PIL import Image
import pytesseract
from pytesseract import Output
import argparse
import cv2
import os
from PyDictionary import PyDictionary

cap = cv2.VideoCapture(0)
while cap.isOpened():
    success, image = cap.read()
    if not success:
        break
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # check to see if we should apply thresholding to preprocess the
    # image
    #if args["preprocess"] == "thresh":
    gray = cv2.threshold(gray, 0, 255,
        cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    # filename = "{}.png".format(os.getpid())
    # cv2.imwrite(filename, gray)
    # # load the image as a PIL/Pillow image, apply OCR, and then delete
    # # the temporary file
    # text = pytesseract.image_to_string(Image.open(filename))
    d = pytesseract.image_to_data(image, output_type=Output.DICT)
    n_boxes = len(d['text'])
    for i in range(n_boxes):
        if int(d['conf'][i]) > 60:
            (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
            # if GivenX>=x and GivenX<=x+w and y-GivenY==1:
            #     print(d['text'][i])
            #     meaning = dict.meaning(d['text'][i]) 
            #     print("meaning=",meaning) 
                #print(x,y)
            img = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # show the output images
    cv2.imshow("Out", img)
    if cv2.waitKey(5) & 0xFF == 27:
        break
cap.release()