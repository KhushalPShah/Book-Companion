import numpy as np
import cv2
import os

import pytesseract
from pytesseract import Output
from PyDictionary import PyDictionary

from gtts import gTTS
#pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# These are the number of pixels, from the centre of the detected finger co-ordinate, for cropping the image.
CROP_DISTANCE_LEFT = 100
CROP_DISTANCE_RIGHT = 100
# The CROP_DISTANCE_BOTTOM is the pixels from the detected finger co-ordinate to the actual word's bottom co-ordinate.
# This is to take into account the sharpness of the user's finger tip.
CROP_DISTANCE_BOTTOM = -5
CROP_DISTANCE_TOP = 50

ZOOM_PERCENTAGE = 1
OCR_CONFIDENCE = 0.6

def ocr(X,Y,image,width):
  # For Cropping the image
  X_left = int(X-CROP_DISTANCE_LEFT)
  X_right = int(X+CROP_DISTANCE_RIGHT)
  print(X,Y)
  #cv2.imshow("image",image)
  if X_left < 0:
      X_right = X_right + X_left
      X_left = 0
  if X_right > width:
      X_left = X_left - X_right + width
      X_right = width
  Y_up=int(Y-CROP_DISTANCE_BOTTOM)
  if Y_up < 0:
      Y_up = 0
  Y_down=int(Y-CROP_DISTANCE_TOP)
  image = image[Y_down:Y_up, X_left:X_right]
  #print("X coordinate of finger", X)
  #print("Y coordinate of finger", Y)
  #print("leftmost coordinate", X_left)
  #print("rightmost coordinate", X_right)

  #To scale the image
  scale_percent = ZOOM_PERCENTAGE * 100
  width = int(image.shape[1] * scale_percent / 100)
  height = int(image.shape[0] * scale_percent / 100)
  dsize = (width, height)
  #resize image
  image = cv2.resize(image, dsize)
  X = width/2
  #X = X*ZOOM_PERCENTAGE
  #Y = Y*ZOOM_PERCENTAGE
  print("Word Coordinates", X)
  #print(Y)
  #cv2.imshow('temp', image)

  d = pytesseract.image_to_data(image, output_type=Output.DICT)
  n_boxes = len(d['text'])
  img = image

  markedWord=""
  for i in range(n_boxes-1, -1, -1):
      if int(d['conf'][i]) > OCR_CONFIDENCE * 100:
          (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
          #print("X=(",x,",",x+w,") Y=(",y,",",y+h,")")
          #print(d['text'][i])
          #putting it outside loop for debugging
          img = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
          if X >= x and X <= x+w :
            # print(d['text'][i])
            # print("X=(",x,",",x+w,") Y=(",y,",",y+h,")")
            # if markedWord=="":
            #print(d['text'][i])
            markedWord=d['text'][i]
            #img = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            break

  print(markedWord)
  language='en'
  if markedWord=="" :
      myobj = gTTS(text="word cannot be found, please place your finger properly", lang=language, slow=False)
      myobj.save("welcome.mp3")
      os.system("mpg123 welcome.mp3")
      return
  
  punc = '''!()-[]{};:'"\, <>./?@#$%^&*_~'''
  
  # Removing punctuations in string
  # Using loop + punctuation string
  for ele in markedWord: 
    if ele in punc: 
        markedWord = markedWord.replace(ele, "")
  dict = PyDictionary()
  meaning = dict.meaning(markedWord)

  print("meaning=",meaning["Noun"][0])
  meaningfinal=meaning["Noun"][0]
  myobj = gTTS(text=meaningfinal, lang=language, slow=False)
  myobj.save("welcome.mp3")
  os.system("mpg123 welcome.mp3")
  # show the output images
  cv2.imshow("Out", img)
  cv2.waitKey(0)


prev_x=0
prev_y=0
count=0

vid = cv2.VideoCapture('test4.mp4')
i = 0
while(True):
    success, frame = vid.read()
    if not success:
        break
    
    original = frame.copy()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array([22, 93, 0], dtype="uint8")
    upper = np.array([45, 255, 255], dtype="uint8")
    mask = cv2.inRange(frame, lower, upper)

    cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cnts = sorted(cnts, key=lambda x: cv2.contourArea(x), reverse = True)
    # for c in cnts:
    #     print(cv2.contourArea(c))

    centre=(0,0)
    done = 0
    for c in cnts:
        perimeter = cv2.arcLength(c,False)
        if (perimeter>100):
            if done == 1:
                break
            done = 1
        
            x,y,w,h = cv2.boundingRect(c)
            cv2.rectangle(original, (x, y), (x + w, y + h), (36,255,12), 2)
    
    
            # compute the center of the contour
            left = tuple(c[c[:, :, 0].argmin()][0])
            right = tuple(c[c[:, :, 0].argmax()][0])
            top = tuple(c[c[:, :, 1].argmin()][0])
            centre=(int((left[0]+right[0])/2),int(top[1]))
            cx,cy=int(centre[0]),int(centre[1])
            cx = cx - 10    # Adding this offset since the contour detected is lot on the right side.
            # frame[cx,cy]=[0,0,255]
            # print(centre)
            if (((cx-prev_x)*2+(cy-prev_y)*2)*0.5) < 3:
                count=count+1
            else:
                count=0
            
            cv2.circle(original,centre,1,(255,0,0),1)
            prev_x = cx
            prev_y = cy

    # cv2.imshow('mask', mask)
    if count == 60:
        image_hight, image_width, _ = original.shape
        ocr(centre[0], centre[1], original, image_width)
        count = 0

    cv2.imshow('original', original)
    cv2.waitKey(1)
    # cv2.imwrite('output.jpeg',original)
    # if (i+1)%50 == 0:
    #     cv2.imwrite("op{0}.jpeg".format(i),original)
    #     image_hight, image_width, _ = original.shape
    #     ocr(centre[0], centre[1], original, image_width)
    # i = i+1
    

vid.release()
cv2.destroyAllWindows()