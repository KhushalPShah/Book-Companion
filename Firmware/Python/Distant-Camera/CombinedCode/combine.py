import cv2
import mediapipe as mp
import pytesseract
from pytesseract import Output
from PyDictionary import PyDictionary
import constants
import argparse
import time

# The below line must be uncommented for executing the code on Windows.
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

def ocr(X,Y,image):

  # For Cropping the image
  XLeft=int(X-constants.CROP_DISTANCE_LEFT)
  X_right=int(X+constants.CROP_DISTANCE_RIGHT)
  Y_up=int(Y-constants.CROP_DISTANCE_BOTTOM)
  Y_down=int(Y-constants.CROP_DISTANCE_TOP)
  image = image[Y_down:Y_up, XLeft:X_right]

  #To scale the image
  scale_percent = constants.ZOOM_PERCENTAGE * 100
  width = int(image.shape[1] * scale_percent / 100)
  height = int(image.shape[0] * scale_percent / 100)
  dsize = (width, height)
  #resize image
  image = cv2.resize(image, dsize)

  d = pytesseract.image_to_data(image, output_type=Output.DICT)
  n_boxes = len(d['text'])
  img = image

  X=constants.CROPPED_IMAGE_CENTRE
  markedWord=""
  for i in range(n_boxes-1, -1, -1):
      if int(d['conf'][i]) > constants.OCR_CONFIDENCE * 100:
          (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
          # print(d['text'][i])
          # print("X=(",x,",",x+w,") Y=(",y,",",y+h,")")
          if X>=x and X<=x+w :
            # print(d['text'][i])
            # print("X=(",x,",",x+w,") Y=(",y,",",y+h,")")
            # if markedWord=="":
            print(d['text'][i])
            markedWord=d['text'][i]
            img = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            break
          
  print(markedWord)
  # dict = PyDictionary() 
  # meaning = dict.meaning(markedWord) 
  # print("meaning=",meaning) 

  # show the output images
  cv2.imshow("Out", img)
  cv2.waitKey(0)

# For webcam input:
prev_x=0
prev_y=0
count=0
hands = mp_hands.Hands(
    min_detection_confidence=constants.FINGER_DETECTION_CONFIDENCE, min_tracking_confidence=constants.FINGER_TRACKING_CONFIDENCE)

#construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",type=str, required=True,help="videos/test.mp4")
args = vars(ap.parse_args())
video = args["video"]


def rescale_frame(frame, percent=75):
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)

cap = cv2.VideoCapture(video)
while cap.isOpened():
  success, image = cap.read()
  if not success:
    break
  # this condition is to be used when there is a need to zoom out.
  # else:
  #   image = rescale_frame(image, percent=80)
  # Flip the image horizontally for a later selfie-view display, and convert
  # the BGR image to RGB.
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  # This following part is to be commented if the orientation of the video is straight, and not rotated.
  image=cv2.rotate(image, cv2.cv2.ROTATE_90_CLOCKWISE)

  # To improve performance, optionally mark the image as not writeable to
  # pass by reference.
  image.flags.writeable = False
  results = hands.process(image)

  # Draw the hand annotations on the image.
  image.flags.writeable = True
  image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
  image_hight, image_width, _ = image.shape
  if results.multi_hand_landmarks:
    for hand_landmarks in results.multi_hand_landmarks:

      cur_x=hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width
      cur_y=hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_hight

      if (((cur_x-prev_x)**2+(cur_y-prev_y)**2)**0.5)<constants.PERMISSIBLE_FINGER_MOVEMENT:
        count=count+1
      else:
        count=0

      if count == constants.MIN_STEADY_FINGER_COUNT:
        ocr(cur_x, cur_y, image)
        count=0

      prev_x=cur_x
      prev_y=cur_y

      mp_drawing.draw_landmarks(
          image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

  cv2.imshow('MediaPipe Hands', image)
  if cv2.waitKey(5) & 0xFF == 27:
    break
hands.close()
cap.release()