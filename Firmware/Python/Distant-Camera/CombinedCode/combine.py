import cv2
import mediapipe as mp
import pytesseract
from pytesseract import Output
from PyDictionary import PyDictionary
import constants
import argparse
import time

# The below line must be uncommented for executing the code on Windows.
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

def ocr(X,Y,image,width):
  # For Cropping the image
  X_left = int(X-constants.CROP_DISTANCE_LEFT)
  X_right = int(X+constants.CROP_DISTANCE_RIGHT)

  if X_left < 0:
      X_right = X_right + X_left
      X_left = 0
  if X_right > width:
      X_left = X_left - X_right + width
      X_right = width
  Y_up=int(Y-constants.CROP_DISTANCE_BOTTOM)
  if Y_up < 0:
      Y_up = 0
  Y_down=int(Y-constants.CROP_DISTANCE_TOP)
  image = image[Y_down:Y_up, X_left:X_right]
  print("X coordinate of finger", X)
  print("Y coordinate of finger", Y)
  print("leftmost coordinate", X_left)
  print("rightmost coordinate", X_right)

  #To scale the image
  scale_percent = constants.ZOOM_PERCENTAGE * 100
  width = int(image.shape[1] * scale_percent / 100)
  height = int(image.shape[0] * scale_percent / 100)
  dsize = (width, height)
  #resize image
  image = cv2.resize(image, dsize)
  X = width/2
  #X = X*constants.ZOOM_PERCENTAGE
  #Y = Y*constants.ZOOM_PERCENTAGE
  print("Word Coordinates", X)
  #print(Y)
  #cv2.imshow('temp', image)

  d = pytesseract.image_to_data(image, output_type=Output.DICT)
  n_boxes = len(d['text'])
  img = image

  markedWord=""
  for i in range(n_boxes-1, -1, -1):
      if int(d['conf'][i]) > constants.OCR_CONFIDENCE * 100:
          (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
          #print("X=(",x,",",x+w,") Y=(",y,",",y+h,")")
          #print(d['text'][i])
          #putting it outside loop for debugging
          img = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
          if X >= x and X <= x+w :
            # print(d['text'][i])
            # print("X=(",x,",",x+w,") Y=(",y,",",y+h,")")
            # if markedWord=="":
            print(d['text'][i])
            markedWord=d['text'][i]
            #img = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
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
ap.add_argument("-d", "--movement", type=int, required=True, help=2)
ap.add_argument("-c", "--count", type=int, required=True, help=5)
ap.add_argument("-rs", "--rescale", type=int, required=True, help=0.3)
ap.add_argument("-rt", "--rotate",type=int ,required=True,help=True)

args = vars(ap.parse_args())
video = args["video"]
constants.PERMISSIBLE_FINGER_MOVEMENT = args["movement"]
constants.MIN_STEADY_FINGER_COUNT = args["count"]
constants.RESCALE_FACTOR = args["rescale"]
if args["rotate"]==1:
  constants.ROTATE = True
else:
  constants.ROTATE = False

print("Permissible Movement : ",constants.PERMISSIBLE_FINGER_MOVEMENT)
print("Steady Count : ",constants.MIN_STEADY_FINGER_COUNT)
print("Rescale Factor : ",constants.RESCALE_FACTOR)
print("Rotate : ",constants.ROTATE)


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
  if constants.RESCALE:
    image = rescale_frame(image, constants.RESCALE_FACTOR * 100)
  # Flip the image horizontally for a later selfie-view display, and convert
  # the BGR image to RGB.
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  if constants.ROTATE:
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
        ocr(cur_x, cur_y, image, image_width)
        # Removing, so that the next word will be detected only when the finger moves farther than the PERMISSIBLE_FINGER_MOVEMENT
        # count=0

      prev_x=cur_x
      prev_y=cur_y

      mp_drawing.draw_landmarks(
          image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

  cv2.imshow('MediaPipe Hands', image)
  if cv2.waitKey(5) & 0xFF == 27:
    break
hands.close()
cap.release()
