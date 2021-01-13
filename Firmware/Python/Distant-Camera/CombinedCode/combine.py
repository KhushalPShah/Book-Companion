import cv2
import mediapipe as mp
import pytesseract
from pytesseract import Output
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
from PyDictionary import PyDictionary

#CONSTANTS
Dis_btw_fingerCoordinates=2


def ocr(X,Y,image):

  # For Cropping the image
  XLeft=int(X-100)
  X_right=int(X+100)
  Y_up=int(Y-15)
  Y_down=int(Y-100)
  image = image[Y_down:Y_up, XLeft:X_right]

  #To scale the image
  scale_percent = 200
  width = int(image.shape[1] * scale_percent / 100)
  height = int(image.shape[0] * scale_percent / 100)
  dsize = (width, height)
  #resize image
  image = cv2.resize(image, dsize)

  d = pytesseract.image_to_data(image, output_type=Output.DICT)
  n_boxes = len(d['text'])
  img = image

  X=200
  markedWord=""
  for i in range(n_boxes):
      if int(d['conf'][i]) > 60:
          (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
          # print(d['text'][i])
          # print("X=(",x,",",x+w,") Y=(",y,",",y+h,")")
          if X>=x and X<=x+w :
            # print(d['text'][i])
            # print("X=(",x,",",x+w,") Y=(",y,",",y+h,")")
            markedWord=d['text'][i]
          img = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
  print(markedWord)
  # dict = PyDictionary() 
  # meaning = dict.meaning(list1[len(list1)-1][1]) 
  # print("meaning=",meaning) 

  # show the output images
  cv2.imshow("Out", img)
  cv2.waitKey(0)

# For webcam input:
prev_x=0
prev_y=0
count=0
hands = mp_hands.Hands(
    min_detection_confidence=0.7, min_tracking_confidence=0.5)
cap = cv2.VideoCapture('test.mp4')
while cap.isOpened():
  success, image = cap.read()
  if not success:
    break

  # Flip the image horizontally for a later selfie-view display, and convert
  # the BGR image to RGB.
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
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

      if (((cur_x-prev_x)**2+(cur_y-prev_y)**2)**0.5)<Dis_btw_fingerCoordinates:
        count=count+1
      else:
        count=0

      if count==10:
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