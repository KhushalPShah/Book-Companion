import time
import cv2
# #Capture video from webcam
# vid_capture = cv2.VideoCapture(0)
# vid_cod = cv2.VideoWriter_fourcc(*'XVID')
# output = cv2.VideoWriter("cam_video.mp4", vid_cod, 1.0, (640,480))
# while(True):
#      # Capture each frame of webcam video
#      ret,frame = vid_capture.read()
#      cv2.imshow("My cam video", frame)
#      output.write(frame)
#      time.sleep(0.5)
#      # Close and break the loop after pressing "x" key
#      if cv2.waitKey(1) &0XFF == ord('x'):
#          break
# # close the already opened camera
# vid_capture.release()
# # close the already opened file
# output.release()
# # close the window and de-allocate any associated memory usage
# cv2.destroyAllWindows()
cap = cv2.VideoCapture('videos/prithvi.mp4')
while cap.isOpened():
    success, image = cap.read()
    image = cv2.rotate(image, cv2.cv2.ROTATE_90_CLOCKWISE)
    if not success:
        break
    cv2.imshow('image', image)
    time.sleep(1)
    if cv2.waitKey(5) & 0xFF == 27:
        break
cap.release()
