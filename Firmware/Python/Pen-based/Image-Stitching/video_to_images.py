import cv2
vidcap = cv2.VideoCapture('video.mp4')
success, image = vidcap.read()
count = 1
while success:
    success, image = vidcap.read()
    if count%20 == 0:
        cv2.imwrite("D:\VJTI\Sem 7\FYP\Test\Book-Companion\Firmware\Python\Image-Stitching\Github\im\image_%d.jpg" % count, image)    
        print('Saved image ', count)
    count += 1