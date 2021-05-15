# Book-Companion - A Dictionary with its own Eyes!
A strong vocabulary is a rarely found skill. Knowing the meaning of each of the words in the English dictionary is not an easy task. 
So, when you encounter an unseen/unknown word while you are reading a written content - you tend to switch to a digital device - maybe a cellphone and search the
meaning. However, the cell phone has an impact - it creates distraction in the form of notifications from
hundreds of apps installed on the device. Thus, the reader is easily lured away.
So, we propose a solution which allows the readers to find the meaning of the word without having to
interact with any other digital device - enabling sheer reading experience.

A companion to provide you with the meanings of difficult words, while you read a book. 

We provide 2 solutions:
- **Finger based Detection(For 64 bit Hardware)**
- **Colored Bookmark based Detection**

## Finger based Detection(For 64 bit Hardware)
In this approach, the user must place his/her finger beneath the word whose meaning is needed. 
We use Google's Mediapipe for detecting the finger (specifically, the tip of the index finger), and Py-tesseract for the OCR.
Once the word is detected, PyDictionary is used to find the meaning of the word.

*Note: This approach is to be used only by 64 bit architecture since Mediapipe has only 64 bit support as of now*

Steps:
- Go inside the Firmware/Python/Distant-Camera/Finger-Detection/
- Make a /videos folders.
- Place the video to be tested inside this new folder.
- Run the following command:
   - python run.py --v videos/video_name.mp4

*The code uses certain constants. You can pass it via CLI or by a file - constants.py. Refer constants.py for description for these constants.*


## Colored Bookmark based Detection

In this approach, the user must keep the bookmark, of a specified color, beneath the word to be detected.
Using OpenCV, the contour for the bookmark is obtained.
Later on, Py-tesseract is used for the OCR.


Steps:
- Go inside the Firmware/Python/Distant-Camera/Finger-Detection/
- Run the following command:
   - python color_detection.py
*This approach uses the live feed from the camera. This approach was used with Raspberry Pi*




