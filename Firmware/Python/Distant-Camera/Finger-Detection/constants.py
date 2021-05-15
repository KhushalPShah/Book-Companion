# This is the number of pixels, along which the finger can move while the user tries to hold the finger steady below the required word.
PERMISSIBLE_FINGER_MOVEMENT = 4
# This is the number of times for which the detected finger co-ordinate must be within the PERMISSIBLE_FINGER_MOVEMENT
MIN_STEADY_FINGER_COUNT = 2

# These are the number of pixels, from the centre of the detected finger co-ordinate, for cropping the image.
CROP_DISTANCE_LEFT = 100
CROP_DISTANCE_RIGHT = 100
# The CROP_DISTANCE_BOTTOM is the pixels from the detected finger co-ordinate to the actual word's bottom co-ordinate.
# This is to take into account the sharpness of the user's finger tip.
CROP_DISTANCE_BOTTOM = 15
CROP_DISTANCE_TOP = 100

# Factor to Zoom the Cropped Image.
ZOOM_PERCENTAGE = 2
# Centre of the Cropped Image
CROPPED_IMAGE_CENTRE = 200

# Confidence Levels
FINGER_DETECTION_CONFIDENCE = 0.7
FINGER_TRACKING_CONFIDENCE = 0.5
OCR_CONFIDENCE = 0.6

# Rescaling Factor:
RESCALE_FACTOR = 0.2

# Rotate Flag:
# 1 for Rotate, -1 for No Rotate
ROTATE = 1
