from pyimagesearch.transform import four_point_transform
from skimage.filters import threshold_local
import argparse
import cv2
import imutils


ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help='Path to the image to be scanned')
args = vars(ap.parse_args())


################################
# Step 1: Edge Detection
print("Step 1: Edge detection")

# load & resize image
original_image = cv2.imread(args['image'])
image = imutils.resize(original_image, height=500)
ratio = original_image.shape[0] / image.shape[0]  # height ratio

# convert to gray & detect edge
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(gray, 75, 200)

# show result
cv2.imshow('Image', image)
cv2.imshow('Edged', edged)
cv2.waitKey(0)
cv2.destroyAllWindows()


################################
# Step 2: Finding Contours
print("Step 2: Finding contours of paper")
screenCnt = None

# find contours
cnts = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

# find largest contour with four points
for c in cnts:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02*peri, True)

    print(f'  ...examining contours. num of points: {len(approx)}')
    if len(approx) == 4:
        screenCnt = approx
        break

# show result
cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
cv2.imshow('Outline', image)
cv2.waitKey(0)
cv2.destroyAllWindows()


###################################################
# Step 3: Perspective Transform & Threshold
print("Step 3: Perspective transform & threshold")

# do perspective transform
warped = four_point_transform(original_image, screenCnt.reshape(4, 2) * ratio)

# convert to gray & threshold
warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

# global threshold
# scanned = cv2.threshold(warped_gray, 155, 255, cv2.THRESH_BINARY)[1]
# scanned = cv2.threshold(warped_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

# adaptive threshold
T = threshold_local(warped_gray, 81, method='gaussian', offset=10)
scanned = (warped_gray > T).astype('uint8') * 255

# show result
cv2.imshow('Original', imutils.resize(original_image, height=650))
cv2.imshow('Transformed', imutils.resize(warped, height=650))
cv2.imshow('Scanned', imutils.resize(scanned, height=650))
cv2.waitKey(0)
