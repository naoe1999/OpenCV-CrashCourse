import argparse
import imutils
import cv2


ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help='path to input image')
args = vars(ap.parse_args())


# read image
image = cv2.imread(args['image'])
print('image shape:', image.shape)
cv2.imshow("Image", image)
cv2.waitKey(0)

# gray scale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
print('gray shape:', gray.shape)
cv2.imshow("Gray", gray)
cv2.waitKey(0)

# edge detection
edged = cv2.Canny(gray, 30, 150)
print('edged shape:', edged.shape)
cv2.imshow("Edged", edged)
cv2.waitKey(0)

# thresholding
thresh = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY_INV)[1]
print('thresh shape:', thresh.shape)
cv2.imshow("Thresh", thresh)
cv2.waitKey(0)


# contours
conts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
conts = imutils.grab_contours(conts)
output = image.copy()

# concurrent drawing
cv2.drawContours(output, conts, -1, (240, 0, 159), 3)
cv2.imshow("Contours", output)
cv2.waitKey(0)

# sequential drawing
# for c in conts:
#     cv2.drawContours(output, [c], -1, (240, 0, 159), 3)
#     cv2.imshow("Contours", output)
#     cv2.waitKey(0)

# text
text = f'I found {len(conts)} objects!'
cv2.putText(output, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 0, 159), 2)
cv2.imshow("Contours", output)
cv2.waitKey(0)


# erosion
mask = thresh.copy()
mask = cv2.erode(mask, None, iterations=5)
cv2.imshow("Eroded", mask)
cv2.waitKey(0)

# dilation
mask = thresh.copy()
mask = cv2.dilate(mask, None, iterations=5)
cv2.imshow("Dilated", mask)
cv2.waitKey(0)


# masking
mask = thresh.copy()
output = cv2.bitwise_and(image, image, mask=mask)
cv2.imshow("Output", output)
cv2.waitKey(0)
