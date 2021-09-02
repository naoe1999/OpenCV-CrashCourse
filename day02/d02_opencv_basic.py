import imutils
import cv2

# read image as numpy array
image = cv2.imread('jp.png')
(h, w, d) = image.shape
print(f'width={w}, height={h}, depth={d}')

# show image on the screen
cv2.imshow("Image", image)
cv2.waitKey(0)

# get color values from a pixel
(B, G, R) = image[100, 50]
print(f'R={R}, G={G}, B={B}')

# extract ROI
roi = image[60:160, 320:420]
cv2.imshow("ROI", roi)
cv2.waitKey(0)


# resize (to fixed size)
resized = cv2.resize(image, (300, 200))
cv2.imshow("Resized", resized)
cv2.waitKey(0)

# resize (keeping its own aspect ratio)
r = 300.0 / w
dim = (300, int(h * r))
resized = cv2.resize(image, dim)
cv2.imshow("Resized keeping its aspect ratio", resized)
cv2.waitKey(0)

# resize (using imutils function)
resized = imutils.resize(image, width=300)
cv2.imshow("Resized using imutils", resized)
cv2.waitKey(0)


# rotate
center = (w // 2, h // 2)
M = cv2.getRotationMatrix2D(center, -45, 1.0)
rotated = cv2.warpAffine(image, M, (w, h))
cv2.imshow("Rotated", rotated)
cv2.waitKey(0)

# rotate (using imutils function)
rotated = imutils.rotate(image, -45)
cv2.imshow("Rotated using imutils", rotated)
cv2.waitKey(0)

# rotate bound (note the sign of angle is reversed)
rotated = imutils.rotate_bound(image, 45)
cv2.imshow("Rotated using imutils.rotate_bound", rotated)
cv2.waitKey(0)


# blur
blurred = cv2.GaussianBlur(image, (11, 11), 0)
cv2.imshow("Blurred", blurred)
cv2.waitKey(0)


# drawing section
# draw a rect
output = image.copy()
cv2.rectangle(output, (320, 60), (420, 160), (0, 0, 255), 2)
cv2.imshow("Rectangle", output)
cv2.waitKey(0)

# draw a circle
output = image.copy()
cv2.circle(output, (300, 150), 20, (255, 0, 0), -1)
cv2.imshow("Circle", output)
cv2.waitKey(0)

# draw a line
output = image.copy()
cv2.line(output, (60, 20), (400, 200), (0, 0, 255), 5)
cv2.imshow("Line", output)
cv2.waitKey(0)

# put text
output = image.copy()
cv2.putText(output, 'OpenCV + Jurassic Park', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
cv2.imshow("Text", output)
cv2.waitKey(0)
