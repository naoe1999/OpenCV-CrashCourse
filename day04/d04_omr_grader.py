from imutils.perspective import four_point_transform
from imutils import contours
import imutils
import cv2
import numpy as np
import argparse


ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help='path to the input image')
args = vars(ap.parse_args())

ANSWER_KEY = {0: 1, 1: 4, 2: 0, 3: 3, 4: 1}
COLORS = [
    (0, 0, 255),    # red
    (0, 255, 0),    # green
    (255, 0, 0),    # blue
    (0, 255, 255),  # yellow
    (255, 0, 255),  # magenta
    (255, 255, 0),  # cyan
]
MARK_THRESH = 400



# load, detect edges & find contours
image = cv2.imread(args['image'])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(blurred, 75, 200)

cnts = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
docCnt = None

if len(cnts) > 0:
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            docCnt = approx
            break

# contoured = image.copy()
# cv2.drawContours(contoured, [docCnt], -1, (0, 0, 255), thickness=2)
#
# cv2.imshow('original', image)
# cv2.imshow('edged', edged)
# cv2.imshow('contoured', contoured)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# apply perspective transform & threshold
paper = four_point_transform(image, docCnt.reshape(4, 2))
warped = four_point_transform(gray, docCnt.reshape(4, 2))
thresh = cv2.threshold(warped, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

# cv2.imshow('transformed', paper)
# cv2.imshow('gray', warped)
# cv2.imshow('thresh', thresh)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# find all bubbles
cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
questionCnts = []

for c in cnts:
    (x, y, w, h) = cv2.boundingRect(c)
    ar = w / h
    if w >= 20 and h >= 20 and 0.9 <= ar <= 1.1:
        questionCnts.append(c)

# bubbled = paper.copy()
# cv2.drawContours(bubbled, questionCnts, -1, (0, 0, 255), thickness=2)
# cv2.imshow('bubbled', bubbled)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# sort bubbles
sortedCnts = contours.sort_contours(questionCnts, method='top-to-bottom')[0]
sortedCnts = list(sortedCnts)
# sorted = paper.copy()

for (q, i) in enumerate(np.arange(0, len(sortedCnts), 5)):
    cnts = contours.sort_contours(sortedCnts[i:i+5])[0]
    sortedCnts[i:i+5] = cnts
    # cv2.drawContours(sorted, cnts, -1, COLORS[q], thickness=2)

# cv2.imshow('sorted', sorted)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# get marked answers & check if it's right
answer_marked = {}
correct = 0

for (q, i) in enumerate(np.arange(0, len(sortedCnts), 5)):
    cnts = sortedCnts[i:i+5]
    marked = None
    for (j, c) in enumerate(cnts):
        mask = np.zeros(thresh.shape, dtype='uint8')
        cv2.drawContours(mask, [c], -1, 255, -1)
        # cv2.imshow('mask', mask)
        # cv2.waitKey(0)

        mask = cv2.bitwise_and(thresh, thresh, mask=mask)
        total = cv2.countNonZero(mask)
        # print(total)
        # cv2.imshow('mask', mask)
        # cv2.waitKey(0)

        if total > MARK_THRESH and (marked is None or total > marked[0]):
            marked = (total, j)

    answer_marked[q] = marked[1] if marked is not None else None

    color = (0, 0, 255)
    k = ANSWER_KEY[q]
    if k == answer_marked[q]:
        color = (0, 255, 0)
        correct += 1

    cv2.drawContours(paper, [cnts[k]], -1, color, 3)


# show result
score = (correct / 5) * 100

print("Correct answer:", ANSWER_KEY)
print("Test taker's:  ", answer_marked)
print(f'[INFO] score: {score:.2f}%')
cv2.putText(paper, f'{score:.2f}%', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
cv2.imshow('original', image)
cv2.imshow('exam', paper)
cv2.waitKey(0)
