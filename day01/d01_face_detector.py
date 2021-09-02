import numpy as np
import argparse
import cv2


ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help='path to input image')
ap.add_argument('-p', '--prototxt', required=True, help='path to Caffe \'deploy\' prototxt file')
ap.add_argument('-m', '--model', required=True, help='path to Caffe pre-trained model')
ap.add_argument('-c', '--confidence', type=float, default=0.5, help='probability threshold')
args = vars(ap.parse_args())

print('loading model...')
net = cv2.dnn.readNetFromCaffe(args['prototxt'], args['model'])

image = cv2.imread(args['image'])
(h, w) = image.shape[:2]
# blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 117.0, 123.0))
blob = cv2.dnn.blobFromImage(cv2.resize(image, (1000, 800)), 1.0, (1000, 800), (104.0, 117.0, 123.0))
# (104.0, 177.0, 123.0) --> (104.0, 117.0, 123.0)

print('detecting...')
net.setInput(blob)
detections = net.forward()
# 4d numpy array

for i in range(0, detections.shape[2]):
    confidence = detections[0, 0, i, 2]
    if confidence >= args['confidence']:
        # draw box
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (x0, y0, x1, y1) = box.astype('int')
        cv2.rectangle(image, (x0, y0), (x1, y1), (0, 0, 255), 1)

        # put text
        text = f'{confidence * 100 :.2f}%'
        yt = y0 - 10 if y0 - 10 > 10 else y0 + 10
        cv2.putText(image, text, (x0, yt), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

# show output image
cv2.imshow('detection', image)
cv2.waitKey(0)
