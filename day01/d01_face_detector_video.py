import numpy as np
import argparse
import cv2
import time
import imutils
from imutils.video import VideoStream


ap = argparse.ArgumentParser()
ap.add_argument('-p', '--prototxt', required=True, help='path to Caffe \'deploy\' prototxt file')
ap.add_argument('-m', '--model', required=True, help='path to Caffe pre-trained model')
ap.add_argument('-c', '--confidence', type=float, default=0.5, help='probability threshold')
args = vars(ap.parse_args())

print('loading model...')
net = cv2.dnn.readNetFromCaffe(args['prototxt'], args['model'])

print('starting video stream...')
vs = VideoStream(src=0).start()
time.sleep(2.0)

while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=400)
    frame = cv2.flip(frame, 1)

    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 117.0, 123.0))

    net.setInput(blob)
    detections = net.forward()

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence < args['confidence']:
            continue

        # draw box
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (x0, y0, x1, y1) = box.astype('int')
        cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 255), 1)

        # put text
        text = f'{confidence * 100 :.2f}%'
        yt = y0 - 10 if y0 - 10 > 10 else y0 + 10
        cv2.putText(frame, text, (x0, yt), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

    cv2.imshow('web cam', frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

cv2.destroyAllWindows()
vs.stop()

time.sleep(2.0)
