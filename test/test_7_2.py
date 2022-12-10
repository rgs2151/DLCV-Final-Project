'''
CLEARING THE ARRAY AFTER EACH PREDICTION AND MAKING ONE PREDICTION EVERY 30 FRAMES
'''

from cv2 import VideoCapture, waitKey, imshow, cvtColor, flip, COLOR_BGR2RGB, COLOR_RGB2BGR, putText, FONT_HERSHEY_SIMPLEX, resize, rectangle, line
from numpy import ndarray, array, argmax, expand_dims
from tensorflow.keras.models import load_model
from collections import deque
import time

import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

from ASL_UTILS.extractor import MediapipeExtractor
from ASL_UTILS.collection import ImageHandler

cap = VideoCapture(0)
me = MediapipeExtractor()
im = ImageHandler()

DQ = deque(maxlen=1)

Q = []
Q_limit = 70
Q_status = 0
# klasses = ['before', 'cool', 'hands_down']
klasses = ['Address', 'Age', 'Class', 'How', 'Movie', 'Name', 'Phone', 'Play', 'Please', 'ThankYou', 'Time', 'What', 'When', 'Where', 'Work', 'You', 'Your']


QR = []
QR_limit = 30
QR_status = 0

#frame rate
prev_frame_time = 0
new_frame_time = 0
avgfps = 0

model = "test/model151"

model = load_model(str(model))

width = 640
height = 480

width_from_center = 175
height_from_center_top = 200
height_from_center_bottom = int(height / 2)

point1 = (int((width/2) - width_from_center), int((height/2) - height_from_center_top))
point2 = (int((width/2) + width_from_center), int((height/2) + height_from_center_bottom))

lheight = int((height/2) - 10)
lpoint1 = (point1[0], lheight)
lpoint2 = (point2[0], lheight)

fwidth = 100
fpoint1 = (int(width/2 - fwidth), point1[1])
fpoint2 = ((int(width/2 + fwidth), lpoint1[1]))

new_result =''
new_result_1 =''
new_result_2 =''
new_result_3 =''
results_forever=''
text1 = ''

with mp_holistic.Holistic(
    min_detection_confidence = 0.5,
    min_tracking_confidence = 0.5
) as holistic:

    while cap.isOpened():
        
        image: ndarray
        success, image = cap.read()

        if not success: continue

        image = flip(image, 1)

        original = image.copy()

        image = cvtColor(image, COLOR_BGR2RGB)

        image.flags.writeable = False
        results = holistic.process(image)

        image = cvtColor(image, COLOR_RGB2BGR)
        image = im.draw_results(image, results)

        landmarks = me.extract_landmarks(results)


        if QR_status == QR_limit:

            QR_vid = expand_dims(array(QR), 0)
            results = model.predict(QR_vid)

            # DQ.append(results[0])
            # results = array(DQ).mean(axis=0)
            results = array(results).mean(axis=0)

            predicted_kls = klasses[results.argmax(axis=0)]
            text1 = f"{predicted_kls}"
            results_forever = results
            QR_status = 0
            QR = []
        else:

            QR.append(landmarks)
            QR_status += 1

            text2 = f"{'|'*QR_status}"

        # use (1280, 1024) below if the output window is too small
        image = resize(image, (640, 512))
        image = putText(image, text1, (30, image.shape[0] - 90), FONT_HERSHEY_SIMPLEX, 2, (11,209,252), 3)
        image = putText(image, text2, (30, image.shape[0] - 30), FONT_HERSHEY_SIMPLEX, 2, (11,209,252), 3)
        z=0
        for x in sorted(zip(klasses, results_forever)):
            if z <= 4:
                new_result = new_result + " " + str(x[0]) + "   " + str(round(x[1],5)) + " "
                z +=1
            elif z>=5 and z<=9:
                new_result_1 = new_result_1 + " " + str(x[0]) + "   " + str(round(x[1],5)) + "  "
                z +=1
            elif z>=10 and z<=14:
                new_result_2 = new_result_2 + " " + str(x[0]) + "   " + str(round(x[1],5)) + "  "
                z +=1
            elif z>=15 and z<=20:
                new_result_3 = new_result_3 + " " + str(x[0]) + "   " + str(round(x[1],5)) + "  "
                z +=1
        try:
            image = putText(image, str(new_result), (30, image.shape[0] - 980), FONT_HERSHEY_SIMPLEX, 0.7, (255,174,0), 2)
            image = putText(image, str(new_result_1), (30, image.shape[0] - 940), FONT_HERSHEY_SIMPLEX, 0.7, (255,174,0), 2)
            image = putText(image, str(new_result_2), (30, image.shape[0] - 900), FONT_HERSHEY_SIMPLEX, 0.7, (255,174,0), 2)
            image = putText(image, str(new_result_3), (30, image.shape[0] - 860), FONT_HERSHEY_SIMPLEX, 0.7, (255,174,0), 2)
            new_result = ""
            new_result_1 = ""
            new_result_2 = ""
            new_result_3 = ""
            image = putText(image, str((sorted(zip(klasses, results), key=lambda x: x[1], reverse=True))), (30, image.shape[0] - 150), FONT_HERSHEY_SIMPLEX, 1, (255,174,0), 2)

        except Exception as err:
            # pass
            image = putText(image, str(results), (30, image.shape[0] - 200), FONT_HERSHEY_SIMPLEX, 1, (255,174,0), 2)
        #fps counter
        new_frame_time = time.time()
        fps = 1/(new_frame_time-prev_frame_time)
        prev_frame_time = new_frame_time
        fps = int(fps)
        avgfps=0.9*float(avgfps)+0.1*float(fps)
        original_text = str(original.shape) + "  FPS: " + str(int(avgfps))
        
        original = putText(original, original_text, (30, 30), FONT_HERSHEY_SIMPLEX, 1, (11,209,252), 2)
        original = rectangle(original, point1, point2, (211, 66, 242), 2)
        original = line(original, lpoint1, lpoint2, (211, 66, 242), 2)
        original = rectangle(original, fpoint1, fpoint2, (92, 206, 17), 2)
        
        imshow('out', image)
        imshow('original', original)

        if waitKey(5) & 0xFF == 27: break # 27 is escape!

    cap.release()