from cv2 import VideoCapture, waitKey, imshow, cvtColor, flip, COLOR_BGR2RGB, COLOR_RGB2BGR, putText, FONT_HERSHEY_SIMPLEX, resize, rectangle, line
from numpy import asscalar, ndarray, array
from pathlib import Path
from shutil import rmtree
import os
from os.path import split
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

from ASL_UTILS.extractor import MediapipeExtractor
from ASL_UTILS.collection import Writer, ImageHandler

cap = VideoCapture(0)
me = MediapipeExtractor()
w = Writer()
im = ImageHandler()

klass = 'Favourite' # <-------------- 
cur_video_count = 1
frame_count = 50
video_count = 100

out_prefix = klass

destination = "D:/ASL_DATASET/recording"
out_video = Path(destination + '/video') 
out_csv = Path(destination + '/csv')

video_path = out_video / klass
csv_path = out_csv / klass
print(video_path,csv_path)
if not video_path.exists():
    video_path.mkdir()
if not csv_path.exists(): 
    csv_path.mkdir()

# def uniquify(path):
#     filename, extension = os.path.splitext(path)
#     counter = 1
#     while os.path.exists(path):
#         path = filename + " (" + str(counter) + ")" + extension
#         counter += 1
#     return path
def uniquify(path):
    counter = 0
    label = split(split(path)[1])[1]
    temp_path = path + '\\' + label + '_' + str(counter) + '.csv'
    print(temp_path)
    while os.path.exists(temp_path):
        counter += 1
        temp_path = path + '\\' + label + '_' + str(counter) + '.csv'
        print(temp_path)
    return counter-1


wait_flag = True
wait_count = 30
count_flag = False

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

video_count+=1

cur_video_count = uniquify(str(csv_path))

with mp_holistic.Holistic(
    min_detection_confidence = 0.5,
    min_tracking_confidence = 0.5
) as holistic:
    
    while cap.isOpened():
        image: ndarray
        success, image = cap.read()

        if not success: continue

        original = image.copy()
        
        image = resize(image, (640, 480))

        image = resize(image, (640, 480))

        image = flip(image, 1)
        
        if not wait_flag:

            image = cvtColor(image, COLOR_BGR2RGB)

            image.flags.writeable = False
            results = holistic.process(image)

            image = cvtColor(image, COLOR_RGB2BGR)
            image = im.draw_results(image, results)

            landmarks = me.extract_landmarks(results)
            
            w.write_to_csv(landmarks, csv_path / f'{out_prefix}_{cur_video_count}.csv')
            w.write_as_video(str(video_path / f'{out_prefix}_{cur_video_count}.avi'), original)


            frame_count -= 1
            if not frame_count: 
                wait_flag = True
                frame_count = 50

            text = f"Taking video {video_count}"

        else:

            wait_count -= 1
            if not wait_count:
                wait_count = 30
                wait_flag = False

                video_count -= 1
                cur_video_count += 1

                if video_count == 0: break
            
            text = "Waiting"

            if wait_count < 100: image = putText(image, 'BE READY' if video_count != 1 else 'STOPPING', (int(image.shape[1]/2 - 100), int(image.shape[1]/2)), FONT_HERSHEY_SIMPLEX, 1, (146,35,255), 2)

        
        image = resize(image, (1280, 1024))
        image = putText(image, text, (int(image.shape[1]/2 - 300), 60), FONT_HERSHEY_SIMPLEX, 2, (255,174,0), 3)
        image = putText(image, f"frame count {frame_count}", (image.shape[1] - 300, image.shape[0] - 90), FONT_HERSHEY_SIMPLEX, 1, (147, 248, 80), 3)
        image = putText(image, f"video count {video_count}", (image.shape[1] - 300, image.shape[0] - 60), FONT_HERSHEY_SIMPLEX, 1, (248, 115,80), 3)
        image = putText(image, f"wait count {wait_count}", (image.shape[1] - 300, image.shape[0] - 30), FONT_HERSHEY_SIMPLEX, 1, (88, 88, 252), 3)

        original = putText(original, str(original.shape), (30, 30), FONT_HERSHEY_SIMPLEX, 1, (11,209,252), 2)
        original = rectangle(original, point1, point2, (211, 66, 242), 2)
        original = line(original, lpoint1, lpoint2, (211, 66, 242), 2)
        original = rectangle(original, fpoint1, fpoint2, (92, 206, 17), 2)

        if video_count == 1:
            image = putText(image, 'LAST VIDEO' if frame_count != 50 else 'COMPLETE', (520, 150), FONT_HERSHEY_SIMPLEX, 1, (13,219,255), 2)
            
        imshow('out', image)
        imshow('original', original)

        if waitKey(5) & 0xFF == 27: break # 27 is escape!

    cap.release()