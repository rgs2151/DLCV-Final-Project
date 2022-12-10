import os
import glob
from pathlib import Path
import cv2
from cv2 import VideoCapture, waitKey, imshow, cvtColor, flip, COLOR_BGR2RGB, COLOR_RGB2BGR, putText, FONT_HERSHEY_SIMPLEX, resize, rectangle, line
from numpy import asscalar, ndarray, array
from pathlib import Path
from shutil import rmtree

import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

from ASL_UTILS.extractor import MediapipeExtractor
from ASL_UTILS.collection import Writer, ImageHandler

me = MediapipeExtractor()
w = Writer()
im = ImageHandler()


source_dir = r"Z:/jupyternotebook/ASL-main/ASL-main/csv maker/content_renamed"
out_dir = "Z:/jupyternotebook/ASL-main/ASL-main/csv maker/content_csv"      #WARNING! MAKE SURE out_dir IS EMPTY OR EVERYTHING WILL BE DELETED




source_directory = Path(source_dir)
classes = list(x.name for x in source_directory.glob('*'))
print(classes)

all_source_ds = []      #source video location list
for category in classes:
    cat_dir = list((source_directory/category).glob("*"))
    all_source_ds.append(cat_dir)
# print(all_source_ds)  #GET READY FOR FLOOOD

counter_class = len(classes)
print(counter_class)        #checking for expected number of classes in source directory

for i in range(len(classes)):
    category = classes[i]
    current_ds = all_source_ds[i]   #source video location of i'th class
    print(len(current_ds))
    out_dir_temp = out_dir
    out_dir_temp = out_dir_temp + '/' + classes[i]
    out_directory = Path(out_dir_temp)      #modified out_dir to out_dir -> class name -> csv
    if out_directory.exists(): rmtree(out_directory)        #WARNING! MAKE SURE out_directory IS EMPTY OR EVERYTHING WILL BE DELETED
    out_directory.mkdir()
    for j in range(len(current_ds)):
        cap = VideoCapture(str(current_ds[j]))
        frame_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))           
        klass = category
        out_prefix = klass 

        video_name = os.path.basename(os.path.normpath(str(current_ds[j])))             
        video_name = str(os.path.splitext(video_name)[0])
        print("New Video loaded" + "/tCategory : " + klass + "/tName : " + video_name )    
        with mp_holistic.Holistic(
            min_detection_confidence = 0.5,
            min_tracking_confidence = 0.5
        ) as holistic:
            while cap.isOpened() and frame_length!=0:
                image: ndarray
                success, image = cap.read()

                if not success: continue
                image = cvtColor(image, COLOR_BGR2RGB)

                image.flags.writeable = False
                results = holistic.process(image)

                image = cvtColor(image, COLOR_RGB2BGR)
                image = im.draw_results(image, results)

                landmarks = me.extract_landmarks(results)

                w.write_to_csv(landmarks, out_directory / f'{out_prefix}_{video_name}.csv')
                frame_length-=1
                print("Progress : Frame " + str(frame_length))


