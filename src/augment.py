
# from pathlib import Path
# from cv2 import VideoCapture

# video_dir = ''
# video_dir = Path(video_dir)

# out_dir = ''
# out_dir = Path(out_dir)

# def augment_video_from_dir(video_dir, out_dir, extension='avi'):
#     video_dir = Path(video_dir)
#     out_dir = Path(out_dir)

#     all_videos = video_dir.glob(f'*.{extension}')
#     for video in all_videos:
#         capture = VideoCapture(str(video))

#         while capture.isOpened(): pass


from pathlib import Path
from tkinter import W
from vidaug import augmentors as va
import cv2
import random
random.seed(1)
from ASL_UTILS.collection import Writer
vr = Writer()
from shutil import rmtree

aug_size_per_class = 302

source = "D:/Bird"

out = "D:/Bird/Bird_aug"

wrong_framelength_videos = "D:/ASL_DATASET/Favourite,ExcuseMe,School,Class"


source_ds = Path(source)
aug_ds = Path(out)
wrong_framelength_videos_ds = Path(wrong_framelength_videos)

classes = list(x.name for x in source_ds.glob('*'))
print(classes)

print(source_ds.exists(),  aug_ds.exists())

all_source_ds = []
for category in classes:
    cat_dir = list((source_ds/category).glob("*"))
    all_source_ds.append(cat_dir)
wrong_count = 0
counter_class = len(classes)
for i in range(len(classes)):
# for i in range(19,20):
    counter_vid = aug_size_per_class
    category = classes[i]
    current_ds = all_source_ds[i]
    print(category)
    make_dir = Path(out)
    make_dir = make_dir / category
    if make_dir.exists(): rmtree(make_dir)
    make_dir.mkdir()
    for _ in range(aug_size_per_class):
        cap = cv2.VideoCapture(str(random.choice(current_ds)))
        video_frames = []

        while True:
            status, frame = cap.read()
            if not status: break
            video_frames.append(frame)

        print(len(video_frames))
        video_shape = video_frames[0].shape

        seq = va.Sequential([va.RandomTranslate(int(video_shape[1]/4),50), va.RandomShear(0.07,0.07)])

        video_aug = seq(video_frames)
        aug_vid_name = aug_ds/category/f"{random.randint(100000,999999)}.avi"
        if len(video_aug) == 50:
            for aug_frame in video_aug:
                vr.write_as_video(str(aug_vid_name), aug_frame)
        else:
            print("WRONG FRAME LENGTH : " + str(len(video_aug)) )
            wrong_count+=1


        print("class:",counter_class,"||","vidno:",counter_vid,"/",aug_size_per_class,"||","filename:",aug_vid_name)
        counter_vid -=1

    counter_class -=1
print("Failed augmentation videos : ",wrong_count)