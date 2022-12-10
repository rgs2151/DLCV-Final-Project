import os
from pathlib import Path
from shutil import rmtree
# Function to rename multiple files

src_folder = r"Z:/jupyternotebook/ASL-main/ASL-main/rohit_dataset/video/address"
dst_folder = r"Z:/jupyternotebook/ASL-main/ASL-main/rohit_dataset/video/address/Renamed"
dst_ds= Path(dst_folder)
dst_ds.mkdir()
for count, filename in enumerate(os.listdir(src_folder)):
    dst = f"{'address_' + str(count+51)}.avi"
    src =f"{src_folder}/{filename}"  # foldername/filename, if .py file is outside folder
    dst =f"{dst_folder}/{dst}"
    os.rename(src, dst)  
    # os.remove(src_folder)
