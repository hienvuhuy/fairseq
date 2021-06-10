import sys, os, shutil, glob

INPUT_PATH = "/home/cl/huyhien-v/Workspace/MT/data/en-ru-baseline-bpe-split/original_full/"
DESTINATION_PATH = "/home/cl/huyhien-v/Workspace/MT/data/en-ru-baseline-bpe-split/train/"
file_only = lambda x: x.strip().split('/')[-1]

list_files=['test.en', 'test.ru', 'valid.en', 'valid.ru']
list_files=[INPUT_PATH+item for item in list_files]

list_folders = glob.glob(DESTINATION_PATH+"*/")

for _file in list_files:
    for _folder in list_folders:
        # print ("copy from ", _file, _folder+file_only(_file))
        shutil.copy(_file, _folder+file_only(_file))

print('done')