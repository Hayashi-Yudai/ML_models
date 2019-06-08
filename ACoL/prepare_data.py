import os

IMAGE_DIR = '/home/yudai/Pictures/raw-img'

def image_files(train=True) -> list:
  suffix = '/train/' if train else '/validation/'
  category_list = os.listdir(IMAGE_DIR + suffix)

  images = []
  for category in category_list:
    tmp = []
    for f in os.listdir(IMAGE_DIR + suffix + category):
      if f.endswith('.png') or f.endswith('.jpg') or f.endswith('.jpeg'):
        tmp.append(f)
    
    images.append(tmp)

  return images