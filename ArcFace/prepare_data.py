import numpy as np
from PIL import Image, ImageOps, ImageEnhance
import os

def make_dataset(data_num):
    imgs = []
    labels = []

    images = os.listdir("/Users/wantedly150/Downloads/Images/")
    for num, img in enumerate(images[:data_num]):
        if not "jpg" in img: continue
        pil_x = Image.open("/Users/wantedly150/Downloads/Images/" + img)
        if pil_x.size[0] < pil_x.size[1]:
            pil_x = pil_x.rotate(90)
        pil_x = pil_x.resize((460, 280))

        label = [0] * data_num
        #num = int(img.split("-")[1].split(".")[0])
        label[num] = 1
        mark = [0, 1]
        for _ in range(50):
            if np.random.choice(mark) == 1:
                pil_x = ImageOps.flip(pil_x)
            if np.random.choice(mark) == 1:
                pil_x = ImageOps.mirror(pil_x)
            
            color_img = ImageEnhance.Color(pil_x)
            change = np.random.rand() / 2.0 + 0.75
            pil_x = color_img.enhance(change)

            top = np.random.rand() * 10
            left = np.random.rand() * 10
            pil_x = pil_x.crop((top, left,
                              top + 450, left + 270))
                
            np_x = np.asarray(pil_x) / 255.0
            if len(np_x.shape) != 3: continue
            imgs.append(np_x)
            labels.append(label)
    
    return imgs, labels