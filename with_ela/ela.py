from PIL import Image, ImageChops, ImageEnhance
import sys, os.path
import os
from pathlib import Path


def convert_to_ela_image(path, quality, filenamea, dir):
    # print(path)
    filename = path
    resaved_filename = filename.split('.')[0] + '.resaved.jpg'
    ELA_filename = filename.split('.')[0] + '.ela.png'
    
    im = Image.open(filename).convert('RGB')
    im.save(resaved_filename, 'JPEG', quality=quality)
    resaved_im = Image.open(resaved_filename)
    
    ela_im = ImageChops.difference(im, resaved_im)
    
    extrema = ela_im.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    if max_diff == 0:
        max_diff = 1
    scale = 255.0 / max_diff
    ela_im = ImageEnhance.Brightness(ela_im).enhance(scale)
    os.remove(resaved_filename)
    print('dataset/ela/'+dir+"/"+filenamea, scale, max_diff)
    ela_im.save('dataset/ela/'+dir+"/"+filenamea)
    # return ela_im

dir = ['fake','real']
i = 1
for dname  in dir:
    for fname in os.listdir('dataset/'+dname):
        # if "jpg" in fname:    
        name = '{:06}.jpg'.format(i)
        convert_to_ela_image('dataset/'+dname+"/"+fname, 90, name, dname)
        i+=1
