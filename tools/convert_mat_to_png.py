import scipy.io as sio
from PIL import Image
import os
import numpy as np
import argparse

def get_pallete(num_cls):
    """
    this function is to get the colormap for visualizing the segmentation mask
    :param num_cls: the number of visulized class
    :return: the pallete
    """
    n = num_cls
    pallete = [0] * (n * 3)
    for j in xrange(0, n):
        lab = j
        pallete[j * 3 + 0] = 0
        pallete[j * 3 + 1] = 0
        pallete[j * 3 + 2] = 0
        i = 0
        while (lab > 0):
            pallete[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            pallete[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            pallete[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i = i + 1
            lab >>= 3
    return pallete

def convert_mat_to_png(input_dir,output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print "Making the output dir:",output_dir

    filenames = os.listdir(input_dir)
    for filename in filenames:
        filepath = os.path.join(input_dir,filename)
        mat = sio.loadmat(filepath)
        label = mat['GTcls'][0]['Segmentation'][0].astype(np.uint8)
        filename_without_ext = os.path.splitext(filename)[0]
        im = Image.fromarray(label)
        im.putpalette(get_pallete(256))
        savepath=os.path.join(output_dir,filename_without_ext+".png")
        im.save(savepath)
        print "convert",filename
    print "convert done"


def convert_mat_to_png_context(context_dir,output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print "Making the output dir:",output_dir

    labels_400 = [label.replace(' ', '') for idx, label in
                       np.genfromtxt(context_dir + '/labels.txt', delimiter=':', dtype=None)]
    labels_59 = [label.replace(' ', '') for idx, label in
                      np.genfromtxt(context_dir + '/59_labels.txt', delimiter=':', dtype=None)]

    data_dir = os.path.join(context_dir, 'trainval')
    filenames = os.listdir(data_dir)
    for filename in filenames:

        filepath = os.path.join(data_dir,filename)
        label_400 = sio.loadmat(filepath)['LabelMap']
        label = np.zeros_like(label_400, dtype=np.uint8)
        print len(labels_59)
        for idx, l in enumerate(labels_59):
            idx_400 = labels_400.index(l) + 1
            label[label_400 == idx_400] = idx + 1

        filename_without_ext = os.path.splitext(filename)[0]
        im = Image.fromarray(label)
        im.putpalette(get_pallete(256))
        savepath=os.path.join(output_dir,filename_without_ext+".png")
        im.save(savepath)
        print "convert",filename

    print "convert done"

def parse_args():
    parser = argparse.ArgumentParser(description='convert the mat to the jpg')
    parser.add_argument('--input_dir', help='input dir', default=".", type=str)
    parser.add_argument('--output_dir', help='output dir ', default=".", type=str)
    return parser.parse_args()

if __name__=="__main__":
    args = parse_args()
    convert_mat_to_png_context(args.input_dir,args.output_dir)

