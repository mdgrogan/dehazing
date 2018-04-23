import os
import numpy as np
import caffe
import sys
from pylab import *
import re
import random
import time
import copy
import matplotlib.pyplot as plt
import cv2
import scipy
import shutil
import csv
from PIL import Image
import datetime


def EditFcnProto(templateFile, height, width):
        with open(templateFile, 'r') as ft:
            template = ft.read()
        print(templateFile)
        outFile = 'DeployT.prototxt'
        with open(outFile, 'w') as fd:
            fd.write(template.format(height=height,width=width))

def CLAHE(bgr, lim, gs):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    lab_planes = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=lim, tileGridSize=(gs, gs))
    lab_planes[0] = clahe.apply(lab_planes[0])
    lab = cv2.merge(lab_planes)
    bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return bgr


def test():
    #caffe.set_mode_gpu()
    #caffe.set_device(0)
    caffe.set_mode_cpu();

    imagesnum=0;
    for i in range(1, 19):
        imagesnum = imagesnum + 1;
        npstore = caffe.io.load_image('../img/%s.png'% str(i))
        label = caffe.io.load_image('../img/%s-clear.png'% str(i))
        height = npstore.shape[0]
        width = npstore.shape[1]


        templateFile = 'test_template.prototxt'
        EditFcnProto(templateFile, height, width)


        model='AOD_Net.caffemodel';

        net = caffe.Net('DeployT.prototxt', model, caffe.TEST);
        batchdata = []
        data = npstore
        data = data.transpose((2, 0, 1))
        batchdata.append(data)
        net.blobs['data'].data[...] = batchdata

        batchdata1 = []
        label = label.transpose((2, 0, 1))
        batchdata1.append(label)
        net.blobs['label'].data[...] = batchdata1

        net.forward();
        print('iteration' + str(i))

        data = net.blobs['sum'].data[0];
        data = data.transpose((1, 2, 0));
        data = data[:, :, ::-1]

        savepath = '../img/' + str(i) + '_AOD-Net.png'
        cv2.imwrite(savepath, data * 255.0)

        # my gut says this isn't good
        data = data*255.0
        data = data.astype(np.uint8)

        data1 = CLAHE(data, 0.6, 8)
        savepath = '../img/' + str(i) + '_CLAHE-0.6-8.png'
        cv2.imwrite(savepath, data1)

        data2 = CLAHE(data, 0.4, 8)
        savepath = '../img/' + str(i) + '_CLAHE-0.4-8.png'
        cv2.imwrite(savepath, data2)

        data3 = CLAHE(data, 0.6, 12)
        savepath = '../img/' + str(i) + '_CLAHE-0.6-12.png'
        cv2.imwrite(savepath, data3)

        data4 = CLAHE(data, 0.4, 12)
        savepath = '../img/' + str(i) + '_CLAHE-0.4-12.png'
        cv2.imwrite(savepath, data4)

        f = open('loss.txt', 'a')
        f.write('{0:d} '.format(i))
        f.write('{0:f}\n'.format(net.blobs['loss'].data))
        f.close();

    print('image numbers:',imagesnum)

def main():
    f = open('loss.txt', 'w+')
    f.close()
    test()


if __name__ == '__main__':
    main();


