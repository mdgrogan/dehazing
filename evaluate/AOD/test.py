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
    for it in range(1, 19):
        imagesnum = imagesnum + 1;
        #npstore = caffe.io.load_image('../img/%s.png'% str(it))
        #label = caffe.io.load_image('../img/%s-clear.png'% str(it))
        haze = cv2.imread('../img/%s.png'% str(it))
        label = cv2.imread('../img/%s-clear.png'% str(it))
        height = haze.shape[0]
        width = haze.shape[1]


        templateFile = 'test_template.prototxt'
        EditFcnProto(templateFile, height, width)


        model='AOD_Net.caffemodel';

        net = caffe.Net('DeployT.prototxt', model, caffe.TEST);
        batchdata = []
        data = haze/255.0
        data = data.transpose((2, 0, 1))
        batchdata.append(data)
        net.blobs['data'].data[...] = batchdata

        batchlabel = []
        label = label/255.0
        label = label.transpose((2, 0, 1))
        batchlabel.append(label)
        net.blobs['label'].data[...] = batchlabel

        net.forward();

        data = net.blobs['sum'].data[0];
        data = data.transpose((1, 2, 0));

        savepath = '../img/' + str(it) + '_AOD-Net.png'
        cv2.imwrite(savepath, data * 255.0)

        f = open('loss.txt', 'a')
        f.write('{0:d} '.format(it))
        f.write('{0:f}\n'.format(net.blobs['loss'].data))
        f.close();

        ################################
        # post processing stuff
        # my gut says this isn't a good way to convert to uint8
        data = data*255.0
        data = data.astype(np.uint8)


        clip = 0.4
        for ii in range(5):
            gs = 4
            for jj in range(4):
                #H = CLAHE(haze, clip, gs)
                #savepath = "../img/clahe/{0}_{1:.1f}-{2}-haze.png".format(it, clip, gs)
                #cv2.imwrite(savepath, H)

                I = CLAHE(data, clip, gs)
                #savepath = "../img/clahe/{0}_{1:.1f}-{2}-AOD.png".format(it, clip, gs)
                savepath = "../img/clahe/{0}_{1:.1f}-{2}.png".format(it, clip, gs)
                cv2.imwrite(savepath, I)

                gs += 4
            clip += 0.1

    print('image numbers:',imagesnum)

def main():
    f = open('loss.txt', 'w+')
    f.close()
    test()


if __name__ == '__main__':
    main();


