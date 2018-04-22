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
from visualize_caffe import *

caffe.set_mode_cpu();

net = caffe.Net('../DeployT.prototxt', '../AOD_Net.caffemodel', caffe.TEST);
visualize_weights(net, 'conv1', filename='conv1-trained.png')
visualize_weights(net, 'conv2', filename='conv2-trained.png')
visualize_weights(net, 'conv3', filename='conv3-trained.png')
visualize_weights(net, 'conv4', filename='conv4-trained.png')
visualize_weights(net, 'conv5', filename='conv5-trained.png')

net = caffe.Net('../DeployT.prototxt', '../model_iter_1300.caffemodel', caffe.TEST);
visualize_weights(net, 'conv1', filename='conv1-1300.png')
visualize_weights(net, 'conv2', filename='conv2-1300.png')
visualize_weights(net, 'conv3', filename='conv3-1300.png')
visualize_weights(net, 'conv4', filename='conv4-1300.png')
visualize_weights(net, 'conv5', filename='conv5-1300.png')

net = caffe.Net('../DeployT.prototxt', '../model_iter_24100.caffemodel', caffe.TEST);
visualize_weights(net, 'conv1', filename='conv1-24100.png')
visualize_weights(net, 'conv2', filename='conv2-24100.png')
visualize_weights(net, 'conv3', filename='conv3-24100.png')
visualize_weights(net, 'conv4', filename='conv4-24100.png')
visualize_weights(net, 'conv5', filename='conv5-24100.png')
