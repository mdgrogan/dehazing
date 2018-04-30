import caffe
import numpy as np
from random import shuffle
import cv2

#prototxt:
#layer {
#    name: "data"
#    type: "Python"
#    top: "data"
#    top: "label"
#    python_param {
#        module: "pairedInputLayer"
#        layer: "pairedInput"
#        param_str :'{"src_file": "/path/train.txt", "batch_size": 8,'\ 
#                   '"width": 550, "height": 413}'
#    }
#}

class pairedInput(caffe.Layer):
    def setup(self, bottom, top):
        if len(top) != 2:
            raise Exception("Need data and label defined")
        if len(bottom) != 0:
            raise Exception("Do not define a bottom")

        # Read parameters
        params = eval(self.param_str)
        src_file = params["src_file"]
        self.batch_size = params["batch_size"]
        self.width = params["width"]
        self.height = params["height"]

        # Reshape top
        top[0].reshape(self.batch_size, 3, self.height, self.width)
        top[1].reshape(self.batch_size, 2, 1, 1)

        # Populate file list
        self.flist = [line.rstrip('\n') for 
                line in open(src_file)]
        shuffle(self.flist)
        self._cur = 0 # to check if need to restart list of images

###############################################################################
    def forward(self, bottom, top):
        # If finished with all images, epoch is finished - time to start over
        for it in range(self.batch_size):
            if self._cur == len(self.flist):
                self._cur = 0
                shuffle(self.flist)
            items = self.flist[self._cur].split(" ")
            #print(items[0])
            #print(items[1])
            #print(items[2])
            im = cv2.imread(items[0])
            # resize
            im = cv2.resize(im, (self.width, self.height))
            # [0,255] -> [0,1]
            im = im/255.0
            # (H,W,C) -> (C,H,W)
            im = im.transpose((2,0,1))
            # label is clip, tile. Plus an attempt at scaling
            label = np.array([float(items[1])/3.0, float(items[2])/30.0])
            label = label.reshape(2, 1, 1)
            # Add directly to top blob
            top[0].data[it, ...] = im
            top[1].data[it, ...] = label
            self._cur += 1

###############################################################################
    def load_next_image(self):
        # If finished with all images, epoch is finished - time to start over
        if self._cur == len(self.flist):
            self._cur = 0
            shuffle(self.flist)
        
        # flist has lines of format 0001_0.8_0.04
        # training images are 0001_0.8_0.04.jpg
        # label images are 0001.jpg
        scene = self.flist[self._cur]
        base = scene.split("_")
        label = cv2.imread(self.label_folder + "/" + base[0] + ".jpg")
        im = cv2.imread(self.data_folder + "/" + scene + ".jpg")
        self._cur += 1
        return im, label

###############################################################################
    def reshape(self, bottom, top):
        # shouldn't need to reshape anything, right?
        pass

###############################################################################
    def backward(self, top, propagate_down, bottom):
        pass
