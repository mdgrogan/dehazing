from __future__ import division
import sys
caffe_root='/home/grogan/Build/caffe/'
sys.path.insert(0, caffe_root+'python')
import caffe
import numpy as np
import cv2

# init
caffe.set_mode_cpu()

solver = caffe.SGDSolver('solver.prototxt')
#solver.net.copy_from('model_iter_300000.caffemodel')
#solver.restore('../models/model_iter_100.solverstate')

niter = 100000 #?
train_loss = np.zeros(niter)

f = open('loss.txt', 'w+')
f.close()

for it in range(niter): 
    solver.step(1)
    print("******************************************")
    #data = solver.net.blobs['data'].data[0]
    #data = data.transpose((1, 2, 0));
    #cv2.imwrite("tmp.png", 255*data)

    #label = solver.net.blobs['label'].data[0]
    #label = label.transpose((1, 2, 0));
    #label = np.repeat(label, 3, axis=2)
    #clear = clear[:, :, ::-1]
    #cv2.imwrite("img/label.jpg", 255*label,[cv2.IMWRITE_JPEG_QUALITY, 100])
    #cv2.imwrite("img/label.jpg", label,[cv2.IMWRITE_JPEG_QUALITY, 100])

    #out = solver.net.blobs['bn3'].data[0]
    #out = out.transpose((1, 2, 0))
    #out = np.repeat(out, 3, axis=2)
    #out = out[:, :, ::-1]
    #cv2.imwrite("img/out.jpg", 255*out,[cv2.IMWRITE_JPEG_QUALITY, 100])
    #cv2.imwrite("img/out.jpg", out,[cv2.IMWRITE_JPEG_QUALITY, 100])

    #print(solver.net.blobs['conv1A'].data[0])
    #print(solver.net.blobs['conv2A'].data[0])
    #print(solver.net.blobs['conv3A'].data[0])
    #print("bn4_\n", solver.net.blobs['bn4_'].data[0])
    print("out\n [{0:.8f} {1:.8f}]".format(solver.net.blobs['out'].data[0,0,0,0,],
                                          solver.net.blobs['out'].data[0,1,0,0,]))
    print("label\n", solver.net.blobs['label'].data[0,:,0,0])


    train_loss[it] = solver.net.blobs['loss'].data
    print("loss:", solver.net.blobs['loss'].data)
    f = open('loss.txt', 'a')
    f.write('{0:d} '.format(it))
    f.write('{0:f}\n'.format(train_loss[it]))
    f.close();



# solver.step(80000)


