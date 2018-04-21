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
solver.net.copy_from('model_iter_300000.caffemodel')
#solver.restore('../models/model_iter_100.solverstate')

niter = 100000 #?
train_loss = np.zeros(niter)

f = open('loss.txt', 'w+')
f.close()

for it in range(niter): 
    solver.step(1)
    print("******************************************")
    data = solver.net.blobs['data'].data[0]
    data = data.transpose((1, 2, 0));
    #data = data[:, :, ::-1]
    cv2.imwrite("../img/data.jpg", data,[cv2.IMWRITE_JPEG_QUALITY, 100])

    label = solver.net.blobs['label'].data[0]
    label = label.transpose((1, 2, 0));
    #clear = clear[:, :, ::-1]
    cv2.imwrite("../img/label.jpg", label,[cv2.IMWRITE_JPEG_QUALITY, 100])

    out = solver.net.blobs['bn3'].data[0]
    out = out.transpose((1, 2, 0))
    out = np.repeat(out, 3, axis=2)
    #out = out[:, :, ::-1]
    cv2.imwrite("../img/out.jpg", out,[cv2.IMWRITE_JPEG_QUALITY, 100])

    train_loss[it] = solver.net.blobs['loss'].data
    print(solver.net.blobs['loss'].data)
    f = open('loss.txt', 'a')
    f.write('{d} '.format(it))
    f.write('{f}\n'.format(train_loss[it]))
    f.close();



# solver.step(80000)


