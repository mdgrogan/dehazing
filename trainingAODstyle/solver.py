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
solver.net.copy_from('AOD_Net.caffemodel')
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
    cv2.imwrite("../img/data.jpg", data * 255.0,[cv2.IMWRITE_JPEG_QUALITY, 100])

    label = solver.net.blobs['label'].data[0]
    label = label.transpose((1, 2, 0));
    #clear = clear[:, :, ::-1]
    cv2.imwrite("../img/label.jpg", label * 255.0,[cv2.IMWRITE_JPEG_QUALITY, 100])

    out = solver.net.blobs['sum'].data[0]
    out = out.transpose((1, 2, 0));
    #out = out[:, :, ::-1]
    cv2.imwrite("../img/out.jpg", out * 255.0,[cv2.IMWRITE_JPEG_QUALITY, 100])

    train_loss[it] = solver.net.blobs['loss'].data
    print(solver.net.blobs['loss'].data)
    f = open('loss.txt', 'a')
    f.write('{0:d} '.format(it))
    f.write('{0:f}\n'.format(train_loss[it]))
    f.close();



# solver.step(80000)


