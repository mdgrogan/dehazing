from __future__ import division
import sys
#caffe_root='/home/grogan/Build/caffe/'
#sys.path.insert(0, caffe_root+'python')
import caffe
import numpy as np

# init
caffe.set_mode_cpu()

solver = caffe.SGDSolver('solver.prototxt')

niter = 1000 #?
train_loss = np.zeros(niter)

f = open('loss.txt', 'w+')

for it in range(niter): 
    solver.step(1)

    print("******************************************")
    t1 = solver.net.blobs['data'].data[0]
    t2 = solver.net.blobs['clear'].data[0]
    t3 = solver.net.blobs['sum'].data[0]

    train_loss[it] = solver.net.blobs['loss'].data
    f.write('{0: d} '.format(it))
    f.write('{0: f}\n'.format(train_loss[it]))

f.close()

# solver.step(80000)


