#coding=utf-8

import os.path as osp
import sys
import copy
import os
import cv2
import numpy as np
from sys import path

path.append('/data1/henryzhong/caffe/python')
import caffe

caffe.set_mode_cpu()
result_net = caffe.Net('./models/deploy_68_new.prototxt', './models/vgg_68_new.caffemodel', caffe.TEST)
#result_net = caffe.Net('./models/result.prototxt', './models/result.caffemodel', caffe.TEST)

im = cv2.imread('10.jpg')
im.resize(60, 60, 3)
im = im.transpose(2, 0, 1)[np.newaxis, :, :, :]
data = result_net.blobs['data']
data.reshape(*im.shape)
print (im)
data.data[...] = im
print (result_net.forward())
print (result_net.blobs['fc6_landmark'].data)
#print result_net.blobs
