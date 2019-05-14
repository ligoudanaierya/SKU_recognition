#!/usr/bin/env python

# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen, based on code from Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import time
import _init_paths
from model.config import cfg
from model.test import im_detect
from model.nms_wrapper import nms
import glob
from utils.timer import Timer
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os, cv2
import argparse

from nets.vgg16 import vgg16
from nets.resnet_v1 import resnetv1
from forward import forward
from KNN import pre
CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')

NETS = {'vgg16': ('vgg16_faster_rcnn_iter_70000.ckpt',),'res101': ('res101_faster_rcnn_iter_70000.ckpt',)}
DATASETS= {'pascal_voc': ('voc_2007_trainval',),'pascal_voc_0712': ('voc_2007_trainval+voc_2012_trainval',)}

def vis_detections(im, class_name, dets, thresh=0.5, im_name=''):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return
    if not os.path.exists('middle_dir/'+im_name):
       os.mkdir('middle_dir/'+im_name)
    file = open('middle_dir/'+im_name+'/'+im_name.replace('.jpg','_map.txt'),'a')

    #im2 = im[:, :, (2, 1, 0)]
    #fig, ax = plt.subplots(figsize=(12, 12))
    #ax.imshow(im2, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]
        img_cut = im[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2]),:]
        file.write(str(i)+'_'+im_name+':'+str(bbox[0])+'_'+str(bbox[1])+'_'+str(bbox[2])+'_'+str(bbox[3])+'\n')
        cv2.imwrite(os.path.join('middle_dir/'+im_name,str(i)+'_'+im_name),img_cut)

        #print(os.path.join('new_cola/' + im_name.split('/')[0], str(i) + '_' + im_name.replace('/', '_')+'.jpg'))
        #cv2.imwrite(os.path.join('no_cola/' + im_name.split('/')[0], str(i) + '_' + im_name.replace('/', '_')), img_cut)

        '''ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')'''
    file.close()
    '''ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                  fontsize=14)'''
    #.axis('off')
    #plt.tight_layout()
    #plt.savefig(os.path.join('middle_dir/'+im_name,im_name.replace('.jpg','.png')))
    #plt.draw()

def demo(sess, net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im_file = os.path.join(cfg.DATA_DIR, 'testset', image_name)
    im_file = im_file.replace('data/','tools/')
    print(im_file)
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(sess, net, im)
    timer.toc()
    print('Detection took {:.3f}s for {:d} object proposals'.format(timer.total_time, boxes.shape[0]))

    # Visualize detections for each class
    CONF_THRESH = 0.8
    NMS_THRESH = 0.3
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        vis_detections(im, cls, dets, thresh=CONF_THRESH,im_name=image_name)


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Tensorflow Faster R-CNN demo')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16 res101]',
                        choices=NETS.keys(), default='res101')
    parser.add_argument('--dataset', dest='dataset', help='Trained dataset [pascal_voc pascal_voc_0712]',
                        choices=DATASETS.keys(), default='pascal_voc')
    args = parser.parse_args()

    return args

if __name__ == '__main__':


    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    args = parse_args()

    # model path
    demonet = args.demo_net
    dataset = args.dataset
    tfmodel = os.path.join('../output', demonet, DATASETS[dataset][0], 'default',
                              NETS[demonet][0])


    if not os.path.isfile(tfmodel + '.meta'):
        raise IOError(('{:s} not found.\nDid you download the proper networks from '
                       'our server and place them properly?').format(tfmodel + '.meta'))

    # set config
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth=True

    # init session
    sess = tf.Session(config=tfconfig)
    # load network
    if demonet == 'vgg16':
        net = vgg16()
    elif demonet == 'res101':
        net = resnetv1(num_layers=101)
    else:
        raise NotImplementedError
    print("nwetnetnetnentente")
    net.create_architecture("TEST", 21,
                          tag='default', anchor_scales=[8, 16, 32])
    print("overoveoroeoreore")
    saver = tf.train.Saver()
    saver.restore(sess, tfmodel)

    print('Loaded network {:s}'.format(tfmodel))

    #im_names = ['0000328.jpg', '0000297.jpg', '0000034.jpg',
    #            '0000049.jpg', '0000291.jpg']


    listglob = []
    listglob = glob.glob(r"testset/*.jpg")
    for i in range (len(listglob)):
        print(listglob[i])
        listglob[i]=listglob[i].split('/')[-1]
    for im_name in listglob:
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('Demo for data/demo/{}'.format(im_name))
        print(im_name)
        demo(sess, net, im_name)

    tf.reset_default_graph()
    forward()
    pre()
    sess.close()
        #plt.show()
    '''list_dir = os.listdir('no_cola')
    for dir in list_dir:
        jpg_names = os.listdir('no_cola/'+dir)
        for jpg_name in jpg_names:
            name = os.path.join(dir,jpg_name)
            print(name)
            demo(sess,net,name)
            os.remove(os.path.join('no_cola',name))'''
