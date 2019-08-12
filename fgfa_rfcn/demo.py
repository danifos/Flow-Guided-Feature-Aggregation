# --------------------------------------------------------
# Flow-Guided Feature Aggregation
# Copyright (c) 2017 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Yuqing Zhu, Shuhao Fu, Xizhou Zhu, Yi Li, Haochen Zhang
# --------------------------------------------------------

import _init_paths

import argparse
import os
import glob
import sys
import time

import logging
import pprint
import cv2
from config.config import config as cfg
cfg.TEST.KEY_FRAME_INTERVAL = 5
from config.config import update_config
from utils.image import resize, transform
import numpy as np
from collections import deque

def preprocess(feat):
    # feat[:, [847, 584, 946, 578, 544, 527, 694, 863, 1004, 749, 539, 910, 322, 964, 849], :, :] = 0
    return
    feat[:, [5, 19, 25, 39, 46, 84, 95, 104, 106, 112, 139, 152, 165, 168, 185, 226, 232, 267, 274, 284, 296, 322, 341, 346, 362, 366, 375, 383, 390, 397, 419, 429, 435, 440, 454, 461, 485, 492, 493, 511, 513, 527, 538, 539, 544, 555, 578, 584, 626, 631, 694, 749, 750, 831, 847, 849, 863, 877, 879, 896, 897, 910, 944, 946, 964, 1004, 1013], :, :] = 0

# get config
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
os.environ['MXNET_ENABLE_GPU_P2P'] = '0'
cur_path = os.path.abspath(os.path.dirname(__file__))
update_config(cur_path + '/../experiments/fgfa_rfcn/cfgs/fgfa_rfcn_vid_demo.yaml')

sys.path.insert(0, os.path.join(cur_path, '../external/mxnet/', cfg.MXNET_VERSION))
import mxnet as mx
import time
from core.tester import im_detect, im_detect_feat, im_detect_rfcn, Predictor, get_resnet_output, prepare_data, prepare_aggregation, draw_all_detection
from symbols import *
from nms.seq_nms import seq_nms
from utils.load_model import load_param
from utils.tictoc import tic, toc
from nms.nms import py_nms_wrapper, cpu_nms_wrapper, gpu_nms_wrapper

def parse_args():
    parser = argparse.ArgumentParser(description='Show Flow-Guided Feature Aggregation demo')
    args = parser.parse_args()
    return args

args = parse_args()




def process_pred_result(classes, pred_result, num_classes, thresh, cfg, nms, all_boxes, idx, max_per_image, vis, center_image, scales):
    for delta, (scores, boxes, data_dict) in enumerate(pred_result):
        for j in range(1,num_classes):
            indexes = np.where(scores[:, j] > thresh)[0]
            cls_scores = scores[indexes, j, np.newaxis]
            cls_boxes = boxes[indexes, 4:8] if cfg.CLASS_AGNOSTIC else boxes[indexes, j * 4:(j + 1) * 4]
            cls_dets = np.hstack((cls_boxes, cls_scores))
            if cfg.TEST.SEQ_NMS:
                all_boxes[j][idx+delta]=cls_dets
            else:
                cls_dets=np.float32(cls_dets)
                keep = nms(cls_dets)
                all_boxes[j][idx + delta] = cls_dets[keep, :]

        if cfg.TEST.SEQ_NMS==False and  max_per_image > 0:
            image_scores = np.hstack([all_boxes[j][idx + delta][:, -1]
                                      for j in range(1, num_classes)])
            if len(image_scores) > max_per_image:
                image_thresh = np.sort(image_scores)[-max_per_image]
                for j in range(1, num_classes):
                    keep = np.where(all_boxes[j][idx + delta][:, -1] >= image_thresh)[0]
                    all_boxes[j][idx + delta] = all_boxes[j][idx + delta][keep, :]

            boxes_this_image = [[]] + [all_boxes[j][idx + delta] for j in range(1, num_classes)]

            out_im = draw_all_detection(center_image, boxes_this_image, classes, scales[delta], cfg, threshold=0.5)

            return out_im
    return 0


def save_image(output_dir, count, out_im):
    filename = '{:04d}.JPEG'.format(count)
    cv2.imwrite(output_dir + filename, out_im)

class VideoLoader:
    def __init__(self, images_names):
        self.images_names = images_names

        # Load one frame first
        self.data_template = self.__getitem__(0)

    def _load_frame(self, idx):
        im_name = self.images_names[idx]
        assert os.path.exists(im_name), ('%s does not exist'.format(im_name))
        im = cv2.imread(im_name, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        target_size = cfg.SCALES[0][0]
        max_size = cfg.SCALES[0][1]
        im, im_scale = resize(im, target_size, max_size, stride=cfg.network.IMAGE_STRIDE)
        im_tensor = transform(im, cfg.network.PIXEL_MEANS)
        im_info = np.array([[im_tensor.shape[2], im_tensor.shape[3], im_scale]], dtype=np.float32)

        feat_stride = float(cfg.network.RCNN_FEAT_STRIDE)
        return mx.nd.array(im_tensor), mx.nd.array(im_info)

    def __getitem__(self, index):
        if index == 0 and hasattr(self, 'data_template'):
            return self.data_template
        im_tensor, im_info = self._load_frame(index)
        return [im_tensor, im_info, im_tensor, im_tensor, im_tensor]

    def __len__(self):
        return len(self.images_names)

def main():
    # get symbol
    pprint.pprint(cfg)
    print '===================='
    print 'SEQ_NMS:', cfg.TEST.SEQ_NMS
    print '===================='
    cfg.symbol = 'resnet_v1_101_flownet_rfcn'
    model = '/../model/rfcn_fgfa_flownet_vid'
    all_frame_interval = cfg.TEST.KEY_FRAME_INTERVAL * 2 + 1
    max_per_image = cfg.TEST.max_per_image
    feat_sym_instance = eval(cfg.symbol + '.' + cfg.symbol)()
    aggr_sym_instance = eval(cfg.symbol + '.' + cfg.symbol)()

    feat_sym = feat_sym_instance.get_feat_symbol(cfg)
    # aggr_sym = aggr_sym_instance.get_aggregation_symbol(cfg)
    # aggr_sym_feat = aggr_sym_instance.get_aggregation_symbol_feat(cfg, 3)
    intervals = [1, 3, 5]
    aggr_sym_feat_array = [aggr_sym_instance.get_aggregation_symbol_feat(cfg, i) for i in intervals]
    aggr_sym_rfcn = aggr_sym_instance.get_aggregation_symbol_rfcn(cfg)

    # set up class names
    num_classes = 31
    classes = ['__background__','airplane', 'antelope', 'bear', 'bicycle',
               'bird', 'bus', 'car', 'cattle',
               'dog', 'domestic_cat', 'elephant', 'fox',
               'giant_panda', 'hamster', 'horse', 'lion',
               'lizard', 'monkey', 'motorcycle', 'rabbit',
               'red_panda', 'sheep', 'snake', 'squirrel',
               'tiger', 'train', 'turtle', 'watercraft',
               'whale', 'zebra']

    # load demo data

    # image_names = glob.glob(cur_path + '/../demo/ILSVRC2015_val_00007010/*.JPEG')
    video_index = 'val_00007010'
    image_names = glob.glob('/home/user/ILSVRC2015/Data/VID/val/ILSVRC2015_{}/*.JPEG'.format(video_index))
    image_names.sort()
    # output_dir = cur_path + '/../demo/rfcn_fgfa/'
    output_dir = cur_path + '/../demo/{}/'.format(video_index)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # data = []
    # for im_name in image_names:
    #     assert os.path.exists(im_name), ('%s does not exist'.format(im_name))
    #     im = cv2.imread(im_name, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    #     target_size = cfg.SCALES[0][0]
    #     max_size = cfg.SCALES[0][1]
    #     im, im_scale = resize(im, target_size, max_size, stride=cfg.network.IMAGE_STRIDE)
    #     im_tensor = transform(im, cfg.network.PIXEL_MEANS)
    #     im_info = np.array([[im_tensor.shape[2], im_tensor.shape[3], im_scale]], dtype=np.float32)

    #     feat_stride = float(cfg.network.RCNN_FEAT_STRIDE)
    #     data.append({'data': im_tensor, 'im_info': im_info, 'aggregated_conv_feat_cache':im_tensor, 'data_cache': im_tensor,    'feat_cache': im_tensor})
    feat_stride = float(cfg.network.RCNN_FEAT_STRIDE)



    # get predictor

    print 'get-predictor'
    data_names = ['data', 'im_info', 'aggregated_conv_feat_cache', 'data_cache', 'feat_cache']
    label_names = []

    t1 = time.time()
    interval = cfg.TEST.KEY_FRAME_INTERVAL * 2 + 1
    # data = [[mx.nd.array(data[i][name]) for name in data_names] for i in xrange(len(data))]
    data = VideoLoader(image_names)
    max_data_shape = [[('data', (1, 3, max([v[0] for v in cfg.SCALES]), max([v[1] for v in cfg.SCALES]))),
                       ('aggregated_conv_feat_cache', ((1, 1024,
                                                np.ceil(max([v[0] for v in cfg.SCALES]) / feat_stride).astype(np.int),
                                                np.ceil(max([v[1] for v in cfg.SCALES]) / feat_stride).astype(np.int)))),
                       ('data_cache', (interval, 3, max([v[0] for v in cfg.SCALES]), max([v[1] for v in cfg.SCALES]))),
                       ('feat_cache', ((interval, cfg.network.FGFA_FEAT_DIM,
                                                np.ceil(max([v[0] for v in cfg.SCALES]) / feat_stride).astype(np.int),
                                                np.ceil(max([v[1] for v in cfg.SCALES]) / feat_stride).astype(np.int))))]]
    provide_data = [[(k, v.shape) for k, v in zip(data_names, data[0])] for _ in xrange(len(data))]
    provide_label = [None for _ in xrange(len(data))]

    arg_params, aux_params = load_param(cur_path + model, 0, process=True)

    feat_predictors = Predictor(feat_sym, data_names, label_names,
                          context=[mx.gpu(0)], max_data_shapes=max_data_shape,
                          provide_data=provide_data, provide_label=provide_label,
                          arg_params=arg_params, aux_params=aux_params)
#     aggr_predictors = Predictor(aggr_sym, data_names, label_names,
#                           context=[mx.gpu(0)], max_data_shapes=max_data_shape,
#                           provide_data=provide_data, provide_label=provide_label,
#                           arg_params=arg_params, aux_params=aux_params)
#     aggr_predictors_feat = Predictor(aggr_sym_feat, data_names, label_names,
#                           context=[mx.gpu(0)], max_data_shapes=max_data_shape,
#                           provide_data=provide_data, provide_label=provide_label,
#                           arg_params=arg_params, aux_params=aux_params)
    aggr_predictors_feat_array = [Predictor(aggr_sym_feat, data_names, label_names,
                          context=[mx.gpu(0)], max_data_shapes=max_data_shape,
                          provide_data=provide_data, provide_label=provide_label,
                          arg_params=arg_params, aux_params=aux_params) \
                                  for aggr_sym_feat in aggr_sym_feat_array]
    aggr_predictors_rfcn = Predictor(aggr_sym_rfcn, data_names, label_names,
                          context=[mx.gpu(0)], max_data_shapes=max_data_shape,
                          provide_data=provide_data, provide_label=provide_label,
                          arg_params=arg_params, aux_params=aux_params)
    nms = py_nms_wrapper(cfg.TEST.NMS)


    # First frame of the video
    idx = 0
    data_batch = mx.io.DataBatch(data=[data[idx]], label=[], pad=0, index=idx,
                                 provide_data=[[(k, v.shape) for k, v in zip(data_names, data[idx])]],
                                 provide_label=[None])
    scales = [data_batch.data[i][1].asnumpy()[0, 2] for i in xrange(len(data_batch.data))]
    all_boxes = [[[] for _ in range(len(data))]
                 for _ in range(num_classes)]
    data_list = deque(maxlen=all_frame_interval)
    feat_list = deque(maxlen=all_frame_interval)
    image, feat = get_resnet_output(feat_predictors, data_batch, data_names)
    # append cfg.TEST.KEY_FRAME_INTERVAL padding images in the front (first frame)
    while len(data_list) < cfg.TEST.KEY_FRAME_INTERVAL:
        data_list.append(image)
        preprocess(feat)
        feat_list.append(feat)

    vis = False
    file_idx = 0
    thresh = 1e-3
    load_end = time.time()
    for idx, element in enumerate(data):

        data_batch = mx.io.DataBatch(data=[element], label=[], pad=0, index=idx,
                                     provide_data=[[(k, v.shape) for k, v in zip(data_names, element)]],
                                     provide_label=[None])
        scales = [data_batch.data[i][1].asnumpy()[0, 2] for i in xrange(len(data_batch.data))]

        if(idx != len(data)-1):

            if len(data_list) < all_frame_interval - 1:
                image, feat = get_resnet_output(feat_predictors, data_batch, data_names)
                data_list.append(image)
                preprocess(feat)
                feat_list.append(feat)

            else:
                #################################################
                # main part of the loop
                #################################################
                image, feat = get_resnet_output(feat_predictors, data_batch, data_names)
                data_list.append(image)
                preprocess(feat)
                feat_list.append(feat)

                prepare_data(data_list, feat_list, data_batch)
                aggr_feat = im_detect_feat(aggr_predictors_feat_array, data_batch, data_names, scales, cfg, intervals)
                aggr_feat = list(aggr_feat)[-1]
                prepare_aggregation(aggr_feat, data_batch)
                pred_result = im_detect_rfcn(aggr_predictors_rfcn, data_batch, data_names, scales, cfg)
                data_batch.data[0][-3] = None
                data_batch.provide_data[0][-3] = ('aggregated_conv_feat_cache', None)
                data_batch.data[0][-2] = None
                data_batch.provide_data[0][-2] = ('data_cache', None)
                data_batch.data[0][-1] = None
                data_batch.provide_data[0][-1] = ('feat_cache', None)

                out_im = process_pred_result(classes, pred_result, num_classes, thresh, cfg, nms, all_boxes, file_idx, max_per_image, vis,
                                    data_list[cfg.TEST.KEY_FRAME_INTERVAL].asnumpy(), scales)
                total_time = time.time()-t1
                if (cfg.TEST.SEQ_NMS==False):
                    save_image(output_dir, file_idx, out_im)
                print 'testing {} {:.4f}s'.format(str(file_idx)+'.JPEG', total_time /(file_idx+1))
                file_idx += 1

        else:
            #################################################
            # end part of a video                           #
            #################################################

            end_counter = 0
            image, feat = get_resnet_output(feat_predictors, data_batch, data_names)
            while end_counter < cfg.TEST.KEY_FRAME_INTERVAL + 1:
                data_list.append(image)
                preprocess(feat)
                feat_list.append(feat)
                prepare_data(data_list, feat_list, data_batch)
                aggr_feat = im_detect_feat(aggr_predictors_feat_array, data_batch, data_names, scales, cfg, intervals)
                aggr_feat = list(aggr_feat)[-1]
                prepare_aggregation(aggr_feat, data_batch)
                pred_result = im_detect_rfcn(aggr_predictors_rfcn, data_batch, data_names, scales, cfg)

                out_im = process_pred_result(classes, pred_result, num_classes, thresh, cfg, nms, all_boxes, file_idx, max_per_image, vis,
                                    data_list[cfg.TEST.KEY_FRAME_INTERVAL].asnumpy(), scales)

                total_time = time.time() - t1
                if (cfg.TEST.SEQ_NMS == False):
                    save_image(output_dir, file_idx, out_im)
                print 'testing {} {:.4f}s'.format(str(file_idx)+'.JPEG', total_time / (file_idx+1))
                file_idx += 1
                end_counter+=1
        load_end = time.time()

    if(cfg.TEST.SEQ_NMS):
        video = [all_boxes[j][:] for j in range(1, num_classes)]
        dets_all = seq_nms(video)
        for cls_ind, dets_cls in enumerate(dets_all):
            for frame_ind, dets in enumerate(dets_cls):
                keep = nms(dets)
                all_boxes[cls_ind + 1][frame_ind] = dets[keep, :]
        for idx in range(len(data)):
            boxes_this_image = [[]] + [all_boxes[j][idx] for j in range(1, num_classes)]
            out_im = draw_all_detection(data[idx][0].asnumpy(), boxes_this_image, classes, scales[0], cfg, threshold=0.5)
            save_image(output_dir, idx, out_im)

    print 'done'

if __name__ == '__main__':
    main()
