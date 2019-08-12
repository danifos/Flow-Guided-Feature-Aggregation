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
from config.config import update_config
from utils.image import resize, transform
import numpy as np
from collections import deque


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
from utils.load_model import load_param
from utils.tictoc import tic, toc


class FGFADetector:

    def __init__(self):
        # get symbol
        # pprint.pprint(cfg)
        cfg.symbol = 'resnet_v1_101_flownet_rfcn'
        model = '/../model/rfcn_fgfa_flownet_vid'
        all_frame_interval = cfg.TEST.KEY_FRAME_INTERVAL * 2 + 1
        max_per_image = cfg.TEST.max_per_image
        feat_sym_instance = eval(cfg.symbol + '.' + cfg.symbol)()
        aggr_sym_instance = eval(cfg.symbol + '.' + cfg.symbol)()

        feat_sym = feat_sym_instance.get_feat_symbol_origin(cfg)
        aggr_sym = aggr_sym_instance.get_aggregation_symbol(cfg)

        self.model = model
        self.all_frame_interval = all_frame_interval
        self.feat_sym = feat_sym
        self.aggr_sym = aggr_sym


    def predict(self, images, feat_output, aggr_feat_output):

        model = self.model
        all_frame_interval = self.all_frame_interval
        feat_sym = self.feat_sym
        aggr_sym = self.aggr_sym

        # load video data
        data = []
        for im in images:
            target_size = cfg.SCALES[0][0]
            max_size = cfg.SCALES[0][1]
            im, im_scale = resize(im, target_size, max_size, stride=cfg.network.IMAGE_STRIDE)
            im_tensor = transform(im, cfg.network.PIXEL_MEANS)
            im_info = np.array([[im_tensor.shape[2], im_tensor.shape[3], im_scale]], dtype=np.float32)

            feat_stride = float(cfg.network.RCNN_FEAT_STRIDE)
            data.append({'data': im_tensor, 'im_info': im_info, 'data_cache': im_tensor,    'feat_cache': im_tensor})


        # get predictor

        data_names = ['data', 'im_info', 'data_cache', 'feat_cache']
        label_names = []

        t1 = time.time()
        interval = cfg.TEST.KEY_FRAME_INTERVAL * 2 + 1
        data = [[mx.nd.array(data[i][name]) for name in data_names] for i in xrange(len(data))]
        max_data_shape = [[('data', (1, 3, max([v[0] for v in cfg.SCALES]), max([v[1] for v in cfg.SCALES]))),
                           ('data_cache', (interval, 3, max([v[0] for v in cfg.SCALES]), max([v[1] for v in cfg.SCALES]))),
                           ('feat_cache', ((interval, cfg.network.FGFA_FEAT_DIM,
                                                    np.ceil(max([v[0] for v in cfg.SCALES]) / feat_stride).astype(np.int),
                                                    np.ceil(max([v[1] for v in cfg.SCALES]) / feat_stride).astype(np.int))))]]
        provide_data = [[(k, v.shape) for k, v in zip(data_names, data[i])] for i in xrange(len(data))]
        provide_label = [None for _ in xrange(len(data))]

        arg_params, aux_params = load_param(cur_path + model, 0, process=True)

        feat_predictors = Predictor(feat_sym, data_names, label_names,
                              context=[mx.gpu(0)], max_data_shapes=max_data_shape,
                              provide_data=provide_data, provide_label=provide_label,
                              arg_params=arg_params, aux_params=aux_params)
        aggr_predictors = Predictor(aggr_sym, data_names, label_names,
                              context=[mx.gpu(0)], max_data_shapes=max_data_shape,
                              provide_data=provide_data, provide_label=provide_label,
                              arg_params=arg_params, aux_params=aux_params)

        # First frame of the video
        idx = 0
        data_batch = mx.io.DataBatch(data=[data[idx]], label=[], pad=0, index=idx,
                                     provide_data=[[(k, v.shape) for k, v in zip(data_names, data[idx])]],
                                     provide_label=[None])
        scales = [data_batch.data[i][1].asnumpy()[0, 2] for i in xrange(len(data_batch.data))]
        data_list = deque(maxlen=all_frame_interval)
        feat_list = deque(maxlen=all_frame_interval)
        image, feat = get_resnet_output(feat_predictors, data_batch, data_names)
        # if feat_output is not None:
        #     feat_output.append(feat.asnumpy()[0][:1024])
        # append cfg.TEST.KEY_FRAME_INTERVAL padding images in the front (first frame)
        while len(data_list) < cfg.TEST.KEY_FRAME_INTERVAL:
            data_list.append(image)
            feat_list.append(feat)

        vis = False
        file_idx = 0
        thresh = 1e-3
        for idx, element in enumerate(data):

            data_batch = mx.io.DataBatch(data=[element], label=[], pad=0, index=idx,
                                         provide_data=[[(k, v.shape) for k, v in zip(data_names, element)]],
                                         provide_label=[None])
            scales = [data_batch.data[i][1].asnumpy()[0, 2] for i in xrange(len(data_batch.data))]

            if(idx != len(data)-1):

                if len(data_list) < all_frame_interval - 1:
                    image, feat = get_resnet_output(feat_predictors, data_batch, data_names)
                    if feat_output is not None:
                        feat_output.append(feat.asnumpy()[0][:1024])
                    data_list.append(image)
                    feat_list.append(feat)

                else:
                    #################################################
                    # main part of the loop
                    #################################################
                    image, feat = get_resnet_output(feat_predictors, data_batch, data_names)
                    if feat_output is not None:
                        feat_output.append(feat.asnumpy()[0][:1024])
                    data_list.append(image)
                    feat_list.append(feat)

                    prepare_data(data_list, feat_list, data_batch)
                    pred_result, aggr_feat = im_detect(aggr_predictors, data_batch, data_names, scales, cfg, aggr_feats=True)
                    assert len(aggr_feat) == 1
                    if aggr_feat_output is not None:
                        aggr_feat_output.append(aggr_feat[0].asnumpy()[0])

                    data_batch.data[0][-2] = None
                    data_batch.provide_data[0][-2] = ('data_cache', None)
                    data_batch.data[0][-1] = None
                    data_batch.provide_data[0][-1] = ('feat_cache', None)

                    print '\r(main) Testing FGFA R-FCN: {} / {}'.format(file_idx, len(images)),
                    file_idx += 1

            else:
                #################################################
                # end part of a video                           #
                #################################################

                end_counter = 0
                image, feat = get_resnet_output(feat_predictors, data_batch, data_names)
                if feat_output is not None:
                    feat_output.append(feat.asnumpy()[0][:1024])
                while end_counter < cfg.TEST.KEY_FRAME_INTERVAL + 1:
                    data_list.append(image)
                    feat_list.append(feat)
                    prepare_data(data_list, feat_list, data_batch)
                    pred_result, aggr_feat = im_detect(aggr_predictors, data_batch, data_names, scales, cfg, aggr_feats=True)
                    assert len(aggr_feat) == 1
                    if aggr_feat_output is not None:
                        aggr_feat_output.append(aggr_feat[0].asnumpy()[0])

                    print '\r(end) Testing FGFA R-FCN: {} / {}'.format(file_idx, len(images)),
                    file_idx += 1
                    end_counter+=1

        print
