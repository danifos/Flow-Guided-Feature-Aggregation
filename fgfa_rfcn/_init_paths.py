# --------------------------------------------------------
# Flow-Guided Feature Aggregation
# Copyright (c) 2017 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Xizhou Zhu
# --------------------------------------------------------

import os
import os.path as osp
import sys

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = osp.dirname(__file__)

lib_path = osp.join(this_dir, '..', 'lib')
add_path(lib_path)

sys.path.insert(0, os.readlink(osp.join(this_dir, 'video_dissect')))
