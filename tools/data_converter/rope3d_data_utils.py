# Copyright (c) OpenMMLab. All rights reserved.
from collections import OrderedDict
from concurrent import futures as futures
from os import path as osp
from pathlib import Path

import mmcv
import numpy as np
from PIL import Image
from skimage import io

import pdb 

def get_rope3d_info_path(name,
                        prefix,
                        info_type='image_2',
                        file_tail='.jpg',
                        training=True,
                        relative_path=True,
                        exist_check=True):
    filename = name + file_tail
    prefix = Path(prefix)
    if info_type in ['image_2', 'depth_2']:
        if training:
            file_path = Path('training-' + info_type) / filename 
        else:
            file_path = Path('validation-' + info_type) / filename 
    elif info_type in ['calib', 'denorm', 'label_2']:
        if training:
            file_path = Path('training') / info_type / filename
        else:
            file_path = Path('validation') / info_type / filename
    else:
        raise TypeError('info type of {} cannot be fetched.'.format(info_type))
    
    if exist_check and not (prefix / file_path).exists():
        raise ValueError('file not exist: {}'.format(file_path))
    if relative_path:
        return str(file_path)
    else:
        return str(prefix / file_path)


def get_image_path(name,
                   prefix,
                   training=True,
                   relative_path=True,
                   exist_check=True,
                   info_type='image_2'):
    return get_rope3d_info_path(name, prefix, info_type, '.jpg', training,
                               relative_path, exist_check)
                            
def get_depth_path(name,
                   prefix,
                   training=True,
                   relative_path=True,
                   exist_check=True,
                   info_type='depth_2'):
    return get_rope3d_info_path(name, prefix, info_type, '.jpg', training,
                               relative_path, exist_check)

def get_label_path(name,
                   prefix,
                   training=True,
                   relative_path=True,
                   exist_check=True,
                   info_type='label_2'):
    return get_rope3d_info_path(name, prefix, info_type, '.txt', training,
                               relative_path, exist_check)

def get_gplane_path(name,
                   prefix,
                   training=True,
                   relative_path=True,
                   exist_check=True,
                   info_type='denorm'):
    return get_rope3d_info_path(name, prefix, info_type, '.txt', training,
                               relative_path, exist_check)

def get_calib_path(name,
                   prefix,
                   training=True,
                   relative_path=True,
                   exist_check=True):
    return get_rope3d_info_path(name, prefix, 'calib', '.txt', training,
                               relative_path, exist_check)

def get_label_anno(label_path):
    annotations = {}
    annotations.update({
        'name': [],
        'truncated': [],
        'occluded': [],
        'alpha': [],
        'bbox': [],
        'dimensions': [],
        'location': [],
        'rotation_y': []
    })
    with open(label_path, 'r') as f:
        lines = f.readlines()
    content = [line.strip().split(' ') for line in lines]
    # re-order objects according to put dontcare at last.
    _content, dontcare = [], []
    for obj in content:
        if 'unknown' in obj[0]:
            dontcare.append(obj)
        else:
            _content.append(obj)
    content = _content + dontcare
    num_objects = len(_content)
    annotations['name'] = np.array([x[0] for x in content])
    num_gt = len(annotations['name'])
    annotations['truncated'] = np.array([float(x[1]) for x in content])
    annotations['occluded'] = np.array([int(x[2]) for x in content])
    annotations['alpha'] = np.array([float(x[3]) for x in content])
    annotations['bbox'] = np.array([[float(info) for info in x[4:8]]
                                    for x in content]).reshape(-1, 4)
    # dimensions will convert hwl format to standard lhw(camera) format.
    annotations['dimensions'] = np.array([[float(info) for info in x[8:11]]
                                          for x in content
                                          ]).reshape(-1, 3)[:, [2, 0, 1]]
    annotations['location'] = np.array([[float(info) for info in x[11:14]]
                                        for x in content]).reshape(-1, 3)
    annotations['rotation_y'] = np.array([float(x[14])
                                          for x in content]).reshape(-1)
    if len(content) != 0 and len(content[0]) == 16:  # have score
        annotations['score'] = np.array([float(x[15]) for x in content])
    else:
        annotations['score'] = np.zeros((annotations['bbox'].shape[0], ))
    index = list(range(num_objects)) + [-1] * (num_gt - num_objects)
    annotations['index'] = np.array(index, dtype=np.int32)
    annotations['group_ids'] = np.arange(num_gt, dtype=np.int32)
    return annotations


def _extend_matrix(mat):
    mat = np.concatenate([mat, np.array([[0., 0., 0., 1.]])], axis=0)
    return mat

def calc_rotp(norm_vec):
    a, b, c = float(norm_vec[0]), float(norm_vec[1]), float(norm_vec[2])
    r12, r22, r32 = a, b, c
    r11, r21, r31 = 1, - a / b, 0
    div = a ** 2 + b ** 2
    r13, r23, r33 = (- a * c) / div, (-b * c) / div, 1
    R = np.array([[r11, r12, r13], [r21, r22, r23], [r31, r32, r33]], dtype=np.float64)
    return R / np.linalg.norm(R, axis=0)

def get_rope3d_image_info(path,
                         training=True,
                         label_info=True,
                         calib=False,
                         with_plane=False,
                         image_set=[],
                         extend_matrix=True,
                         num_worker=8,
                         relative_path=True,
                         with_imageshape=True,
                         scale=1):
    """
    rope3d annotation format version 2:
    {
        [optional]points: [N, 3+] point cloud
        [optional, for kitti]image: {
            image_idx: ...
            image_path: ...
            image_shape: ...
        }
        point_cloud: {
            num_features: 4
            velodyne_path: ...
        }
        [optional, for kitti]calib: {
            R0_rect: ...
            Tr_velo_to_cam: ...
            P2: ...
        }
        annos: {
            location: [num_gt, 3] array
            dimensions: [num_gt, 3] array
            rotation_y: [num_gt] angle array
            name: [num_gt] ground truth name array
            [optional]difficulty: kitti difficulty
            [optional]group_ids: used for multi-part object
        }
    }
    """
    root_path = Path(path)
    assert isinstance(image_set, list)
    name2idx = {name: idx for idx, name in enumerate(image_set)}

    def map_func(name: str):
        info = {}
        calib_info = {}
        image_info = {'image_id': name2idx[name]}
        annotations = None
        image_info['image_path'] = get_image_path(name, path, training,
                                                  relative_path)
        image_info['depth_path'] = get_depth_path(name, path, training, 
                                                    relative_path)
        if with_imageshape:
            img_path = image_info['image_path']
            if relative_path:
                img_path = str(root_path / img_path)
            image_info['image_shape'] = np.array(
                io.imread(img_path).shape[:2], dtype=np.int32)
        if label_info:
            label_path = get_label_path(name, path, training, relative_path)
            if relative_path:
                label_path = str(root_path / label_path)
            annotations = get_label_anno(label_path)
        info['image'] = image_info

        if with_plane:
            plane_path = get_gplane_path(name, path, training, relative_path=False)
            handler = open(plane_path, 'r')
            lines = [line.strip().split(' ') for line in handler.readlines()]
            coef = list(map(lambda x: float(x), lines[0]))
            info['gplane'] = np.array(coef, dtype=np.float32)

        R_p = calc_rotp(- info['gplane'])

        if calib:
            calib_path = get_calib_path(
                name, path, training, relative_path=False)
            with open(calib_path, 'r') as f:
                lines = f.readlines()
            P2 = np.array([float(info) for info in lines[0].split(' ')[1:13]
                           ]).reshape([3, 4])            
            # transform cam-intrinstic with plane-rotation
            view_pad, R_pad, scale_pad = np.eye(4), np.eye(4), np.eye(4)
            view_pad[:3, :4] = P2
            R_pad[:3, :3] = R_p
            view_pad = np.dot(view_pad, R_pad)
            if scale != 1:
                scale_pad[:, 0] /= scale 
                scale_pad[:, 1] /= scale 
                view_pad = np.dot(scale_pad, view_pad)
            calib_info['P2'] = view_pad
            info['calib'] = calib_info
        
        # transform center from rope-coord to kitti-coord
        # (3, 3) \dot (3, n) -> (3, n) -> (n, 3)
        annotations['location'] = np.dot(np.linalg.inv(R_p), annotations['location'].transpose(1, 0)).transpose(1, 0)

        if annotations is not None:
            info['annos'] = annotations
            add_difficulty_to_annos(info)
        return info

    with futures.ThreadPoolExecutor(num_worker) as executor:
        image_infos = executor.map(map_func, image_set)

    return list(image_infos)

def add_difficulty_to_annos(info):
    min_height = [40, 25,
                  25]  # minimum height for evaluated groundtruth/detections
    max_occlusion = [
        0, 1, 2
    ]  # maximum occlusion level of the groundtruth used for evaluation
    max_trunc = [
        0.15, 0.3, 0.5
    ]  # maximum truncation level of the groundtruth used for evaluation
    annos = info['annos']
    dims = annos['dimensions']  # lhw format
    bbox = annos['bbox']
    height = bbox[:, 3] - bbox[:, 1]
    occlusion = annos['occluded']
    truncation = annos['truncated']
    diff = []
    easy_mask = np.ones((len(dims), ), dtype=np.bool)
    moderate_mask = np.ones((len(dims), ), dtype=np.bool)
    hard_mask = np.ones((len(dims), ), dtype=np.bool)
    i = 0
    for h, o, t in zip(height, occlusion, truncation):
        if o > max_occlusion[0] or h <= min_height[0] or t > max_trunc[0]:
            easy_mask[i] = False
        if o > max_occlusion[1] or h <= min_height[1] or t > max_trunc[1]:
            moderate_mask[i] = False
        if o > max_occlusion[2] or h <= min_height[2] or t > max_trunc[2]:
            hard_mask[i] = False
        i += 1
    is_easy = easy_mask
    is_moderate = np.logical_xor(easy_mask, moderate_mask)
    is_hard = np.logical_xor(hard_mask, moderate_mask)

    for i in range(len(dims)):
        # if is_easy[i]:
        #     diff.append(0)
        # elif is_moderate[i]:
        #     diff.append(1)
        # elif is_hard[i]:
        #     diff.append(2)
        # else:
        #     diff.append(-1)
        diff.append(0)
    annos['difficulty'] = np.array(diff, np.int32)
    return diff


def kitti_result_line(result_dict, precision=4):
    prec_float = '{' + ':.{}f'.format(precision) + '}'
    res_line = []
    all_field_default = OrderedDict([
        ('name', None),
        ('truncated', -1),
        ('occluded', -1),
        ('alpha', -10),
        ('bbox', None),
        ('dimensions', [-1, -1, -1]),
        ('location', [-1000, -1000, -1000]),
        ('rotation_y', -10),
        ('score', 0.0),
    ])
    res_dict = [(key, None) for key, val in all_field_default.items()]
    res_dict = OrderedDict(res_dict)
    for key, val in result_dict.items():
        if all_field_default[key] is None and val is None:
            raise ValueError('you must specify a value for {}'.format(key))
        res_dict[key] = val

    for key, val in res_dict.items():
        if key == 'name':
            res_line.append(val)
        elif key in ['truncated', 'alpha', 'rotation_y', 'score']:
            if val is None:
                res_line.append(str(all_field_default[key]))
            else:
                res_line.append(prec_float.format(val))
        elif key == 'occluded':
            if val is None:
                res_line.append(str(all_field_default[key]))
            else:
                res_line.append('{}'.format(val))
        elif key in ['bbox', 'dimensions', 'location']:
            if val is None:
                res_line += [str(v) for v in all_field_default[key]]
            else:
                res_line += [prec_float.format(v) for v in val]
        else:
            raise ValueError('unknown key. supported key:{}'.format(
                res_dict.keys()))
    return ' '.join(res_line)
