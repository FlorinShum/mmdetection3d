# Copyright (c) OpenMMLab. All rights reserved.
from collections import OrderedDict
from pathlib import Path

import mmcv
import numpy as np
from nuscenes.utils.geometry_utils import view_points

from mmdet3d.core.bbox import box_np_ops, points_cam2img
from .rope3d_data_utils import get_rope3d_image_info
from .nuscenes_converter import post_process_coords

rope3d_categories = ('car', 'pedestrian', 'motorcyclist', 'cyclist', 
                    'van', 'bus', 'tricyclist', 'truck', 'trafficcone')

def convert_to_kitti_info_version2(info):
    """convert kitti info v1 to v2 if possible.

    Args:
        info (dict): Info of the input kitti data.
            - image (dict): image info
            - calib (dict): calibration info
            - point_cloud (dict): point cloud info
    """
    if 'image' not in info or 'calib' not in info or 'point_cloud' not in info:
        info['image'] = {
            'image_shape': info['img_shape'],
            'image_idx': info['image_idx'],
            'image_path': info['img_path'],
        }
        info['calib'] = {
            'R0_rect': info['calib/R0_rect'],
            'Tr_velo_to_cam': info['calib/Tr_velo_to_cam'],
            'P2': info['calib/P2'],
        }
        info['point_cloud'] = {
            'velodyne_path': info['velodyne_path'],
        }

def create_rope3d_info_file(data_path,
                           pkl_prefix='rope3d',
                           with_plane=False,
                           save_path=None,
                           relative_path=True):
    """Create info file of rope3d dataset.

    Given the raw data, generate its related info file in pkl format.

    Args:
        data_path (str): Path of the data root.
        pkl_prefix (str, optional): Prefix of the info file to be generated.
            Default: 'rope3d'.
        with_plane (bool, optional): Whether to use plane information.
            Default: False.
        save_path (str, optional): Path to save the info file.
            Default: None.
        relative_path (bool, optional): Whether to use relative path.
            Default: True.
    """
    root_path = Path(data_path)
    
    def _load_set(filename):
        handler = open(filename, 'r')
        return [line.strip() for line in handler.readlines()]
    
    train_img_set = _load_set(str(root_path / 'training/train.txt'))
    val_img_set = _load_set(str(root_path / 'validation/val.txt'))

    print('Generate info. this may take several minutes.')
    if save_path is None:
        save_path = Path(data_path)
    else:
        save_path = Path(save_path)
    rope3d_infos_train = get_rope3d_image_info(
        data_path,
        training=True,
        calib=True,
        with_plane=with_plane,
        image_set=train_img_set,
        relative_path=relative_path)
    filename = save_path / f'{pkl_prefix}_infos_train.pkl'
    print(f'Rope3D info train file is saved to {filename}')
    mmcv.dump(rope3d_infos_train, filename)
    rope3d_infos_val = get_rope3d_image_info(
        data_path,
        training=False,
        calib=True,
        with_plane=with_plane,
        image_set=val_img_set,
        relative_path=relative_path)
    filename = save_path / f'{pkl_prefix}_infos_val.pkl'
    print(f'Rope3D info val file is saved to {filename}')
    mmcv.dump(rope3d_infos_val, filename)

def export_2d_annotation(root_path, info_path, mono3d=True):
    """Export 2d annotation from the info file and raw data.

    Args:
        root_path (str): Root path of the raw data.
        info_path (str): Path of the info file.
        mono3d (bool, optional): Whether to export mono3d annotation.
            Default: True.
    """
    # get bbox annotations for camera
    rope3d_infos = mmcv.load(info_path)
    cat2Ids = [
        dict(id=rope3d_categories.index(cat_name), name=cat_name)
        for cat_name in rope3d_categories
    ]
    coco_ann_id = 0
    coco_2d_dict = dict(annotations=[], images=[], categories=cat2Ids)
    from os import path as osp
    for info in mmcv.track_iter_progress(rope3d_infos):
        coco_infos = get_2d_boxes(info, occluded=[0, 1, 2, 3], mono3d=mono3d)
        (height, width,
         _) = mmcv.imread(osp.join(root_path,
                                   info['image']['image_path'])).shape
        coco_2d_dict['images'].append(
            dict(
                file_name=info['image']['image_path'],
                id=info['image']['image_id'],
                cam_intrinsic=info['calib']['P2'],
                gplane=info['gplane'],
                width=width,
                height=height))
        for coco_info in coco_infos:
            if coco_info is None:
                continue
            # add an empty key for coco format
            coco_info['segmentation'] = []
            coco_info['id'] = coco_ann_id
            coco_2d_dict['annotations'].append(coco_info)
            coco_ann_id += 1
    if mono3d:
        json_prefix = f'{info_path[:-4]}_mono3d'
    else:
        json_prefix = f'{info_path[:-4]}'
    mmcv.dump(coco_2d_dict, f'{json_prefix}.coco.json')

def get_2d_boxes(info, occluded, mono3d=True):
    """Get the 2D annotation records for a given info.

    Args:
        info: Information of the given sample data.
        occluded: Integer (0, 1, 2, 3) indicating occlusion state:
            0 = fully visible, 1 = partly occluded, 2 = largely occluded,
            3 = unknown, -1 = DontCare
        mono3d (bool): Whether to get boxes with mono3d annotation.

    Return:
        list[dict]: List of 2D annotation record that belongs to the input
            `sample_data_token`.
    """
    # Get calibration information
    P2 = info['calib']['P2']
    # Get ground-gplane information and rotation matrix.
    # for roadside dataset only.
    GP = info['gplane'][np.newaxis, :].astype(np.float128)
    a, b, c = - GP[:, 0], - GP[:, 1], - GP[:, 2]
    y_rot = np.stack([a, b, c], axis=0)
    x_rot = np.stack([np.ones_like(a), - a / b, np.zeros_like(a)], axis=0)
    div = a ** 2 + b ** 2
    z_rot = np.stack([(- a * c) / div, (-b * c) / div, np.ones_like(a)], axis=0)
    R_p = np.stack([x_rot, y_rot, z_rot], axis=0)
    R_p = R_p / np.linalg.norm(R_p, axis=1)

    repro_recs = []
    # if no annotations in info (test dataset), then return
    if 'annos' not in info:
        return repro_recs

    # Get all the annotation with the specified visibilties.
    ann_dicts = info['annos']
    mask = [(ocld in occluded) for ocld in ann_dicts['occluded']]
    for k in ann_dicts.keys():
        ann_dicts[k] = ann_dicts[k][mask]

    # convert dict of list to list of dict
    ann_recs = []
    for i in range(len(ann_dicts['occluded'])):
        ann_rec = {}
        for k in ann_dicts.keys():
            ann_rec[k] = ann_dicts[k][i]
        ann_recs.append(ann_rec)

    for ann_idx, ann_rec in enumerate(ann_recs):
        # Augment sample_annotation with token information.
        ann_rec['sample_annotation_token'] = \
            f"{info['image']['image_id']}.{ann_idx}"
        ann_rec['sample_data_token'] = info['image']['image_id']
        sample_data_token = info['image']['image_id']

        loc = ann_rec['location'][np.newaxis, :]
        dim = ann_rec['dimensions'][np.newaxis, :]
        rot = ann_rec['rotation_y'][np.newaxis, np.newaxis]
        # transform the center from [0.5, 1.0, 0.5] to [0.5, 0.5, 0.5]
        dst = np.array([0.5, 0.5, 0.5])
        src = np.array([0.5, 1.0, 0.5])
        loc = loc + dim * (dst - src)
        offset = (info['calib']['P2'][0, 3]) / info['calib']['P2'][0, 0]
        loc_3d = np.copy(loc)
        loc_3d[0, 0] += offset
        gt_bbox_3d = np.concatenate([loc, dim, rot], axis=1).astype(np.float32)

        # Filter out the corners that are not in front of the calibrated
        # sensor.
        corners_3d = box_np_ops.center_to_corner_box3d(
            gt_bbox_3d[:, :3],
            gt_bbox_3d[:, 3:6],
            gt_bbox_3d[:, 6], [0.5, 0.5, 0.5],
            axis=1)
        corners_3d = corners_3d[0].T  # (1, 8, 3) -> (3, 8)
        in_front = np.argwhere(corners_3d[2, :] > 0).flatten()
        corners_3d = corners_3d[:, in_front]

        # Project 3d box to 2d.
        camera_intrinsic = P2
        # view_points(corners_3d, camera_intrinsic, True).T[:, :2]
        corner_coords = view_points(corners_3d, camera_intrinsic,
                                    True).T[:, :2].tolist()

        # Keep only corners that fall within the image.
        final_coords = post_process_coords(corner_coords, imsize=(1920, 1080))

        # Skip if the convex hull of the re-projected corners
        # does not intersect the image canvas.
        if final_coords is None:
            continue
        else:
            min_x, min_y, max_x, max_y = final_coords

        # Generate dictionary record to be included in the .json file.
        repro_rec = generate_record(ann_rec, min_x, min_y, max_x, max_y,
                                    sample_data_token,
                                    info['image']['image_path'])

        # If mono3d=True, add 3D annotations in camera coordinates
        if mono3d and (repro_rec is not None):
            repro_rec['bbox_cam3d'] = np.concatenate(
                [loc_3d, dim, rot],
                axis=1).astype(np.float32).squeeze().tolist()
            repro_rec['velo_cam3d'] = -1  # no velocity in rope3d

            center3d = np.array(loc).reshape([1, 3])
            center2d = points_cam2img(
                center3d, camera_intrinsic, with_depth=True)
            repro_rec['center2d'] = center2d.squeeze().tolist()
            # normalized center2D + depth
            # samples with depth < 0 will be removed
            if repro_rec['center2d'][2] <= 0:
                continue

            repro_rec['attribute_name'] = -1  # no attribute in rope3d
            repro_rec['attribute_id'] = -1

        repro_recs.append(repro_rec)

    return repro_recs

def generate_record(ann_rec, x1, y1, x2, y2, sample_data_token, filename):
    """Generate one 2D annotation record given various information on top of
    the 2D bounding box coordinates.

    Args:
        ann_rec (dict): Original 3d annotation record.
        x1 (float): Minimum value of the x coordinate.
        y1 (float): Minimum value of the y coordinate.
        x2 (float): Maximum value of the x coordinate.
        y2 (float): Maximum value of the y coordinate.
        sample_data_token (str): Sample data token.
        filename (str):The corresponding image file where the annotation
            is present.

    Returns:
        dict: A sample 2D annotation record.
            - file_name (str): file name
            - image_id (str): sample data token
            - area (float): 2d box area
            - category_name (str): category name
            - category_id (int): category id
            - bbox (list[float]): left x, top y, x_size, y_size of 2d box
            - iscrowd (int): whether the area is crowd
    """
    repro_rec = OrderedDict()
    repro_rec['sample_data_token'] = sample_data_token
    coco_rec = dict()

    key_mapping = {
        'name': 'category_name',
        'num_points_in_gt': 'num_lidar_pts',
        'sample_annotation_token': 'sample_annotation_token',
        'sample_data_token': 'sample_data_token',
    }

    for key, value in ann_rec.items():
        if key in key_mapping.keys():
            repro_rec[key_mapping[key]] = value

    repro_rec['bbox_corners'] = [x1, y1, x2, y2]
    repro_rec['filename'] = filename

    coco_rec['file_name'] = filename
    coco_rec['image_id'] = sample_data_token
    coco_rec['area'] = (y2 - y1) * (x2 - x1)

    if repro_rec['category_name'] not in rope3d_categories:
        return None
    cat_name = repro_rec['category_name']
    coco_rec['category_name'] = cat_name
    coco_rec['category_id'] = rope3d_categories.index(cat_name)
    coco_rec['bbox'] = [x1, y1, x2 - x1, y2 - y1]
    coco_rec['iscrowd'] = 0

    return coco_rec
