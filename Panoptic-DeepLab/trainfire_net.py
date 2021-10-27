#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

"""
Panoptic-DeepLab Training Script.
This script is a simplified version of the training script in detectron2/tools.
"""

import os
import torch

import detectron2.data.transforms as T
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, build_detection_train_loader
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
)
from detectron2.projects.deeplab import build_lr_scheduler
from detectron2.projects.panoptic_deeplab import (
    PanopticDeeplabDatasetMapper,
    add_panoptic_deeplab_config,
)
from detectron2.solver import get_default_optimizer_params
from detectron2.solver.build import maybe_add_gradient_clipping


def build_sem_seg_train_aug(cfg):
    augs = [
        T.ResizeShortestEdge(
            cfg.INPUT.MIN_SIZE_TRAIN, cfg.INPUT.MAX_SIZE_TRAIN, cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
        )
    ]
    if cfg.INPUT.CROP.ENABLED:
        augs.append(T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE))
    augs.append(T.RandomFlip())
    return augs


class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains a number pre-defined logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can use the cleaner
    "SimpleTrainer", or write your own training loop.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if cfg.MODEL.PANOPTIC_DEEPLAB.BENCHMARK_NETWORK_SPEED:
            return None
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type in ["cityscapes_panoptic_seg", "coco_panoptic_seg"]:
            evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
        if evaluator_type == "cityscapes_panoptic_seg":
            assert (
                torch.cuda.device_count() > comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            evaluator_list.append(CityscapesSemSegEvaluator(dataset_name))
            evaluator_list.append(CityscapesInstanceEvaluator(dataset_name))
        if evaluator_type == "coco_panoptic_seg":
            # `thing_classes` in COCO panoptic metadata includes both thing and
            # stuff classes for visualization. COCOEvaluator requires metadata
            # which only contains thing classes, thus we map the name of
            # panoptic datasets to their corresponding instance datasets.
            dataset_name_mapper = {
                "coco_2017_val_panoptic": "coco_2017_val",
                "coco_2017_val_100_panoptic": "coco_2017_val_100",
                "fire_val_panoptic": "fire_val",
                "mycoco_2017_val_panoptic": "mycoco_2017_val",
            }
            evaluator_list.append(
                COCOEvaluator(dataset_name_mapper[dataset_name], output_dir=output_folder)
            )
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def build_train_loader(cls, cfg):
        mapper = PanopticDeeplabDatasetMapper(cfg, augmentations=build_sem_seg_train_aug(cfg))
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        """
        It now calls :func:`detectron2.solver.build_lr_scheduler`.
        Overwrite it if you'd like a different scheduler.
        """
        return build_lr_scheduler(cfg, optimizer)

    @classmethod
    def build_optimizer(cls, cfg, model):
        """
        Build an optimizer from config.
        """
        params = get_default_optimizer_params(
            model,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
            weight_decay_norm=cfg.SOLVER.WEIGHT_DECAY_NORM,
        )

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            return maybe_add_gradient_clipping(cfg, torch.optim.SGD)(
                params,
                cfg.SOLVER.BASE_LR,
                momentum=cfg.SOLVER.MOMENTUM,
                nesterov=cfg.SOLVER.NESTEROV,
            )
        elif optimizer_type == "ADAM":
            return maybe_add_gradient_clipping(cfg, torch.optim.Adam)(params, cfg.SOLVER.BASE_LR)
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_panoptic_deeplab_config(cfg)

    #config_file="./Panoptic-DeepLab/configs/COCO-PanopticSegmentation/panoptic_deeplab_R_52_SyncBN.yaml"
    cfg.merge_from_file(args.config_file)
    #cfg.merge_from_file(args.config_file)
    #cfg.merge_from_list(args.opts)
    #cfg.freeze()
    default_setup(cfg, args)
    return cfg

from detectron2.data.datasets import register_coco_instances, load_coco_json, register_coco_panoptic_separated, load_sem_seg
#from detectron2.data.datasets import _get_builtin_metadata
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import Visualizer
from PIL import Image
import copy
import numpy as np
import os, json, cv2, random
from matplotlib import pyplot as plt

from detectron2.utils.file_io import PathManager
def load_coco_panoptic_json(json_file, image_dir, gt_dir, meta):
    """
    Args:
        image_dir (str): path to the raw dataset. e.g., "~/coco/train2017".
        gt_dir (str): path to the raw annotations. e.g., "~/coco/panoptic_train2017".
        json_file (str): path to the json file. e.g., "~/coco/annotations/panoptic_train2017.json".
    Returns:
        list[dict]: a list of dicts in Detectron2 standard format. (See
        `Using Custom Datasets </tutorials/datasets.html>`_ )
    """

    def _convert_category_id(segment_info, meta):
        if segment_info["category_id"] in meta["thing_dataset_id_to_contiguous_id"]:
            segment_info["category_id"] = meta["thing_dataset_id_to_contiguous_id"][
                segment_info["category_id"]
            ]
            segment_info["isthing"] = True
        else:
            segment_info["category_id"] = meta["stuff_dataset_id_to_contiguous_id"][
                segment_info["category_id"]
            ]
            segment_info["isthing"] = False
        return segment_info

    with PathManager.open(json_file) as f:
        json_info = json.load(f)

    ret = []
    for ann in json_info["annotations"]:
        image_id = int(ann["image_id"])
        # TODO: currently we assume image and label has the same filename but
        # different extension, and images have extension ".jpg" for COCO. Need
        # to make image extension a user-provided argument if we extend this
        # function to support other COCO-like datasets.
        image_file = os.path.join(image_dir, os.path.splitext(ann["file_name"])[0] + ".jpg")
        label_file = os.path.join(gt_dir, ann["file_name"])
        segments_info = [_convert_category_id(x, meta) for x in ann["segments_info"]]
        #segments_info = [x for x in ann["segments_info"]]
        ret.append(
            {
                "file_name": image_file,
                "image_id": image_id,
                "pan_seg_file_name": label_file,
                "segments_info": segments_info,
            }
        )
    assert len(ret), f"No images found in {image_dir}!"
    assert PathManager.isfile(ret[0]["file_name"]), ret[0]["file_name"]
    assert PathManager.isfile(ret[0]["pan_seg_file_name"]), ret[0]["pan_seg_file_name"]
    return ret

# panoptic_coco_categories = './Detectron2COCOPanoptic/panoptic_coco_categories.json'
# with open(panoptic_coco_categories, 'r') as f:
#     categories_list = json.load(f)
# COCO_CATEGORIES =categories_list# {category['id']: category for category in categories_list}

COCO_CATEGORIES = [
   {"color": [209, 226, 140], "isthing": 1, "id": 1, "name": "Fire"},
    {"color": [64, 170, 64], "isthing": 1, "id": 2, "name": "NoFire"},
    {"color": [216, 186, 171], "isthing": 1, "id": 3, "name": "Smoke"},
    {"color": [206, 186, 171], "isthing": 1, "id": 4, "name": "BurntArea"},]

def cv2_imshow(img, outputfilename='./outputs/result.png'):
    rgb=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    fig = plt.figure(figsize = (20, 10))
    plt.imshow(rgb)
    fig.savefig(outputfilename)


def get_coco_panoptic_standard():
    meta = {}
    # The following metadata maps contiguous id from [0, #thing categories +
    # #stuff categories) to their names and colors. We have to replica of the
    # same name and color under "thing_*" and "stuff_*" because the current
    # visualization function in D2 handles thing and class classes differently
    # due to some heuristic used in Panoptic FPN. We keep the same naming to
    # enable reusing existing visualization functions.
    thing_classes = [k["name"] for k in COCO_CATEGORIES]
    thing_colors = [k["color"] for k in COCO_CATEGORIES]
    stuff_classes = [k["name"] for k in COCO_CATEGORIES]
    stuff_colors = [k["color"] for k in COCO_CATEGORIES]

    meta["thing_classes"] = thing_classes
    meta["thing_colors"] = thing_colors
    meta["stuff_classes"] = stuff_classes
    meta["stuff_colors"] = stuff_colors

    # Convert category id for training:
    #   category id: like semantic segmentation, it is the class id for each
    #   pixel. Since there are some classes not used in evaluation, the category
    #   id is not always contiguous and thus we have two set of category ids:
    #       - original category id: category id in the original dataset, mainly
    #           used for evaluation.
    #       - contiguous category id: [0, #classes), in order to train the linear
    #           softmax classifier.
    thing_dataset_id_to_contiguous_id = {}
    stuff_dataset_id_to_contiguous_id = {}

    for i, cat in enumerate(COCO_CATEGORIES):
        if cat["isthing"]:
            thing_dataset_id_to_contiguous_id[cat["id"]] = i
        else:
            stuff_dataset_id_to_contiguous_id[cat["id"]] = i

    meta["thing_dataset_id_to_contiguous_id"] = thing_dataset_id_to_contiguous_id
    meta["stuff_dataset_id_to_contiguous_id"] = stuff_dataset_id_to_contiguous_id

    return meta

def _get_coco_instances_meta():
    thing_ids = [k["id"] for k in COCO_CATEGORIES if k["isthing"] == 1]
    thing_colors = [k["color"] for k in COCO_CATEGORIES if k["isthing"] == 1]
    #assert len(thing_ids) == 80, len(thing_ids)
    # Mapping from the incontiguous COCO category id to an id in [0, 79]
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in COCO_CATEGORIES if k["isthing"] == 1]
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
        "thing_colors": thing_colors,
    }
    return ret

def mypanopticdatasetregister(panoptic_name, image_root, panoptic_root, panoptic_json, instances_json, instancedataset_name="mycoco_2017_val"):
    mymeta=get_coco_panoptic_standard()#_get_coco_panoptic_separated_meta()
    print(mymeta["thing_dataset_id_to_contiguous_id"])
    #ref to register_coco_panoptic in https://github.com/facebookresearch/detectron2/blob/main/detectron2/data/datasets/coco_panoptic.py
    DatasetCatalog.register(
        panoptic_name,
        lambda: load_coco_panoptic_json(panoptic_json, image_root, panoptic_root, mymeta),
    )
    MetadataCatalog.get(panoptic_name).set(
        panoptic_root=panoptic_root,
        image_root=image_root,
        panoptic_json=panoptic_json,
        json_file=instances_json,
        evaluator_type="coco_panoptic_seg",
        ignore_label=255,
        label_divisor=1000,
        **mymeta,
    )

    metalist=MetadataCatalog.list()
    panotic_metadata = MetadataCatalog.get(panoptic_name)

    
    register_coco_instances(instancedataset_name, _get_coco_instances_meta(), instances_json, image_root)
    instance_metadata = MetadataCatalog.get(instancedataset_name)

    return panotic_metadata, instance_metadata

def main(args):
    cfg = setup(args)

    metalist=MetadataCatalog.list()

    #panoptic_name="fire_panoptic"#"coco_2017_train_panoptic_separated"#"coco_2017_train_panoptic"
    image_root='./Dataset/FireDataset/train'
    panoptic_root='./Dataset/FireDataset/panoptic_train/' #directory which contains panoptic annotation images
    panoptic_json='./Dataset/FireDataset/annotations/fire-net-panoptic-format-train.json'
    #sem_seg_root= '/DATA5T2/Datasets/COCO2017/coco/panoptic_stuff_train2017/'#directory which contains all the ground truth segmentation annotations.
    instances_json='./Dataset/FireDataset/annotations/fire-net-instances-format-train.json'

    mypanoptic_name='fire_train_panoptic'
    mymetadataCatalog, instance_train_meta=mypanopticdatasetregister(mypanoptic_name, image_root, panoptic_root, panoptic_json, instances_json, instancedataset_name="fire_train")
    
    valimage_root='./Dataset/FireDataset/val'
    valpanoptic_root='./Dataset/FireDataset/panoptic_val/' #directory which contains panoptic annotation images
    valpanoptic_json='./Dataset/FireDataset/annotations/fire-net-panoptic-format-val.json'
    #sem_seg_root= '/DATA5T2/Datasets/COCO2017/coco/panoptic_stuff_train2017/'#directory which contains all the ground truth segmentation annotations.
    valinstances_json='./Dataset/FireDataset/annotations/fire-net-instances-format-val.json'
    myvalpanoptic_name='fire_val_panoptic'
    myvalmetadataCatalog, instance_train_meta=mypanopticdatasetregister(myvalpanoptic_name, valimage_root, valpanoptic_root, valpanoptic_json, valinstances_json, instancedataset_name="fire_val")
    

    #Visualize a few images
    dataset_dicts = DatasetCatalog.get(mypanoptic_name)
    for d in random.sample(dataset_dicts, 3):
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=mymetadataCatalog, scale=0.5)
        vis = visualizer.draw_dataset_dict(d)
        cv2_imshow(vis.get_image()[:, :, ::-1],'./outputs/panopticdeeplab'+str(d["image_id"]))

    cfg.MODEL.WEIGHTS = './outputs/panoticdeeplab_model_final.pkl'
    cfg.OUTPUT_DIR='./output/firepanopticdeeplab'
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()

import argparse
if __name__ == "__main__":
    #cd /path/to/detectron2/projects/Panoptic-DeepLab
#python train_net.py --config-file configs/Cityscapes-PanopticSegmentation/panoptic_deeplab_R_52_os16_mg124_poly_90k_bs32_crop_512_1024_dsconv.yaml --num-gpus 8

    args = default_argument_parser().parse_args()
    # parser = argparse.ArgumentParser(description='PyTorch Distributed Training')
    # parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    # parser.add_argument(
    #     "--resume",
    #     action="store_true",
    #     help="Whether to attempt to resume from the checkpoint directory. "
    #     "See documentation of `DefaultTrainer.resume_or_load()` for what it means.",
    # )
    # parser.add_argument("--eval-only", action="store_true", help="perform evaluation only")
    # parser.add_argument("--num-gpus", type=int, default=1, help="number of gpus *per machine*")
    # args = parser.parse_args()

    print(args)
    #args.config_file="/Panoptic-DeepLab/configs/COCO-PanopticSegmentation/panoptic_deeplab_R_52_os16_mg124_poly_200k_bs64_crop_640_640_coco_dsconv.yaml"
    # args.config_file="./Panoptic-DeepLab/configs/COCO-PanopticSegmentation/panoptic_deeplab_R_52_os16_mg124_poly_200k_bs64_crop_640_640_coco_dsconv.yaml"
    args.config_file="./Panoptic-DeepLab/configs/COCO-PanopticSegmentation/firepanoptic_deeplab.yaml"
    print("Command Line Args:", args)

    # import torch.distributed as dist
    # dist.init_process_group('gloo', init_method="env://", rank=0, world_size=1)
    # dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
    #                             world_size=args.world_size, rank=args.rank)

    #main(args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,#1
        machine_rank=args.machine_rank,#0
        dist_url=args.dist_url,
        args=(args,),
    )