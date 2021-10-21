import torch
print(torch.__version__) #1.8.0+cu111
import torchvision
print(torchvision.__version__) #0.9.0+cu111
# check if CUDA is available
train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print('CUDA is not available.  Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...')

import detectron2 #detectron2 version: 0.4+cu111
#from detectron2.data import MetadataCatalog
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode

from matplotlib import pyplot as plt

#For training
from detectron2.engine import DefaultTrainer

#import json
# import os
# import numpy as np
# import cv2

panoptic_coco_categories = './Detectron2COCOPanoptic/panoptic_coco_categories.json'
with open(panoptic_coco_categories, 'r') as f:
    categories_list = json.load(f)
COCO_CATEGORIES =categories_list# {category['id']: category for category in categories_list}


from pycocotools.coco import COCO
def generate_segmentation_file(img_dir, outputpath):
    json_file = os.path.join(img_dir, "FireClassification.json")
    coco=COCO(json_file)
    #print(coco)
    cats = coco.loadCats(coco.getCatIds())
    nms=[cat['name'] for cat in cats] #['Fire', 'NoFire', 'Smoke', 'BurntArea']
    print('COCO categories: \n{}\n'.format(' '.join(nms)))
    imgIds_1 = coco.getImgIds()
    print(imgIds_1)
    for i in imgIds_1:
        imgIds = coco.getImgIds(imgIds = i) ##Image id part in the json
        img = coco.loadImgs(imgIds)[0]
        print(img)
        annIds = coco.getAnnIds(imgIds=img['id'], iscrowd=None)
        anns = coco.loadAnns(annIds)
        mask = coco.annToMask(anns[0])#(1080, 1920)
        for i in range(len(anns)):
          mask += coco.annToMask(anns[i])
        print(mask)
        output = os.path.join(outputpath, img["file_name"])
        cv2.imwrite(output, mask)

def cv2_imshow(img, outputfilename='./outputs/result.png'):
    rgb=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    fig = plt.figure(figsize = (20, 10))
    plt.imshow(rgb)
    fig.savefig(outputfilename)

def init_cfg(config_file: str, datasettrain):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(config_file))
    cfg.DATASETS.TRAIN = (datasettrain,)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config_file)  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    cfg.SOLVER.MAX_ITER = 500  # 300 iterations seems good enough for this toy dataset; you may need to train longer for a practical dataset
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128  # faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4  # only has one class (ballon)
    return cfg

def get_predictor(cfg, model_name: str):
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, model_name)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.1  # set the testing threshold for this model
   # cfg.DATASETS.TEST = ("balloon_val",)
    predictor = DefaultPredictor(cfg)
    return predictor

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
        #segments_info = [_convert_category_id(x, meta) for x in ann["segments_info"]]
        segments_info = [x for x in ann["segments_info"]]
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


def _get_coco_panoptic_separated_meta():
    """
    Returns metadata for "separated" version of the panoptic segmentation dataset.
    """
    stuff_ids = [k["id"] for k in COCO_CATEGORIES if k["isthing"] == 0]
    assert len(stuff_ids) == 53, len(stuff_ids)

    # For semantic segmentation, this mapping maps from contiguous stuff id
    # (in [0, 53], used in models) to ids in the dataset (used for processing results)
    # The id 0 is mapped to an extra category "thing".
    stuff_dataset_id_to_contiguous_id = {k: i + 1 for i, k in enumerate(stuff_ids)}
    # When converting COCO panoptic annotations to semantic annotations
    # We label the "thing" category to 0
    stuff_dataset_id_to_contiguous_id[0] = 0

    # 54 names for COCO stuff categories (including "things")
    stuff_classes = ["things"] + [
        k["name"].replace("-other", "").replace("-merged", "")
        for k in COCO_CATEGORIES
        if k["isthing"] == 0
    ]

    # NOTE: I randomly picked a color for things
    stuff_colors = [[82, 18, 128]] + [k["color"] for k in COCO_CATEGORIES if k["isthing"] == 0]
    ret = {
        "stuff_dataset_id_to_contiguous_id": stuff_dataset_id_to_contiguous_id,
        "stuff_classes": stuff_classes,
        "stuff_colors": stuff_colors,
    }
    ret.update(_get_coco_instances_meta())
    return ret

def _get_coco_instances_meta():
    thing_ids = [k["id"] for k in COCO_CATEGORIES if k["isthing"] == 1]
    thing_colors = [k["color"] for k in COCO_CATEGORIES if k["isthing"] == 1]
    assert len(thing_ids) == 80, len(thing_ids)
    # Mapping from the incontiguous COCO category id to an id in [0, 79]
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in COCO_CATEGORIES if k["isthing"] == 1]
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
        "thing_colors": thing_colors,
    }
    return ret

from detectron2.data.datasets import register_coco_instances, load_coco_json, register_coco_panoptic_separated, load_sem_seg
#from detectron2.data.datasets import _get_builtin_metadata
from PIL import Image
import copy

def merge_to_panoptic(detection_dicts, sem_seg_dicts):
    """
    Create dataset dicts for panoptic segmentation, by
    merging two dicts using "file_name" field to match their entries.
    Args:
        detection_dicts (list[dict]): lists of dicts for object detection or instance segmentation.
        sem_seg_dicts (list[dict]): lists of dicts for semantic segmentation.
    Returns:
        list[dict] (one per input image): Each dict contains all (key, value) pairs from dicts in
            both detection_dicts and sem_seg_dicts that correspond to the same image.
            The function assumes that the same key in different dicts has the same value.
    """
    results = []
    sem_seg_file_to_entry = {x["file_name"]: x for x in sem_seg_dicts}
    assert len(sem_seg_file_to_entry) > 0

    for det_dict in detection_dicts:
        dic = copy.copy(det_dict)
        dic.update(sem_seg_file_to_entry[dic["file_name"]])
        results.append(dic)
    return results

if __name__ == "__main__":
    metalist=MetadataCatalog.list()

    panoptic_name="coco_2017_train_panoptic_separated"#"coco_2017_train_panoptic"
    image_root='/DATA5T2/Datasets/COCO2017/coco/images/train2017/'
    panoptic_root='/DATA5T2/Datasets/COCO2017/coco/annotations/panoptic_train2017/' #directory which contains panoptic annotation images
    panoptic_json='/DATA5T2/Datasets/COCO2017/coco/annotations/panoptic_train2017.json'
    sem_seg_root= '/DATA5T2/Datasets/COCO2017/coco/panoptic_stuff_train2017/'#directory which contains all the ground truth segmentation annotations.
    instances_json='/DATA5T2/Datasets/COCO2017/coco/annotations/instances_train2017.json'
    
    # MetadataCatalog.get(panoptic_name).set(
    #     panoptic_root=panoptic_root,
    #     image_root=image_root,
    #     panoptic_json=panoptic_json,
    #     sem_seg_root=sem_seg_root,
    #     json_file=instances_json,  # TODO rename
    #     evaluator_type="coco_panoptic_seg",
    #     ignore_label=255,
    # )
    meta = MetadataCatalog.get(panoptic_name) #"coco_2017_train_panoptic"
        
    mypanoptic_name='mycoco_2017_train_panoptic'
    mymeta=_get_coco_panoptic_separated_meta()
    DatasetCatalog.register(
        mypanoptic_name,
        lambda: merge_to_panoptic(
            load_coco_json(instances_json, image_root, panoptic_name),
            load_sem_seg(sem_seg_root, image_root),
        ),
    )
    MetadataCatalog.get(mypanoptic_name).set(
        panoptic_root=panoptic_root,
        image_root=image_root,
        panoptic_json=panoptic_json,
        sem_seg_root=sem_seg_root,
        json_file=instances_json,  # TODO rename
        evaluator_type="coco_panoptic_seg",
        ignore_label=255,
        **mymeta,
    )
    #register_coco_panoptic_separated('mycoco_2017_train_panoptic', mymeta, image_root, panoptic_root, panoptic_json, sem_seg_root, instances_json)
    metalist=MetadataCatalog.list()
    mymeta = MetadataCatalog.get(mypanoptic_name)

    #Visualize a few images
    dataset_dicts = DatasetCatalog.get(mypanoptic_name)
    for d in random.sample(dataset_dicts, 3):
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=mymeta, scale=0.5)
        vis = visualizer.draw_dataset_dict(d)
        cv2_imshow(vis.get_image()[:, :, ::-1],'./outputs/'+str(d["image_id"]))

    #dicts=load_coco_panoptic_json('/DATA5T2/Datasets/COCO2017/coco/annotations/panoptic_train2017.json', '/DATA5T2/Datasets/COCO2017/coco/images/train2017/', '/DATA5T2/Datasets/COCO2017/coco/annotations/panoptic_train2017/', meta)#len: 118287

    #dicts = load_coco_json('/DATA5T2/Datasets/COCO2017/coco/annotations/panoptic_train2017.json', '/DATA5T2/Datasets/COCO2017/coco/images/train2017/', 'coco_2017_train_panoptic') #path/to/json path/to/image_root dataset_name
    #dicts = load_coco_json('/DATA5T2/Datasets/COCO2017/coco/annotations/instances_train2017.json', '/DATA5T2/Datasets/COCO2017/coco/images/train2017/', 'coco_2017_train') #path/to/json path/to/image_root dataset_name


    
    #cfg = init_cfg("COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml", mypanoptic_name)
    cfg = get_cfg()
    config_file="COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml"
    cfg.merge_from_file(model_zoo.get_config_file(config_file))
    cfg.DATASETS.TRAIN = (mypanoptic_name,)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config_file)  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    cfg.SOLVER.MAX_ITER = 500  # 300 iterations seems good enough for this toy dataset; you may need to train longer for a practical dataset
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128  # faster, and good enough for this toy dataset (default: 512)
    #cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4  # only has one class (ballon)
    print(cfg.MODEL.ROI_HEADS.NUM_CLASSES)#80
    cfg.OUTPUT_DIR='./output/cocopanoptic'
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    predictor = get_predictor(cfg, "model_final.pth")
    inputs = cv2.imread('./Dataset/input.jpg')
    panoptic_seg, segments_info = predictor(inputs)["panoptic_seg"]

    print("segments_info")
    print(segments_info)
    print("panoptic_seg")
    print(panoptic_seg)
    datasetname=cfg.DATASETS.TRAIN[0]
    metadata=MetadataCatalog.get(datasetname)
    v = Visualizer(inputs[:, :, ::-1], metadata, scale=1.2)
    v = v.draw_panoptic_seg_predictions(panoptic_seg.to("cpu"),segments_info)
    cv2_imshow(v.get_image()[:, :, ::-1])

    for d in random.sample(dataset_dicts, 5):
        img = cv2.imread(d["file_name"])
        panoptic_seg, segments_info = predictor(img)["panoptic_seg"]

        v = Visualizer(img[:, :, ::-1], metadata=metadata, scale=0.5)
        v = v.draw_panoptic_seg_predictions(panoptic_seg.to("cpu"),segments_info)
        #v = v.draw_dataset_dict(d)
        cv2_imshow(v.get_image()[:, :, ::-1],'./outputs/inference'+str(d["image_id"]))

