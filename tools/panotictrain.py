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

def cv2_imshow(img, outputfilename='./outputs/result.png'):
    rgb=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    fig = plt.figure(figsize = (20, 10))
    plt.imshow(rgb)
    fig.savefig(outputfilename)

def get_balloon_dicts(img_dir):
    json_file = os.path.join(img_dir, "via_region_data.json")
    with open(json_file) as f:
        imgs_anns = json.load(f)

    dataset_dicts = []
    for idx, v in enumerate(imgs_anns.values()):
        record = {}

        filename = os.path.join(img_dir, v["filename"])
        height, width = cv2.imread(filename).shape[:2]

        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width
        # Pixel-wise segmentation
        record["sem_seg_file_name"] = os.path.join(img_dir, "segmentation", v["filename"])#

        annos = v["regions"]
        objs = []
        for _, anno in annos.items():
            assert not anno["region_attributes"]
            anno = anno["shape_attributes"]
            px = anno["all_points_x"]
            py = anno["all_points_y"]
            poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
            poly = [p for x in poly for p in x]

            obj = {
                "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": 0,
                # "Things" are well-defined countable objects,
                # while "stuff" is amorphous something with a different label than the background.
                "isthing": True,
                "iscrowd": 0
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts

if __name__ == "__main__":
    metalist=MetadataCatalog.list()#len=54

    image1 = cv2.imread('./Dataset/input.jpg') #array (480, 640, 3)
    rgb=cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
        
    # Inference with a panoptic segmentation model
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml")
    predictor = DefaultPredictor(cfg)
    result=predictor(rgb)#three keys: sem_seg, instances, panoptic_seg
    panoptic_seg, segments_info = result["panoptic_seg"]
    metadata=MetadataCatalog.get(cfg.DATASETS.TRAIN[0])#TRAIN[0]:coco_2017_train_panoptic_separated, metadata=evaluator_type:'coco_panoptic_seg'
    v = Visualizer(rgb[:, :, ::-1], metadata, scale=1.2)
    out = v.draw_panoptic_seg_predictions(panoptic_seg.to("cpu"), segments_info)
    cv2_imshow(out.get_image()[:, :, ::-1],'./outputs/panoptic.jpg')

    for d in ["train", "val"]:
        DatasetCatalog.register("balloon_" + d, lambda d=d: get_balloon_dicts("./Dataset/balloon/" + d))
        # For semantic / panoptic segmentation, add a stuff class.
        MetadataCatalog.get("balloon_" + d).set(thing_classes=["balloon"], stuff_classes=["background"])
    #balloon_metadata = MetadataCatalog.get("balloon_train")

    DatasetCatalog.register("balloon_train_new", lambda d=d:get_balloon_dicts("./Dataset/balloon/train"))
    MetadataCatalog.get("balloon_train_new").thing_classes=["balloon"]
    MetadataCatalog.get("balloon_train_new").stuff_classes=["background"]
    ##registering again as have modified the dicts obatained from the COCO format , added the segmenattion info
    #DatasetCatalog.register("balloon_train", lambda d=d:get_balloon_dicts("./Dataset/balloon/train"))

    balloon_metadata = MetadataCatalog.get("balloon_train_new")

    #Training
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml"))
    cfg.DATASETS.TRAIN = ("balloon_train",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml")  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    cfg.SOLVER.MAX_ITER = 300    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
    cfg.SOLVER.STEPS = []        # do not decay learning rate
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.
    #cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 2 #sets the number of stuff classes for Semantic FPN & Panoptic FPN.

    cfg.OUTPUT_DIR='./output/panotic3/'
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)#./output
    trainer = DefaultTrainer(cfg) 
    trainer.resume_or_load(resume=False)
    trainer.train()

    #inference
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, 'model_final.pth')
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set the testing threshold for this model
    cfg.DATASETS.TEST = ("balloon_val",)
    predictor = DefaultPredictor(cfg)
    panoptic_seg, segments_info = predictor(rgb)["panoptic_seg"]
    v = Visualizer(rgb[:, :, ::-1], balloon_metadata, scale=1.2)
    out = v.draw_panoptic_seg_predictions(panoptic_seg.to("cpu"), segments_info)
    cv2_imshow(out.get_image()[:, :, ::-1],'./outputs/panoptic_inference.jpg')