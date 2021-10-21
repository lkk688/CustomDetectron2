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

def init_cfg(config_file: str):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(config_file))
    cfg.DATASETS.TRAIN = ("firedataset_train_new",)
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

if __name__ == "__main__":
    for d in ["train"]:
        dataroot="./Dataset/CMPE_295_All_images/"
        outputpath=os.path.join(dataroot, d, "segmentation")
        os.makedirs(outputpath, exist_ok=True)
        #Change path to point to the shared Images Folder /content/drive/MyDrive/Panoptic_Segmentation/CMPE_295_All_images/Images
        generate_segmentation_file(os.path.join(dataroot, "Images"), outputpath)
    
    #if your dataset is in COCO format, this cell can be replaced by the following three lines:
    from detectron2.data.datasets import register_coco_instances
    register_coco_instances("firedataset_train", {}, './Dataset/CMPE_295_All_images/Images/FireClassification.json', "./Dataset/CMPE_295_All_images/Images")
    dataset_dicts = DatasetCatalog.get("firedataset_train")

    ##Added the extra sem_seg_file_name in the train
    print(len(dataset_dicts))
    for i in range(len(dataset_dicts)):
        tem = dataset_dicts[i]["file_name"]
        first, second = tem.rsplit('/', 1)
        dataset_dicts[i]["sem_seg_file_name"] = os.path.join(outputpath,second)
    
    from detectron2.data import MetadataCatalog
    ##registering again as have modified the dicts obatained from the COCO format , added the segmenattion info
    DatasetCatalog.register("firedataset_train_new", lambda d=d:dataset_dicts)
    MetadataCatalog.get("firedataset_train_new").thing_classes = ['Fire', 'NoFire', 'Smoke', 'BurntArea']
    MetadataCatalog.get("firedataset_train_new").stuff_classes = ['Fire', 'NoFire', 'Smoke', 'BurntArea']

    dataset_metadata = MetadataCatalog.get("firedataset_train_new")
    # Check whether dataset is correctly initialised
    #visualise_dataset("train")
    #dataset_dicts = get_balloon_dicts(os.path.join("balloon", d))
    for d in random.sample(dataset_dicts, 3):
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=dataset_metadata, scale=0.5)
        vis = visualizer.draw_dataset_dict(d)
        cv2_imshow(vis.get_image()[:, :, ::-1],'./outputs/'+str(d["image_id"]))
    
    cfg = init_cfg("COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml")
    cfg.OUTPUT_DIR='./output/firepanoptic'
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    predictor = get_predictor(cfg, "model_final.pth")
    inputs = cv2.imread('./Dataset/CMPE_295_All_images/Images/ChoppervideoCameronPeakFirestartsnearChambersLakeinwesternLarimerCounty-223.jpg')
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

