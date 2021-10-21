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
from detectron2.utils.visualizer import ColorMode

from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

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
        record["sem_seg_file_name"] = os.path.join(img_dir, "segmentation", v["filename"])

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

def cv2_imshow(img, outputfilename='./outputs/result.png'):
    rgb=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    fig = plt.figure(figsize = (20, 10))
    plt.imshow(rgb)
    fig.savefig(outputfilename)

from detectron2.utils.visualizer import (
    ColorMode,
    Visualizer,
    _create_text_labels,
    _PanopticPrediction,)

def testpanotic(panoptic_seg, segments_info, balloon_metadata):
    pred = _PanopticPrediction(panoptic_seg.to("cpu"), segments_info, balloon_metadata)
    for mask, sinfo in pred.semantic_masks():
        category_idx = sinfo["category_id"]
        try:
            mask_color = [x / 255 for x in balloon_metadata.stuff_colors[category_idx]]
        except AttributeError:
            mask_color = None

        v.draw_binary_mask(
            mask,
            color=mask_color,
            text=balloon_metadata.stuff_classes[category_idx]
        )

    all_instances = list(pred.instance_masks())

if __name__ == "__main__":
    # Inference should use the config with parameters that are used in training
    # cfg now already contains everything we've set previously. We changed it a little bit for inference:
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml"))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.OUTPUT_DIR = './output/panotic/'
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
    predictor = DefaultPredictor(cfg)

    dataset_dicts = get_balloon_dicts("./Dataset/balloon/val")

    for d in ["train", "val"]:
        DatasetCatalog.register("balloon_" + d, lambda d=d: get_balloon_dicts("./Dataset/balloon/" + d))
        # For semantic / panoptic segmentation, add a stuff class.
        MetadataCatalog.get("balloon_" + d).set(thing_classes=["balloon"], stuff_classes=["background0","background1"])
    balloon_metadata = MetadataCatalog.get("balloon_train")

    #inference
    image1 = cv2.imread('./Dataset/input.jpg') #array (480, 640, 3)
    rgb=cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, 'model_final.pth')
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set the testing threshold for this model
    cfg.DATASETS.TEST = ("balloon_val",)
    predictor = DefaultPredictor(cfg)
    panoptic_seg, segments_info = predictor(rgb)["panoptic_seg"]
    v = Visualizer(rgb[:, :, ::-1], balloon_metadata, scale=1.2)
    testpanotic(panoptic_seg, segments_info, balloon_metadata)
    out = v.draw_panoptic_seg_predictions(panoptic_seg.to("cpu"), segments_info)
    cv2_imshow(out.get_image()[:, :, ::-1],'./outputs/panoptic_inference.jpg')
    

    for d in random.sample(dataset_dicts, 3):    
        im = cv2.imread(d["file_name"])
        panoptic_seg, segments_info = predictor(im)["panoptic_seg"]
        v = Visualizer(im[:, :, ::-1], balloon_metadata, scale=1.2)
        testpanotic(panoptic_seg, segments_info, balloon_metadata)
        out = v.draw_panoptic_seg_predictions(panoptic_seg.to("cpu"), segments_info)
        cv2_imshow(out.get_image()[:, :, ::-1],'./outputs/'+str(d["image_id"]))
    
    evaluator = COCOEvaluator("balloon_val", output_dir="./output")
    val_loader = build_detection_test_loader(cfg, "balloon_val")
    print(inference_on_dataset(predictor.model, val_loader, evaluator))
    # another equivalent way to evaluate the model is to use `trainer.test`