#!/usr/bin/env bash
import json
import os
import numpy as np
import cv2

#ref: https://www.celantur.com/blog/panoptic-segmentation-in-detectron2/
#The setup for panoptic segmentation is very similar to instance segmentation. However, as in semantic segmentation, you have to tell Detectron2 the pixel-wise labelling of the whole image, e.g. using an image where the colours encode the labels.
def generate_segmentation_file(img_dir):
    json_file = os.path.join(img_dir, "via_region_data.json")
    with open(json_file) as f:
        imgs_anns = json.load(f)

    for idx, v in enumerate(imgs_anns.values()):
        print(v)
        filename = os.path.join(img_dir, v["filename"])
        height, width = cv2.imread(filename).shape[:2]

        # Because we only have one object category (balloon) to train,
        # 1 is the category of the background
        segmentation = np.ones((height, width), dtype=np.uint8)
        annos = v["regions"]
        for _, anno in annos.items():
            assert not anno["region_attributes"]
            anno = anno["shape_attributes"]
            px = anno["all_points_x"]#37 len
            py = anno["all_points_y"]#37 len
            poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
            poly = np.array(poly, np.int32)#(37, 2)
            category_id = 0  # change to 255 for better visualisation
            cv2.fillPoly(segmentation, [poly], category_id)
            output = os.path.join(img_dir, "segmentation", v["filename"])
            cv2.imwrite(output, segmentation)

if "__main__" == __name__:
    for d in ["train", "val"]:
        #create segmentation folder under balloon/train{val} to save png segmentation files
        os.makedirs(os.path.join("./Dataset/balloon", d, "segmentation"), exist_ok=True)
        generate_segmentation_file(os.path.join("./Dataset/balloon", d))