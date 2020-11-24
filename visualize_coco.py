# You may need to restart your runtime prior to this, to let your installation take effect
# Some basic setup
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import matplotlib.pyplot as plt
import numpy as np
import cv2
import yaml
import os

# import some common detectron2 utilities
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

from detectron2.data.datasets import register_coco_instances
register_coco_instances("fruits_nuts", {}, "./data/trainval.json", "./data/images")

fruits_nuts_metadata = MetadataCatalog.get("fruits_nuts")
dataset_dicts = DatasetCatalog.get("fruits_nuts")

cfg = get_cfg()
yaml_str = cfg.dump()

with open('input_model.yaml', 'w', encoding='utf-8') as output_file:
    output_file.write(yaml_str)

import random

for d in random.sample(dataset_dicts, 3):
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=fruits_nuts_metadata, scale=0.5)
    vis = visualizer.draw_dataset_dict(d)
    cv2.imshow("test", vis.get_image()[:, :, ::-1])

    if cv2.waitKey(1000) & 0xFF == ord('q'):
        break

cfg.merge_from_file("input_model.yaml")
print(cfg.dump())
