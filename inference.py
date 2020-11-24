from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
import os
import cv2

import random
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.data.datasets import register_coco_instances

from detectron2.utils.visualizer import ColorMode
import numpy as np

register_coco_instances("fruits_nuts", {}, "./data/trainval.json", "./data/images")

fruits_nuts_metadata = MetadataCatalog.get("fruits_nuts")
dataset_dicts = DatasetCatalog.get("fruits_nuts")

cfg = get_cfg()
cfg.merge_from_file("../detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.DATASETS.TRAIN = ("fruits_nuts",)
cfg.DATASETS.TEST = ()   # no metrics implemented for this dataset
cfg.DATALOADER.NUM_WORKERS = 2
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.02
cfg.SOLVER.MAX_ITER = 300    # 300 iterations seems good enough, but you can certainly train longer
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3  # 3 classes (data, fig, hazelnut)


cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set the testing threshold for this model
cfg.DATASETS.TEST = ("fruits_nuts", )

yaml_str = cfg.dump()

with open(os.path.join(cfg.OUTPUT_DIR, 'retrained_config.yaml'), 'w', encoding='utf-8') as output_file:
    output_file.write(yaml_str)

predictor = DefaultPredictor(cfg)


for d in random.sample(dataset_dicts, 3):    
    im = cv2.imread(d["file_name"])
    outputs = predictor(im)
    outputs_cpu = outputs["instances"].to("cpu")
    print(outputs_cpu)
    v = Visualizer(im[:, :, ::-1],
                   metadata=fruits_nuts_metadata, 
                   scale=0.8, 
                   instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels
    )
    v = v.draw_instance_predictions(outputs_cpu)
    cv2.imshow("test", v.get_image()[:, :, ::-1])

    if cv2.waitKey(1000) & 0xFF == ord('q'):
        break

    import time
    times = []
    for i in range(20):
        start_time = time.time()
        outputs = predictor(im)
        delta = time.time() - start_time
        times.append(delta)
    mean_delta = np.array(times).mean()
    fps = 1 / mean_delta
    print("Average(sec):{:.2f},fps:{:.2f}".format(mean_delta, fps))