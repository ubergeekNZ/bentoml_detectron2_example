import bentoml
import json
import cv2
import torch
import random
from bentoml.frameworks.detectron import DetectronModelArtifact

from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode
from detectron2.data.datasets import register_coco_instances
register_coco_instances("fruits_nuts", {}, "./data/trainval.json", "./data/images")

fruits_nuts_metadata = MetadataCatalog.get("fruits_nuts")
dataset_dicts = DatasetCatalog.get("fruits_nuts")

config = {
    'cfg' : "output/retrained_config.yaml",
    'device' : 0,
    'name' : 'model_final'
}

detectron_artifact = DetectronModelArtifact("model_final")

loaded_artifact = detectron_artifact.load("output")
model = loaded_artifact._model
aug = loaded_artifact._aug

with torch.no_grad():
    for d in random.sample(dataset_dicts, 3): 
        original_image = cv2.imread(d["file_name"])
        original_image = original_image[:, :, ::-1]
        height, width = original_image.shape[:2]
        image = aug.get_transform(original_image).apply_image(original_image)
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

        inputs = {"image": image, "height": height, "width": width}
        predictions = model([inputs])[0]
        print(type(predictions))
        instances = predictions["instances"].to("cpu")
        scores = instances.get("scores")
        score = scores[0]
        small_mask = instances.get("pred_masks")[0, ...].unsqueeze(0)
        v = Visualizer(original_image,
                   metadata=fruits_nuts_metadata, 
                   scale=0.8, 
                   instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels
        )
        v = v.draw_instance_predictions(instances)
        cv2.imshow("test", v.get_image()[:, :, ::-1])

        if cv2.waitKey(1000) & 0xFF == ord('q'):
            break

