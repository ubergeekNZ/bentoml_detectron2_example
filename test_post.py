import requests
import PIL
import cv2
import numpy as np
from PIL import Image
from detectron2.data import MetadataCatalog, DatasetCatalog
from customVisualizer import Visualizer
from detectron2.structures.instances import Instances
from detectron2.utils.visualizer import ColorMode
from detectron2.data.datasets import register_coco_instances
register_coco_instances("fruits_nuts", {}, "./data/trainval.json", "./data/images")

fruits_nuts_metadata = MetadataCatalog.get("fruits_nuts")
dataset_dicts = DatasetCatalog.get("fruits_nuts")
print(fruits_nuts_metadata)

original_image = open('data/images/0.jpg', 'rb')

files = {
    "image":  ('images.jpg', original_image, 'image/jpg'),
}
response = requests.post("http://localhost:5000/predict", files=files)

original_image = cv2.imread('data/images/0.jpg')


original_image = original_image[:, :, ::-1]

v = Visualizer(original_image,
                   metadata=fruits_nuts_metadata, 
                   scale=0.8, 
                   instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels
        )
v = v.draw_instance_predictions(response.json())
cv2.imshow("test", v.get_image()[:, :, ::-1])

if cv2.waitKey(1000) & 0xFF == ord('q'):
    pass