# DETECTRON2 Example running on BentoML

    The example is based on the coco example in 
    https://www.dlology.com/blog/how-to-train-detectron2-with-custom-coco-datasets/


## Folders 

* data (dataset)
* output (generated files)

## Based on the coco example for training and inference

* train_coco.py
* visualize_coco.py
* inference.py

## Bentoml Detectron2

* input_model.yaml (Detectron2 config file)
* test_detectron2_artifact.py (Testing the arctifact)
* classificationService.py (Bentoml service)
* saveToBento.py (Save detectron2 service to the bentoml repo)

## Test files for posting to the bentoml service

* test_post.py
* customVisualizer.py

## Commands

    Curl command to hit the bentoml service
    curl -X POST "http://localhost:5000/predict" -F image=@"data/images/0.jpg"

    Start the bentoml service
    bentoml serve CocoDetectronClassifier:latest --enable-microbatch