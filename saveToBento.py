from detectron2.config import get_cfg
from detectron2.checkpoint import DetectionCheckpointer  # noqa # pylint: disable=unused-import
from detectron2.modeling import META_ARCH_REGISTRY
from detectron2.config import get_cfg
from detectron2.data import transforms as T
from classificationService import CocoDetectronClassifier


cfg = get_cfg()
cfg.merge_from_file("input_model.yaml")        
meta_arch = META_ARCH_REGISTRY.get(cfg.MODEL.META_ARCHITECTURE)
model = meta_arch(cfg)
model.eval()
device = "cuda:{}".format(0)
model.to(device)
checkpointer = DetectionCheckpointer(model)
checkpointer.load("output/model_final.pth")

bento_svc = CocoDetectronClassifier()
bento_svc.pack('model', model)
saved_path = bento_svc.save()

