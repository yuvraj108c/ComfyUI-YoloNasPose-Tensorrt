# pip install super_gradients

from super_gradients.training import models
from super_gradients.common.object_names import Models

model = models.get(Models.YOLO_NAS_POSE_L, pretrained_weights="coco_pose")
export_result = model.export("yolo_nas_pose_l.onnx", confidence_threshold=0.5)

print(export_result)
