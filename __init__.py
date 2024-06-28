import os
import folder_paths
import numpy as np
import torch.nn.functional as F
import torch
from comfy.utils import ProgressBar
from .utilities import Engine
from .yolo_nas_pose.pose_estimation import PoseVisualization
import cv2

ENGINE_DIR = os.path.join(folder_paths.models_dir,
                          "tensorrt", "yolo-nas-pose")

black_image = np.zeros((640, 640, 3))


def iterate_over_batch_predictions(predictions, batch_size):
    num_detections, batch_boxes, batch_scores, batch_joints = predictions
    for image_index in range(batch_size):
        num_detection_in_image = num_detections[image_index, 0]

        pred_scores = batch_scores[image_index, :num_detection_in_image]
        pred_boxes = batch_boxes[image_index, :num_detection_in_image]
        pred_joints = batch_joints[image_index, :num_detection_in_image].reshape(
            (len(pred_scores), -1, 3))

        yield image_index, pred_boxes, pred_scores, pred_joints


def show_predictions_from_batch_format(predictions):

    image_index, pred_boxes, pred_scores, pred_joints = next(
        iter(iterate_over_batch_predictions(predictions, 1)))

    edge_links = [[0, 17], [13, 15], [14, 16],
                  [12, 14], [12, 17], [5, 6], [11, 13], [7, 9], [5, 7], [17, 11], [6, 8], [8, 10], [1, 3], [0, 1], [0, 2], [2, 4]]
    edge_colors = [
        [255, 0, 0], [255, 85, 0],  [170, 255, 0], [85, 255, 0], [
            85, 255, 0], [85, 0, 255], [255, 170, 0], [0, 177, 58], [0, 179, 119], [179, 179, 0], [0, 119, 179], [0, 179, 179], [119, 0, 179], [179, 0, 179], [178, 0, 118], [178, 0, 118]
    ]
    new_pred_joints = []

    for i in range(pred_joints.shape[0]):
        list1 = pred_joints[i][5]
        list2 = pred_joints[i][6]
        middle_list = [(a + b) / 2 for a, b in zip(list1, list2)]
        middle_data_np = np.array(middle_list)
        row = np.expand_dims(middle_data_np, axis=0)

        row = np.concatenate((pred_joints[i], row), axis=0)
        new_pred_joints.append(row)

    new_pred_joints = np.array(new_pred_joints)

    image = PoseVisualization.draw_poses(
        image=black_image, poses=new_pred_joints, scores=None, boxes=None,
        edge_links=edge_links, edge_colors=edge_colors, keypoint_colors=None, is_crowd=None, joint_thickness=10, box_thickness=2, keypoint_radius=10  # ovewritten in function
    )
    return image


class YoloNasPoseTensorrt:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "engine": (os.listdir(ENGINE_DIR),),
            }
        }
    RETURN_NAMES = ("IMAGE",)
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "main"
    CATEGORY = "tensorrt"

    def main(self, images, engine):
        # setup tensorrt engine
        engine = Engine(os.path.join(ENGINE_DIR, engine))
        engine.load()
        engine.activate()
        engine.allocate_buffers()
        cudaStream = torch.cuda.current_stream().cuda_stream

        pbar = ProgressBar(images.shape[0])
        images_bchw = images.permute(0, 3, 1, 2)
        images_resized = F.interpolate(images_bchw, size=(
            640, 640), mode='bilinear', align_corners=False)
        images_resized_uint8 = (images_resized * 255.0).type(torch.uint8)
        images_list = list(torch.split(
            images_resized_uint8, split_size_or_sections=1))

        pose_frames = []

        for img in images_list:
            try:
                result = engine.infer({"input": img}, cudaStream)

                predictions = []
                for key in result.keys():
                    if key != 'input':
                        predictions.append(result[key].cpu().numpy())

                result = show_predictions_from_batch_format(predictions)
            except Exception as e:
                result = black_image

            result = (result.clip(0, 255)).astype(np.uint8)
            result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
            # result = cv2.resize(result, (images_bchw.shape[3], images_bchw.shape[2])) # quite slow
            pose_frames.append(result)
            pbar.update(1)

        pose_frames_np = np.array(pose_frames).astype(np.float32) / 255
        pose_frames_tensor = torch.from_numpy(pose_frames_np)

        resized_pose_frames_tensor = torch.nn.functional.interpolate(
            # (B, H, W, C) -> (B, C, H, W)
            pose_frames_tensor.permute(0, 3, 1, 2),
            size=(images_bchw.shape[2], images_bchw.shape[3]),
            mode='bilinear'
        ).permute(0, 2, 3, 1)  # (B, C, H, W) -> (B, H, W, C)
        return (resized_pose_frames_tensor,)


NODE_CLASS_MAPPINGS = {
    "YoloNasPoseTensorrt": YoloNasPoseTensorrt,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "YoloNasPoseTensorrt": "Yolo Nas Pose Tensorrt",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
