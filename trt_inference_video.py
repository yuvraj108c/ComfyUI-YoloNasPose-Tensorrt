# H100 benchmark 200 FPS

import cv2
import numpy as np
from utilities import Engine
import torch
import timeit
from yolo_nas_pose.pose_estimation import PoseVisualization
import timeit


def iterate_over_batch_predictions(predictions, batch_size):
    num_detections, batch_boxes, batch_scores, batch_joints = predictions
    for image_index in range(batch_size):
        num_detection_in_image = num_detections[image_index, 0]

        pred_scores = batch_scores[image_index, :num_detection_in_image]
        pred_boxes = batch_boxes[image_index, :num_detection_in_image]
        pred_joints = batch_joints[image_index, :num_detection_in_image].reshape(
            (len(pred_scores), -1, 3))

        yield image_index, pred_boxes, pred_scores, pred_joints


def show_predictions_from_batch_format(image, predictions):
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

    # blank_image = np.zeros_like(image)
    blank_image = image
    image = PoseVisualization.draw_poses(
        image=blank_image, poses=new_pred_joints, scores=None, boxes=None,
        edge_links=edge_links, edge_colors=edge_colors, keypoint_colors=None, is_crowd=None,        joint_thickness=5,
        box_thickness=2,
        keypoint_radius=5
    )
    return image


# image = cv2.imread("./sitting_16.jpg")
# og_height, og_width, _ = image.shape
# image = cv2.resize(image, (640, 640))

video = cv2.VideoCapture("feastables.mp4")


width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(video.get(cv2.CAP_PROP_FPS))

# Create a VideoWriter object to save the video
output_file = 'output_video.mp4'
video_writer = cv2.VideoWriter(
    output_file, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

engine = Engine("./yolo_nas_pose_l.engine")
engine.load()
engine.activate()
engine.allocate_buffers()
cudaStream = torch.cuda.current_stream().cuda_stream


idx = 0

start = timeit.default_timer()
while True:
    success, frame = video.read()
    if not success:
        break
    idx += 1

    try:
        image = cv2.resize(frame, (640, 640))
        image_bchw = np.transpose(np.expand_dims(image, 0), (0, 3, 1, 2))
        image_torch = torch.from_numpy(image_bchw)

        result = engine.infer({"input": image_torch}, cudaStream)

        predictions = []

        for key in result.keys():
            if key != 'input':
                predictions.append(result[key].cpu().numpy())

        result = show_predictions_from_batch_format(image, predictions)
        upscaled = cv2.resize(result, (width, height))
        video_writer.write(upscaled)

    except Exception as e:
        # black_image = np.zeros_like(frame)
        video_writer.write(frame)
        # print(e)
        continue


end = timeit.default_timer()
print('FPS: ', idx/(end-start), 'Frames: ', idx)

# Release the video capture object
video.release()
video_writer.release()
