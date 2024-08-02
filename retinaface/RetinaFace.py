import os
import warnings
import logging
from typing import Union, Any, Optional, Dict

# this has to be set before importing tf
os.environ["TF_USE_LEGACY_KERAS"] = "1"

# pylint: disable=wrong-import-position
import numpy as np
import tensorflow as tf

from retinaface import __version__
from retinaface.model import retinaface_model
from retinaface.commons import preprocess, postprocess
from retinaface.commons.logger import Logger
from retinaface.commons import package_utils

# users should install tf_keras package if they are using tf 2.16 or later versions
package_utils.validate_for_keras3()

logger = Logger(module="retinaface/RetinaFace.py")

# pylint: disable=global-variable-undefined, no-name-in-module, unused-import, too-many-locals, redefined-outer-name,
# too-many-statements, too-many-arguments

# ---------------------------

# configurations
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# Limit the amount of reserved VRAM so that other scripts can be run in the same GPU as well
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

tf_version = int(tf.__version__.split(".", maxsplit=1)[0])

if tf_version == 2:
    tf.get_logger().setLevel(logging.ERROR)
    from tensorflow.keras.models import Model
else:
    from keras.models import Model


# ---------------------------


class ModelBuilder:
    _model_instance = None

    @staticmethod
    def build_model() -> Model:
        """
        Builds retinaface model once and stores it in memory.
        Ensures thread safety and avoids global variables.
        """
        if ModelBuilder._model_instance is None:
            ModelBuilder._model_instance = retinaface_model.build_model()
        return ModelBuilder._model_instance


# Implementing the extract_faces function
def extract_faces(
        img_path: Union[str, np.ndarray],
        threshold: float = 0.9,
        alt_model: Optional[Model] = None,
        align: bool = True,
        allow_upscaling: bool = True,
        expand_face_area: int = 0,
) -> list:
    resp = []

    img = preprocess.get_image(img_path)

    if alt_model is None:
        model = ModelBuilder.build_model()
    else:
        model = alt_model

    obj = detect_faces(
        img_path=img, threshold=threshold, alt_model=model, allow_upscaling=allow_upscaling
    )

    if not isinstance(obj, dict):
        return resp

    for _, identity in obj.items():
        facial_area = identity["facial_area"]
        rotate_angle = 0
        rotate_direction = 1

        x = int(facial_area[0])
        y = int(facial_area[1])
        w = int(facial_area[2] - x)
        h = int(facial_area[3] - y)

        if expand_face_area > 0:
            expanded_w = w + int(w * expand_face_area / 100)
            expanded_h = h + int(h * expand_face_area / 100)

            x = max(0, x - int((expanded_w - w) / 2))
            y = max(0, y - int((expanded_h - h) / 2))
            w = min(img.shape[1] - x, expanded_w)
            h = min(img.shape[0] - y, expanded_h)

        facial_img = img[y: y + h, x: x + w]

        if align is True:
            landmarks = identity["landmarks"]
            left_eye = landmarks["left_eye"]
            right_eye = landmarks["right_eye"]
            nose = landmarks["nose"]

            aligned_img, rotate_angle, rotate_direction = postprocess.alignment_procedure(
                img=img, left_eye=right_eye, right_eye=left_eye, nose=nose
            )

            rotated_x1, rotated_y1, rotated_x2, rotated_y2 = postprocess.rotate_facial_area(
                (x, y, x + w, y + h), rotate_angle, rotate_direction, (img.shape[0], img.shape[1])
            )
            facial_img = aligned_img[
                         int(rotated_y1): int(rotated_y2), int(rotated_x1): int(rotated_x2)
                         ]

        resp.append(facial_img[:, :, ::-1])

    return resp


def detect_faces(
        img_path: Union[str, np.ndarray],
        threshold: float = 0.9,
        alt_model: Optional[Model] = None,
        allow_upscaling: bool = True,
) -> Dict[str, Any]:
    resp = {}
    img = preprocess.get_image(img_path)

    if alt_model is None:
        model = ModelBuilder.build_model()
    else:
        model = alt_model

    nms_threshold = 0.4
    decay4 = 0.5

    _feat_stride_fpn = [32, 16, 8]

    _anchors_fpn = {
        "stride32": np.array(
            [[-248.0, -248.0, 263.0, 263.0], [-120.0, -120.0, 135.0, 135.0]], dtype=np.float32
        ),
        "stride16": np.array(
            [[-56.0, -56.0, 71.0, 71.0], [-24.0, -24.0, 39.0, 39.0]], dtype=np.float32
        ),
        "stride8": np.array([[-8.0, -8.0, 23.0, 23.0], [0.0, 0.0, 15.0, 15.0]], dtype=np.float32),
    }

    _num_anchors = {"stride32": 2, "stride16": 2, "stride8": 2}

    proposals_list = []
    scores_list = []
    landmarks_list = []
    im_tensor, im_info, im_scale = preprocess.preprocess_image(img, allow_upscaling)
    net_out = model(im_tensor)
    net_out = [elt.numpy() for elt in net_out]
    sym_idx = 0

    for _, s in enumerate(_feat_stride_fpn):
        scores = net_out[sym_idx]
        scores = scores[:, :, :, _num_anchors[f"stride{s}"]:]

        bbox_deltas = net_out[sym_idx + 1]
        height, width = bbox_deltas.shape[1], bbox_deltas.shape[2]

        A = _num_anchors[f"stride{s}"]
        K = height * width
        anchors_fpn = _anchors_fpn[f"stride{s}"]
        anchors = postprocess.anchors_plane(height, width, s, anchors_fpn)
        anchors = anchors.reshape((K * A, 4))
        scores = scores.reshape((-1, 1))

        bbox_stds = [1.0, 1.0, 1.0, 1.0]
        bbox_pred_len = bbox_deltas.shape[3] // A
        bbox_deltas = bbox_deltas.reshape((-1, bbox_pred_len))
        bbox_deltas[:, 0::4] = bbox_deltas[:, 0::4] * bbox_stds[0]
        bbox_deltas[:, 1::4] = bbox_deltas[:, 1::4] * bbox_stds[1]
        bbox_deltas[:, 2::4] = bbox_deltas[:, 2::4] * bbox_stds[2]
        bbox_deltas[:, 3::4] = bbox_deltas[:, 3::4] * bbox_stds[3]
        proposals = postprocess.bbox_pred(anchors, bbox_deltas)

        proposals = postprocess.clip_boxes(proposals, im_info[:2])

        if s == 4 and decay4 < 1.0:
            scores *= decay4

        scores_ravel = scores.ravel()
        order = np.where(scores_ravel >= threshold)[0]
        proposals = proposals[order, :]
        scores = scores[order]

        proposals[:, 0:4] /= im_scale
        proposals_list.append(proposals)
        scores_list.append(scores)

        landmark_deltas = net_out[sym_idx + 2]
        landmark_pred_len = landmark_deltas.shape[3] // A
        landmark_deltas = landmark_deltas.reshape((-1, 5, landmark_pred_len // 5))
        landmarks = postprocess.landmark_pred(anchors, landmark_deltas)
        landmarks = landmarks[order, :]

        landmarks[:, :, 0:2] /= im_scale
        landmarks_list.append(landmarks)
        sym_idx += 3

    proposals = np.vstack(proposals_list)

    if proposals.shape[0] == 0:
        return resp

    scores = np.vstack(scores_list)
    scores_ravel = scores.ravel()
    order = scores_ravel.argsort()[::-1]

    proposals = proposals[order, :]
    scores = scores[order]
    landmarks = np.vstack(landmarks_list)
    landmarks = landmarks[order].astype(np.float32, copy=False)

    pre_det = np.hstack((proposals[:, 0:4], scores)).astype(np.float32, copy=False)
    keep = postprocess.cpu_nms(pre_det, nms_threshold)

    det = np.hstack((pre_det, proposals[:, 4:]))
    det = det[keep, :]
    landmarks = landmarks[keep]

    for idx, face in enumerate(det):
        label = "face_" + str(idx + 1)
        resp[label] = {}
        resp[label]["score"] = face[4]
        resp[label]["facial_area"] = list(face[0:4].astype(int))
        resp[label]["landmarks"] = {}
        resp[label]["landmarks"]["right_eye"] = list(landmarks[idx][0])
        resp[label]["landmarks"]["left_eye"] = list(landmarks[idx][1])
        resp[label]["landmarks"]["nose"] = list(landmarks[idx][2])
        resp[label]["landmarks"]["mouth_right"] = list(landmarks[idx][3])
        resp[label]["landmarks"]["mouth_left"] = list(landmarks[idx][4])

    return resp


# Vergleichsfunktion
def compare_faces(known_face, face_data):
    """
    Compare two faces based on their landmarks by normalizing the landmarks
    using the bounding box and calculating the Euclidean distance between them.

    Args:
        known_face (dict): A dictionary containing the landmarks and facial area of a known face.
        face_data (dict): A dictionary containing the landmarks and facial area of the face to compare.

    Returns:
        float: The Euclidean distance between the normalized landmarks of the two faces.
    """

    # Extract landmarks from both faces
    known_landmarks = known_face['landmarks']
    landmarks = face_data['landmarks']

    # Extract bounding boxes for both faces
    # Default bounding box is [0, 0, 1, 1] if not provided
    known_bbox = known_face.get('facial_area', [0, 0, 1, 1])
    bbox = face_data.get('facial_area', [0, 0, 1, 1])

    # Normalize the landmarks based on the bounding box
    # This is done by translating the coordinates such that the top-left corner
    # of the bounding box is (0, 0) and scaling the coordinates by the width and height of the bounding box.
    known_landmarks_array = np.array([
        [(x - known_bbox[0]) / (known_bbox[2] - known_bbox[0]),
         (y - known_bbox[1]) / (known_bbox[3] - known_bbox[1])]
        for x, y in known_landmarks.values()
    ])
    landmarks_array = np.array([
        [(x - bbox[0]) / (bbox[2] - bbox[0]),
         (y - bbox[1]) / (bbox[3] - bbox[1])]
        for x, y in landmarks.values()
    ])

    # Log the normalized landmarks for debugging purposes
    logging.debug(f"Normalized known landmarks: {known_landmarks_array}")
    logging.debug(f"Normalized input landmarks: {landmarks_array}")

    # Calculate the Euclidean distance between the normalized landmarks of the two faces
    # The Euclidean distance is a measure of the "straight line" distance between two points in Euclidean space.
    distance = np.linalg.norm(known_landmarks_array - landmarks_array)

    return distance
