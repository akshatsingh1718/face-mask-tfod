import os
import cv2
import numpy as np
import tensorflow as tf
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

WORKSPACE_PATH = r"Tensorflow/workspace"
SCRIPTS_PATH = r"Tensorflow/scripts"
APIMODEL_PATH = r"Tensorflow/models"
CUSTOM_MODEL_NAME = "my_ssd_mobilenet_320"
PRETRAINED_MODEL_NAME = 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8'
ANNOTATION_PATH = WORKSPACE_PATH + "/training_demo/annotations"
IMAGE_PATH = WORKSPACE_PATH + "/training_demo/images"
MODEL_PATH = WORKSPACE_PATH + "/training_demo/models"
PRETRAINED_MODEL_PATH = WORKSPACE_PATH + "/training_demo/pre-trained-models"
CONFIG_PATH = MODEL_PATH + "/my_ssd_resnet50_v1_fpn/pipeline.config"
CHECKPOINT_PATH = MODEL_PATH + "/my_ssd_mobilenet_320/"
OUTPUT_PATH = os.path.join(MODEL_PATH, CUSTOM_MODEL_NAME, 'export')
TFJS_PATH = os.path.join(MODEL_PATH, CUSTOM_MODEL_NAME, 'tfjsexport')
TFLITE_PATH = os.path.join(MODEL_PATH, CUSTOM_MODEL_NAME, 'tfliteexport')
PROTOC_PATH = os.path.join('Tensorflow', 'protoc')
OUTPUT_PATH = os.path.join(MODEL_PATH, CUSTOM_MODEL_NAME, 'export')
CUSTOM_MODEL_PATH = os.path.join(MODEL_PATH, CUSTOM_MODEL_NAME)
EXPORT_PATH = os.path.join(CUSTOM_MODEL_PATH, 'export')
SAVED_MODEL_PATH = os.path.join(EXPORT_PATH, 'saved_model')
LABELMAP_PATH = os.path.join(ANNOTATION_PATH, "label_map.pbtxt")
PB_FILE_PATH = os.path.join(EXPORT_PATH, 'saved_model', 'saved_model.pb')

detection_model = tf.saved_model.load(SAVED_MODEL_PATH)


def run_inference_for_single_image(model, image):
    image = np.asarray(image)
    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]

    # Run inference
    model_fn = model.signatures["serving_default"]
    output_dict = model_fn(input_tensor)
    #   print(output_dict)
    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(output_dict.pop("num_detections"))
    output_dict = {
        key: value[0, :num_detections].numpy() for key, value in output_dict.items()
    }
    output_dict["num_detections"] = num_detections

    # detection_classes should be ints.
    output_dict["detection_classes"] = output_dict["detection_classes"].astype(np.int64)

    # Handle models with masks:
    if "detection_masks" in output_dict:
        # Reframe the the bbox mask to the image size.
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            output_dict["detection_masks"],
            output_dict["detection_boxes"],
            image.shape[0],
            image.shape[1],
        )
        detection_masks_reframed = tf.cast(detection_masks_reframed > 0.8, tf.uint8)
        output_dict["detection_masks_reframed"] = detection_masks_reframed.numpy()

    return output_dict


def show_inference(model, image_np):
    # the array based representation of the image will be used later in order to prepare the
    # result image with boxes and labels on it.
    #   image_np = np.array(Image.open(image_path))
    # Actual detection.
    output_dict = run_inference_for_single_image(model, image_np)

    #   print(category_index)
    # Visualization of the results of a detection.
    final_img = vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        output_dict["detection_boxes"],
        output_dict["detection_classes"],
        output_dict["detection_scores"],
        category_index,
        instance_masks=output_dict.get("detection_masks_reframed", None),
        use_normalized_coordinates=True,
        line_thickness=8,
    )
    return final_img


cap = cv2.VideoCapture(0)
category_index = label_map_util.create_category_index_from_labelmap(
    LABELMAP_PATH, use_display_name=True
)

while 1:
    _, img = cap.read()

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    final_img = show_inference(detection_model, img)

    final_img = cv2.cvtColor(final_img, cv2.COLOR_RGB2BGR)

    cv2.imshow("Face Mask Detection", final_img)

    #     cv2.imshow('img',img)
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
