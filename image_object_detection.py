import cv2
import numpy as np
import onnxruntime as ort


def generate_grids_and_strides(input_size, strides):
  """Generates grid coordinates and corresponding stride values for YOLOX raw output decoding.

  Total anchors generated: (80x80) + (40x40) + (20x20) = 8400 grid cells.
  """
  grid_strides = []
  for stride in strides:
    feat_size = input_size // stride
    for y in range(feat_size):
      for x in range(feat_size):
        grid_strides.append((x, y, stride))
  return np.array(grid_strides)


# 1. Load the target image and COCO dataset class labels
cat_image = cv2.imread("Images/leon.jpg")

class_names = []
with open("coco.txt", "rt") as file:
  class_names = file.read().rstrip("\n").split("\n")

# 2. Initialize ONNX Runtime inference session
session = ort.InferenceSession("object_detection_yolox_2022nov.onnx")
input_name = session.get_inputs()[0].name

# 3. Preprocess the image (Resize, Color Conversion, Layout Transpose, Batch Dimension)
resized_image = cv2.resize(cat_image, (640, 640))
rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
input_data = np.transpose(rgb_image, (2, 0, 1)).astype("float32")
input_data = np.expand_dims(input_data, axis=0)

# 4. Execute forward pass model inference
outputs = session.run(None, {input_name: input_data})
predictions = outputs[0][0]  # Raw output tensor shape: (8400, 85)

# 5. Generate spatial anchor grids matching YOLOX feature maps
grid_strides = generate_grids_and_strides(640, [8, 16, 32])

# 6. Calculate scaling factors to map bounding boxes back to the original image resolution
h_orig, w_orig, _ = cat_image.shape
x_scale = w_orig / 640.0
y_scale = h_orig / 640.0

boxes = []
confidences = []
class_ids = []

# 7. Decode raw prediction vectors (8400 iterations)
for i, pred in enumerate(predictions):
  obj_score = pred[4]  # Objectness score
  class_scores = pred[5:]  # 80 class probability scores

  class_id = np.argmax(class_scores)
  confidence = obj_score * class_scores[class_id]

  # Filter out low-confidence predictions
  if confidence > 0.4:
    grid_x, grid_y, stride = grid_strides[i]

    # YOLOX raw coordinate decoding formulas
    cx = (pred[0] + grid_x) * stride
    cy = (pred[1] + grid_y) * stride
    w = np.exp(pred[2]) * stride
    h = np.exp(pred[3]) * stride

    # Convert center coordinates (cx, cy, w, h) to top-left bounding box (xmin, ymin, width, height)
    xmin = int((cx - (w / 2.0)) * x_scale)
    ymin = int((cy - (h / 2.0)) * y_scale)
    width = int(w * x_scale)
    height = int(h * y_scale)

    boxes.append([xmin, ymin, width, height])
    confidences.append(float(confidence))
    class_ids.append(class_id)

# 8. Apply Non-Maximum Suppression (NMS) to eliminate redundant boxes
indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold = 0.4, nms_threshold = 0.45)

# 9. Draw final detections on the original image
if len(indices) > 0:
  for i in indices.flatten():
    x, y, w, h = boxes[i]
    label = f"{class_names[class_ids[i]]}: {confidences[i]:.2f}"

    cv2.rectangle(cat_image, (x, y), (x + w, y + h), (0, 255, 0), thickness = 2)
    cv2.putText(
        cat_image,
        label,
        (x, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        2,
        (0, 0, 255),
        thickness = 2
    )

# 10. Display the result window
cv2.namedWindow("Cat Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Cat Detection", 800, 600)
cv2.imshow("Cat Detection", cat_image)
cv2.waitKey(0)
cv2.destroyAllWindows()