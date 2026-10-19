import cv2
import numpy as np
import onnxruntime as ort


def generate_grids_and_strides(input_size=640, strides=[8, 16, 32]):
  """Generates grid coordinates and corresponding stride values for YOLOX raw output decoding."""
  grid_strides = []
  for stride in strides:
    feat_size = input_size // stride
    for y in range(feat_size):
      for x in range(feat_size):
        grid_strides.append((x, y, stride))
  return np.array(grid_strides)


# ------------------------------------------------------------------
# 1. Choose Input Source:
#    Use 0 for Live Webcam
#    OR pass the video path like: "Videos/test_video.mp4"
# ------------------------------------------------------------------
VIDEO_SOURCE = 0  # Change to "your_video.mp4" for video files

cap = cv2.VideoCapture(VIDEO_SOURCE)

# 2. Load COCO class labels
class_names = []
with open("coco.txt", "rt") as file:
  class_names = file.read().rstrip("\n").split("\n")

# 3. Initialize ONNX Runtime Inference Session
session = ort.InferenceSession("object_detection_yolox_2022nov.onnx")
input_name = session.get_inputs()[0].name

# Generate spatial grid points matching model output
grid_strides = generate_grids_and_strides(640, [8, 16, 32])

# 4. Processing Loop (Frame-by-Frame)
while cap.isOpened():
  ret, frame = cap.read()
  if not ret:
    print("Video stream ended or camera unavailable.")
    break

  # Scale factors mapping bounding boxes back to frame size
  h_orig, w_orig, _ = frame.shape
  x_scale = w_orig / 640.0
  y_scale = h_orig / 640.0

  # Preprocess current frame
  resized_image = cv2.resize(frame, (640, 640))
  rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
  input_data = np.transpose(rgb_image, (2, 0, 1)).astype("float32")
  input_data = np.expand_dims(input_data, axis=0)

  # Run inference
  outputs = session.run(None, {input_name: input_data})
  predictions = outputs[0][0]

  boxes = []
  confidences = []
  class_ids = []

  # Decode raw predictions
  for i, pred in enumerate(predictions):
    obj_score = pred[4]
    class_scores = pred[5:]
    class_id = np.argmax(class_scores)
    confidence = obj_score * class_scores[class_id]

    if confidence > 0.4:
      grid_x, grid_y, stride = grid_strides[i]

      cx = (pred[0] + grid_x) * stride
      cy = (pred[1] + grid_y) * stride
      w = np.exp(pred[2]) * stride
      h = np.exp(pred[3]) * stride

      xmin = int((cx - (w / 2.0)) * x_scale)
      ymin = int((cy - (h / 2.0)) * y_scale)
      width = int(w * x_scale)
      height = int(h * y_scale)

      boxes.append([xmin, ymin, width, height])
      confidences.append(float(confidence))
      class_ids.append(class_id)

  # Apply Non-Maximum Suppression (NMS)
  indices = cv2.dnn.NMSBoxes(
      boxes, confidences, score_threshold=0.4, nms_threshold=0.45
  )

  # Draw detections on current frame
  if len(indices) > 0:
    for i in indices.flatten():
      x, y, w, h = boxes[i]
      label = f"{class_names[class_ids[i]]}: {confidences[i]:.2f}"

      cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
      cv2.putText(
          frame,
          label,
          (x, y - 10),
          cv2.FONT_HERSHEY_SIMPLEX,
          0.5,
          (0, 255, 0),
          2,
      )

  # Display the live detections
  cv2.imshow("Real-Time Object Detection", frame)

  # Press 'q' on keyboard to quit the loop
  if cv2.waitKey(1) & 0xFF == ord("q"):
    break

# Release video resources
cap.release()
cv2.destroyAllWindows()