import argparse
import os
import cv2
import numpy as np
import onnxruntime

from tracker.byte_tracker import BYTETracker as YourTracker
from tracking_utils.timer import Timer


from tracking_utils.classes import classes
from tracking_utils.utils import mkdir, tlwh_to_tlbr, plot_tracking


def make_parser():
    parser = argparse.ArgumentParser("onnxruntime inference sample")
    parser.add_argument("-m", "--model", type=str, default="yolov8s.onnx", help="Input your onnx model.")
    parser.add_argument("-i", "--video_path", type=str, default="input.mp4", help="Path to your input video.")
    parser.add_argument("-o", "--output_dir", type=str, default="demo_output", help="Path to your output directory.")
    parser.add_argument("-s", "--score_thr", type=float, default=0.5, help="Score threshold to filter the result.")
    parser.add_argument("-n", "--nms_thr", type=float, default=0.5, help="NMS threshold.")
    parser.add_argument("--input_shape", type=str, default="640,640", help="Specify an input shape for inference.")

    # tracking args
    parser.add_argument("--track_thresh", type=float, default=0.5, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keeping lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.5, help="matching threshold for tracking")
    parser.add_argument("--min-box-area", type=float, default=10, help="filter out tiny boxes")
    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")
    return parser


class YOLOv8:
    """YOLOv8 object detection model class for handling inference and visualization."""

    def __init__(self, args):
        """
        Initializes an instance of the YOLOv8 class.

        Args:
            args: Parsed command-line arguments.
        """
        self.args = args
        self.onnx_model = args.model
        self.input_video = args.video_path
        self.output_video = args.output_dir
        self.confidence_thres = args.score_thr
        self.iou_thres = args.nms_thr

        # Load the class names
        self.classes = classes

        # Generate a color palette for the classes
        self.color_palette = np.random.uniform(0, 255, size=(len(self.classes), 3))

        # Create an inference session using the ONNX model and specify execution providers
        self.session = onnxruntime.InferenceSession(self.onnx_model, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])

        # Get the model inputs
        model_inputs = self.session.get_inputs()

        # Store the shape of the input for later use
        input_shape = model_inputs[0].shape
        self.input_width = input_shape[2]
        self.input_height = input_shape[3]

        self.tracker = YourTracker(args, frame_rate=30)
        self.timer = Timer()

    def draw_detections(self, img, box, score, class_id):
        """
        Draws bounding boxes and labels on the input image based on the detected objects.

        Args:
            img: The input image to draw detections on.
            box: Detected bounding box.
            score: Corresponding detection score.
            class_id: Class ID for the detected
            object.

        Returns:
            None
        """

        # Extract the coordinates of the bounding box
        x1, y1, w, h = box

        # Retrieve the color for the class ID
        color = self.color_palette[class_id]

        # Draw the bounding box on the image
        cv2.rectangle(img, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 2)

        # Create the label text with class name and score
        label = f"{self.classes[class_id]}: {score:.2f}"

        # Calculate the dimensions of the label text
        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        # Calculate the position of the label text
        label_x = x1
        label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10

        # Draw a filled rectangle as the background for the label text
        cv2.rectangle(img, (label_x, label_y - label_height), (label_x + label_width, label_y + label_height), color, cv2.FILLED)

        # Draw the label text on the image
        cv2.putText(img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    def preprocess(self, frame):
        """
        Preprocesses the input frame before performing inference.

        Args:
            frame: The input video frame.

        Returns:
            image_data: Preprocessed frame data ready for inference.
        """
        # Get the height and width of the input frame
        self.img_height, self.img_width = frame.shape[:2]

        # Convert the image color space from BGR to RGB
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Resize the image to match the input shape
        img = cv2.resize(img, (self.input_width, self.input_height))

        # Normalize the image data by dividing it by 255.0
        image_data = np.array(img) / 255.0

        # Transpose the image to have the channel dimension as the first dimension
        image_data = np.transpose(image_data, (2, 0, 1))  # Channel first

        # Expand the dimensions of the image data to match the expected input shape
        image_data = np.expand_dims(image_data, axis=0).astype(np.float32)

        return image_data

    def postprocess(self, input_image, output):
        """
        Performs post-processing on the model's output to extract bounding boxes, scores, and class IDs.

        Args:
            input_image (numpy.ndarray): The input image.
            output (numpy.ndarray): The output of the model.

        Returns:
            numpy.ndarray: The input image with detections drawn on it.
        """

        # Transpose and squeeze the output to match the expected shape
        outputs = np.transpose(np.squeeze(output[0]))

        # Get the number of rows in the outputs array
        rows = outputs.shape[0]

        # Lists to store the bounding boxes, scores, and class IDs of the detections
        boxes = []
        scores = []
        class_ids = []

        # Calculate the scaling factors for the bounding box coordinates
        x_factor = self.img_width / self.input_width
        y_factor = self.img_height / self.input_height

        # Iterate over each row in the outputs array
        for i in range(rows):
            # Extract the class scores from the current row
            classes_scores = outputs[i][4:]

            # Find the maximum score among the class scores
            max_score = np.amax(classes_scores)

            # If the maximum score is above the confidence threshold
            if max_score >= self.confidence_thres:
                # Get the class ID with the highest score
                class_id = np.argmax(classes_scores)

                # Extract the bounding box coordinates from the current row
                x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]

                # Calculate the scaled coordinates of the bounding box
                left = int((x - w / 2) * x_factor)
                top = int((y - h / 2) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)

                # Add the class ID, score, and box coordinates to the respective lists
                class_ids.append(class_id)
                scores.append(max_score)
                boxes.append([left, top, width, height])

        # Apply non-maximum suppression to filter out overlapping bounding boxes
        indices = cv2.dnn.NMSBoxes(boxes, scores, self.confidence_thres, self.iou_thres)
        boxes_new = []
        scores_new = []
        class_ids_new = []
        # Iterate over the selected indices after non-maximum suppression
        for i in indices:
            # Get the box, score, and class ID corresponding to the index
            box = boxes[i]
            score = scores[i]
            class_id = class_ids[i]

            class_ids_new.append(class_id)
            scores_new.append(score)
            boxes_new.append(box)

            # Draw the detection on the input image
            # self.draw_detections(input_image, box, score, class_id)

        # Return the modified input image
        return boxes_new, scores_new, class_ids_new

    def detect_objects(self, frame):
        """
        Detects objects in the input frame using the YOLOv8 model.

        Args:
            frame: The input video frame.

        Returns:
            bboxes: List of bounding boxes for detected objects.
            scores: List of confidence scores for detected objects.
            class_ids: List of class IDs for detected objects.
        """
        # Preprocess the input frame
        image_data = self.preprocess(frame)

        # Perform inference using the ONNX model
        outputs = self.session.run(None, {self.session.get_inputs()[0].name: image_data})

        # Postprocess the model outputs to get the final detections
        bboxes, scores, class_ids = self.postprocess(frame, outputs)

        return bboxes, scores, class_ids

    def run(self):
        """
        Runs the object detection and tracking pipeline on the input video.

        Returns:
            None
        """
        # Create the output directory if it doesn't exist
        mkdir(self.output_video)

        # Open the input video file
        cap = cv2.VideoCapture(self.input_video)

        # Get the video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Define the codec and create the VideoWriter object to save the output video
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(os.path.join(self.output_video, "output.mp4"), fourcc, fps, (width, height))

        frame_id = 0
        # Process each frame of the video
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Start the timer
            self.timer.tic()

            # Detect objects in the frame
            bboxes, scores, class_ids = self.detect_objects(frame)

            # Prepare detection results for tracking
            dets = []

            # 将 scores 转换为列向量
            scores = np.array(scores)
            scores = scores[:, np.newaxis]

            # 将 bboxes 和 scores 水平堆叠

            bboxes = tlwh_to_tlbr(bboxes)
            dets = np.hstack((bboxes, scores))

            # Update the tracker with detection results
            online_targets = self.tracker.update(np.array(dets), [height, width], [height, width])
            print("online_targets:", online_targets)
            online_tlwhs = []
            online_ids = []
            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id

                online_tlwhs.append(tlwh)
                online_ids.append(tid)
            frame_id += 1

            # Calculate the elapsed time
            self.timer.toc()

            # Display FPS on the frame
            fps = 1.0 / self.timer.average_time
            im = plot_tracking(frame, online_tlwhs, online_ids, scores=None, frame_id=frame_id, fps=fps, ids2=None)
            # Write the frame with detection and tracking results to the output video
            out.write(im)
            cv2.imshow("YOLOv8 track", im)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        # Release resources
        cap.release()
        out.release()
        print("Object detection and tracking completed successfully.")


if __name__ == "__main__":
    args = make_parser().parse_args()
    model = YOLOv8(args)
    model.run()
