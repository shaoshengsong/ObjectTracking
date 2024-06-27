# Object Tracking with Python

Only cv2, numpy, and onnxruntime are required, making it very simple. If you don't want to use onnxruntime, you can refer to the YOLOv8_det_img_opencv.py example.
The detector and tracker have been decoupled. In this code repository, the detector uses YOLOv8 and the tracker uses ByteTrack. It is very easy to replace the detector and tracker in the code. For example, you can replace YOLOv8 with YOLOv9 or YOLOv10, and ByteTrack with other advanced trackers.


You only need two steps to add a tracker to the detector. The sample code is YOLOv8_track.py.

tracker = YourTracker()
tracker.update()

## YOLOv8 - ONNX Runtime

This project implements YOLOv8 using ONNX Runtime or OpenCV.



### Installing `onnxruntime-gpu`

If you have an NVIDIA GPU and want to leverage GPU acceleration, you can install the onnxruntime-gpu package using the following command:

```bash
pip install onnxruntime-gpu
```

Note: Make sure you have the appropriate GPU drivers installed on your system.

### Installing `onnxruntime` (CPU version)

If you don't have an NVIDIA GPU or prefer to use the CPU version of onnxruntime, you can install the onnxruntime package using the following command:

```bash
pip install onnxruntime
```

