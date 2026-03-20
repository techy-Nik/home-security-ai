"""YOLOv8 Object Detector with configuration support"""
import cv2
import numpy as np
from pathlib import Path
import onnxruntime as ort
from typing import List, Tuple


class SecurityDetector:
    """YOLOv8 object detector"""
    
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        
        self.conf_threshold = config.detection.confidence_threshold
        self.nms_threshold = config.detection.nms_threshold
        self.model_path = config.detection.model_path
        
        # Ensure model exists
        self._ensure_model()
        
        # Initialize ONNX Runtime
        providers = ['CPUExecutionProvider']
        if 'CUDAExecutionProvider' in ort.get_available_providers():
            providers.insert(0, 'CUDAExecutionProvider')
        
        self.session = ort.InferenceSession(self.model_path, providers=providers)
        
        # Get input details
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]
        
        # COCO classes
        self.class_names = self._load_coco_names()
        
        self.logger.info(f"[OK] YOLOv8 loaded: {self.model_path}")
        self.logger.info(f"  Input size: {self.input_width}x{self.input_height}")
        self.logger.info(f"  Providers: {providers}")
    
    def _ensure_model(self):
        """Download model if not present"""
        model_path = Path(self.model_path)
        
        if not model_path.exists():
            self.logger.info("Model not found. Downloading YOLOv8n...")
            model_path.parent.mkdir(exist_ok=True, parents=True)
            
            import subprocess
            import sys
            
            try:
                import ultralytics
            except ImportError:
                self.logger.info("Installing ultralytics...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "ultralytics"])
            
            from ultralytics import YOLO
            self.logger.info("Exporting model to ONNX...")
            model = YOLO('yolov8n.pt')
            model.export(format='onnx', simplify=True)
            
            # Move to correct location
            import shutil
            if Path('yolov8n.onnx').exists():
                shutil.move('yolov8n.onnx', str(model_path))
            
            # Cleanup
            if Path('yolov8n.pt').exists():
                Path('yolov8n.pt').unlink()
            
            self.logger.info(f"[OK] Model downloaded to {model_path}")
    
    def _load_coco_names(self) -> List[str]:
        """Load COCO class names"""
        return [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
            'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
            'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
            'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
            'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
            'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
            'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
    
    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame for YOLO"""
        img = cv2.resize(frame, (self.input_width, self.input_height))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        return img
    
    def postprocess(self, outputs: np.ndarray, orig_shape: Tuple[int, int]) -> List[dict]:
        """Postprocess YOLO outputs"""
        if isinstance(outputs, list):
            predictions = outputs[0]
        else:
            predictions = outputs
        
        if len(predictions.shape) == 3:
            predictions = predictions[0].T
        elif len(predictions.shape) == 2:
            if predictions.shape[0] < predictions.shape[1]:
                predictions = predictions.T
        
        boxes = predictions[:, :4]
        scores = predictions[:, 4:]
        
        class_ids = np.argmax(scores, axis=1)
        confidences = scores[np.arange(len(scores)), class_ids]
        
        mask = confidences > self.conf_threshold
        boxes = boxes[mask]
        confidences = confidences[mask]
        class_ids = class_ids[mask]
        
        if len(boxes) > 0:
            boxes_xyxy = np.zeros_like(boxes)
            boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
            boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
            boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
            boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
            
            indices = self._nms(boxes_xyxy, confidences, self.nms_threshold)
            boxes = boxes[indices]
            confidences = confidences[indices]
            class_ids = class_ids[indices]
        
        orig_h, orig_w = orig_shape
        scale_x = orig_w / self.input_width
        scale_y = orig_h / self.input_height
        
        detections = []
        for box, conf, cls_id in zip(boxes, confidences, class_ids):
            x_center, y_center, w, h = box
            
            x1 = int((x_center - w/2) * scale_x)
            y1 = int((y_center - h/2) * scale_y)
            x2 = int((x_center + w/2) * scale_x)
            y2 = int((y_center + h/2) * scale_y)
            
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(orig_w, x2), min(orig_h, y2)
            
            detections.append({
                'bbox': (x1, y1, x2, y2),
                'confidence': float(conf),
                'class_id': int(cls_id),
                'class_name': self.class_names[cls_id],
            })
        
        return detections
    
    def _nms(self, boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
        """Non-Maximum Suppression"""
        if len(boxes) == 0:
            return np.array([], dtype=int)
        
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            
            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]
        
        return np.array(keep, dtype=int)
    
    def detect(self, frame: np.ndarray) -> List[dict]:
        """Run detection on frame"""
        orig_shape = frame.shape[:2]
        input_tensor = self.preprocess(frame)
        outputs = self.session.run(None, {self.input_name: input_tensor})
        detections = self.postprocess(outputs[0], orig_shape)
        return detections