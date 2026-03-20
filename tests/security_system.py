"""
Real-time AI Security System
Uses YOLOv8 for object detection - fully local, no cloud APIs
"""
import cv2
import numpy as np
import time
from pathlib import Path
import onnxruntime as ort
from typing import List, Tuple
import urllib.request
from datetime import datetime


class SecurityDetector:
    """Real ML-based object detector using YOLOv8"""
    
    def __init__(self, model_path: str = None, conf_threshold: float = 0.5):
        self.conf_threshold = conf_threshold
        self.model_path = model_path or self._download_model()
        
        # Initialize ONNX Runtime session
        providers = ['CPUExecutionProvider']
        # Try to use GPU if available
        if 'CUDAExecutionProvider' in ort.get_available_providers():
            providers.insert(0, 'CUDAExecutionProvider')
            
        self.session = ort.InferenceSession(self.model_path, providers=providers)
        
        # Get model input details
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]
        
        # COCO class names (YOLOv8 is trained on COCO dataset)
        self.class_names = self._load_coco_names()
        
        # Security-relevant classes to track
        self.alert_classes = {'person', 'car', 'truck', 'dog', 'cat', 'backpack', 'handbag'}
        
        print(f"✓ Model loaded: {self.model_path}")
        print(f"✓ Input size: {self.input_width}x{self.input_height}")
        print(f"✓ Providers: {providers}")
        
    def _download_model(self) -> str:
        """Download YOLOv8n ONNX model if not present"""
        model_dir = Path("models")
        model_dir.mkdir(exist_ok=True)
        model_path = model_dir / "yolov8n.onnx"
        
        if not model_path.exists():
            print("Downloading YOLOv8n model (~6MB)...")
            print("Installing ultralytics to export model...")
            
            # Install ultralytics if not present
            import subprocess
            import sys
            try:
                import ultralytics
            except ImportError:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "ultralytics"])
            
            # Export YOLOv8n to ONNX format
            from ultralytics import YOLO
            print("Exporting YOLOv8n to ONNX format...")
            model = YOLO('yolov8n.pt')  # This will auto-download the PyTorch model
            model.export(format='onnx', simplify=True)
            
            # Move the exported model to our models directory
            import shutil
            exported_path = Path('yolov8n.onnx')
            if exported_path.exists():
                shutil.move(str(exported_path), str(model_path))
            
            # Clean up the .pt file
            pt_file = Path('yolov8n.pt')
            if pt_file.exists():
                pt_file.unlink()
                
            print(f"✓ Model exported to {model_path}")
        
        return str(model_path)
    
    def _load_coco_names(self) -> List[str]:
        """Load COCO dataset class names"""
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
        """Preprocess frame for YOLOv8 input"""
        # Resize and pad to maintain aspect ratio
        img = cv2.resize(frame, (self.input_width, self.input_height))
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1] and transpose to CHW format
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # HWC to CHW
        
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        
        return img
    
    def postprocess(self, outputs: np.ndarray, orig_shape: Tuple[int, int]) -> List[dict]:
        """Post-process YOLOv8 outputs to get detections"""
        # Handle different output formats
        if isinstance(outputs, list):
            predictions = outputs[0]
        else:
            predictions = outputs
            
        # Debug: print shape to understand the format
        # print(f"Output shape: {predictions.shape}")
        
        # YOLOv8 ONNX output can be [batch, 84, 8400] or [1, 84, 8400]
        # We need to reshape to [8400, 84]
        if len(predictions.shape) == 3:
            # Remove batch dimension and transpose
            predictions = predictions[0]  # [84, 8400]
            predictions = predictions.T    # [8400, 84]
        elif len(predictions.shape) == 2:
            # Already in correct format
            if predictions.shape[0] < predictions.shape[1]:
                predictions = predictions.T
        else:
            raise ValueError(f"Unexpected output shape: {predictions.shape}")
        
        # Now predictions should be [8400, 84]
        # First 4 values are bbox (x, y, w, h), remaining 80 are class scores
        boxes = predictions[:, :4]
        scores = predictions[:, 4:]
        
        # Get class with highest score for each detection
        class_ids = np.argmax(scores, axis=1)
        confidences = scores[np.arange(len(scores)), class_ids]
        
        # Filter by confidence threshold
        mask = confidences > self.conf_threshold
        boxes = boxes[mask]
        confidences = confidences[mask]
        class_ids = class_ids[mask]
        
        # Apply Non-Maximum Suppression (NMS) to remove overlapping boxes
        if len(boxes) > 0:
            # Convert to [x1, y1, x2, y2] format for NMS
            boxes_xyxy = np.zeros_like(boxes)
            boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x1
            boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y1
            boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2  # x2
            boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2  # y2
            
            # Simple NMS
            indices = self._nms(boxes_xyxy, confidences, iou_threshold=0.45)
            boxes = boxes[indices]
            confidences = confidences[indices]
            class_ids = class_ids[indices]
        
        # Convert boxes from center format to corner format
        # and scale to original image size
        orig_h, orig_w = orig_shape
        scale_x = orig_w / self.input_width
        scale_y = orig_h / self.input_height
        
        detections = []
        for box, conf, cls_id in zip(boxes, confidences, class_ids):
            x_center, y_center, w, h = box
            
            # Convert to corner coordinates
            x1 = int((x_center - w/2) * scale_x)
            y1 = int((y_center - h/2) * scale_y)
            x2 = int((x_center + w/2) * scale_x)
            y2 = int((y_center + h/2) * scale_y)
            
            # Clip to image bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(orig_w, x2), min(orig_h, y2)
            
            class_name = self.class_names[cls_id]
            
            detections.append({
                'bbox': (x1, y1, x2, y2),
                'confidence': float(conf),
                'class_id': int(cls_id),
                'class_name': class_name,
                'is_alert': class_name in self.alert_classes
            })
        
        return detections
    
    def _nms(self, boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.45) -> np.ndarray:
        """Non-Maximum Suppression to remove overlapping boxes"""
        if len(boxes) == 0:
            return np.array([], dtype=int)
        
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        
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
        """Run detection on a frame"""
        orig_shape = frame.shape[:2]
        
        # Preprocess
        input_tensor = self.preprocess(frame)
        
        # Run inference
        outputs = self.session.run(None, {self.input_name: input_tensor})
        
        # Post-process (outputs is a list, we need the first element)
        detections = self.postprocess(outputs[0], orig_shape)
        
        return detections


class SecuritySystem:
    """Main security system with camera feed and detection"""
    
    def __init__(self, camera_id: int = 0, target_fps: int = 30):
        self.camera_id = camera_id
        self.target_fps = target_fps
        self.detector = SecurityDetector(conf_threshold=0.4)
        
        # Recording settings
        self.recording_dir = Path("recordings")
        self.recording_dir.mkdir(exist_ok=True)
        
        # Statistics
        self.frame_count = 0
        self.fps = 0
        self.last_alert_time = 0
        self.alert_cooldown = 2.0  # seconds
        
    def draw_detections(self, frame: np.ndarray, detections: List[dict]) -> np.ndarray:
        """Draw bounding boxes and labels on frame"""
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            conf = det['confidence']
            class_name = det['class_name']
            is_alert = det['is_alert']
            
            # Color: red for alert objects, green for others
            color = (0, 0, 255) if is_alert else (0, 255, 0)
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label background
            label = f"{class_name}: {conf:.2f}"
            (label_w, label_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(frame, (x1, y1 - label_h - 10), 
                         (x1 + label_w, y1), color, -1)
            
            # Draw label text
            cv2.putText(frame, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def draw_stats(self, frame: np.ndarray, detections: List[dict], 
                   inference_time: float) -> np.ndarray:
        """Draw statistics overlay"""
        h, w = frame.shape[:2]
        
        # Semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (300, 120), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        # Statistics text
        stats = [
            f"FPS: {self.fps:.1f}",
            f"Inference: {inference_time*1000:.1f}ms",
            f"Detections: {len(detections)}",
            f"Alerts: {sum(1 for d in detections if d['is_alert'])}"
        ]
        
        y_offset = 30
        for stat in stats:
            cv2.putText(frame, stat, (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y_offset += 25
        
        return frame
    
    def handle_alert(self, frame: np.ndarray, detections: List[dict]):
        """Handle security alerts"""
        current_time = time.time()
        
        # Check if cooldown period has passed
        if current_time - self.last_alert_time < self.alert_cooldown:
            return
        
        alert_objects = [d for d in detections if d['is_alert']]
        
        if alert_objects:
            # Save frame
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = self.recording_dir / f"alert_{timestamp}.jpg"
            cv2.imwrite(str(filename), frame)
            
            alert_names = [d['class_name'] for d in alert_objects]
            print(f"🚨 ALERT: Detected {', '.join(alert_names)} - Saved to {filename}")
            
            self.last_alert_time = current_time
    
    def run(self):
        """Main loop"""
        cap = cv2.VideoCapture(self.camera_id)
        
        if not cap.isOpened():
            print(f"❌ Failed to open camera {self.camera_id}")
            return
        
        # Set camera properties for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, self.target_fps)
        
        print(f"\n🎥 Security System Active")
        print(f"Camera: {self.camera_id}")
        print(f"Target FPS: {self.target_fps}")
        print(f"Press 'q' to quit, 's' to save frame\n")
        
        fps_start_time = time.time()
        fps_frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("❌ Failed to read frame")
                    break
                
                # Run detection
                t1 = time.time()
                detections = self.detector.detect(frame)
                inference_time = time.time() - t1
                
                # Handle alerts
                self.handle_alert(frame, detections)
                
                # Draw results
                frame = self.draw_detections(frame, detections)
                frame = self.draw_stats(frame, detections, inference_time)
                
                # Calculate FPS
                fps_frame_count += 1
                if time.time() - fps_start_time >= 1.0:
                    self.fps = fps_frame_count / (time.time() - fps_start_time)
                    fps_frame_count = 0
                    fps_start_time = time.time()
                
                # Display
                cv2.imshow('AI Security System', frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = self.recording_dir / f"manual_{timestamp}.jpg"
                    cv2.imwrite(str(filename), frame)
                    print(f"📸 Saved frame to {filename}")
                
                self.frame_count += 1
                
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print(f"\n✓ System stopped. Processed {self.frame_count} frames")


if __name__ == "__main__":
    # Create and run the security system
    system = SecuritySystem(camera_id=0, target_fps=60)
    system.run()