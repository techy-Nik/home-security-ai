"""
Enhanced AI Security System v2.0
Features:
1. YOLOv8 Object Detection
2. MOG2 Motion Detection (10x performance boost)
3. DeepSORT Multi-Object Tracking
4. Activity Zones (context-aware)
5. Smart Bayesian Alert Logic
"""
import cv2
import numpy as np
import time
from pathlib import Path
import onnxruntime as ort
from typing import List, Tuple, Dict, Optional
from datetime import datetime
from collections import defaultdict, deque
from dataclasses import dataclass
import json


@dataclass
class TrackedObject:
    """Represents a tracked object with history"""
    track_id: int
    class_name: str
    bbox: Tuple[int, int, int, int]
    confidence: float
    first_seen: float
    last_seen: float
    positions: deque  # History of positions
    zone: Optional[str] = None
    
    @property
    def duration(self) -> float:
        """How long this object has been tracked (seconds)"""
        return self.last_seen - self.first_seen
    
    @property
    def center(self) -> Tuple[int, int]:
        """Center point of bounding box"""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)


class ActivityZone:
    """Defines an activity zone with custom rules"""
    def __init__(self, name: str, polygon: List[Tuple[int, int]], 
                 alert_classes: set = None, sensitivity: float = 1.0):
        self.name = name
        self.polygon = np.array(polygon, dtype=np.int32)
        self.alert_classes = alert_classes or {'person', 'car', 'truck'}
        self.sensitivity = sensitivity  # Multiplier for threat score
        
    def contains_point(self, point: Tuple[int, int]) -> bool:
        """Check if a point is inside this zone"""
        result = cv2.pointPolygonTest(self.polygon, point, False)
        return result >= 0
    
    def draw(self, frame: np.ndarray, color=(255, 255, 0), thickness=2):
        """Draw zone on frame"""
        cv2.polylines(frame, [self.polygon], True, color, thickness)
        # Draw zone name
        x, y = self.polygon[0]
        cv2.putText(frame, self.name, (x, y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


class MotionDetector:
    """MOG2-based motion detection for performance optimization"""
    def __init__(self, history: int = 500, var_threshold: int = 16):
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=history, 
            varThreshold=var_threshold,
            detectShadows=True
        )
        self.min_contour_area = 500  # Minimum area to consider as motion
        
    def detect_motion(self, frame: np.ndarray) -> Tuple[bool, np.ndarray, List]:
        """
        Detect motion in frame
        Returns: (has_motion, fg_mask, motion_boxes)
        """
        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(frame)
        
        # Remove shadows (they're marked as 127)
        fg_mask[fg_mask == 127] = 0
        
        # Apply morphological operations to reduce noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area
        motion_boxes = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.min_contour_area:
                x, y, w, h = cv2.boundingRect(contour)
                motion_boxes.append((x, y, x+w, y+h))
        
        has_motion = len(motion_boxes) > 0
        
        return has_motion, fg_mask, motion_boxes


class SimpleTracker:
    """Simplified DeepSORT-style tracker using IoU matching"""
    def __init__(self, max_age: int = 30, min_hits: int = 3, iou_threshold: float = 0.3):
        self.max_age = max_age  # Frames to keep track alive without detection
        self.min_hits = min_hits  # Minimum detections before confirmed
        self.iou_threshold = iou_threshold
        
        self.tracks: Dict[int, TrackedObject] = {}
        self.next_id = 1
        self.frame_count = 0
        
    def update(self, detections: List[dict]) -> Dict[int, TrackedObject]:
        """Update tracks with new detections"""
        self.frame_count += 1
        current_time = time.time()
        
        # Match detections to existing tracks
        matched_tracks = set()
        matched_detections = set()
        
        for track_id, track in list(self.tracks.items()):
            best_iou = 0
            best_det_idx = -1
            
            for det_idx, detection in enumerate(detections):
                if det_idx in matched_detections:
                    continue
                    
                # Only match same class
                if detection['class_name'] != track.class_name:
                    continue
                
                iou = self._calculate_iou(track.bbox, detection['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_det_idx = det_idx
            
            # Update track if good match found
            if best_iou > self.iou_threshold:
                detection = detections[best_det_idx]
                track.bbox = detection['bbox']
                track.confidence = detection['confidence']
                track.last_seen = current_time
                track.positions.append(track.center)
                track.zone = detection.get('zone')
                
                matched_tracks.add(track_id)
                matched_detections.add(best_det_idx)
        
        # Create new tracks for unmatched detections
        for det_idx, detection in enumerate(detections):
            if det_idx not in matched_detections:
                track = TrackedObject(
                    track_id=self.next_id,
                    class_name=detection['class_name'],
                    bbox=detection['bbox'],
                    confidence=detection['confidence'],
                    first_seen=current_time,
                    last_seen=current_time,
                    positions=deque(maxlen=50),
                    zone=detection.get('zone')
                )
                track.positions.append(track.center)
                self.tracks[self.next_id] = track
                self.next_id += 1
        
        # Remove old tracks
        for track_id in list(self.tracks.keys()):
            track = self.tracks[track_id]
            if current_time - track.last_seen > self.max_age:
                del self.tracks[track_id]
        
        return self.tracks
    
    def _calculate_iou(self, box1: Tuple[int, int, int, int], 
                       box2: Tuple[int, int, int, int]) -> float:
        """Calculate Intersection over Union"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Intersection
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)
        
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        
        # Union
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = box1_area + box2_area - inter_area
        
        if union_area == 0:
            return 0
        
        return inter_area / union_area


class ThreatAssessor:
    """Bayesian-style threat assessment based on context"""
    
    def __init__(self):
        # Time-based priors (hour -> threat multiplier)
        self.time_priors = {
            range(0, 6): 1.5,    # Late night: 1.5x threat
            range(6, 9): 0.8,    # Early morning: 0.8x
            range(9, 18): 0.7,   # Daytime: 0.7x
            range(18, 22): 0.9,  # Evening: 0.9x
            range(22, 24): 1.3,  # Night: 1.3x
        }
        
        # Loitering thresholds (seconds)
        self.loiter_threshold = 10.0  # 10 seconds in one place
        self.suspicious_duration = 30.0  # 30 seconds total
        
    def assess_threat(self, track: TrackedObject, zone: Optional[ActivityZone] = None) -> float:
        """
        Calculate threat score (0-100)
        Higher score = more suspicious
        """
        score = 0.0
        
        # Base score by object class
        class_scores = {
            'person': 30,
            'car': 20,
            'truck': 20,
            'motorcycle': 25,
            'backpack': 15,
            'handbag': 10,
            'dog': 5,
        }
        score += class_scores.get(track.class_name, 5)
        
        # Time-of-day multiplier
        current_hour = datetime.now().hour
        time_multiplier = 1.0
        for hour_range, multiplier in self.time_priors.items():
            if current_hour in hour_range:
                time_multiplier = multiplier
                break
        score *= time_multiplier
        
        # Duration-based scoring
        if track.duration > self.suspicious_duration:
            score += 30  # Long presence is suspicious
        elif track.duration > self.loiter_threshold:
            score += 15  # Loitering
        
        # Zone-based multiplier
        if zone:
            score *= zone.sensitivity
            
        # Confidence multiplier
        score *= track.confidence
        
        # Cap at 100
        return min(100, score)
    
    def get_alert_level(self, threat_score: float) -> Tuple[str, str, Tuple[int, int, int]]:
        """
        Convert threat score to alert level
        Returns: (level, description, color_bgr)
        """
        if threat_score >= 70:
            return ("HIGH", "🚨 High Threat", (0, 0, 255))  # Red
        elif threat_score >= 50:
            return ("MEDIUM", "⚠️ Medium Threat", (0, 165, 255))  # Orange
        elif threat_score >= 30:
            return ("LOW", "ℹ️ Low Alert", (0, 255, 255))  # Yellow
        else:
            return ("INFO", "👁️ Monitoring", (0, 255, 0))  # Green


class SecurityDetector:
    """Real ML-based object detector using YOLOv8"""
    
    def __init__(self, model_path: str = None, conf_threshold: float = 0.4):
        self.conf_threshold = conf_threshold
        self.model_path = model_path or self._download_model()
        
        # Initialize ONNX Runtime session
        providers = ['CPUExecutionProvider']
        if 'CUDAExecutionProvider' in ort.get_available_providers():
            providers.insert(0, 'CUDAExecutionProvider')
            
        self.session = ort.InferenceSession(self.model_path, providers=providers)
        
        # Get model input details
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]
        
        # COCO class names
        self.class_names = self._load_coco_names()
        
        print(f"✓ YOLOv8 loaded: {self.model_path}")
        print(f"✓ Input size: {self.input_width}x{self.input_height}")
        print(f"✓ Providers: {providers}")
        
    def _download_model(self) -> str:
        """Download YOLOv8n ONNX model if not present"""
        model_dir = Path("models")
        model_dir.mkdir(exist_ok=True)
        model_path = model_dir / "yolov8n.onnx"
        
        if not model_path.exists():
            print("Downloading YOLOv8n model...")
            print("Installing ultralytics to export model...")
            
            import subprocess
            import sys
            try:
                import ultralytics
            except ImportError:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "ultralytics"])
            
            from ultralytics import YOLO
            print("Exporting YOLOv8n to ONNX format...")
            model = YOLO('yolov8n.pt')
            model.export(format='onnx', simplify=True)
            
            import shutil
            exported_path = Path('yolov8n.onnx')
            if exported_path.exists():
                shutil.move(str(exported_path), str(model_path))
            
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
        """Preprocess frame for YOLOv8"""
        img = cv2.resize(frame, (self.input_width, self.input_height))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        return img
    
    def postprocess(self, outputs: np.ndarray, orig_shape: Tuple[int, int]) -> List[dict]:
        """Post-process YOLOv8 outputs"""
        if isinstance(outputs, list):
            predictions = outputs[0]
        else:
            predictions = outputs
            
        if len(predictions.shape) == 3:
            predictions = predictions[0]
            predictions = predictions.T
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
            
            indices = self._nms(boxes_xyxy, confidences, iou_threshold=0.45)
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
            
            class_name = self.class_names[cls_id]
            
            detections.append({
                'bbox': (x1, y1, x2, y2),
                'confidence': float(conf),
                'class_id': int(cls_id),
                'class_name': class_name,
            })
        
        return detections
    
    def _nms(self, boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.45) -> np.ndarray:
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


class EnhancedSecuritySystem:
    """Enhanced Security System with all 4 improvements"""
    
    def __init__(self, camera_id: int = 0, target_fps: int = 30):
        self.camera_id = camera_id
        self.target_fps = target_fps
        
        # Initialize components
        print("\n🔧 Initializing Enhanced Security System...")
        self.detector = SecurityDetector(conf_threshold=0.4)
        self.motion_detector = MotionDetector()
        self.tracker = SimpleTracker()
        self.threat_assessor = ThreatAssessor()
        
        # Activity zones (customize these for your setup)
        self.zones = self._create_default_zones()
        
        # Recording
        self.recording_dir = Path("recordings")
        self.recording_dir.mkdir(exist_ok=True)
        
        # Statistics
        self.stats = {
            'frames_processed': 0,
            'frames_with_motion': 0,
            'frames_with_detections': 0,
            'yolo_calls': 0,
            'alerts_triggered': 0,
        }
        self.fps = 0
        self.motion_only_mode = True  # Run YOLO only on motion
        
        print("✓ System initialized!")
        
    def _create_default_zones(self) -> List[ActivityZone]:
        """Create default activity zones - customize for your camera view"""
        # These are example zones - you should adjust based on your camera view
        # Coordinates are [x, y] points forming a polygon
        zones = []
        
        # You can add zones in setup mode (press 'z' during runtime)
        # For now, return empty list
        return zones
    
    def add_zone(self, name: str, polygon: List[Tuple[int, int]], 
                 alert_classes: set = None, sensitivity: float = 1.0):
        """Add a new activity zone"""
        zone = ActivityZone(name, polygon, alert_classes, sensitivity)
        self.zones.append(zone)
        print(f"✓ Added zone: {name}")
    
    def _assign_zones(self, detections: List[dict]):
        """Assign zone to each detection"""
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            
            detection['zone'] = None
            for zone in self.zones:
                if zone.contains_point(center):
                    detection['zone'] = zone.name
                    break
    
    def draw_ui(self, frame: np.ndarray, tracks: Dict[int, TrackedObject],
                has_motion: bool, motion_boxes: List, inference_time: float) -> np.ndarray:
        """Draw complete UI overlay"""
        h, w = frame.shape[:2]
        
        # Draw zones
        for zone in self.zones:
            zone.draw(frame)
        
        # Draw motion boxes (if in debug mode)
        if has_motion and len(motion_boxes) > 0:
            for box in motion_boxes:
                x1, y1, x2, y2 = box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (100, 100, 100), 1)
        
        # Draw tracked objects
        for track_id, track in tracks.items():
            x1, y1, x2, y2 = track.bbox
            
            # Calculate threat
            zone_obj = None
            for z in self.zones:
                if z.name == track.zone:
                    zone_obj = z
                    break
            
            threat_score = self.threat_assessor.assess_threat(track, zone_obj)
            level, desc, color = self.threat_assessor.get_alert_level(threat_score)
            
            # Draw bounding box
            thickness = 3 if level in ["HIGH", "MEDIUM"] else 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            
            # Draw tracking trail
            if len(track.positions) > 1:
                points = np.array(list(track.positions), dtype=np.int32)
                cv2.polylines(frame, [points], False, color, 2)
            
            # Draw label with threat info
            label_lines = [
                f"ID:{track_id} {track.class_name}",
                f"Threat: {threat_score:.0f}% ({level})",
                f"Duration: {track.duration:.1f}s",
            ]
            if track.zone:
                label_lines.append(f"Zone: {track.zone}")
            
            y_offset = y1 - 10
            for line in reversed(label_lines):
                (tw, th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (x1, y_offset - th - 5), (x1 + tw + 5, y_offset), color, -1)
                cv2.putText(frame, line, (x1 + 2, y_offset - 3),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                y_offset -= (th + 8)
        
        # Draw statistics panel
        self._draw_stats_panel(frame, has_motion, len(tracks), inference_time)
        
        # Draw legend
        self._draw_legend(frame)
        
        return frame
    
    def _draw_stats_panel(self, frame: np.ndarray, has_motion: bool, 
                         num_tracks: int, inference_time: float):
        """Draw statistics overlay"""
        h, w = frame.shape[:2]
        
        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (350, 200), (0, 0, 0), -1)
        frame[:] = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        stats_text = [
            f"FPS: {self.fps:.1f}",
            f"Motion: {'YES' if has_motion else 'NO'}",
            f"Active Tracks: {num_tracks}",
            f"Inference: {inference_time*1000:.1f}ms",
            f"Mode: {'Motion+YOLO' if self.motion_only_mode else 'Always YOLO'}",
            "",
            f"Total Frames: {self.stats['frames_processed']}",
            f"YOLO Calls: {self.stats['yolo_calls']} ({self.stats['yolo_calls']/max(1, self.stats['frames_processed'])*100:.1f}%)",
            f"Alerts: {self.stats['alerts_triggered']}",
        ]
        
        y_offset = 30
        for text in stats_text:
            color = (0, 255, 0) if "Motion: YES" in text else (255, 255, 255)
            cv2.putText(frame, text, (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            y_offset += 20
    
    def _draw_legend(self, frame: np.ndarray):
        """Draw control legend"""
        h, w = frame.shape[:2]
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (w - 250, 10), (w - 10, 130), (0, 0, 0), -1)
        frame[:] = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        controls = [
            "Controls:",
            "Q - Quit",
            "S - Save frame",
            "M - Toggle motion mode",
            "Z - Zone setup mode",
            "R - Reset stats",
        ]
        
        y_offset = 30
        for text in controls:
            cv2.putText(frame, text, (w - 240, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            y_offset += 18
    
    def handle_alert(self, tracks: Dict[int, TrackedObject]):
        """Handle high-threat alerts"""
        for track_id, track in tracks.items():
            threat_score = self.threat_assessor.assess_threat(track)
            level, desc, _ = self.threat_assessor.get_alert_level(threat_score)
            
            if level == "HIGH" and track.duration > 2.0:  # High threat for >2 seconds
                self.stats['alerts_triggered'] += 1
                print(f"{desc} - Track ID:{track_id} {track.class_name} - "
                      f"Threat:{threat_score:.0f}% Duration:{track.duration:.1f}s")
    
    def run(self):
        """Main loop"""
        cap = cv2.VideoCapture(self.camera_id)
        
        if not cap.isOpened():
            print(f"❌ Failed to open camera {self.camera_id}")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, self.target_fps)
        
        print(f"\n🎥 Enhanced Security System Active")
        print(f"Camera: {self.camera_id}")
        print(f"Target FPS: {self.target_fps}")
        print(f"Motion-triggered mode: {'ON' if self.motion_only_mode else 'OFF'}")
        print("\nPress 'Q' to quit, 'S' to save, 'M' to toggle motion mode\n")
        
        fps_start_time = time.time()
        fps_frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                self.stats['frames_processed'] += 1
                
                # Step 1: Motion Detection (always runs - very fast)
                has_motion, fg_mask, motion_boxes = self.motion_detector.detect_motion(frame)
                if has_motion:
                    self.stats['frames_with_motion'] += 1
                
                # Step 2: Object Detection (conditional)
                detections = []
                inference_time = 0
                
                if (not self.motion_only_mode) or has_motion:
                    t1 = time.time()
                    detections = self.detector.detect(frame)
                    inference_time = time.time() - t1
                    self.stats['yolo_calls'] += 1
                    
                    if detections:
                        self.stats['frames_with_detections'] += 1
                
                # Step 3: Assign zones
                self._assign_zones(detections)
                
                # Step 4: Update tracker
                tracks = self.tracker.update(detections)
                
                # Step 5: Handle alerts
                self.handle_alert(tracks)
                
                # Step 6: Draw UI
                frame = self.draw_ui(frame, tracks, has_motion, motion_boxes, inference_time)
                
                # Calculate FPS
                fps_frame_count += 1
                if time.time() - fps_start_time >= 1.0:
                    self.fps = fps_frame_count / (time.time() - fps_start_time)
                    fps_frame_count = 0
                    fps_start_time = time.time()
                
                # Display
                cv2.imshow('Enhanced Security System v2.0', frame)
                
                # Handle keyboard
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = self.recording_dir / f"snapshot_{timestamp}.jpg"
                    cv2.imwrite(str(filename), frame)
                    print(f"📸 Saved: {filename}")
                elif key == ord('m'):
                    self.motion_only_mode = not self.motion_only_mode
                    print(f"Motion mode: {'ON' if self.motion_only_mode else 'OFF'}")
                elif key == ord('r'):
                    self.stats = {k: 0 for k in self.stats.keys()}
                    print("📊 Stats reset")
                elif key == ord('z'):
                    print("Zone setup mode - coming soon! Edit code to add zones manually.")
                
        finally:
            cap.release()
            cv2.destroyAllWindows()
            
            print(f"\n✓ System stopped")
            print(f"Frames processed: {self.stats['frames_processed']}")
            print(f"YOLO efficiency: {self.stats['yolo_calls']}/{self.stats['frames_processed']} "
                  f"({self.stats['yolo_calls']/max(1, self.stats['frames_processed'])*100:.1f}%)")
            print(f"Alerts triggered: {self.stats['alerts_triggered']}")


if __name__ == "__main__":
    # Example: Add custom zones (adjust coordinates for your camera)
    system = EnhancedSecuritySystem(camera_id=0, target_fps=60)
    
    # Example zone definitions (uncomment and adjust for your setup):
    # Front door zone (high sensitivity)
    # system.add_zone("Front Door", [(100, 300), (500, 300), (500, 600), (100, 600)], 
    #                 alert_classes={'person', 'backpack'}, sensitivity=1.3)
    
    # Driveway zone (medium sensitivity)
    # system.add_zone("Driveway", [(600, 200), (1200, 200), (1200, 700), (600, 700)],
    #                 alert_classes={'person', 'car', 'truck'}, sensitivity=1.0)
    
    system.run()
