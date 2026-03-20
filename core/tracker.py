"""Advanced tracking with Kalman filtering and appearance"""
import numpy as np
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
import cv2


@dataclass
class Track:
    """Represents a tracked object with state estimation"""
    track_id: int
    class_name: str
    bbox: Tuple[int, int, int, int]
    confidence: float
    first_seen: float
    last_seen: float
    last_detected: float  # Last time actually detected (vs predicted)
    
    # State
    state: str = "tentative"  # tentative -> confirmed -> lost
    hit_streak: int = 0  # Consecutive detections
    time_since_update: int = 0  # Frames since last detection
    
    # History
    positions: deque = field(default_factory=lambda: deque(maxlen=100))
    confidences: deque = field(default_factory=lambda: deque(maxlen=30))
    zones: deque = field(default_factory=lambda: deque(maxlen=30))
    
    # Kalman filter state
    kalman: Optional[cv2.KalmanFilter] = None
    predicted_bbox: Optional[Tuple[int, int, int, int]] = None
    
    @property
    def duration(self) -> float:
        return self.last_seen - self.first_seen
    
    @property
    def center(self) -> Tuple[int, int]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)
    
    @property
    def avg_confidence(self) -> float:
        if not self.confidences:
            return self.confidence
        return float(np.mean(self.confidences))
    
    @property
    def velocity(self) -> float:
        """Average velocity in pixels/frame"""
        if len(self.positions) < 2:
            return 0.0
        
        distances = []
        for i in range(1, len(self.positions)):
            p1 = np.array(self.positions[i-1])
            p2 = np.array(self.positions[i])
            distances.append(np.linalg.norm(p2 - p1))
        
        return float(np.mean(distances)) if distances else 0.0
    
    @property
    def is_stationary(self) -> bool:
        """Check if object is not moving much"""
        return self.velocity < 5.0  # Less than 5 pixels/frame


class ImprovedTracker:
    """
    Multi-object tracker with:
    - Kalman filtering for motion prediction
    - Tentative/confirmed track states
    - Better handling of occlusions
    """
    
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        
        self.tracks: Dict[int, Track] = {}
        self.next_id = 1
        self.frame_count = 0
        
        self.max_age = config.tracking.max_age_frames
        self.min_hits = config.tracking.min_hits
        self.iou_threshold = config.tracking.iou_threshold
        self.use_kalman = config.tracking.use_kalman
        self.max_tracks = config.tracking.max_tracks
        
    def update(self, detections: List[dict]) -> Dict[int, Track]:
        """Update tracks with new detections"""
        self.frame_count += 1
        current_time = time.time()
        
        # Predict new positions for existing tracks using Kalman filter
        self._predict_tracks()
        
        # Match detections to tracks
        matched_pairs, unmatched_dets, unmatched_tracks = self._match(detections)
        
        # Update matched tracks
        for track_id, det_idx in matched_pairs:
            self._update_track(track_id, detections[det_idx], current_time)
        
        # Handle unmatched tracks (predict or mark for deletion)
        for track_id in unmatched_tracks:
            self._handle_unmatched_track(track_id, current_time)
        
        # Create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            detection = detections[det_idx]
            # Only create track if confidence is high enough
            if detection['confidence'] >= self.config.detection.min_confidence_for_tracking:
                self._create_track(detection, current_time)
        
        # Remove dead tracks
        self._prune_tracks()
        
        # Limit number of tracks
        if len(self.tracks) > self.max_tracks:
            self._limit_tracks()
        
        return {tid: t for tid, t in self.tracks.items() if t.state == "confirmed"}
    
    def _predict_tracks(self):
        """Predict next position for all tracks using Kalman filter"""
        for track in self.tracks.values():
            if self.use_kalman and track.kalman is not None:
                try:
                    # Kalman prediction
                    prediction = track.kalman.predict()
                    
                    # Extract coordinates (handle shape correctly)
                    cx = float(prediction[0, 0])
                    cy = float(prediction[1, 0])
                    
                    # Use last known size
                    x1, y1, x2, y2 = track.bbox
                    w, h = x2 - x1, y2 - y1
                    
                    track.predicted_bbox = (
                        int(cx - w/2), int(cy - h/2),
                        int(cx + w/2), int(cy + h/2)
                    )
                except Exception as e:
                    track.predicted_bbox = track.bbox
                    # Don't spam logs with Kalman errors
                    pass
    
    def _match(self, detections: List[dict]) -> Tuple[List, List, List]:
        """
        Match detections to tracks using greedy IoU matching
        
        Returns: (matched_pairs, unmatched_detections, unmatched_tracks)
        """
        if len(self.tracks) == 0:
            return [], list(range(len(detections))), []
        
        if len(detections) == 0:
            return [], [], list(self.tracks.keys())
        
        # Build cost matrix
        track_ids = list(self.tracks.keys())
        cost_matrix = np.zeros((len(track_ids), len(detections)))
        
        for i, track_id in enumerate(track_ids):
            track = self.tracks[track_id]
            track_bbox = track.predicted_bbox if track.predicted_bbox else track.bbox
            
            for j, detection in enumerate(detections):
                # Only match same class
                if detection['class_name'] != track.class_name:
                    cost_matrix[i, j] = 1e6  # Very high cost
                else:
                    iou = self._calculate_iou(track_bbox, detection['bbox'])
                    cost_matrix[i, j] = 1 - iou  # Convert IoU to cost
        
        # Greedy matching
        matched_pairs = []
        unmatched_dets = set(range(len(detections)))
        unmatched_tracks = set(track_ids)
        
        for i, track_id in enumerate(track_ids):
            best_j = None
            best_cost = self.iou_threshold
            
            for j in range(len(detections)):
                if j not in unmatched_dets:
                    continue
                    
                cost = cost_matrix[i, j]
                iou = 1 - cost
                
                if iou > best_cost:
                    best_cost = iou
                    best_j = j
            
            if best_j is not None:
                matched_pairs.append((track_id, best_j))
                unmatched_dets.remove(best_j)
                unmatched_tracks.remove(track_id)
        
        return matched_pairs, list(unmatched_dets), list(unmatched_tracks)
    
    def _update_track(self, track_id: int, detection: dict, current_time: float):
        """Update track with new detection"""
        track = self.tracks[track_id]
        
        track.bbox = detection['bbox']
        track.confidence = detection['confidence']
        track.last_seen = current_time
        track.last_detected = current_time
        track.time_since_update = 0
        track.hit_streak += 1
        
        # Update Kalman filter
        if self.use_kalman and track.kalman is not None:
            try:
                cx, cy = track.center
                measurement = np.array([
                    [np.float32(cx)], 
                    [np.float32(cy)]
                ], dtype=np.float32)
                track.kalman.correct(measurement)
            except Exception:
                pass
        
        # Update history
        track.positions.append(track.center)
        track.confidences.append(detection['confidence'])
        
        if 'zone' in detection:
            track.zones.append(detection['zone'])
        
        # State transition: tentative -> confirmed
        if track.state == "tentative" and track.hit_streak >= self.min_hits:
            track.state = "confirmed"
            self.logger.info(f"Track {track_id} confirmed: {track.class_name}")
    
    def _handle_unmatched_track(self, track_id: int, current_time: float):
        """Handle track that wasn't matched to any detection"""
        track = self.tracks[track_id]
        
        track.time_since_update += 1
        track.hit_streak = 0
        track.last_seen = current_time
        
        # Use Kalman prediction for position
        if track.predicted_bbox:
            track.bbox = track.predicted_bbox
            track.positions.append(track.center)
    
    def _create_track(self, detection: dict, current_time: float):
        """Create new track from detection"""
        # Initialize Kalman filter
        kalman = None
        if self.use_kalman:
            kalman = self._init_kalman(detection['bbox'])
        
        track = Track(
            track_id=self.next_id,
            class_name=detection['class_name'],
            bbox=detection['bbox'],
            confidence=detection['confidence'],
            first_seen=current_time,
            last_seen=current_time,
            last_detected=current_time,
            state="tentative",
            hit_streak=1,
            kalman=kalman
        )
        
        track.positions.append(track.center)
        track.confidences.append(detection['confidence'])
        
        if 'zone' in detection:
            track.zones.append(detection['zone'])
        
        self.tracks[self.next_id] = track
        self.next_id += 1
    
    def _init_kalman(self, bbox: Tuple[int, int, int, int]) -> cv2.KalmanFilter:
        """Initialize Kalman filter for tracking"""
        kalman = cv2.KalmanFilter(4, 2)  # 4 state vars, 2 measurements
        kalman.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], np.float32)
        
        kalman.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], np.float32)
        
        kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        
        # Initialize state with bbox center
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        
        kalman.statePre = np.array([
            [np.float32(cx)], 
            [np.float32(cy)], 
            [np.float32(0)], 
            [np.float32(0)]
        ], dtype=np.float32)
        
        kalman.statePost = np.array([
            [np.float32(cx)], 
            [np.float32(cy)], 
            [np.float32(0)], 
            [np.float32(0)]
        ], dtype=np.float32)
        
        return kalman
    
    def _prune_tracks(self):
        """Remove tracks that are too old"""
        to_remove = []
        
        for track_id, track in self.tracks.items():
            if track.time_since_update > self.max_age:
                to_remove.append(track_id)
        
        for track_id in to_remove:
            del self.tracks[track_id]
    
    def _limit_tracks(self):
        """Limit number of active tracks to prevent memory issues"""
        if len(self.tracks) <= self.max_tracks:
            return
        
        # Sort by priority
        sorted_tracks = sorted(
            self.tracks.items(),
            key=lambda x: (
                x[1].state == "confirmed",
                -x[1].time_since_update,
                x[1].avg_confidence
            ),
            reverse=True
        )
        
        # Keep top N tracks
        to_keep = sorted_tracks[:self.max_tracks]
        
        self.tracks = {tid: track for tid, track in to_keep}
    
    def _calculate_iou(self, box1: Tuple[int, int, int, int],
                       box2: Tuple[int, int, int, int]) -> float:
        """Calculate Intersection over Union"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)
        
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = box1_area + box2_area - inter_area
        
        if union_area == 0:
            return 0
        
        return inter_area / union_area
    
    def get_stats(self) -> dict:
        """Get tracker statistics"""
        return {
            'total_tracks': len(self.tracks),
            'confirmed_tracks': sum(1 for t in self.tracks.values() if t.state == "confirmed"),
            'tentative_tracks': sum(1 for t in self.tracks.values() if t.state == "tentative"),
            'next_id': self.next_id,
        }