"""
Enhanced Security System v3.0 - Main Entry Point
Production-grade security camera system with AI detection
"""
import cv2
import numpy as np
import time
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from utils.config import ConfigLoader
from utils.logger import StructuredLogger
from core.detector import SecurityDetector
from core.motion import SmartMotionDetector
from core.tracker import ImprovedTracker
from core.zones import ZoneManager
from core.threat import ImprovedThreatAssessor, ThreatLevel


class ProductionSecuritySystem:
    """
    Production-grade security system with:
    - Smart motion detection
    - Object tracking with Kalman filtering
    - Zone-based monitoring
    - Learning threat assessment
    - Structured logging
    - Error recovery
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        # Load configuration
        self.config = ConfigLoader.load(config_path)
        
        # Initialize logger
        self.logger = StructuredLogger(self.config)
        self.logger.info("="*60)
        self.logger.info("Enhanced Security System v3.0 Starting")
        self.logger.info("="*60)
        
        # Initialize components
        try:
            self.detector = SecurityDetector(self.config, self.logger)
            self.motion_detector = SmartMotionDetector(self.config, self.logger)
            self.tracker = ImprovedTracker(self.config, self.logger)
            self.zone_manager = ZoneManager(self.config, self.logger)
            self.threat_assessor = ImprovedThreatAssessor(self.config, self.logger)
        except Exception as e:
            self.logger.error(f"Failed to initialize components: {e}")
            raise
        
        # Recording setup
        self.recording_dir = Path(self.config.recording.output_dir)
        self.recording_dir.mkdir(exist_ok=True)
        
        # Performance tracking
        self.stats = {
            'frames_processed': 0,
            'frames_with_motion': 0,
            'frames_with_valid_motion': 0,
            'yolo_calls': 0,
            'detections_total': 0,
            'alerts_triggered': 0,
            'errors': 0,
        }
        
        self.fps = 0.0
        self.inference_times = []
        
        # Camera state
        self.cap = None
        self.frame_skip_counter = 0
        
        self.logger.info("[OK] System initialized successfully")
    
    def _init_camera(self) -> bool:
        """Initialize camera with retry logic"""
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                self.cap = cv2.VideoCapture(self.config.camera.id)
                
                if not self.cap.isOpened():
                    raise RuntimeError(f"Failed to open camera {self.config.camera.id}")
                
                # Set camera properties
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.camera.width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.camera.height)
                self.cap.set(cv2.CAP_PROP_FPS, self.config.camera.fps)
                
                # Test read
                ret, frame = self.cap.read()
                if not ret:
                    raise RuntimeError("Failed to read test frame")
                
                self.logger.info(f"[OK] Camera initialized: {self.config.camera.id}")
                self.logger.info(f"  Resolution: {frame.shape[1]}x{frame.shape[0]}")
                self.logger.info(f"  Target FPS: {self.config.camera.fps}")
                
                return True
                
            except Exception as e:
                self.logger.error(f"Camera init attempt {attempt+1}/{max_retries} failed: {e}")
                if self.cap:
                    self.cap.release()
                    self.cap = None
                
                if attempt < max_retries - 1:
                    time.sleep(2)
        
        return False
    
    def _process_frame(self, frame: np.ndarray) -> Dict:
        """Process a single frame"""
        result = {
            'motion': None,
            'detections': [],
            'tracks': {},
            'inference_time': 0,
        }
        
        # Step 1: Motion detection (always runs)
        motion_result = self.motion_detector.detect_motion(frame)
        result['motion'] = motion_result
        
        if motion_result.has_motion:
            self.stats['frames_with_motion'] += 1
        
        if motion_result.is_valid:
            self.stats['frames_with_valid_motion'] += 1
        
        # Step 2: Object detection (conditional)
        run_yolo = False
        
        if self.config.motion.enabled:
            # Motion-triggered mode
            run_yolo = motion_result.has_motion and motion_result.is_valid
        else:
            # Always run mode
            run_yolo = True
        
        if run_yolo:
            try:
                t1 = time.time()
                detections = self.detector.detect(frame)
                inference_time = time.time() - t1
                
                result['detections'] = detections
                result['inference_time'] = inference_time
                
                self.stats['yolo_calls'] += 1
                self.stats['detections_total'] += len(detections)
                self.inference_times.append(inference_time)
                
                # Keep only last 100 inference times
                if len(self.inference_times) > 100:
                    self.inference_times = self.inference_times[-100:]
                
            except Exception as e:
                self.logger.error(f"Detection failed: {e}")
                self.stats['errors'] += 1
        
        # Step 3: Assign zones
        if result['detections']:
            self.zone_manager.assign_zones(result['detections'])
        
        # Step 4: Update tracker
        try:
            tracks = self.tracker.update(result['detections'])
            result['tracks'] = tracks
            
            # Update zone occupancy
            self.zone_manager.update_occupancy(tracks)
            
        except Exception as e:
            self.logger.error(f"Tracking failed: {e}")
            self.stats['errors'] += 1
        
        return result
    
    def _handle_alerts(self, tracks: Dict, frame: np.ndarray):
        """Process alerts and save recordings"""
        for track_id, track in tracks.items():
            # Get zone object
            zone = None
            if track.zones and len(track.zones) > 0:
                zone_name = track.zones[-1]
                zone = self.zone_manager.get_zone_by_name(zone_name)
            
            # Assess threat
            threat_score, breakdown = self.threat_assessor.assess_threat(track, zone)
            level, desc, color = self.threat_assessor.get_alert_level(threat_score)
            
            # Check if we should alert
            if self.threat_assessor.should_alert(track_id, threat_score, level):
                self.stats['alerts_triggered'] += 1
                
                # Log structured event
                self.logger.log_event('alert', {
                    'track_id': track_id,
                    'class': track.class_name,
                    'level': level,
                    'score': round(threat_score, 2),
                    'duration': round(track.duration, 2),
                    'zone': zone.name if zone else None,
                    'breakdown': {k: round(v, 2) if isinstance(v, float) else v 
                                 for k, v in breakdown.items()},
                })
                
                # Console alert
                self.logger.warning(
                    f"{desc} - Track #{track_id} ({track.class_name}) - "
                    f"Score: {threat_score:.0f} - Duration: {track.duration:.1f}s - "
                    f"Zone: {zone.name if zone else 'None'}"
                )
                
                # Save frame if configured
                if self.config.recording.save_on_alert:
                    threshold_level = ThreatLevel.THRESHOLDS.get(
                        self.config.recording.alert_level_threshold, 50
                    )
                    
                    if threat_score >= threshold_level:
                        self._save_alert_frame(frame, track_id, level, threat_score)
    
    def _save_alert_frame(self, frame: np.ndarray, track_id: int, 
                         level: str, score: float):
        """Save alert frame to disk"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = self.recording_dir / f"alert_{level}_{track_id}_{timestamp}_score{int(score)}.jpg"
            
            cv2.imwrite(str(filename), frame)
            self.logger.info(f"[SNAP] Saved alert: {filename.name}")
            
        except Exception as e:
            self.logger.error(f"Failed to save alert frame: {e}")
    
    def _draw_ui(self, frame: np.ndarray, result: Dict) -> np.ndarray:
        """Draw complete UI overlay"""
        # Draw zones
        self.zone_manager.draw_zones(frame)
        
        # Draw motion boxes (debug)
        if result['motion'] and result['motion'].has_motion:
            for box in result['motion'].motion_boxes[:5]:  # Max 5 boxes
                x1, y1, x2, y2 = box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (128, 128, 128), 1)
        
        # Draw tracks
        for track_id, track in result['tracks'].items():
            x1, y1, x2, y2 = track.bbox
            
            # Get zone
            zone = None
            if track.zones and len(track.zones) > 0:
                zone_name = track.zones[-1]
                zone = self.zone_manager.get_zone_by_name(zone_name)
            
            # Calculate threat
            threat_score, _ = self.threat_assessor.assess_threat(track, zone)
            level, desc, color = self.threat_assessor.get_alert_level(threat_score)
            
            # Draw bounding box
            thickness = 3 if level in [ThreatLevel.HIGH, ThreatLevel.MEDIUM] else 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            
            # Draw track trail with fading effect
            if len(track.positions) > 1:
                points = list(track.positions)[-30:]  # Last 30 points
                
                # Draw fading trail
                num_points = len(points)
                for i in range(1, num_points):
                    # Calculate alpha (fade from 30% to 100%)
                    alpha = 0.3 + (0.7 * i / num_points)
                    line_thickness = max(1, int(3 * alpha))
                    
                    # Get points as integers
                    pt1 = (int(points[i-1][0]), int(points[i-1][1]))
                    pt2 = (int(points[i][0]), int(points[i][1]))
                    
                    # Fade color
                    faded_color = tuple(int(c * alpha) for c in color)
                    
                    try:
                        cv2.line(frame, pt1, pt2, faded_color, line_thickness)
                    except cv2.error:
                        # Skip this line if any issue
                        pass
            
            # Draw label
            labels = [
                f"#{track_id} {track.class_name} ({track.state})",
                f"{level}: {threat_score:.0f}%",
                f"[TIME] {track.duration:.1f}s",
            ]
            
            if zone:
                labels.append(f"[LOC] {zone.name}")
            
            y_offset = y1 - 10
            for label in reversed(labels):
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                
                # Background
                cv2.rectangle(frame, (x1, y_offset - th - 5), 
                            (x1 + tw + 5, y_offset), color, -1)
                
                # Text
                cv2.putText(frame, label, (x1 + 2, y_offset - 3),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                y_offset -= (th + 8)
        
        # Draw stats panel
        self._draw_stats_panel(frame, result)
        
        # Draw controls
        self._draw_controls(frame)
        
        return frame
    
    def _draw_stats_panel(self, frame: np.ndarray, result: Dict):
        """Draw statistics panel"""
        h, w = frame.shape[:2]
        
        # Background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 280), (0, 0, 0), -1)
        frame[:] = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        # Calculate efficiency
        yolo_efficiency = (self.stats['yolo_calls'] / max(1, self.stats['frames_processed'])) * 100
        
        avg_inference = np.mean(self.inference_times) if self.inference_times else 0
        
        # Stats text
        stats = [
            f"FPS: {self.fps:.1f}",
            f"Motion: {'YES' if result['motion'] and result['motion'].has_motion else 'NO'} "
            f"(Valid: {'YES' if result['motion'] and result['motion'].is_valid else 'NO'})",
            f"Active Tracks: {len(result['tracks'])}",
            f"Inference: {result['inference_time']*1000:.1f}ms (avg: {avg_inference*1000:.1f}ms)",
            "",
            "=== Totals ===",
            f"Frames: {self.stats['frames_processed']}",
            f"YOLO Calls: {self.stats['yolo_calls']} ({yolo_efficiency:.1f}%)",
            f"Detections: {self.stats['detections_total']}",
            f"Alerts: {self.stats['alerts_triggered']}",
            f"Errors: {self.stats['errors']}",
            "",
            "=== Tracker ===",
        ]
        
        tracker_stats = self.tracker.get_stats()
        stats.extend([
            f"Total: {tracker_stats['total_tracks']}",
            f"Confirmed: {tracker_stats['confirmed_tracks']}",
            f"Tentative: {tracker_stats['tentative_tracks']}",
        ])
        
        y_offset = 30
        for text in stats:
            if "===" in text:
                color = (0, 255, 255)  # Yellow for headers
                font_scale = 0.5
            else:
                color = (255, 255, 255)
                font_scale = 0.45
            
            cv2.putText(frame, text, (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1)
            y_offset += 18
    
    def _draw_controls(self, frame: np.ndarray):
        """Draw control legend"""
        h, w = frame.shape[:2]
        
        # Background
        overlay = frame.copy()
        cv2.rectangle(overlay, (w - 280, 10), (w - 10, 180), (0, 0, 0), -1)
        frame[:] = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        controls = [
            "=== Controls ===",
            "Q - Quit",
            "S - Save frame",
            "M - Toggle motion mode",
            "R - Reset stats",
            "P - Save profile",
            "D - Toggle debug",
            "SPACE - Pause",
        ]
        
        y_offset = 30
        for text in controls:
            if "===" in text:
                color = (0, 255, 255)
            else:
                color = (255, 255, 255)
            
            cv2.putText(frame, text, (w - 270, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
            y_offset += 20
    
    def run(self):
        """Main system loop"""
        # Initialize camera
        if not self._init_camera():
            self.logger.error("Failed to initialize camera. Exiting.")
            return
        
        self.logger.info("\n" + "="*60)
        self.logger.info("System Running - Press Q to quit")
        self.logger.info("="*60 + "\n")
        
        fps_start_time = time.time()
        fps_frame_count = 0
        paused = False
        
        try:
            while True:
                if not paused:
                    # Read frame
                    ret, frame = self.cap.read()
                    
                    if not ret:
                        self.logger.error("Failed to read frame")
                        # Try to reconnect
                        if self._init_camera():
                            continue
                        else:
                            break
                    
                    self.stats['frames_processed'] += 1
                    
                    # Frame skipping
                    if self.config.performance.skip_frames > 0:
                        self.frame_skip_counter += 1
                        if self.frame_skip_counter % (self.config.performance.skip_frames + 1) != 0:
                            continue
                    
                    # Process frame
                    result = self._process_frame(frame)
                    
                    # Handle alerts
                    self._handle_alerts(result['tracks'], frame)
                    
                    # Draw UI
                    frame = self._draw_ui(frame, result)
                    
                    # Calculate FPS
                    fps_frame_count += 1
                    if time.time() - fps_start_time >= 1.0:
                        self.fps = fps_frame_count / (time.time() - fps_start_time)
                        fps_frame_count = 0
                        fps_start_time = time.time()
                        
                        # Periodic cleanup
                        self.threat_assessor.cleanup_old_alerts()
                
                # Display
                cv2.imshow('Enhanced Security System v3.0', frame)
                
                # Handle keyboard
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = self.recording_dir / f"manual_{timestamp}.jpg"
                    cv2.imwrite(str(filename), frame)
                    self.logger.info(f"[SNAP] Saved: {filename.name}")
                elif key == ord('m'):
                    self.config.motion.enabled = not self.config.motion.enabled
                    self.logger.info(f"Motion mode: {'ON' if self.config.motion.enabled else 'OFF'}")
                elif key == ord('r'):
                    self.stats = {k: 0 for k in self.stats.keys()}
                    self.logger.info("[STATS] Stats reset")
                elif key == ord('p'):
                    self.threat_assessor.save_profile()
                    self.logger.info("[SAVE] Activity profile saved")
                elif key == ord(' '):
                    paused = not paused
                    self.logger.info(f"{'[PAUSE] Paused' if paused else '[PLAY] Resumed'}")
                
        except KeyboardInterrupt:
            self.logger.info("\n[WARN] Interrupted by user")
        except Exception as e:
            self.logger.error(f"Fatal error: {e}", exc_info=True)
        finally:
            self._cleanup()
    
    def _cleanup(self):
        """Cleanup resources"""
        self.logger.info("\n" + "="*60)
        self.logger.info("Shutting down...")
        
        # Save activity profile
        self.threat_assessor.save_profile()
        
        # Release camera
        if self.cap:
            self.cap.release()
        
        cv2.destroyAllWindows()
        
        # Print final stats
        self.logger.info("\n=== Final Statistics ===")
        for key, value in self.stats.items():
            self.logger.info(f"{key}: {value}")
        
        tracker_stats = self.tracker.get_stats()
        self.logger.info(f"\nTracker: {tracker_stats}")
        
        threat_stats = self.threat_assessor.get_stats()
        self.logger.info(f"Threats: {threat_stats}")
        
        zone_stats = self.zone_manager.get_stats()
        self.logger.info(f"Zones: {zone_stats}")
        
        self.logger.info("\n[OK] System stopped cleanly")
        self.logger.info("="*60)


if __name__ == "__main__":
    system = ProductionSecuritySystem("config.yaml")
    system.run()