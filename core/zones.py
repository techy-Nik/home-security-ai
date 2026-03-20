"""Intelligent activity zone management"""
import cv2
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class Zone:
    """Represents an activity zone with rules"""
    name: str
    polygon: np.ndarray
    alert_classes: set
    sensitivity: float
    priority: int
    
    def contains_point(self, point: Tuple[int, int]) -> bool:
        """Check if point is inside zone"""
        result = cv2.pointPolygonTest(self.polygon, point, False)
        return result >= 0
    
    def draw(self, frame: np.ndarray, active: bool = False):
        """Draw zone on frame"""
        color = (0, 255, 255) if active else (255, 255, 0)  # Yellow/Cyan
        thickness = 3 if active else 2
        
        cv2.polylines(frame, [self.polygon], True, color, thickness)
        
        # Draw label
        x, y = self.polygon[0]
        label = f"{self.name} (P{self.priority})"
        
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x, y - th - 10), (x + tw + 5, y), color, -1)
        cv2.putText(frame, label, (x + 2, y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)


class ZoneManager:
    """Manages multiple zones with priority and overlap handling"""
    
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.zones: List[Zone] = []
        
        # Initialize zone occupancy BEFORE loading zones
        self.zone_occupancy = {}  # zone_name -> set of track_ids
        
        # Load zones from config
        self._load_zones()
    
    def _load_zones(self):
        """Load zones from configuration"""
        for zone_config in self.config.zones:
            polygon = np.array(zone_config.polygon, dtype=np.int32)
            alert_classes = set(zone_config.alert_classes) if zone_config.alert_classes else set()
            
            zone = Zone(
                name=zone_config.name,
                polygon=polygon,
                alert_classes=alert_classes,
                sensitivity=zone_config.sensitivity,
                priority=zone_config.priority
            )
            
            self.zones.append(zone)
            self.zone_occupancy[zone.name] = set()  # Now this works!
            
            self.logger.info(f"Loaded zone: {zone.name} (priority {zone.priority})")
    
    def assign_zones(self, detections: List[dict]):
        """
        Assign zones to detections
        
        Handles overlapping zones by priority
        """
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            
            # Find all zones containing this point
            containing_zones = []
            for zone in self.zones:
                if zone.contains_point(center):
                    containing_zones.append(zone)
            
            # Assign highest priority zone
            if containing_zones:
                best_zone = min(containing_zones, key=lambda z: z.priority)
                detection['zone'] = best_zone.name
                detection['zone_obj'] = best_zone
            else:
                detection['zone'] = None
                detection['zone_obj'] = None
    
    def update_occupancy(self, tracks: dict):
        """Update zone occupancy tracking"""
        # Reset occupancy
        for zone_name in self.zone_occupancy:
            self.zone_occupancy[zone_name] = set()
        
        # Count tracks in each zone
        for track_id, track in tracks.items():
            if track.zones and len(track.zones) > 0:
                current_zone = track.zones[-1]  # Most recent zone
                if current_zone and current_zone in self.zone_occupancy:
                    self.zone_occupancy[current_zone].add(track_id)
    
    def get_zone_by_name(self, name: str) -> Optional[Zone]:
        """Get zone by name"""
        for zone in self.zones:
            if zone.name == name:
                return zone
        return None
    
    def draw_zones(self, frame: np.ndarray):
        """Draw all zones on frame"""
        for zone in self.zones:
            # Check if zone is active (has objects in it)
            active = len(self.zone_occupancy.get(zone.name, set())) > 0
            zone.draw(frame, active)
    
    def get_stats(self) -> dict:
        """Get zone statistics"""
        stats = {}
        for zone_name, track_ids in self.zone_occupancy.items():
            stats[zone_name] = {
                'occupancy': len(track_ids),
                'track_ids': list(track_ids)
            }
        return stats