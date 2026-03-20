"""Data-driven threat assessment with anomaly detection"""
import numpy as np
from datetime import datetime, time as dt_time
from typing import Optional, Tuple
from collections import defaultdict
import json
from pathlib import Path


class ThreatLevel:
    """Threat level constants"""
    INFO = "INFO"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    
    COLORS = {
        INFO: (0, 255, 0),      # Green
        LOW: (0, 255, 255),     # Yellow
        MEDIUM: (0, 165, 255),  # Orange
        HIGH: (0, 0, 255),      # Red
    }
    
    THRESHOLDS = {
        INFO: 0,
        LOW: 30,
        MEDIUM: 50,
        HIGH: 70,
    }


class ActivityProfile:
    """Learn normal activity patterns for anomaly detection"""
    
    def __init__(self, profile_file: str = "activity_profile.json"):
        self.profile_file = Path(profile_file)
        self.hourly_activity = defaultdict(list)  # hour -> list of activity counts
        self.zone_activity = defaultdict(lambda: defaultdict(list))  # zone -> hour -> counts
        
        self._load_profile()
    
    def _load_profile(self):
        """Load existing activity profile"""
        if self.profile_file.exists():
            try:
                with open(self.profile_file, 'r') as f:
                    data = json.load(f)
                    self.hourly_activity = defaultdict(list, {
                        int(k): v for k, v in data.get('hourly', {}).items()
                    })
                    self.zone_activity = defaultdict(
                        lambda: defaultdict(list),
                        {k: defaultdict(list, v) for k, v in data.get('zones', {}).items()}
                    )
            except Exception as e:
                print(f"⚠️  Failed to load activity profile: {e}")
    
    def save_profile(self):
        """Save activity profile to disk"""
        try:
            data = {
                'hourly': {str(k): v for k, v in self.hourly_activity.items()},
                'zones': {k: dict(v) for k, v in self.zone_activity.items()},
            }
            with open(self.profile_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"⚠️  Failed to save activity profile: {e}")
    
    def record_activity(self, hour: int, zone: Optional[str] = None):
        """Record activity at given hour"""
        self.hourly_activity[hour].append(1)
        
        if zone:
            self.zone_activity[zone][hour].append(1)
        
        # Keep only last 1000 entries per hour
        if len(self.hourly_activity[hour]) > 1000:
            self.hourly_activity[hour] = self.hourly_activity[hour][-1000:]
    
    def get_anomaly_score(self, hour: int, zone: Optional[str] = None) -> float:
        """
        Calculate anomaly score for current hour
        
        Returns: 0.0 (normal) to 1.0 (highly unusual)
        """
        # Check hourly activity
        if hour not in self.hourly_activity:
            return 0.5  # Unknown hour, medium anomaly
        
        activity_counts = self.hourly_activity[hour]
        if not activity_counts:
            return 0.5
        
        # Calculate statistics
        mean_activity = np.mean(activity_counts)
        std_activity = np.std(activity_counts)
        
        # Current activity (1 detection)
        if std_activity == 0:
            return 0.0  # No variance, this is normal
        
        z_score = abs((1 - mean_activity) / std_activity)
        
        # Convert z-score to 0-1 range (z > 3 is very unusual)
        anomaly_score = min(z_score / 3.0, 1.0)
        
        # Check zone-specific activity if available
        if zone and zone in self.zone_activity:
            zone_counts = self.zone_activity[zone].get(hour, [])
            if zone_counts:
                zone_mean = np.mean(zone_counts)
                zone_std = np.std(zone_counts)
                
                if zone_std > 0:
                    zone_z = abs((1 - zone_mean) / zone_std)
                    zone_anomaly = min(zone_z / 3.0, 1.0)
                    
                    # Average with hourly anomaly
                    anomaly_score = (anomaly_score + zone_anomaly) / 2
        
        return anomaly_score


class ImprovedThreatAssessor:
    """
    Enhanced threat assessment with:
    - Learning-based anomaly detection
    - Movement pattern analysis
    - Historical context
    """
    
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        
        self.activity_profile = ActivityProfile()
        
        # Track recent alerts to prevent spam
        self.recent_alerts = {}  # track_id -> last_alert_time
        self.alert_cooldown = 5.0  # seconds
        
        # Alert history for analysis
        self.alert_history = []
    
    def assess_threat(self, track, zone=None) -> Tuple[float, dict]:
        """
        Calculate threat score with detailed breakdown
        
        Returns: (score, breakdown_dict)
        """
        breakdown = {}
        score = 0.0
        
        # 1. Base class score
        base_score = self.config.threat.class_base_scores.get(track.class_name, 5)
        score += base_score
        breakdown['base_class'] = base_score
        
        # 2. Time-based multiplier
        time_mult = self._get_time_multiplier()
        breakdown['time_multiplier'] = time_mult
        
        # 3. Duration-based scoring
        duration_score = 0
        if track.duration > self.config.threat.suspicious_duration_seconds:
            duration_score = 35
        elif track.duration > self.config.threat.loiter_threshold_seconds:
            duration_score = 20
        
        # Additional bonus for stationary objects
        if track.is_stationary and track.duration > 10:
            duration_score += 15
        
        score += duration_score
        breakdown['duration'] = duration_score
        
        # 4. Zone-based multiplier
        zone_mult = 1.0
        if zone:
            zone_mult = zone.sensitivity
        breakdown['zone_multiplier'] = zone_mult
        
        # 5. Anomaly detection
        current_hour = datetime.now().hour
        zone_name = zone.name if zone else None
        anomaly_score = self.activity_profile.get_anomaly_score(current_hour, zone_name)
        anomaly_bonus = anomaly_score * 25  # Up to +25 points for unusual activity
        
        score += anomaly_bonus
        breakdown['anomaly'] = anomaly_bonus
        
        # 6. Movement pattern analysis
        movement_score = self._analyze_movement(track)
        score += movement_score
        breakdown['movement'] = movement_score
        
        # Apply multipliers
        score *= time_mult
        score *= zone_mult
        
        # 7. Confidence adjustment
        confidence_factor = track.avg_confidence
        score *= confidence_factor
        breakdown['confidence_factor'] = confidence_factor
        
        # 8. Track state adjustment
        if track.state == "tentative":
            score *= 0.5  # Reduce score for unconfirmed tracks
            breakdown['tentative_penalty'] = 0.5
        
        # Cap at 100
        final_score = min(100, score)
        breakdown['final'] = final_score
        
        # Record activity for learning
        self.activity_profile.record_activity(current_hour, zone_name)
        
        return final_score, breakdown
    
    def _get_time_multiplier(self) -> float:
        """Get time-based threat multiplier"""
        current_hour = datetime.now().hour
        
        config = self.config.threat.time_multipliers
        
        if 22 <= current_hour or current_hour < 6:
            return config.get('night', 1.4)
        elif 6 <= current_hour < 9:
            return config.get('early_morning', 0.9)
        elif 9 <= current_hour < 18:
            return config.get('day', 0.7)
        else:  # 18-22
            return config.get('evening', 1.0)
    
    def _analyze_movement(self, track) -> float:
        """
        Analyze movement patterns for suspicious behavior
        
        Returns: bonus score (0-20)
        """
        score = 0.0
        
        # Check if object is pacing (back and forth)
        if len(track.positions) >= 10:
            positions = list(track.positions)
            
            # Calculate direction changes
            direction_changes = 0
            for i in range(2, len(positions)):
                p0, p1, p2 = positions[i-2], positions[i-1], positions[i]
                
                v1 = np.array(p1) - np.array(p0)
                v2 = np.array(p2) - np.array(p1)
                
                # Dot product to check if direction reversed
                if np.dot(v1, v2) < 0:
                    direction_changes += 1
            
            # Many direction changes = pacing
            pacing_ratio = direction_changes / len(positions)
            if pacing_ratio > 0.3:  # >30% of movements are reversals
                score += 15
                self.logger.debug(f"Track {track.track_id}: Pacing detected ({pacing_ratio:.1%})")
        
        # Check for circular movement (casing the area)
        if len(track.positions) >= 20:
            positions = np.array(list(track.positions))
            center = np.mean(positions, axis=0)
            distances = np.linalg.norm(positions - center, axis=1)
            
            # If distances are consistent, might be circular
            if np.std(distances) < 10 and np.mean(distances) > 20:
                score += 10
                self.logger.debug(f"Track {track.track_id}: Circular movement detected")
        
        return score
    
    def get_alert_level(self, threat_score: float) -> Tuple[str, str, Tuple[int, int, int]]:
        """
        Convert threat score to alert level
        
        Returns: (level, description, color_bgr)
        """
        if threat_score >= ThreatLevel.THRESHOLDS[ThreatLevel.HIGH]:
            level = ThreatLevel.HIGH
            desc = "🚨 High Threat"
        elif threat_score >= ThreatLevel.THRESHOLDS[ThreatLevel.MEDIUM]:
            level = ThreatLevel.MEDIUM
            desc = "⚠️ Medium Threat"
        elif threat_score >= ThreatLevel.THRESHOLDS[ThreatLevel.LOW]:
            level = ThreatLevel.LOW
            desc = "ℹ️ Low Alert"
        else:
            level = ThreatLevel.INFO
            desc = "👁️ Monitoring"
        
        color = ThreatLevel.COLORS[level]
        
        return level, desc, color
    
    def should_alert(self, track_id: int, threat_score: float, level: str) -> bool:
        """
        Determine if we should trigger an alert
        
        Prevents alert spam with cooldown and level thresholds
        """
        # Only alert for MEDIUM or HIGH
        if level not in [ThreatLevel.MEDIUM, ThreatLevel.HIGH]:
            return False
        
        # Check cooldown
        current_time = datetime.now().timestamp()
        if track_id in self.recent_alerts:
            last_alert = self.recent_alerts[track_id]
            if current_time - last_alert < self.alert_cooldown:
                return False
        
        # Update last alert time
        self.recent_alerts[track_id] = current_time
        
        # Record in history
        self.alert_history.append({
            'timestamp': datetime.now().isoformat(),
            'track_id': track_id,
            'level': level,
            'score': threat_score,
        })
        
        # Keep only last 1000 alerts
        if len(self.alert_history) > 1000:
            self.alert_history = self.alert_history[-1000:]
        
        return True
    
    def cleanup_old_alerts(self):
        """Clean up old alert records"""
        current_time = datetime.now().timestamp()
        cutoff = current_time - 3600  # 1 hour ago
        
        self.recent_alerts = {
            tid: t for tid, t in self.recent_alerts.items()
            if t > cutoff
        }
    
    def save_profile(self):
        """Save learned activity profile"""
        self.activity_profile.save_profile()
    
    def get_stats(self) -> dict:
        """Get threat assessment statistics"""
        return {
            'total_alerts': len(self.alert_history),
            'high_alerts': sum(1 for a in self.alert_history if a['level'] == ThreatLevel.HIGH),
            'medium_alerts': sum(1 for a in self.alert_history if a['level'] == ThreatLevel.MEDIUM),
            'active_cooldowns': len(self.recent_alerts),
        }