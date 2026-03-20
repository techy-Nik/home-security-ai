"""Core security system components"""

from .detector import SecurityDetector
from .motion import SmartMotionDetector, MotionResult
from .tracker import ImprovedTracker, Track
from .zones import ZoneManager, Zone
from .threat import ImprovedThreatAssessor, ThreatLevel, ActivityProfile

__all__ = [
    # Detection
    'SecurityDetector',
    
    # Motion detection
    'SmartMotionDetector',
    'MotionResult',
    
    # Tracking
    'ImprovedTracker',
    'Track',
    
    # Zones
    'ZoneManager',
    'Zone',
    
    # Threat assessment
    'ImprovedThreatAssessor',
    'ThreatLevel',
    'ActivityProfile',
]

__version__ = '3.0.0'