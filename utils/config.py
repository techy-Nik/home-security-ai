"""Configuration management with validation"""
import yaml
from pathlib import Path
from typing import Any, Dict
from dataclasses import dataclass, field
from enum import Enum


class AlertLevel(Enum):
    INFO = 0
    LOW = 30
    MEDIUM = 50
    HIGH = 70


@dataclass
class CameraConfig:
    id: int = 0
    width: int = 1280
    height: int = 720
    fps: int = 30


@dataclass
class DetectionConfig:
    model_path: str = "models/yolov8n.onnx"
    confidence_threshold: float = 0.5
    nms_threshold: float = 0.45
    min_confidence_for_tracking: float = 0.6


@dataclass
class MotionConfig:
    enabled: bool = True
    var_threshold: int = 25
    min_contour_area: int = 800
    learning_rate: float = 0.001
    motion_percentage_threshold: float = 0.4


@dataclass
class TrackingConfig:
    iou_threshold: float = 0.3
    max_age_frames: int = 30
    min_hits: int = 3
    max_tracks: int = 50
    use_kalman: bool = True


@dataclass
class ThreatConfig:
    loiter_threshold_seconds: float = 15.0
    suspicious_duration_seconds: float = 45.0
    confidence_weight: float = 0.3
    time_multipliers: Dict[str, float] = field(default_factory=dict)
    class_base_scores: Dict[str, int] = field(default_factory=dict)


@dataclass
class ZoneConfig:
    name: str
    polygon: list
    alert_classes: list = field(default_factory=list)
    sensitivity: float = 1.0
    priority: int = 99


@dataclass
class RecordingConfig:
    enabled: bool = True
    output_dir: str = "recordings"
    save_on_alert: bool = True
    alert_level_threshold: str = "MEDIUM"
    clip_duration_seconds: int = 10


@dataclass
class LoggingConfig:
    level: str = "INFO"
    file: str = "security.log"
    max_bytes: int = 10485760
    backup_count: int = 5


@dataclass
class PerformanceConfig:
    skip_frames: int = 0
    max_fps: int = 30
    enable_profiling: bool = False


@dataclass
class Config:
    camera: CameraConfig
    detection: DetectionConfig
    motion: MotionConfig
    tracking: TrackingConfig
    threat: ThreatConfig
    zones: list
    recording: RecordingConfig
    logging: LoggingConfig
    performance: PerformanceConfig


class ConfigLoader:
    """Load and validate configuration"""
    
    @staticmethod
    def load(config_path: str = "config.yaml") -> Config:
        """Load configuration from YAML file"""
        config_file = Path(config_path)
        
        if not config_file.exists():
            print(f"⚠️  Config file not found: {config_path}")
            print("Creating default configuration...")
            ConfigLoader._create_default_config(config_file)
        
        with open(config_file, 'r') as f:
            data = yaml.safe_load(f)
        
        return Config(
            camera=CameraConfig(**data.get('camera', {})),
            detection=DetectionConfig(**data.get('detection', {})),
            motion=MotionConfig(**data.get('motion', {})),
            tracking=TrackingConfig(**data.get('tracking', {})),
            threat=ThreatConfig(**data.get('threat', {})),
            zones=[ZoneConfig(**z) for z in data.get('zones', [])],
            recording=RecordingConfig(**data.get('recording', {})),
            logging=LoggingConfig(**data.get('logging', {})),
            performance=PerformanceConfig(**data.get('performance', {})),
        )
    
    @staticmethod
    def _create_default_config(config_file: Path):
        """Create default configuration file"""
        default_config = {
            'camera': {'id': 0, 'width': 1280, 'height': 720, 'fps': 30},
            'detection': {
                'model_path': 'models/yolov8n.onnx',
                'confidence_threshold': 0.5,
                'nms_threshold': 0.45,
                'min_confidence_for_tracking': 0.6,
            },
            'motion': {
                'enabled': True,
                'var_threshold': 25,
                'min_contour_area': 800,
                'learning_rate': 0.001,
                'motion_percentage_threshold': 0.4,
            },
            'tracking': {
                'iou_threshold': 0.3,
                'max_age_frames': 30,
                'min_hits': 3,
                'max_tracks': 50,
                'use_kalman': True,
            },
            'threat': {
                'loiter_threshold_seconds': 15.0,
                'suspicious_duration_seconds': 45.0,
                'confidence_weight': 0.3,
                'time_multipliers': {
                    'night': 1.4,
                    'early_morning': 0.9,
                    'day': 0.7,
                    'evening': 1.0,
                },
                'class_base_scores': {
                    'person': 35,
                    'car': 22,
                    'truck': 22,
                    'motorcycle': 28,
                    'backpack': 18,
                    'handbag': 12,
                    'dog': 8,
                },
            },
            'zones': [],
            'recording': {
                'enabled': True,
                'output_dir': 'recordings',
                'save_on_alert': True,
                'alert_level_threshold': 'MEDIUM',
                'clip_duration_seconds': 10,
            },
            'logging': {
                'level': 'INFO',
                'file': 'security.log',
                'max_bytes': 10485760,
                'backup_count': 5,
            },
            'performance': {
                'skip_frames': 0,
                'max_fps': 30,
                'enable_profiling': False,
            },
        }
        
        with open(config_file, 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False, sort_keys=False)
        
        print(f"✓ Created default config: {config_file}")