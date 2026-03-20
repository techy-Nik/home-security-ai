# 🏠 Enhanced Security System v3.0

AI-powered home security camera system with intelligent threat detection and real-time monitoring.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green.svg)
![YOLOv8](https://img.shields.io/badge/YOLOv8-ONNX-orange.svg)

## 🎯 Features

- **🤖 YOLOv8 Object Detection** - Detect 80 object classes in real-time using ONNX Runtime
- **⚡ Smart Motion Detection** - 10x performance boost with intelligent false positive filtering
- **🎯 Multi-Object Tracking** - Persistent IDs across frames with Kalman filtering
- **📍 Activity Zones** - Context-aware monitoring with customizable security zones
- **🧠 Learning Threat Assessment** - Adapts to your environment and detects anomalies
- **📊 Structured Logging** - Machine-readable event logs for analysis
- **⚙️ Configuration Management** - Full control via YAML config file

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Webcam or IP camera
- Windows/Linux/Mac

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/tech-Nik/home-security-ai.git
cd home-security-ai
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Run the system:**
```bash
python main.py
```

The system will:
- Auto-download YOLOv8n model (~6MB)
- Create default `config.yaml`
- Initialize your camera
- Start monitoring

## 🎮 Controls

| Key | Action |
|-----|--------|
| `Q` | Quit system |
| `S` | Save current frame |
| `M` | Toggle motion detection mode |
| `R` | Reset statistics |
| `P` | Save activity profile |
| `SPACE` | Pause/Resume |

## ⚙️ Configuration

Customize behavior by editing `config.yaml`:
```yaml
camera:
  id: 0              # Camera ID (0 = default webcam)
  width: 1280
  height: 720
  fps: 30

detection:
  confidence_threshold: 0.5    # Higher = fewer detections

motion:
  enabled: true                # Motion-triggered YOLO (10x speedup)
  var_threshold: 25            # Lower = more sensitive

zones:
  - name: "Front Door"
    polygon: [[200, 300], [600, 300], [600, 650], [200, 650]]
    alert_classes: ["person", "backpack"]
    sensitivity: 1.3           # 30% higher threat scores
    priority: 1                # Lower = higher priority
```

## 🎨 Activity Zones

Define custom security zones with different rules:
```yaml
zones:
  - name: "Restricted Area"
    polygon: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
    alert_classes: ["person"]
    sensitivity: 1.5
    priority: 1
```

**Zone Features:**
- Point-in-polygon detection
- Priority-based overlap handling
- Custom sensitivity multipliers
- Per-zone alert class filtering

## 📊 Threat Scoring System

Threat scores (0-100) calculated from multiple factors:

| Factor | Impact |
|--------|--------|
| **Object Class** | person=35, car=22, dog=8 |
| **Time of Day** | night=1.4x, day=0.7x |
| **Duration** | loitering >30s = +35 points |
| **Zone Sensitivity** | high-security zones = 1.3x |
| **Anomaly Detection** | unusual activity = +25 points |
| **Movement Patterns** | pacing/circling = +15 points |

**Alert Levels:**
- 🟢 **0-29 (INFO)** - Normal monitoring
- 🟡 **30-49 (LOW)** - Minor alert
- 🟠 **50-69 (MEDIUM)** - Concerning activity
- 🔴 **70+ (HIGH)** - Immediate threat

## 📁 Output Files

| File | Description |
|------|-------------|
| `recordings/alert_*.jpg` | Alert snapshots with threat scores |
| `security.log` | Human-readable system log |
| `events.jsonl` | Machine-readable events (JSON format) |
| `activity_profile.json` | Learned activity patterns |

## 📈 Performance

**Motion Detection Optimization:**
- Static scene: ~0.3% YOLO usage (3 calls / 1000 frames)
- Active scene: ~10% YOLO usage
- **10x speedup** vs always-on detection

**Hardware Requirements:**
- CPU: Any modern processor (2015+)
- RAM: 2GB minimum
- GPU: Optional (CUDA for 5-10x speedup)

## 🧪 Data Analysis

Analyze events with Python:
```python
import pandas as pd
import json

# Load events
events = []
with open('events.jsonl', 'r') as f:
    for line in f:
        events.append(json.loads(line))

df = pd.DataFrame(events)

# Analysis
high_alerts = df[df['level'] == 'HIGH']
hourly_activity = df.groupby(df['timestamp'].str[:13]).size()
zone_stats = df.groupby('zone')['track_id'].count()
```

## 🏗️ Project Structure
```
home-security-ai/
├── main.py                  # Main entry point
├── config.yaml              # Configuration file
├── requirements.txt         # Dependencies
├── core/
│   ├── detector.py         # YOLOv8 object detection
│   ├── motion.py           # Smart motion detection
│   ├── tracker.py          # Multi-object tracking
│   ├── zones.py            # Activity zone management
│   └── threat.py           # Threat assessment engine
├── utils/
│   ├── config.py           # Configuration loader
│   ├── logger.py           # Structured logging
│   └── video.py            # Video recording utilities
└── tests/
    ├── test_tracker.py     # Tracking tests
    ├── test_zones.py       # Zone tests
    └── test_threat.py      # Threat assessment tests
```

## 🔧 Advanced Usage

### Frame Skipping (Performance Boost)
```yaml
performance:
  skip_frames: 2   # Process every 3rd frame
```

### Disable Kalman Filter
```yaml
tracking:
  use_kalman: false
```

### Custom Threat Scores
```yaml
threat:
  class_base_scores:
    person: 35
    knife: 90
    package: 15
```

## 🐛 Troubleshooting

**Camera not found:**
```bash
# List cameras (Linux/Mac)
ls /dev/video*

# Windows: Check Device Manager
```

**Low FPS:**
- Enable motion detection (`motion.enabled: true`)
- Reduce resolution in config
- Increase `skip_frames`

**Too many false alerts:**
- Increase `confidence_threshold` (0.5 → 0.7)
- Adjust zone sensitivity
- Increase `loiter_threshold_seconds`

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **YOLOv8** - Ultralytics
- **ONNX Runtime** - Microsoft
- **OpenCV** - OpenCV Team

## 📧 Contact

For questions or issues, please open a GitHub issue.

---

**Built with ❤️ for home security**

⭐ Star this repo if you find it useful!