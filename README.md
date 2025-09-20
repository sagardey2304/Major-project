# Multi-Agent Navigation System for Visually Impaired

A sophisticated desktop prototype of a multi-agent navigation assistance system designed to help visually impaired users navigate safely through their environment.

## System Overview

This system uses a modular multi-agent architecture with three specialized agents:

- **Perception Agent**: Handles computer vision processing, object detection (YOLOv8), and depth estimation (MiDaS)
- **Navigation Agent**: Processes spatial information and provides directional guidance and route planning
- **Communication Agent**: Manages text-to-speech audio feedback and visual alerts

## Features

### Core Capabilities
- Real-time object detection using YOLOv8
- Depth estimation using MiDaS for distance calculation
- Intelligent path analysis with trapezoid region detection
- Audio guidance via text-to-speech
- Visual alerts and overlays
- Multi-agent coordination via message bus architecture

### Advanced Features
- Configurable risk assessment and thresholds
- Adaptive communication based on priority levels
- Visual enhancement using Transformer architecture
- Comprehensive logging and debugging support
- Graceful system shutdown and error handling

## Prerequisites

- Python 3.8 or higher
- Camera (webcam or USB camera)
- Audio output device for TTS
- Windows, macOS, or Linux operating system

### Hardware Requirements
- Minimum 8GB RAM (16GB recommended for optimal performance)
- GPU support recommended (NVIDIA with CUDA) for faster processing
- Camera with resolution support of at least 640x480

## Installation

### 1. Clone or Download the Project
```bash
git clone <repository-url>
cd "Major Project Agent"
```

### 2. Create Virtual Environment (Recommended)
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux  
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download Required Models
The system will automatically download required models on first run:
- YOLOv8s model for object detection
- MiDaS DPT_Hybrid model for depth estimation

## Configuration

The system uses a JSON-based configuration file (`config.json`) that is automatically created with default settings on first run.

### Key Configuration Options

#### Perception Agent
- `camera_index`: Camera device index (default: 1)
- `proximity_threshold`: Distance threshold for alerts in meters (default: 2.0)
- `confidence_threshold`: Minimum detection confidence (default: 0.5)
- `frame_width` / `frame_height`: Camera resolution (default: 640x480)

#### Navigation Agent
- `safe_distance`: Safe navigation distance in meters (default: 2.0)
- `warning_distance`: Warning distance threshold (default: 3.0)
- `update_frequency`: Navigation update frequency in seconds (default: 0.5)

#### Communication Agent
- `tts_enabled`: Enable/disable text-to-speech (default: true)
- `speech_rate`: TTS speech rate (default: 160)
- `delay_seconds`: Minimum delay between repeated announcements (default: 4)

### Customizing Configuration
```python
from config import config_manager

# Update perception settings
config_manager.update_perception_config(
    camera_index=0,
    proximity_threshold=1.5
)

# Update communication settings
config_manager.update_communication_config(
    speech_rate=180,
    tts_enabled=True
)
```

## Usage

### Running the System
```bash
python main.py
```

### Keyboard Controls
- `q` - Quit the application
- `s` - Save screenshot of current frame
- `c` - Clear all visual alerts
- `h` - Show help information

### System Startup
1. The system initializes all three agents
2. Camera connection is established
3. AI models are loaded (may take a few moments on first run)
4. Audio confirmation announces system activation
5. Real-time processing begins

### Understanding the Interface

#### Main Window
- Live camera feed with object detection overlays
- Bounding boxes (red = in path, green = safe)
- Navigation alerts and instructions
- System status information

#### Depth Map Window (optional)
- Color-coded depth visualization
- Warmer colors indicate closer objects
- Can be disabled in configuration

#### Audio Feedback
- Immediate alerts for obstacles in path
- Directional guidance ("turn left", "obstacle ahead")
- Priority-based announcement system
- Configurable speech rate and volume

## Agent Communication

The system uses a sophisticated message bus architecture:

### Message Types
- `OBSTACLE_ALERT`: Critical obstacle detection
- `NAVIGATION_UPDATE`: Directional guidance
- `USER_COMMUNICATION`: TTS messages
- `SYSTEM_STATUS`: Status updates between agents

### Priority Levels
1. **Low**: General information
2. **Medium**: Navigation suggestions  
3. **High**: Important warnings
4. **Critical**: Immediate safety alerts

## Troubleshooting

### Common Issues

#### Camera Not Detected
```bash
# Check available cameras
python -c "import cv2; print([i for i in range(5) if cv2.VideoCapture(i).isOpened()])"
```
Update `camera_index` in configuration if needed.

#### TTS Not Working
- Ensure audio output is connected and working
- On Linux, install espeak: `sudo apt-get install espeak`
- Check TTS engine initialization in console output

#### Model Download Issues
- Ensure stable internet connection
- Models are cached in `~/.cache/torch/hub/` and `~/.ultralytics/`
- Clear cache and restart if experiencing issues

#### Performance Issues
- Reduce camera resolution in configuration
- Disable depth map visualization
- Enable GPU acceleration if available
- Close other resource-intensive applications

### Error Messages

**"Cannot open camera"**
- Check camera index in configuration
- Ensure camera isn't used by another application
- Verify camera permissions

**"Model initialization failed"**
- Check internet connection for model download
- Verify PyTorch installation
- Clear model cache and retry

## Development and Extension

### Adding New Agents
1. Inherit from `BaseAgent` class
2. Implement required methods (`_run`, `handle_message`)
3. Subscribe to relevant message types
4. Add agent to main system initialization

### Customizing Detection
- Modify trapezoid region parameters
- Adjust confidence thresholds
- Add new object classes to track
- Implement custom risk assessment logic

### Extending Communication
- Add new TTS voices or languages
- Implement haptic feedback
- Create custom visual alert types
- Add audio spatial processing

## System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────────┐
│ Perception      │    │ Navigation       │    │ Communication       │
│ Agent           │    │ Agent            │    │ Agent               │
├─────────────────┤    ├──────────────────┤    ├─────────────────────┤
│ • YOLO Detection│    │ • Spatial Analysis│    │ • Text-to-Speech    │
│ • Depth Estimation  │ • Path Planning   │    │ • Visual Alerts     │
│ • Object Tracking│    │ • Risk Assessment│    │ • User Interface    │
└─────────────────┘    └──────────────────┘    └─────────────────────┘
         │                       │                        │
         └───────────────────────┼────────────────────────┘
                                 │
                    ┌─────────────────────────┐
                    │      Message Bus        │
                    │                         │
                    │ • Inter-agent comms     │
                    │ • Priority queuing      │
                    │ • Event distribution    │
                    └─────────────────────────┘
```

## Performance Metrics

### Target Performance
- **Frame Rate**: 15-30 FPS depending on hardware
- **Detection Latency**: < 100ms for obstacle alerts
- **Audio Response**: < 200ms for critical warnings
- **Memory Usage**: < 2GB RAM typical operation

### Benchmarking
The system includes FPS monitoring and performance logging for optimization analysis.

## Future Enhancements

- GPS integration for outdoor navigation
- Machine learning personalization
- Mobile application development
- Cloud-based model updates
- Multi-language TTS support
- Integration with smart city infrastructure

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/new-feature`)
3. Implement changes with appropriate tests
4. Update documentation as needed
5. Submit pull request with detailed description

## License

This project is developed for educational and research purposes. Please ensure compliance with all applicable licenses for dependencies and models used.

## Support

For issues, questions, or contributions:
- Check troubleshooting guide above
- Review system logs for error details
- Ensure all dependencies are properly installed
- Test with minimal configuration first

## Acknowledgments

- YOLOv8 by Ultralytics for object detection
- MiDaS by Intel ISL for depth estimation  
- OpenCV for computer vision processing
- PyTorch ecosystem for deep learning support
