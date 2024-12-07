# handtrack-websockets 🤚 

Real-time hand tracking over WebSockets using MediaPipe, OpenCV, and FastAPI. Track hand position and rotation in 3D space with a cyberpunk visualization layer.

## Features

- Real-time hand skeleton tracking and visualization
- WebSocket streaming of hand position/rotation data
- Smooth motion interpolation
- Cyberpunk-style HUD overlay
- ~30 FPS performance on modern hardware

## Requirements

- Python 3.9+
- OpenCV
- MediaPipe 
- FastAPI
- Modern browser with WebSocket support

## Quick Start

```bash
# Create conda environment
conda create -n handtrack python=3.9
conda activate handtrack

# Install dependencies
conda install -c conda-forge mediapipe opencv numpy
pip install fastapi uvicorn

# Run the tracker
python main.py
```

Visit `http://localhost:8000` to see the visualization.

## Protocol

WebSocket endpoint streams JSON packets containing:
- Palm center coordinates (x,y,z)
- Hand rotation (pitch/yaw)
- Raw landmark data
- Motion smoothing

## License

BSD 3-Clause. Use it. Hack it. Share it.

---
*"The hand is the visible part of the brain." - Immanuel Kant*