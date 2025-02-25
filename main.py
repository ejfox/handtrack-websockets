import cv2
import mediapipe as mp
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
import uvicorn
import json
from pathlib import Path
import asyncio
from collections import deque
import math
import logging
import argparse
from fastapi.responses import FileResponse

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize MediaPipe Hand tracking
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=1
)

# Create FastAPI app
app = FastAPI()

# Create a directory for static files if it doesn't exist
static_dir = Path("static")
static_dir.mkdir(exist_ok=True)

# Constants for hand tracking visualization
MIN_CIRCLE_SIZE = 50  # minimum circle radius in pixels
MAX_CIRCLE_SIZE = 200  # maximum circle radius in pixels
Z_RANGE = [0.2, 0.8]  # typical z-values range we want to map to circle size

# Smoothing buffer for hand tracking data
SMOOTH_FRAMES = 5
position_buffer = deque(maxlen=SMOOTH_FRAMES)

def smooth_positions(new_pos):
    """Apply exponential moving average smoothing to positions."""
    position_buffer.append(new_pos)
    if len(position_buffer) < 2:
        return new_pos
    
    alpha = 0.7  # Smoothing factor
    smoothed = np.array(position_buffer[-1])
    for pos in reversed(list(position_buffer)[:-1]):
        smoothed = alpha * np.array(pos) + (1 - alpha) * smoothed
    return smoothed.tolist()

def process_frame(hand_landmarks):
    """Process a single frame and return hand tracking data."""
    # Extract raw landmarks
    raw_landmarks = []
    for landmark in hand_landmarks.landmark:
        raw_landmarks.append({
            'x': landmark.x,
            'y': landmark.y,
            'z': landmark.z
        })
    
    # Calculate palm center (average of specific landmarks)
    palm_landmarks = [0, 5, 9, 13, 17]  # Landmark indices for palm
    palm_center = np.mean([[
        hand_landmarks.landmark[idx].x,
        hand_landmarks.landmark[idx].y,
        hand_landmarks.landmark[idx].z
    ] for idx in palm_landmarks], axis=0)
    
    # Smooth palm center position
    smooth_palm = smooth_positions(palm_center.tolist())
    
    # Calculate hand rotation (simplified)
    wrist = np.array([
        hand_landmarks.landmark[0].x,
        hand_landmarks.landmark[0].y,
        hand_landmarks.landmark[0].z
    ])
    middle_finger = np.array([
        hand_landmarks.landmark[12].x,
        hand_landmarks.landmark[12].y,
        hand_landmarks.landmark[12].z
    ])
    
    # Calculate direction vector
    direction = middle_finger - wrist
    direction = direction / np.linalg.norm(direction)
    
    # Convert to angles
    pitch = math.atan2(direction[1], direction[2])
    yaw = math.atan2(direction[0], direction[2])
    
    return {
        'raw_landmarks': raw_landmarks,
        'palm_center': smooth_palm,
        'rotation': {
            'pitch': pitch,
            'yaw': yaw
        }
    }

# Create HTML content with proper escaping for format strings
html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>Hand Tracking Test</title>
    <style>
        body {{
            margin: 0;
            overflow: hidden;
            background: #1a1a1a;
            font-family: system-ui, -apple-system, sans-serif;
        }}
        .hud {{
            position: fixed;
            font-family: monospace;
            color: rgba(0, 255, 255, 0.7);
            text-shadow: 0 0 5px rgba(0, 255, 255, 0.5);
            pointer-events: none;
            font-size: 12px;
            line-height: 1.4;
        }}
        .hud-top-right {{
            top: 20px;
            right: 20px;
            text-align: right;
        }}
        .hud-bottom-left {{
            bottom: 20px;
            left: 20px;
        }}
        .hud-label {{
            color: rgba(0, 255, 255, 0.5);
        }}
        .hud-value {{
            font-weight: bold;
        }}
        .hud-box {{
            border: 1px solid rgba(0, 255, 255, 0.3);
            padding: 10px;
            background: rgba(0, 0, 0, 0.5);
            margin-bottom: 10px;
        }}
        #debug {{
            position: fixed;
            top: 20px;
            left: 20px;
            color: #fff;
            background: rgba(0, 0, 0, 0.5);
            padding: 10px;
            border-radius: 8px;
            font-size: 14px;
            pointer-events: none;
        }}
        .circle-container {{
            filter: blur(8px);
            transition: filter 0.3s ease;
        }}
        .circle-container.tracking {{
            filter: blur(0);
        }}
    </style>
</head>
<body style="margin: 0; overflow: hidden; background: #1a1a1a;">
    <div id="debug">Waiting for hand tracking...</div>
    <div class="hud hud-top-right">
        <div class="hud-box">
            <div>PALM CENTER</div>
            <div>X: <span id="palm-x" class="hud-value">0.000</span></div>
            <div>Y: <span id="palm-y" class="hud-value">0.000</span></div>
            <div>Z: <span id="palm-z" class="hud-value">0.000</span></div>
        </div>
        <div class="hud-box">
            <div>ROTATION</div>
            <div>PITCH: <span id="rot-pitch" class="hud-value">0.000</span></div>
            <div>YAW: <span id="rot-yaw" class="hud-value">0.000</span></div>
        </div>
    </div>
    <div class="hud hud-bottom-left">
        <div class="hud-box">
            <div>SYSTEM STATUS</div>
            <div>FPS: <span id="hud-fps" class="hud-value">0</span></div>
            <div>TRACKING: <span id="hud-status" class="hud-value">INACTIVE</span></div>
        </div>
    </div>
    <svg width="100%" height="100%" style="position: fixed; inset: 0;">
        <defs>
            <radialGradient id="circleGradient">
                <stop offset="0%" stop-color="rgba(255, 100, 100, 0.8)"/>
                <stop offset="100%" stop-color="rgba(255, 50, 50, 0.2)"/>
            </radialGradient>
        </defs>
        <g class="circle-container">
            <circle id="testCircle" 
                cx="50%" 
                cy="50%" 
                r="100" 
                fill="url(#circleGradient)"
                filter="drop-shadow(0 0 10px rgba(255,0,0,0.5))"
            />
        </g>
    </svg>
    
    <script>
        const debug = document.getElementById('debug');
        const container = document.querySelector('.circle-container');
        let lastUpdate = Date.now();
        let frameCount = 0;
        let wsRetryCount = 0;
        const MAX_RETRIES = 5;
        
        // Constants for z-mapping
        const Z_MIN = {z_min};
        const Z_MAX = {z_max};
        const CIRCLE_MIN = {circle_min};
        const CIRCLE_MAX = {circle_max};
        
        function mapRange(value, inMin, inMax, outMin, outMax) {{
            return (value - inMin) * (outMax - outMin) / (inMax - inMin) + outMin;
        }}
        
        function clamp(value, min, max) {{
            return Math.min(Math.max(value, min), max);
        }}
        
        function connectWebSocket() {{
            debug.textContent = 'Connecting to WebSocket...';
            const ws = new WebSocket('ws://localhost:8000/ws');
            const circle = document.getElementById('testCircle');
            let currentX = window.innerWidth / 2;
            let currentY = window.innerHeight / 2;
            let currentR = 100;
            const smoothingFactor = 0.3;
            
            let isConnected = false;
            
            ws.onopen = function() {{
                debug.textContent = 'WebSocket Connected';
                wsRetryCount = 0;
                isConnected = true;
            }};
            
            ws.onmessage = function(event) {{
                if (!isConnected) return;
                
                frameCount++;
                const now = Date.now();
                if (now - lastUpdate >= 1000) {{
                    debug.textContent = `FPS: ${{frameCount}} | Hand Tracking Active`;
                    frameCount = 0;
                    lastUpdate = now;
                }}
                
                const data = JSON.parse(event.data);
                
                // Handle status messages
                if (data.status) {{
                    debug.textContent = data.status;
                    return;
                }}
                
                // Handle error messages
                if (data.error) {{
                    debug.textContent = `Error: ${{data.error}}`;
                    return;
                }}
                
                if (data.palm_center) {{
                    container.classList.add('tracking');
                    const [x, y, z] = data.palm_center;
                    
                    // Update HUD values
                    document.getElementById('palm-x').textContent = x.toFixed(3);
                    document.getElementById('palm-y').textContent = y.toFixed(3);
                    document.getElementById('palm-z').textContent = z.toFixed(3);
                    document.getElementById('rot-pitch').textContent = data.rotation.pitch.toFixed(3);
                    document.getElementById('rot-yaw').textContent = data.rotation.yaw.toFixed(3);
                    document.getElementById('hud-status').textContent = 'ACTIVE';
                    document.getElementById('hud-fps').textContent = frameCount.toString();
                    
                    // Smooth position updates
                    const targetX = x * window.innerWidth;
                    const targetY = y * window.innerHeight;
                    currentX = currentX + (targetX - currentX) * smoothingFactor;
                    currentY = currentY + (targetY - currentY) * smoothingFactor;
                    
                    // Map z to radius with limits
                    const targetR = mapRange(
                        clamp(z, Z_MIN, Z_MAX),
                        Z_MAX, Z_MIN,  // Invert z range for intuitive scaling
                        CIRCLE_MIN, CIRCLE_MAX
                    );
                    currentR = currentR + (targetR - currentR) * smoothingFactor;
                    
                    circle.setAttribute('cx', currentX);
                    circle.setAttribute('cy', currentY);
                    circle.setAttribute('r', currentR);
                }} else {{
                    container.classList.remove('tracking');
                    debug.textContent = 'No hand detected';
                }}
            }};
            
            ws.onclose = function() {{
                isConnected = false;
                wsRetryCount++;
                if (wsRetryCount < MAX_RETRIES) {{
                    debug.textContent = `WebSocket closed, attempt ${{wsRetryCount}}/${{MAX_RETRIES}} to reconnect...`;
                    setTimeout(connectWebSocket, 1000);
                }} else {{
                    debug.textContent = 'WebSocket connection failed after multiple attempts';
                }}
                container.classList.remove('tracking');
            }};
            
            ws.onerror = function(err) {{
                debug.textContent = `WebSocket error: ${{err.message || 'Unknown error'}}`;
                container.classList.remove('tracking');
                ws.close();
            }};
        }}
        
        connectWebSocket();
        
        window.onfocus = function() {{
            if (wsRetryCount >= MAX_RETRIES) {{
                wsRetryCount = 0;
                connectWebSocket();
            }}
        }};
    </script>
</body>
</html>
""".format(
    z_min=Z_RANGE[0],
    z_max=Z_RANGE[1],
    circle_min=MIN_CIRCLE_SIZE,
    circle_max=MAX_CIRCLE_SIZE
)

# Write the formatted HTML to file
with open(static_dir / "index.html", "w") as f:
    f.write(html_content)

# Add this at the module level
camera = None
frame_processor = None

class FrameProcessor:
    def __init__(self, source=0):
        """Initialize with camera index or video path."""
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError("Failed to open camera!")
        
        # Get video properties
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        
        # Set video properties for better processing
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        if isinstance(source, str):  # If video file
            logger.info(f"Loaded video file: {source}")
            logger.info(f"Video FPS: {self.fps}")
            logger.info(f"Video dimensions: {self.width}x{self.height}")
        else:
            logger.info("Camera opened successfully!")
        
        # For video files, loop playback
        self.is_video = isinstance(source, str)
        
    def process_next_frame(self):
        success, frame = self.cap.read()
        if not success:
            if self.is_video:  # Loop video
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                success, frame = self.cap.read()
                if not success:
                    return None, None
            else:
                return None, None
        
        # Preprocess frame for better detection
        frame = cv2.resize(frame, (640, 480))
        frame = cv2.flip(frame, 1)
        
        # Convert to HSV to handle the red glove
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create mask for red/orange colors
        lower_red = np.array([0, 50, 50])
        upper_red = np.array([30, 255, 255])
        mask1 = cv2.inRange(hsv, lower_red, upper_red)
        
        # Handle wrap-around hue values for red
        lower_red = np.array([150, 50, 50])
        upper_red = np.array([180, 255, 255])
        mask2 = cv2.inRange(hsv, lower_red, upper_red)
        
        # Combine masks and dilate to get full glove
        mask = mask1 + mask2
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
        
        # Apply mask to original image
        glove_region = cv2.bitwise_and(frame, frame, mask=mask)
        
        # Enhance contrast in non-glove regions
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        enhanced = cv2.merge((cl,a,b))
        frame = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        # Combine enhanced frame with glove region
        frame = cv2.addWeighted(frame, 0.7, glove_region, 0.3, 0)
        
        # Convert to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb = cv2.GaussianBlur(frame_rgb, (3,3), 0)  # Lighter blur
        
        results = hands.process(frame_rgb)
        tracking_data = None
        
        if results.multi_hand_landmarks:
            logger.info(f"Detected {len(results.multi_hand_landmarks)} hands")
            # Draw landmarks first
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(),
                    connection_drawing_spec=mp_drawing_styles.get_default_hand_connections_style()
                )
            # Process tracking data with the first hand
            tracking_data = process_frame(results.multi_hand_landmarks[0])
        else:
            logger.debug("No hands detected")
        
        # Add status text
        cv2.putText(
            frame,
            "Video Playing - Press 'Q' to quit" if self.is_video else "Camera Active - Press 'Q' to quit",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )
        
        # Add debug info
        if tracking_data:
            cv2.putText(
                frame,
                "Hand Detected!",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )
            # Draw palm center
            palm_x = int(tracking_data['palm_center'][0] * self.width)
            palm_y = int(tracking_data['palm_center'][1] * self.height)
            cv2.circle(frame, (palm_x, palm_y), 10, (0, 255, 0), -1)
        
        return frame, tracking_data
        
    def cleanup(self):
        self.cap.release()

async def camera_display_loop():
    global frame_processor
    
    while True:
        if frame_processor is None:
            await asyncio.sleep(0.1)
            continue
            
        frame, _ = frame_processor.process_next_frame()
        if frame is not None:
            cv2.imshow('Debug View', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        await asyncio.sleep(1/30)  # Target 30 FPS

# WebSocket endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global frame_processor
    
    try:
        logger.info("New WebSocket connection request")
        await websocket.accept()
        logger.info("WebSocket connection accepted")
        
        try:
            while True:
                if frame_processor is None:
                    await asyncio.sleep(0.1)
                    continue
                    
                frame, tracking_data = frame_processor.process_next_frame()
                
                if tracking_data:
                    try:
                        await websocket.send_json(tracking_data)
                        logger.debug("Sent tracking data")
                    except Exception as e:
                        logger.error(f"Failed to send tracking data: {e}")
                        break
                else:
                    logger.debug("No hand detected in frame")
                
                await asyncio.sleep(1/30)
                
        except WebSocketDisconnect:
            logger.info("WebSocket disconnected")
            
    except Exception as e:
        logger.error(f"Failed to establish WebSocket connection: {e}")
        await websocket.close()

# Mount static files after defining routes
app.mount("/static", StaticFiles(directory="static"), name="static")

# Add a root route to serve the HTML
@app.get("/")
async def root():
    return FileResponse(static_dir / "index.html")

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Hand tracking with webcam or video file')
    parser.add_argument('--movie', type=str, help='Path to video file')
    args = parser.parse_args()
    
    # Initialize camera access before starting server
    logger.info("Testing camera access...")
    try:
        source = args.movie if args.movie else 0
        frame_processor = FrameProcessor(source)
    except Exception as e:
        logger.error(f"Failed to initialize camera: {e}")
        exit(1)
    
    # Create and get event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    # Create FastAPI config
    config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=8000,
        loop=loop
    )
    
    # Create server
    server = uvicorn.Server(config)
    
    # Run everything
    try:
        loop.create_task(camera_display_loop())
        loop.run_until_complete(server.serve())
    except KeyboardInterrupt:
        if frame_processor:
            frame_processor.cleanup()
        cv2.destroyAllWindows()