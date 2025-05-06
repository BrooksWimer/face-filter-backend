import sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import os
import requests
import time

# Add project root to import path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from faceLandmarkPredictor import FaceLandmarkPredictor  # noqa: E402

# Model URL - You'll need to update this with a real URL where you've hosted the model
MODEL_URL = os.environ.get(
    "MODEL_URL", 
    "https://github.com/brookswimer/face-filter-model/releases/download/v1.0/landmark_model.pt"
)

def download_model(model_path):
    """Download the model file if it doesn't exist."""
    print(f"[INFO] Downloading model from {MODEL_URL}")
    try:
        response = requests.get(MODEL_URL, stream=True)
        response.raise_for_status()  # Raise exception for HTTP errors
        
        # Create directory if it doesn't exist
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save the file
        with open(model_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"[INFO] Model downloaded to {model_path}")
        # Small delay to ensure file is fully written
        time.sleep(1)
        return True
    except Exception as e:
        print(f"[ERROR] Failed to download model: {str(e)}")
        return False

def process_video(video_path: Path, mask_path: Path, output_path: Path):
    """Apply mask overlay to each frame using our custom facial landmark model."""
    # Check if model file exists
    model_path = PROJECT_ROOT / "landmark_model.pt"
    
    # Flag to indicate if we're using the full model or fallback
    using_fallback = False
    
    if not model_path.exists():
        print(f"[INFO] Model not found at {model_path}, attempting to download...")
        success = download_model(model_path)
        if not success or not model_path.exists():
            print(f"[WARN] Cannot find or download landmark model to {model_path}, using fallback mode")
            using_fallback = True
    
    if not using_fallback:
        print(f"[DEBUG] Loading model from {model_path}")
        try:
            # Load our PyTorch model
            predictor = FaceLandmarkPredictor(str(model_path))
        except Exception as e:
            print(f"[ERROR] Failed to load landmark model: {str(e)}")
            print("[INFO] Switching to fallback mode")
            using_fallback = True

    # Load mask image with alpha channel
    mask_rgba = Image.open(mask_path).convert("RGBA")
    mask_np = np.array(mask_rgba)
    mask_h0, mask_w0 = mask_np.shape[:2]

    # Set up video reader and get basic info
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Try different codecs until we find one that works
    # Different OS/OpenCV builds support different codecs
    preferred_codecs = ["avc1", "mp4v", "H264", "XVID", "MJPG"]
    out = None
    for codec in preferred_codecs:
        vw = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*codec), fps, (width, height))
        if vw.isOpened():
            out = vw
            print(f"[overlay_processor] Using codec {codec}")
            break
    if out is None:
        raise RuntimeError("Failed to open VideoWriter with any supported codec. Install ffmpeg/libx264 etc.")

    # Fallback mode uses OpenCV's face detector instead of our custom model
    if using_fallback:
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        
    # Process each frame
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if using_fallback:
            # Simple fallback using OpenCV's face detector
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            
            if len(faces) > 0:
                # Choose the largest face
                x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                
                # Calculate simplified landmarks
                left_eye = (x + w//4, y + h//3)
                right_eye = (x + 3*w//4, y + h//3)
                nose_tip = (x + w//2, y + 2*h//3)
                
                # Calculate angle between eyes for rotation
                dx, dy = right_eye[0] - left_eye[0], right_eye[1] - left_eye[1]
                angle = np.degrees(np.arctan2(dy, dx))
                
                # Use eye distance to scale mask
                eye_dist = np.hypot(dx, dy)
                scale = (eye_dist * 3.0) / mask_w0
                
                # Resize mask based on face size
                new_w, new_h = int(mask_w0 * scale), int(mask_h0 * scale)
                resized_mask = cv2.resize(mask_np, (new_w, new_h), interpolation=cv2.INTER_AREA)
                
                # Rotation and positioning code is the same as the original
                M = cv2.getRotationMatrix2D((new_w / 2, new_h / 2), angle, 1.0)
                abs_cos, abs_sin = abs(M[0, 0]), abs(M[0, 1])
                rot_w = int(new_h * abs_sin + new_w * abs_cos)
                rot_h = int(new_h * abs_cos + new_w * abs_sin)
                M[0, 2] += (rot_w / 2) - new_w / 2
                M[1, 2] += (rot_h / 2) - new_h / 2
                
                rotated_mask = cv2.warpAffine(
                    resized_mask,
                    M,
                    (rot_w, rot_h),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=(0, 0, 0, 0)
                )
                
                # Position mask over face
                center_x = int((left_eye[0] + right_eye[0]) / 2)
                center_y = int(nose_tip[1] - rot_h * 0.8)
                
                # Calculate mask corners
                x1, y1 = center_x - rot_w // 2, center_y
                x2, y2 = x1 + rot_w, y1 + rot_h
                
                # Handle mask regions outside frame
                x1c, y1c = max(0, x1), max(0, y1)
                x2c, y2c = min(width, x2), min(height, y2)
                
                # Calculate corresponding mask portion
                mask_x1, mask_y1 = x1c - x1, y1c - y1
                mask_x2, mask_y2 = mask_x1 + (x2c - x1c), mask_y1 + (y2c - y1c)
                
                # Only blend if we have valid regions
                if mask_x2 > mask_x1 and mask_y2 > mask_y1:
                    mask_crop = rotated_mask[mask_y1:mask_y2, mask_x1:mask_x2]
                    frame_crop = frame[y1c:y2c, x1c:x2c]
                    # Alpha blending using mask's alpha channel
                    alpha = mask_crop[..., 3:] / 255.0
                    frame[y1c:y2c, x1c:x2c] = (
                        alpha * mask_crop[..., :3] + (1 - alpha) * frame_crop
                    ).astype(np.uint8)
        else:
            # Use our custom model as before
            landmarks, _ = predictor.predict(frame)
            if landmarks is not None:
                try:
                    # Extract key points for mask positioning
                    left_eye, right_eye, nose_tip = landmarks[0], landmarks[1], landmarks[2]
                except IndexError:
                    # If landmarks are detected but don't have enough points,
                    # just write the frame without mask
                    out.write(frame)
                    continue

                # Calculate angle between eyes for rotation
                dx, dy = right_eye[0] - left_eye[0], right_eye[1] - left_eye[1]
                angle = np.degrees(np.arctan2(dy, dx))
                
                # Use eye distance to scale mask
                eye_dist = np.hypot(dx, dy)
                scale = (eye_dist * 3.0) / mask_w0

                # Resize mask based on face size
                new_w, new_h = int(mask_w0 * scale), int(mask_h0 * scale)
                resized_mask = cv2.resize(mask_np, (new_w, new_h), interpolation=cv2.INTER_AREA)

                # Rotate mask to match face orientation
                # Need to handle the bounds expansion from rotation
                M = cv2.getRotationMatrix2D((new_w / 2, new_h / 2), angle, 1.0)
                abs_cos, abs_sin = abs(M[0, 0]), abs(M[0, 1])
                rot_w = int(new_h * abs_sin + new_w * abs_cos)
                rot_h = int(new_h * abs_cos + new_w * abs_sin)
                M[0, 2] += (rot_w / 2) - new_w / 2
                M[1, 2] += (rot_h / 2) - new_h / 2

                rotated_mask = cv2.warpAffine(
                    resized_mask,
                    M,
                    (rot_w, rot_h),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=(0, 0, 0, 0)
                )

                # Position mask over face 
                # Center between eyes horizontally
                center_x = int((left_eye[0] + right_eye[0]) / 2)
                # Position vertically above nose tip
                center_y = int(nose_tip[1] - rot_h * 0.8)

                # Calculate mask corners
                x1, y1 = center_x - rot_w // 2, center_y
                x2, y2 = x1 + rot_w, y1 + rot_h

                # Handle mask regions outside frame
                x1c, y1c = max(0, x1), max(0, y1)
                x2c, y2c = min(width, x2), min(height, y2)

                # Calculate corresponding mask portion
                mask_x1, mask_y1 = x1c - x1, y1c - y1
                mask_x2, mask_y2 = mask_x1 + (x2c - x1c), mask_y1 + (y2c - y1c)

                # Only blend if we have valid regions
                if mask_x2 > mask_x1 and mask_y2 > mask_y1:
                    mask_crop = rotated_mask[mask_y1:mask_y2, mask_x1:mask_x2]
                    frame_crop = frame[y1c:y2c, x1c:x2c]
                    # Alpha blending using mask's alpha channel
                    alpha = mask_crop[..., 3:] / 255.0
                    frame[y1c:y2c, x1c:x2c] = (
                        alpha * mask_crop[..., :3] + (1 - alpha) * frame_crop
                    ).astype(np.uint8)

        out.write(frame)

    cap.release()
    out.release()


def main():
    if len(sys.argv) != 4:
        print("Usage: python overlay_processor.py <input_video> <mask_png> <output_video>")
        sys.exit(1)

    in_video = Path(sys.argv[1])
    mask_png = Path(sys.argv[2])
    out_video = Path(sys.argv[3])
    process_video(in_video, mask_png, out_video)


if __name__ == "__main__":
    main() 