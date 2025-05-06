from flask import Flask, request, jsonify, send_from_directory, Response
from werkzeug.utils import secure_filename
import subprocess, uuid, pathlib
from flask_cors import CORS
import tempfile, os

# Set up file paths - need to handle uploads & processed videos
BASE_DIR = pathlib.Path(__file__).resolve().parent.parent  # project root
UPLOAD_DIR = BASE_DIR / "uploads"
PROCESSED_DIR = BASE_DIR / "processed"
MASK_PATH = BASE_DIR / "masks" / "cat.png"  # Default mask

# Make sure dirs exist
UPLOAD_DIR.mkdir(exist_ok=True)
PROCESSED_DIR.mkdir(exist_ok=True)

app = Flask(__name__, static_folder=None)
# Fix CORS configuration
CORS(app, resources={r"/*": {"origins": "https://brookswimer.github.io"}})

# Add a simple root route
@app.route("/")
def index():
    return jsonify({"status": "online", "message": "Face Filter API is running"})

@app.route("/upload", methods=["POST"])
def upload():
    """Receive a webm video, process with custom model, return processed URL."""
    if "video" not in request.files:
        return jsonify({"error": "No video field in form"}), 400

    video_file = request.files["video"]

    # Use UUIDs for filenames to avoid collisions
    in_name = secure_filename(f"{uuid.uuid4()}.webm")
    input_path = UPLOAD_DIR / in_name
    video_file.save(input_path)

    # Output keeps same UUID but adds _mask and changes ext
    out_name = input_path.stem + "_mask.mp4"
    output_path = PROCESSED_DIR / out_name

    # Run processor in a separate process to avoid memory issues
    cmd = [
        "python",
        str(BASE_DIR / "backend" / "overlay_processor.py"),
        str(input_path),
        str(MASK_PATH),
        str(output_path)
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        print("[ERROR] overlay_processor failed:", e.stderr.decode())
        return jsonify({"error": "Processing failed", "details": e.stderr.decode()}), 500

    return jsonify({"processed_url": f"/processed/{out_name}"})

@app.route("/processed/<path:fname>")
def processed(fname):
    """Serve processed video files."""
    return send_from_directory(PROCESSED_DIR, fname, as_attachment=False)

@app.route("/process-inline", methods=["POST"])
def process_inline():
    """Receive a video, run overlay processor, stream the processed MP4 back."""
    if "video" not in request.files:
        return jsonify({"error": "No video field in form"}), 400

    # Get mask name from form data, sanitize input
    mask_name = request.form.get("mask", "cat")
    # Only allow simple filenames (no directory traversal)
    if not mask_name.isalnum():
        return jsonify({"error": "Invalid mask name"}), 400
    mask_path = BASE_DIR / "masks" / f"{mask_name}.png"
    if not mask_path.exists():
        print(f"[ERROR] Mask not found: {mask_path}")
        return jsonify({"error": "Mask not found"}), 400

    # Save uploaded webm to temp file
    src_file = request.files["video"]
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp_in:
        src_file.save(tmp_in)
        input_path = tmp_in.name

    # WebM → MP4 conversion (OpenCV needs MP4)
    interm_fd, interm_mp4 = tempfile.mkstemp(suffix=".mp4")
    os.close(interm_fd)
    try:
        print(f"[DEBUG] Running ffmpeg conversion from {input_path} to {interm_mp4}")
        subprocess.run([
            "ffmpeg", "-y",
            "-i", input_path,
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-preset", "veryfast",
            "-movflags", "+faststart",
            interm_mp4,
        ], check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        os.unlink(input_path)
        os.unlink(interm_mp4)
        print(f"[ERROR] ffmpeg conversion failed: {e.stderr.decode()}")
        return jsonify({"error": "Failed to convert WebM", "details": e.stderr.decode()}), 500

    # Create temp file for processed output
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_out:
        output_path = tmp_out.name

    # Call overlay processor to apply mask
    cmd = [
        "python",
        str(BASE_DIR / "backend" / "overlay_processor.py"),
        interm_mp4,
        str(mask_path),
        output_path,
    ]
    try:
        print(f"[DEBUG] Running overlay_processor with: {cmd}")
        result = subprocess.run(cmd, check=True, capture_output=True)
        print(f"[DEBUG] Overlay processor stdout: {result.stdout.decode()}")
    except subprocess.CalledProcessError as e:
        # Clean up our mess
        os.unlink(input_path)
        os.unlink(interm_mp4)
        os.unlink(output_path)
        print(f"[ERROR] Processing failed: {e.stderr.decode()}")
        return jsonify({"error": "Processing failed", "details": e.stderr.decode()}), 500

    # Final ffmpeg pass for browser-compatible output
    # Web browsers are super picky about MP4 compatibility
    fd, final_mp4 = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)  # Close fd so ffmpeg can write on Windows
    try:
        subprocess.run([
            "ffmpeg",
            "-y",  # overwrite if exists
            "-i", str(output_path),
            "-vf", "fps=30",  # ensure constant fps
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-preset", "veryfast",
            "-movflags", "+faststart",
            final_mp4,
        ], check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        # Fallback to unprocessed file if ffmpeg fails
        print("[WARN] ffmpeg transcode failed, serving raw output", e.stderr.decode())
        final_mp4 = output_path

    # Load file into memory
    with open(final_mp4, "rb") as fh:
        video_bytes = fh.read()

    # Clean up all temp files
    os.unlink(input_path)
    os.unlink(interm_mp4)
    os.unlink(output_path)
    if os.path.exists(final_mp4) and final_mp4 != output_path:
        os.unlink(final_mp4)

    # Send video back to browser
    resp = Response(video_bytes, mimetype="video/mp4")
    resp.headers["Content-Disposition"] = "inline; filename=processed.mp4"
    # Explicitly set CORS headers on this response
    resp.headers["Access-Control-Allow-Origin"] = "https://brookswimer.github.io"
    resp.headers["Access-Control-Allow-Methods"] = "POST, GET, OPTIONS"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    print("[DEBUG] returning", len(video_bytes), "bytes from process-inline")
    return resp

@app.route("/", methods=["OPTIONS"])
@app.route("/process-inline", methods=["OPTIONS"])
@app.route("/upload", methods=["OPTIONS"])
def handle_options():
    """Handle preflight CORS requests"""
    response = app.make_default_options_response()
    response.headers["Access-Control-Allow-Origin"] = "https://brookswimer.github.io"
    response.headers["Access-Control-Allow-Methods"] = "POST, GET, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response

@app.after_request
def add_cors_headers(resp):
    # Only add CORS headers if they're not already present
    if "Access-Control-Allow-Origin" not in resp.headers:
        resp.headers["Access-Control-Allow-Origin"] = "https://brookswimer.github.io"
        resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
        resp.headers["Access-Control-Allow-Methods"] = "POST, GET, OPTIONS"
    return resp

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, threaded=True) 