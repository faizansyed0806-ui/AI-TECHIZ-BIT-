# app.py
"""
Flask app that auto-discovers and loads:
 - face_shape.py (any folder)
 - skin_tone.py (any folder)
 - diet_plan.py (any folder)

Place app.py in your project root and keep the module files anywhere under the tree
(e.g. "face shape/face_shape.py", "skin tone/skin_tone.py", "plan/diet_plan.py").
"""

import os
import json
import traceback
from pathlib import Path
from datetime import datetime
import importlib.util

from flask import Flask, request, redirect, url_for, send_file, render_template_string, flash

# ---------- Configuration ----------
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)
ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# If you want the app to serve your project brief, set this path (uploaded earlier).
PROJECT_DOC_PATH = "/mnt/data/298ebfbe-7e20-40ef-9ec9-fd1dab6cdebc.docx"

app = Flask(__name__)
app.secret_key = "dev-secret-key"

# ---------- Utility: find file in tree ----------
def find_first(filename_candidates):
    """
    Search current working directory recursively for any of the given candidate filenames.
    Returns Path or None.
    """
    cwd = Path.cwd()
    for candidate in filename_candidates:
        for p in cwd.rglob(candidate):
            if p.is_file():
                return p
    return None

def load_module_from_path(name, path: Path):
    """Load a module given its filename path and return the module object."""
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create spec for {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# ---------- Discover & load modules ----------
face_candidates = ["face_shape.py", "face-shape.py"]
skin_candidates = ["skin_tone.py", "skin-tone.py"]
diet_candidates = ["diet_plan.py", "dietplan.py", "tdee.py"]

face_path = find_first(face_candidates)
skin_path = find_first(skin_candidates)
diet_path = find_first(diet_candidates)

face_module = None
skin_module = None
diet_module = None

if face_path:
    try:
        face_module = load_module_from_path("face_shape_dyn", face_path)
        print("Loaded face module from:", face_path)
    except Exception as e:
        print("Failed loading face module:", e)
else:
    print("face_shape.py not found under project tree.")

if skin_path:
    try:
        skin_module = load_module_from_path("skin_tone_dyn", skin_path)
        print("Loaded skin module from:", skin_path)
    except Exception as e:
        print("Failed loading skin module:", e)
else:
    print("skin_tone.py not found under project tree.")

if diet_path:
    try:
        diet_module = load_module_from_path("diet_plan_dyn", diet_path)
        print("Loaded diet module from:", diet_path)
    except Exception as e:
        print("Failed loading diet module:", e)
else:
    print("diet_plan.py / tdee.py not found under project tree.")

# Bind functions if available (safe)
def get_attr(m, name):
    return getattr(m, name) if m and hasattr(m, name) else None

analyze_face_shape = get_attr(face_module, "analyze_face_shape")
# skin_tone may provide analyze_image or analyze_skin_tone
analyze_skin_tone = get_attr(skin_module, "analyze_image") or get_attr(skin_module, "analyze_skin_tone")
generate_weekly_plan = get_attr(diet_module, "generate_weekly_plan")

# ---------- HTML Templates (well-formed triple-quoted strings) ----------
HOME_HTML = """
<!doctype html>
<html>
  <body>
    <h1>Health & CV Toolkit</h1>
    <h2>1) Analyze face (shape + skin tone)</h2>
    <form action="/analyze_image" method="post" enctype="multipart/form-data">
      <input type="file" name="image" required><br><br>
      <label>Prefer cheek side:</label>
      <select name="side">
        <option value="left">Left</option><option value="right">Right</option>
      </select><br><br>
      <label><input type="checkbox" name="visualize" value="1"> Produce debug visualization</label><br><br>
      <button type="submit">Analyze Image</button>
    </form>

    <hr>
    <h2>2) Generate 7-day diet plan</h2>
    <form method="post" action="/generate_diet">
      <label>TDEE (kcal):</label><br>
      <input type="number" name="tdee" required placeholder="e.g. 2400"><br><br>
      <label>Goal:</label><br>
      <select name="goal">
        <option value="maintenance">Maintenance</option>
        <option value="weight_loss">Weight Loss</option>
        <option value="muscle_gain">Muscle Gain</option>
      </select><br><br>
      <label>Exclusions (comma-separated):</label><br>
      <input type="text" name="exclusions" placeholder="vegetarian,dairy_free"><br><br>
      <button type="submit">Generate Diet</button>
    </form>

    <hr>
    <h2>3) Project brief</h2>
    <p><a href="/project_brief">Download project brief (docx)</a></p>
  </body>
</html>
"""

RESULT_HTML = """
<!doctype html>
<html>
  <body>
    <h1>Analysis Results</h1>
    {% if error %}
      <h3 style="color:red">{{ error }}</h3>
      <pre>{{ trace }}</pre>
      <a href="/">Back</a>
    {% else %}
      <h2>Face Shape</h2>
      <pre>{{ face_json }}</pre>
      {% if debug_url %}
        <h3>Debug image</h3>
        <img src="{{ debug_url }}" style="max-width:600px"><br>
      {% endif %}
      <h2>Skin Tone</h2>
      <pre>{{ skin_json }}</pre>
      <a href="/">Back</a>
    {% endif %}
  </body>
</html>
"""

DIET_HTML = """
<!doctype html>
<html>
  <body>
    <h1>Diet Plan</h1>
    {% if error %}
      <h3 style="color:red">{{ error }}</h3>
      <pre>{{ trace }}</pre>
    {% else %}
      <pre>{{ plan_json }}</pre>
    {% endif %}
    <a href="/">Back</a>
  </body>
</html>
"""

# ---------- Helpers ----------
def allowed_filename(fname):
    return Path(fname).suffix.lower() in ALLOWED_EXT

def save_upload(file_storage):
    filename = file_storage.filename
    ext = Path(filename).suffix
    out_name = f"upload_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}{ext}"
    out_path = UPLOAD_DIR / out_name
    file_storage.save(out_path)
    return out_path

# ---------- Routes ----------
@app.route("/")
def index():
    return render_template_string(HOME_HTML)

@app.route("/analyze_image", methods=["POST"])
def analyze_image_route():
    try:
        if "image" not in request.files:
            return redirect(url_for("index"))
        f = request.files["image"]
        if f.filename == "":
            return redirect(url_for("index"))
        if not allowed_filename(f.filename):
            return "Unsupported file type", 400

        saved = save_upload(f)
        visualize = bool(request.form.get("visualize"))
        side = request.form.get("side", "left")

        if analyze_face_shape is None:
            raise RuntimeError("face_shape analyzer not available. Make sure face_shape.py exists.")
        face_res = analyze_face_shape(str(saved), visualize=visualize)

        # Prepare debug image if available
        debug_url = None
        if visualize and isinstance(face_res, dict) and "debug_image" in face_res:
            dbg = Path(face_res["debug_image"])
            if dbg.exists():
                dest = UPLOAD_DIR / dbg.name
                if not dest.exists():
                    dest.write_bytes(dbg.read_bytes())
                debug_url = url_for("uploaded_file", filename=dest.name)

        # call skin tone (best-effort, handle signature differences)
        if analyze_skin_tone is None:
            raise RuntimeError("skin_tone analyzer not available. Make sure skin_tone.py exists.")
        try:
            # try common signature
            skin_res = analyze_skin_tone(str(saved), face_bbox=None, prefer_side=side)
        except TypeError:
            # fallback to simpler signature
            skin_res = analyze_skin_tone(str(saved))

        return render_template_string(RESULT_HTML,
                                      face_json=json.dumps(face_res, indent=2),
                                      skin_json=json.dumps(skin_res, indent=2),
                                      debug_url=debug_url,
                                      error=None)
    except Exception as e:
        return render_template_string(RESULT_HTML, error=str(e), trace=traceback.format_exc(), face_json="", skin_json="", debug_url=None)

@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    file_path = UPLOAD_DIR / filename
    if not file_path.exists():
        return "Not found", 404
    return send_file(file_path)

@app.route("/generate_diet", methods=["POST"])
def generate_diet_route():
    try:
        if generate_weekly_plan is None:
            raise RuntimeError("Diet generator not available. Make sure diet_plan.py exists.")
        tdee = int(request.form.get("tdee", 0))
        goal = request.form.get("goal", "maintenance")
        exclusions_raw = request.form.get("exclusions", "")
        exclusions = [s.strip() for s in exclusions_raw.split(",") if s.strip()]
        plan = generate_weekly_plan(tdee, goal=goal, exclusions=exclusions)
        return render_template_string(DIET_HTML, plan_json=json.dumps(plan, indent=2), error=None)
    except Exception as e:
        return render_template_string(DIET_HTML, error=str(e), trace=traceback.format_exc(), plan_json="")

@app.route("/project_brief")
def project_brief():
    if not os.path.isfile(PROJECT_DOC_PATH):
        return "Project brief not found on server.", 404
    return send_file(PROJECT_DOC_PATH, as_attachment=True, download_name=os.path.basename(PROJECT_DOC_PATH))

# ---------- Run ----------
if __name__ == "__main__":
    print("Starting app on http://127.0.0.1:5000")
    app.run(debug=True)
