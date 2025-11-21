# face_shape.py
"""
Face shape classification (heuristic) using OpenCV Haar cascades.

- Detects face and eyes using OpenCV built-in Haar cascades.
- Estimates cheekbone width and jaw width from the face bounding box and eye positions.
- Classifies approximate face shape: Oval, Round, Square, Heart, Diamond, Oblong.
- Returns style suggestions for hair and eyewear.

Dependencies:
    pip install opencv-python numpy

Usage:
    python face_shape.py --image PATH_TO_IMAGE [--side left|right] [--out result.json]

Notes:
- This is a heuristic approach (not a research-grade ML classifier).
- Results depend on frontal face photos, neutral expression, and reasonable lighting.
"""
import argparse
import json
import math
import os
from typing import Dict, Any, Optional, Tuple, List

import cv2
import numpy as np

# Path to OpenCV's haar cascades (should be available in cv2.data.haarcascades)
HAAR_FACE = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
HAAR_EYE = cv2.data.haarcascades + "haarcascade_eye.xml"
HAAR_NOSE = cv2.data.haarcascades + "haarcascade_mcs_nose.xml"  # may or may not exist depending on opencv build
HAAR_MOUTH = cv2.data.haarcascades + "haarcascade_mcs_mouth.xml"  # optional

# Safety: fallback to face cascade only if some cascades missing
_face_cascade = cv2.CascadeClassifier(HAAR_FACE)
_eye_cascade = cv2.CascadeClassifier(HAAR_EYE)


def detect_primary_face(img_gray: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """
    Detect the largest face in a grayscale image.
    Returns bbox (x, y, w, h) or None.
    """
    faces = _face_cascade.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
    if len(faces) == 0:
        return None
    # choose the largest face by area
    faces = sorted(faces, key=lambda r: r[2] * r[3], reverse=True)
    x, y, w, h = faces[0]
    return int(x), int(y), int(w), int(h)


def detect_eyes_in_face(img_gray: np.ndarray, face_bbox: Tuple[int, int, int, int]) -> List[Tuple[int, int, int, int]]:
    """
    Detect eyes within the face box and return list of bboxes (relative to whole image).
    """
    x, y, w, h = face_bbox
    face_roi = img_gray[y:y + h, x:x + w]
    eyes = _eye_cascade.detectMultiScale(face_roi, scaleFactor=1.1, minNeighbors=4, minSize=(15, 15))
    eyes_global = []
    for (ex, ey, ew, eh) in eyes:
        eyes_global.append((x + int(ex), y + int(ey), int(ew), int(eh)))
    # sort eyes by x (left->right)
    eyes_global = sorted(eyes_global, key=lambda e: e[0])
    return eyes_global


def midpoint(p1: Tuple[float, float], p2: Tuple[float, float]) -> Tuple[float, float]:
    return ((p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0)


def estimate_measures_from_bbox_and_eyes(face_bbox: Tuple[int, int, int, int],
                                         eyes: List[Tuple[int, int, int, int]]) -> Dict[str, float]:
    """
    Estimate:
      - face_width, face_height (pixels)
      - aspect_ratio = face_height / face_width
      - eye_distance (center-to-center)
      - cheekbone_width_est (approx)
      - jaw_width_est (approx)
    These are heuristic estimates based on bbox geometry and eye positions.
    """
    x, y, w, h = face_bbox
    face_width = float(w)
    face_height = float(h)
    aspect_ratio = face_height / (face_width + 1e-9)

    # compute eye centers if available
    eye_centers = []
    for (ex, ey, ew, eh) in eyes[:2]:
        cx = ex + ew / 2.0
        cy = ey + eh / 2.0
        eye_centers.append((cx, cy))

    eye_distance = 0.0
    if len(eye_centers) >= 2:
        (x1, y1), (x2, y2) = eye_centers[0], eye_centers[1]
        eye_distance = math.hypot(x2 - x1, y2 - y1)

    # cheekbone width estimate:
    # assume cheekbones roughly sit across face at ~30% down from top of bbox
    cheek_y = y + int(0.30 * h)
    # estimate left and right cheek x positions as small inset from bbox edges
    cheek_left_x = x + int(0.12 * w)
    cheek_right_x = x + int(0.88 * w)
    cheekbone_width_est = float(cheek_right_x - cheek_left_x)

    # jaw width estimate:
    # approximate at ~85% down from top (near jawline)
    jaw_y = y + int(0.85 * h)
    jaw_left_x = x + int(0.18 * w)
    jaw_right_x = x + int(0.82 * w)
    jaw_width_est = float(jaw_right_x - jaw_left_x)

    # forehead width estimate at ~12% down from top
    forehead_y = y + int(0.12 * h)
    forehead_left_x = x + int(0.20 * w)
    forehead_right_x = x + int(0.80 * w)
    forehead_width_est = float(forehead_right_x - forehead_left_x)

    # normalized ratios
    cheek_to_face_ratio = cheekbone_width_est / (face_width + 1e-9)
    jaw_to_cheek_ratio = jaw_width_est / (cheekbone_width_est + 1e-9)
    forehead_to_cheek_ratio = forehead_width_est / (cheekbone_width_est + 1e-9)

    return {
        "face_width": face_width,
        "face_height": face_height,
        "aspect_ratio": aspect_ratio,
        "eye_distance": eye_distance,
        "cheekbone_width": cheekbone_width_est,
        "jaw_width": jaw_width_est,
        "forehead_width": forehead_width_est,
        "cheek_to_face_ratio": cheek_to_face_ratio,
        "jaw_to_cheek_ratio": jaw_to_cheek_ratio,
        "forehead_to_cheek_ratio": forehead_to_cheek_ratio,
    }


def classify_face_shape_from_measures(measures: Dict[str, float]) -> str:
    """
    Heuristic classification based on ratios.
    These heuristics are intentionally simple and conservative.
    """
    ar = measures["aspect_ratio"]
    jaw_cheek = measures["jaw_to_cheek_ratio"]
    fore_cheek = measures["forehead_to_cheek_ratio"]
    cheek_ratio = measures["cheek_to_face_ratio"]

    # Basic meaning of ratios:
    # - ar > 1.2 => long face (oblong)
    # - ar ~ 1.0 => balanced (square/oval/round)
    # - cheek_ratio near 0.8..0.9 means wide cheekbones relative to face width
    # - jaw_cheek < 0.9 means jaw narrower than cheekbones (possible heart/diamond)
    # - forehead wider than jaw -> heart
    # - cheekbones widest -> diamond
    # - jaw & forehead similar and angle strong -> square
    # - short height relative to width -> round

    # Apply ordered rules (most specific first)
    # 1) Oblong / rectangular: clearly tall
    if ar >= 1.25:
        return "Oblong"

    # 2) Round: low aspect ratio and cheekbones/width high
    if ar <= 0.95 and cheek_ratio >= 0.76:
        return "Round"

    # 3) Square: face height ~ width and jaw approx equal to cheek
    if 0.95 < ar < 1.15 and 0.92 <= jaw_cheek <= 1.08:
        return "Square"

    # 4) Heart: forehead wider than cheek/jaw and jaw noticeably narrower
    if fore_cheek > 1.05 and jaw_cheek < 0.90:
        return "Heart"

    # 5) Diamond: cheekbones are the widest region and forehead + jaw narrower
    # We check jaw_cheek < 0.95 and fore_cheek < 0.95 and cheek_ratio relatively high
    if jaw_cheek < 0.95 and fore_cheek < 0.95 and cheek_ratio >= 0.70:
        return "Diamond"

    # 6) Oval: moderate ratios (fallback, common)
    return "Oval"


def style_recommendations_for_shape(shape: str) -> Dict[str, str]:
    """
    High-level hairstyle and eyewear advice by shape.
    """
    recs = {}
    if shape == "Oval":
        recs["hair"] = "Most styles suit an oval face; try layers or soft waves."
        recs["eyewear"] = "Almost any frame shape works; square or geometric frames add contrast."
    elif shape == "Round":
        recs["hair"] = "Add height/volume on top and avoid heavy bangs; long layers elongate the face."
        recs["eyewear"] = "Angular frames (rectangular) help add definition."
    elif shape == "Square":
        recs["hair"] = "Soften the jawline with textured layers and side-swept bangs."
        recs["eyewear"] = "Round or oval frames soften strong jawlines."
    elif shape == "Heart":
        recs["hair"] = "Chin-length bobs or side-parted styles balance a narrower jaw."
        recs["eyewear"] = "Bottom-heavy frames or aviators complement heart shapes."
    elif shape == "Diamond":
        recs["hair"] = "Try soft chin-length cuts or styles that add width at the forehead/chin."
        recs["eyewear"] = "Cat-eye or rimless frames balance cheekbones."
    elif shape == "Oblong":
        recs["hair"] = "Avoid too much height; use bangs and soft curls to shorten the face visually."
        recs["eyewear"] = "Tall frames or styles with decorative temples help add width."
    else:
        recs["hair"] = "Experiment with styles; consider consulting a stylist."
        recs["eyewear"] = "Try different frame shapes to find contrast with your face."
    return recs


def analyze_face_shape(image_path: str, visualize: bool = False) -> Dict[str, Any]:
    """
    Main entry point: analyze an image and return results dict.
    If visualize=True, a debug image will be written next to the input file.
    """
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise IOError(f"Failed to load image (cv2) at: {image_path}")

    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    face_bbox = detect_primary_face(img_gray)
    debug_boxes = []

    if face_bbox is None:
        # fallback: try a larger search (scale parameters) or return early
        face_bbox = _face_cascade.detectMultiScale(img_gray, scaleFactor=1.05, minNeighbors=3, minSize=(60, 60))
        if isinstance(face_bbox, (list, tuple)) and len(face_bbox) > 0:
            x, y, w, h = face_bbox[0]
            face_bbox = (int(x), int(y), int(w), int(h))
        else:
            return {"error": "No face detected"}

    debug_boxes.append(("face", face_bbox))

    eyes = detect_eyes_in_face(img_gray, face_bbox)
    for e in eyes:
        debug_boxes.append(("eye", e))

    measures = estimate_measures_from_bbox_and_eyes(face_bbox, eyes)
    shape = classify_face_shape_from_measures(measures)
    recs = style_recommendations_for_shape(shape)

    result = {
        "image_path": image_path,
        "face_bbox": {"x": int(face_bbox[0]), "y": int(face_bbox[1]), "w": int(face_bbox[2]), "h": int(face_bbox[3])},
        "eyes_detected": [{"x": int(e[0]), "y": int(e[1]), "w": int(e[2]), "h": int(e[3])} for e in eyes],
        "measures": {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in measures.items()},
        "face_shape": shape,
        "recommendations": recs,
    }

    # Optional visualization
    if visualize:
        vis = img_bgr.copy()
        # draw face bbox
        x, y, w, h = face_bbox
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(vis, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 2)
        # write shape text
        cv2.putText(vis, f"Shape: {shape}", (x, max(10, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        out_path = os.path.splitext(image_path)[0] + "_face_shape_debug.jpg"
        cv2.imwrite(out_path, vis)
        result["debug_image"] = out_path

    return result


def parse_args_and_run():
    p = argparse.ArgumentParser(description="Face shape detection (heuristic) using OpenCV")
    p.add_argument("--image", "-i", required=True, help="Path to input image (frontal face recommended)")
    p.add_argument("--out", "-o", help="Optional JSON output path")
    p.add_argument("--visualize", "-v", action="store_true", help="Write debug visualization image")
    args = p.parse_args()

    try:
        res = analyze_face_shape(args.image, visualize=args.visualize)
    except Exception as e:
        print("Error:", e)
        return

    if args.out:
        try:
            with open(args.out, "w") as f:
                json.dump(res, f, indent=2)
            print(f"Wrote JSON to: {args.out}")
        except Exception as e:
            print("Failed to write JSON:", e)

    # pretty print summary
    if "error" in res:
        print("Error:", res["error"])
        return
    print("=== Face Shape Analysis ===")
    print("Image:", res["image_path"])
    fb = res["face_bbox"]
    print(f"Face bbox: x={fb['x']} y={fb['y']} w={fb['w']} h={fb['h']}")
    print("Detected eyes:", len(res["eyes_detected"]))
    m = res["measures"]
    print(f"Face width: {m['face_width']:.1f}px  height: {m['face_height']:.1f}px  aspect_ratio(h/w): {m['aspect_ratio']:.2f}")
    print("Cheekbone width est: {:.1f}px  Jaw width est: {:.1f}px".format(m["cheekbone_width"], m["jaw_width"]))
    print("Inferred face shape:", res["face_shape"])
    print("Recommendations:")
    for k, v in res["recommendations"].items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    parse_args_and_run()
