# skin_tone.py
"""
Lightweight skin tone & undertone analyzer using Pillow (no OpenCV required).

Usage:
    python skin_tone.py --image PATH_TO_IMAGE [--bbox x,y,w,h] [--side left|right] [--out result.json]

Examples:
    python skin_tone.py --image face.jpg
    python skin_tone.py --image face.jpg --bbox 100,120,300,300 --side right --out result.json

Dependencies:
    pip install pillow

Notes:
    - This is an approximate, heuristic-based analyzer. Lighting/camera/makeup affect results.
    - For best results provide a neutral-lit frontal photo or pass a face bbox (if you have landmarks).
"""
from PIL import Image
import argparse
import json
import math
import os
from typing import Optional, Tuple, Dict, Any

def parse_bbox(bbox_str: str) -> Optional[Tuple[int,int,int,int]]:
    try:
        parts = [int(p) for p in bbox_str.split(",")]
        if len(parts) == 4:
            return tuple(parts)
    except Exception:
        pass
    return None

def crop_patch_from_bbox(img: Image.Image, bbox: Tuple[int,int,int,int], side: str="left", frac: float=0.12) -> Image.Image:
    x, y, w, h = bbox
    img_w, img_h = img.size
    patch_w = max(8, int(w * frac))
    patch_h = max(8, int(h * frac * 0.7))
    # cheek offsets relative to face bbox
    if side.lower().startswith("l"):
        cx = x + int(w * 0.18)
    else:
        cx = x + int(w * 0.62)
    cy = y + int(h * 0.48)
    x1 = max(0, cx - patch_w // 2)
    y1 = max(0, cy - patch_h // 2)
    x2 = min(img_w, x1 + patch_w)
    y2 = min(img_h, y1 + patch_h)
    patch = img.crop((x1, y1, x2, y2))
    if patch.size[0] == 0 or patch.size[1] == 0:
        # fallback to center of bbox
        cx = x + w // 2
        cy = y + int(h * 0.6)
        x1 = max(0, cx - patch_w // 2)
        y1 = max(0, cy - patch_h // 2)
        x2 = min(img_w, x1 + patch_w)
        y2 = min(img_h, y1 + patch_h)
        patch = img.crop((x1, y1, x2, y2))
    return patch

def crop_center_patch(img: Image.Image, frac: float=0.12) -> Image.Image:
    w, h = img.size
    pw = max(8, int(min(w, h) * frac))
    ph = pw
    cx = w // 2
    cy = int(h * 0.55)  # slightly lower than center to approximate cheek/neck
    x1 = max(0, cx - pw // 2)
    y1 = max(0, cy - ph // 2)
    x2 = min(w, x1 + pw)
    y2 = min(h, y1 + ph)
    return img.crop((x1, y1, x2, y2))

def mean_rgb_from_patch(patch: Image.Image) -> Tuple[int,int,int]:
    # Ensure RGB
    patch_rgb = patch.convert("RGB")
    pixels = list(patch_rgb.getdata())
    if not pixels:
        return (0,0,0)
    # sum channels
    r_sum = g_sum = b_sum = 0
    count = 0
    for r,g,b in pixels:
        r_sum += r
        g_sum += g
        b_sum += b
        count += 1
    # careful rounding digit-by-digit
    mean_r = int(math.floor((r_sum / count) + 0.5))
    mean_g = int(math.floor((g_sum / count) + 0.5))
    mean_b = int(math.floor((b_sum / count) + 0.5))
    return (mean_r, mean_g, mean_b)

def perceived_luminance(r: int, g: int, b: int) -> float:
    return 0.2126 * r + 0.7152 * g + 0.0722 * b

def classify_skin_tone(lum: float) -> str:
    # heuristic thresholds (tune with dataset if available)
    if lum >= 190:
        return "Fair"
    if 160 <= lum < 190:
        return "Light"
    if 115 <= lum < 160:
        return "Medium"
    return "Deep"

def detect_undertone(r:int, g:int, b:int) -> str:
    # Normalize to reduce brightness effects
    total = max(1, r + g + b)
    rn, gn, bn = r / total, g / total, b / total
    diff_rb = r - b
    # simple heuristics
    if diff_rb > 12 and rn > bn:
        return "Warm"
    if (b - r) > 10 and bn > rn:
        return "Cool"
    if abs(r - g) < 8 and abs(g - b) < 8:
        return "Neutral"
    # fallback using hue approximation from RGB
    # convert to HSV-ish hue
    mx = max(r,g,b)
    mn = min(r,g,b)
    d = mx - mn
    if d == 0:
        hue = 0
    else:
        if mx == r:
            hue = ((g - b) / d) % 6
        elif mx == g:
            hue = ((b - r) / d) + 2
        else:
            hue = ((r - g) / d) + 4
        hue = hue * 60
        if hue < 0:
            hue += 360
    if 20 <= hue <= 60:
        return "Warm"
    if 180 <= hue <= 300:
        return "Cool"
    return "Neutral"

def cosmetic_recommendations(tone: str, undertone: str) -> Dict[str, str]:
    if undertone == "Warm":
        base = "Yellow/Golden base"
    elif undertone == "Cool":
        base = "Pink/Neutral-pink base"
    else:
        base = "Neutral base (mix of yellow & pink)"
    if tone == "Fair":
        depth = "Very light to Light shades"
        lipstick = "Soft pinks, peaches, coral"
    elif tone == "Light":
        depth = "Light shades"
        lipstick = "Rose, mauve, warm pinks"
    elif tone == "Medium":
        depth = "Medium shades"
        lipstick = "Berry, brick red, warm mauves"
    else:
        depth = "Deep shades"
        lipstick = "Deep berries, plums, bold reds"
    return {
        "foundation_base": base,
        "foundation_depth": depth,
        "lipstick_recommendation": lipstick,
        "note": "General guidelines — test swatches under neutral daylight."
    }

def analyze_image(image_path: str, bbox: Optional[Tuple[int,int,int,int]]=None, side: str="left") -> Dict[str,Any]:
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    img = Image.open(image_path)
    # choose patch
    if bbox:
        patch = crop_patch_from_bbox(img, bbox, side=side)
        mode = "bbox_patch"
    else:
        patch = crop_center_patch(img)
        mode = "center_patch"
    mean_r, mean_g, mean_b = mean_rgb_from_patch(patch)
    lum = perceived_luminance(mean_r, mean_g, mean_b)
    tone = classify_skin_tone(lum)
    undertone = detect_undertone(mean_r, mean_g, mean_b)
    recs = cosmetic_recommendations(tone, undertone)
    return {
        "image_path": image_path,
        "mode": mode,
        "mean_rgb": {"r": mean_r, "g": mean_g, "b": mean_b},
        "luminance": round(float(lum), 2),
        "skin_tone": tone,
        "undertone": undertone,
        "recommendations": recs
    }

def main():
    p = argparse.ArgumentParser(description="Lightweight skin tone analyzer (Pillow only)")
    p.add_argument("--image", "-i", required=True, help="Path to input image")
    p.add_argument("--bbox", "-b", help="Optional face bbox as x,y,w,h", default=None)
    p.add_argument("--side", "-s", help="Which cheek to sample: left or right", default="left")
    p.add_argument("--out", "-o", help="Save JSON output to path", default=None)
    args = p.parse_args()
    bbox = parse_bbox(args.bbox) if args.bbox else None
    try:
        res = analyze_image(args.image, bbox=bbox, side=args.side)
    except Exception as e:
        print("Error:", e)
        return
    print("=== Skin Tone Analysis ===")
    print(f"Image: {res['image_path']}")
    m = res["mean_rgb"]
    print(f"Mean RGB (patch): R={m['r']} G={m['g']} B={m['b']}")
    print(f"Luminance: {res['luminance']}")
    print(f"Skin tone category: {res['skin_tone']}")
    print(f"Undertone: {res['undertone']}")
    print("Recommendations:")
    for k,v in res["recommendations"].items():
        print(f"  {k}: {v}")
    if args.out:
        try:
            with open(args.out, "w") as f:
                json.dump(res, f, indent=2)
            print(f"Saved JSON to: {args.out}")
        except Exception as e:
            print("Failed to save JSON:", e)

if __name__ == "__main__":
    main()
