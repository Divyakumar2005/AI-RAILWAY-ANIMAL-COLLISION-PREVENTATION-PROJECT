"""
generate_sample_images.py - Generates realistic simulated track surveillance camera
images (Elephant in forest corridor, Cattle on track, and Clear track) for evaluation.
"""

import os
from PIL import Image, ImageDraw, ImageFont

def ensure_sample_dir():
    os.makedirs("sample_images", exist_ok=True)

def draw_railway_scene(filename, scene_type="elephant"):
    # 640x360 16:9 surveillance camera frame
    w, h = 640, 360
    img = Image.new("RGB", (w, h), color=(15, 23, 42))
    draw = ImageDraw.Draw(img)

    # 1. Sky & Forest/Background
    if scene_type == "elephant":
        # Dense forest background (deep green / misty morning)
        draw.rectangle([0, 0, w, 160], fill=(20, 45, 30))
        for x in range(0, w, 40):
            draw.polygon([(x, 160), (x+20, 80), (x+40, 160)], fill=(12, 35, 22))
    elif scene_type == "cattle":
        # Agricultural field (golden-brown / open pasture)
        draw.rectangle([0, 0, w, 160], fill=(60, 50, 30))
        draw.rectangle([0, 120, w, 160], fill=(45, 65, 35))
    else: # clear
        # Open railway horizon
        draw.rectangle([0, 0, w, 160], fill=(30, 41, 59))
        draw.ellipse([450, 40, 500, 90], fill=(226, 232, 240)) # morning moon/sun

    # 2. Track Ballast (Gravel embankment)
    draw.polygon([(220, 160), (420, 160), (600, h), (40, h)], fill=(71, 85, 105))

    # 3. Railway Sleepers (Ties)
    for y in range(170, h, 14):
        scale = (y - 160) / (h - 160)
        x_left = int(250 - scale * 180)
        x_right = int(390 + scale * 180)
        draw.line([(x_left, y), (x_right, y)], fill=(100, 116, 139), width=3)

    # 4. Steel Rails (Converging perspective)
    # Left Rail
    draw.line([(280, 160), (140, h)], fill=(203, 213, 225), width=4)
    draw.line([(282, 160), (142, h)], fill=(241, 245, 249), width=2)
    # Right Rail
    draw.line([(360, 160), (500, h)], fill=(203, 213, 225), width=4)
    draw.line([(358, 160), (498, h)], fill=(241, 245, 249), width=2)

    # 5. Draw Target Scene Object
    if scene_type == "elephant":
        # Draw Elephant silhouette on the track around (280, 180) to (370, 240)
        # Body
        draw.ellipse([270, 185, 360, 235], fill=(55, 65, 81))
        # Head & Trunk
        draw.ellipse([250, 195, 290, 230], fill=(50, 60, 75))
        draw.line([(260, 220), (250, 250), (262, 255)], fill=(50, 60, 75), width=7)
        # Ears
        draw.ellipse([275, 190, 295, 215], fill=(40, 50, 65))
        # Legs
        draw.rectangle([280, 225, 295, 258], fill=(45, 55, 70))
        draw.rectangle([305, 225, 320, 258], fill=(45, 55, 70))
        draw.rectangle([335, 225, 350, 258], fill=(45, 55, 70))
        # Tusks (White)
        draw.line([(255, 225), (242, 236)], fill=(255, 255, 255), width=2)

        # AI Detection Bounding Box overlay (Neon Red)
        draw.rectangle([235, 175, 375, 265], outline=(239, 68, 68), width=2)
        draw.rectangle([235, 155, 375, 175], fill=(239, 68, 68))
        draw.text((240, 158), "ELEPHANT DETECTED (98.4%)", fill=(255, 255, 255))

    elif scene_type == "cattle":
        # Draw Cow/Cattle silhouette on track
        # Body
        draw.ellipse([290, 205, 355, 245], fill=(220, 220, 220)) # White/spotted cattle
        draw.ellipse([305, 210, 325, 230], fill=(30, 30, 30)) # black patch
        # Head
        draw.ellipse([275, 205, 298, 230], fill=(220, 220, 220))
        # Horns
        draw.line([(280, 205), (275, 195)], fill=(50, 50, 50), width=2)
        draw.line([(288, 205), (292, 195)], fill=(50, 50, 50), width=2)
        # Legs
        draw.rectangle([295, 240, 302, 265], fill=(180, 180, 180))
        draw.rectangle([340, 240, 347, 265], fill=(180, 180, 180))

        # AI Detection Bounding Box overlay (Orange)
        draw.rectangle([268, 190, 365, 270], outline=(249, 115, 22), width=2)
        draw.rectangle([268, 170, 365, 190], fill=(249, 115, 22))
        draw.text((272, 173), "CATTLE ON TRACK (94.1%)", fill=(255, 255, 255))

    else: # clear
        # Signal post on right
        draw.line([(550, 120), (550, 240)], fill=(148, 163, 184), width=4)
        draw.rectangle([542, 90, 558, 125], fill=(15, 23, 42), outline=(148, 163, 184))
        draw.ellipse([546, 94, 554, 102], fill=(16, 185, 129)) # Green aspect light
        # Clear track status badge
        draw.rectangle([220, 180, 420, 220], outline=(16, 185, 129), width=2)
        draw.rectangle([220, 180, 420, 200], fill=(16, 185, 129))
        draw.text((230, 184), "TRACK CLEAR - ALL SENSORS GREEN", fill=(255, 255, 255))

    # 6. Surveillance HUD Overlay Elements (Timestamp, Cam ID, Resolution)
    draw.text((15, 12), "CAM-402W [WALAYAR ELEPHANT CORRIDOR - TOWER 14]", fill=(203, 213, 225))
    draw.text((15, 28), "OPTICAL / THERMAL TELEMETRY FEED | SIL-4", fill=(148, 163, 184))
    draw.text((w - 150, 12), "LIVE 1080p 30FPS", fill=(239, 68, 68))
    draw.ellipse([w - 165, 15, w - 155, 25], fill=(239, 68, 68)) # Red recording dot

    img.save(filename, quality=95)
    print(f"Generated {filename}")

if __name__ == "__main__":
    ensure_sample_dir()
    draw_railway_scene("sample_images/demo_elephant_track.jpg", "elephant")
    draw_railway_scene("sample_images/demo_cattle_track.jpg", "cattle")
    draw_railway_scene("sample_images/demo_clear_track.jpg", "clear")

