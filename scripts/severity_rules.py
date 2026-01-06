def estimate_severity(box, confidence, damage_type, img_shape):
    x1, y1, x2, y2 = box
    h, w = img_shape[:2]

    # Base severity from size
    area_ratio = ((x2 - x1) * (y2 - y1)) / (h * w)
    score = area_ratio

    # CNN confidence adjustment
    if confidence > 0.7:
        score += 0.02

    # Damage-type-based weights
    damage_weights = {
        "pothole": 0.03,
        "alligator_crack": 0.025,
        "longitudinal_crack": 0.02,
        "transverse_crack": 0.02,
        "block_crack": 0.02,
        "repair": 0.01
    }

    score += damage_weights.get(damage_type.lower(), 0.015)

    # Final severity decision
    if score < 0.02:
        return "Low"
    elif score < 0.06:
        return "Medium"
    else:
        return "High"
