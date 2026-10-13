import numpy as np

def calculate_endplate_angle(p1, p2):
    """Calculates endplate orientation relative to horizontal axis in degrees."""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    angle_rad = np.arctan2(dy, dx)
    return np.degrees(angle_rad)

def compute_cobb_angle(angle1, angle2):
    """Computes the primary Cobb angle between two endplate orientation angles."""
    return abs(angle1 - angle2)

def classify_severity(cobb_angle):
    """Categorizes scoliosis severity based on Cobb angle thresholds."""
    if cobb_angle < 10.0:
        return "Physiological Alignment / Normal", "green"
    elif 10.0 <= cobb_angle < 25.0:
        return "Mild AIS (Observe)", "yellow"
    elif 25.0 <= cobb_angle < 40.0:
        return "Moderate AIS (Orthotic Bracing)", "orange"
    else:
        return "Severe AIS (Surgical Evaluation)", "red"
