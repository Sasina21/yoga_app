import math

CUSTOM_POSE_CONNECTIONS = frozenset([
    (11, 13), (13, 15),  
    (11, 23),
    (12, 14), (14, 16),  
    (12, 24),
    (23, 25), (25, 27),
    (24, 26), (26, 28)
])

def calculate_direction(a, b):
    return {
        'x': b['x'] - a['x'],
        'y': b['y'] - a['y'],
        'z': b['z'] - a['z']
    }

def calculate_angle(a, b, c):
    ba = [(a['x'] - b['x']), (a['y'] - b['y'])]
    bc = [(c['x'] - b['x']), (c['y'] - b['y'])]

    dot_product = sum(ba[i] * bc[i] for i in range(2))
    magnitude_ba = math.sqrt(sum(ba[i] ** 2 for i in range(2)))
    magnitude_bc = math.sqrt(sum(bc[i] ** 2 for i in range(2)))

    if magnitude_ba == 0 or magnitude_bc == 0:
        return 0

    cos_angle = dot_product / (magnitude_ba * magnitude_bc)
    angle = math.acos(max(-1, min(1, cos_angle)))
    return math.degrees(angle)

def calculate_distance(x1, y1, x2, y2, z1=0, z2=0):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)