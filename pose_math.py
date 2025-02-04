import math

class PoseMath:
    
    CUSTOM_POSE_CONNECTIONS = frozenset([
        # # body
        # (11, 12),  # ไหล่ซ้ายเชื่อมกับไหล่ขวา
        # (23, 24),  # สะโพกซ้ายเชื่อมกับสะโพกขวา

        # left arm
        (11, 13), (13, 15),  
        (11, 23),

        # right arm
        (12, 14), (14, 16),  
        (12, 24),

        # left leg
        (23, 25), (25, 27),

        # right leg
        (24, 26), (26, 28)
    ])

    @staticmethod
    def calculate_direction(a, b):
        return {
            'x': b['x'] - a['x'],
            'y': b['y'] - a['y'],
            'z': b['z'] - a['z']
        }

    @staticmethod
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

    @staticmethod
    def calculate_distance(x1, y1, x2, y2, z1=0, z2=0):
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)

    @staticmethod
    def calculate_twist_angle(landmarks):

        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]
        left_hip = landmarks[23]
        right_hip = landmarks[24]
        
        shoulder_vector = [
            right_shoulder.x - left_shoulder.x,
            right_shoulder.y - left_shoulder.y,
            right_shoulder.z - left_shoulder.z
        ]
        
        hip_vector = [
            right_hip.x - left_hip.x,
            right_hip.y - left_hip.y,
            right_hip.z - left_hip.z
        ]
        
        # คำนวณ dot product และ magnitude
        dot_product = sum(a * b for a, b in zip(shoulder_vector, hip_vector))
        shoulder_magnitude = math.sqrt(sum(a**2 for a in shoulder_vector))
        hip_magnitude = math.sqrt(sum(b**2 for b in hip_vector))
        
        if shoulder_magnitude == 0 or hip_magnitude == 0:
            twist_angle = 0
        else:
            cos_angle = dot_product / (shoulder_magnitude * hip_magnitude)
            cos_angle = max(-1.0, min(1.0, cos_angle))  # Clamp ค่าให้อยู่ในช่วง [-1, 1]
            twist_angle = math.degrees(math.acos(cos_angle)) 

        center_x = (left_shoulder.x + right_shoulder.x) / 2
        center_y = (left_shoulder.y + right_shoulder.y) / 2
        center_z = (left_shoulder.z + right_shoulder.z) / 2

        return twist_angle, center_x, center_y, center_z