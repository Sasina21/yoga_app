# pip install mediapipe opencv-python pandas numpy
import os
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from typing import Optional, Dict, List
from pose_math import PoseMath

# ====== CONFIG ======
CSV_IN        = "dataset_manifest.csv"   # ต้องมีคอลัมน์ 'img_path'
CSV_OUT       = "keypoints_out0.4_dir.csv"
MIN_DET_CONF  = 0.50                     # เกณฑ์ตรวจจับทั้งภาพ (Pose)
VIS_PASS_THR  = 0.50                     # คีย์พอยต์ "ผ่าน" เมื่อ visibility >= ค่านี้
OK_THR        = 0.80                     # coverage >= 0.8 → keypoint_ok = 1
NUM_JOINTS    = 33
SAVE_XY       = True                     # True=เก็บ x_i,y_i,vis_i / False=เก็บเฉพาะ vis_i
TEST_RATIO = 0.4
RAND_SEED  = 42
STRAT_KEYS = ["pose", "view", "risk_level"]

def assign_split_stratified(df: pd.DataFrame,
                            keys: List[str] = ["pose","view","risk_level"],
                            test_ratio: float = 0.20,
                            seed: int = 42) -> pd.DataFrame:
    """
    ใส่คอลัมน์ 'split' = 'train'/'test' โดยแยกภายในแต่ละคอมโบ (pose,view,risk_level)
    กฎ: ถ้าในกลุ่มมี n>=2 จะกันอย่างน้อย 1 ไป test
    """
    rng = np.random.default_rng(seed)
    out = df.copy()

    # เผื่อไฟล์ไม่มีบางคอลัมน์
    for k in keys:
        if k not in out.columns:
            out[k] = "unknown"

    out["split"] = "train"

    # รวม 3 ตัวเป็น strata เดียว
    groups = out.groupby(keys, dropna=False).groups
    idx_test = []
    for _, idx in groups.items():
        idx = list(idx)
        n = len(idx)
        if n <= 1:
            continue

        # 80/20 แบบปัดใกล้สุด และบังคับ >=1 แต่ไม่เท่ากับ n
        n_test = max(1, int(round(n * test_ratio)))
        if n_test >= n:
            n_test = max(1, n - 1)

        chosen = rng.choice(idx, size=n_test, replace=False)
        idx_test.extend(chosen)

    out.loc[idx_test, "split"] = "test"
    return out

# ====== ดึงคีย์พอยต์จากภาพ 1 ใบ ======
def extract_pose_keypoints(img_bgr: np.ndarray) -> Optional[np.ndarray]:
    """
    เปิด Pose ใหม่สำหรับภาพนี้ → คืน (33,3): [x, y, visibility] (normalized 0..1)
    หรือ None ถ้าไม่พบ landmarks
    """
    mp_pose = mp.solutions.pose
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=1,
        min_detection_confidence=MIN_DET_CONF
    ) as pose:
        res = pose.process(img_rgb)
        if not res.pose_landmarks:
            return None
        coords = np.array([[lm.x, lm.y, lm.visibility] for lm in res.pose_landmarks.landmark],
                          dtype=np.float32)
        return coords

def compute_coverage(vis: np.ndarray, thr: float = VIS_PASS_THR) -> float:
    """สัดส่วนของคีย์พอยต์ที่ visibility >= thr"""
    if vis.size == 0:
        return 0.0
    return float((vis >= thr).mean())

def ensure_columns(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """รับประกันว่าคอลัมน์ที่ต้องการมีอยู่ใน df (ถ้าไม่มีก็สร้างค่าว่าง)"""
    for c in cols:
        if c not in df.columns:
            # สร้างคอลัมน์ว่าง (NaN) เพื่อให้สคริปต์ลงค่าทับได้
            df[c] = np.nan
    return df

# ====== ประมวลผลทีละภาพ ======
def process_one_image(img_path: str) -> Dict:
    rec: Dict = {
        "img_path": img_path,
        "keypoint_coverage": 0.0,
        "keypoint_ok": 0,
        "error_reason": ""  # คอลัมน์ใหม่ที่ช่วย debug (ถ้าไม่อยากเก็บก็ลบได้)
    }

    # 1) มีไฟล์ไหม
    if not os.path.exists(img_path):
        print(f"  ❌ file_not_found: {img_path}")
        rec["error_reason"] = "file_not_found"
        for j in range(NUM_JOINTS):
            if SAVE_XY:
                rec[f"x_{j}"] = np.nan
                rec[f"y_{j}"] = np.nan
            # rec[f"vis_{j}"] = np.nan
        return rec

    # 2) อ่านภาพ
    img = cv2.imread(img_path)
    if img is None:
        print(f"  ❌ imread_failed: {img_path}")
        rec["error_reason"] = "imread_failed"
        for j in range(NUM_JOINTS):
            if SAVE_XY:
                rec[f"x_{j}"] = np.nan
                rec[f"y_{j}"] = np.nan
            # rec[f"vis_{j}"] = np.nan
        return rec

    # 3) ดึงคีย์พอยต์
    coords = extract_pose_keypoints(img)
    if coords is None:
        print(f"  ⚠️  no_landmarks: {img_path}")
        rec["error_reason"] = "no_landmarks"
        for j in range(NUM_JOINTS):
            if SAVE_XY:
                rec[f"x_{j}"] = np.nan
                rec[f"y_{j}"] = np.nan
            # rec[f"vis_{j}"] = np.nan
        return rec

    # 4) coverage & ok
    vis = coords[:, 2]
    coverage = compute_coverage(vis, VIS_PASS_THR)
    rec["keypoint_coverage"] = coverage
    rec["keypoint_ok"] = 1 if coverage >= OK_THR else 0

    # 5) เก็บคีย์พอยต์
    for j in range(NUM_JOINTS):
        if SAVE_XY:
            rec[f"x_{j}"] = float(coords[j, 0])
            rec[f"y_{j}"] = float(coords[j, 1])
        # rec[f"vis_{j}"] = float(coords[j, 2])

    # 6) angle เฉพาะ 8 จุดที่กำหนด
    rec["angle_11_13_15"] = PoseMath.calculate_angle(
        {'x': float(coords[11, 0]), 'y': float(coords[11, 1])},
        {'x': float(coords[13, 0]), 'y': float(coords[13, 1])},
        {'x': float(coords[15, 0]), 'y': float(coords[15, 1])},
    )

    rec["angle_12_14_16"] = PoseMath.calculate_angle(
        {'x': float(coords[12, 0]), 'y': float(coords[12, 1])},
        {'x': float(coords[14, 0]), 'y': float(coords[14, 1])},
        {'x': float(coords[16, 0]), 'y': float(coords[16, 1])},
    )

    rec["angle_23_25_27"] = PoseMath.calculate_angle(
        {'x': float(coords[23, 0]), 'y': float(coords[23, 1])},
        {'x': float(coords[25, 0]), 'y': float(coords[25, 1])},
        {'x': float(coords[27, 0]), 'y': float(coords[27, 1])},
    )

    rec["angle_24_26_28"] = PoseMath.calculate_angle(
        {'x': float(coords[24, 0]), 'y': float(coords[24, 1])},
        {'x': float(coords[26, 0]), 'y': float(coords[26, 1])},
        {'x': float(coords[28, 0]), 'y': float(coords[28, 1])},
    )

    rec["angle_11_23_25"] = PoseMath.calculate_angle(
        {'x': float(coords[11, 0]), 'y': float(coords[11, 1])},
        {'x': float(coords[23, 0]), 'y': float(coords[23, 1])},
        {'x': float(coords[25, 0]), 'y': float(coords[25, 1])},
    )

    rec["angle_12_24_26"] = PoseMath.calculate_angle(
        {'x': float(coords[12, 0]), 'y': float(coords[12, 1])},
        {'x': float(coords[24, 0]), 'y': float(coords[24, 1])},
        {'x': float(coords[26, 0]), 'y': float(coords[26, 1])},
    )

    rec["angle_13_11_23"] = PoseMath.calculate_angle(
        {'x': float(coords[13, 0]), 'y': float(coords[13, 1])},
        {'x': float(coords[11, 0]), 'y': float(coords[11, 1])},
        {'x': float(coords[23, 0]), 'y': float(coords[23, 1])},
    )

    rec["angle_14_12_24"] = PoseMath.calculate_angle(
        {'x': float(coords[14, 0]), 'y': float(coords[14, 1])},
        {'x': float(coords[12, 0]), 'y': float(coords[12, 1])},
        {'x': float(coords[24, 0]), 'y': float(coords[24, 1])},
    )

    # 7) dir
    for a, b in sorted(PoseMath.CUSTOM_POSE_CONNECTIONS):
        rec[f"dir_x_{a}_{b}"] = float(coords[a, 0]) - float(coords[b, 0])
        rec[f"dir_y_{a}_{b}"] = float(coords[a, 1]) - float(coords[b, 1])
            
    # direction_shoulder 
    rec[f"dir_x_{11}_{12}"] = float(coords[11, 0]) - float(coords[12, 0])
    rec[f"dir_y_{11}_{12}"] = float(coords[11, 1]) - float(coords[12, 1])
    # direction_waist 
    rec[f"dir_x_{23}_{24}"] = float(coords[23, 0]) - float(coords[24, 0])
    rec[f"dir_y_{23}_{24}"] = float(coords[23, 1]) - float(coords[24, 1])

    print(f"  ✅ coverage={coverage:.2f}, keypoint_ok={rec['keypoint_ok']}")
    return rec

# ====== Runner ======
def run_manifest(csv_in: str = CSV_IN, csv_out: str = CSV_OUT):
    # โหลด csv
    try:
        df = pd.read_csv(csv_in)
    except Exception as e:
        raise SystemExit(f"[FATAL] อ่านไฟล์ CSV_IN ไม่ได้: {csv_in} -> {e}")
    if "img_path" not in df.columns:
        raise SystemExit("[FATAL] ไม่พบคอลัมน์ 'img_path' ใน CSV")

    # ===== ทำ stratified split โดยใช้คอมโบ (pose, view, risk_level) =====
    df = assign_split_stratified(
        df,
        keys=STRAT_KEYS,
        test_ratio=TEST_RATIO,
        seed=RAND_SEED
    )

    total = len(df)

    # ให้แน่ใจว่ามีคอลัมน์ที่จะเขียนทับได้อยู่แล้ว
    must_have = [
        "keypoint_coverage",
        "keypoint_ok",
        "notes"
    ]
    df = ensure_columns(df, must_have)

    out_rows = []
    c_ok = c_nf = c_rd = c_nl = 0

    for i, row in df.iterrows():
        img_path = str(row["img_path"]).strip()
        print(f"[{i+1}/{total}] {img_path}", flush=True)
        rec = process_one_image(img_path)

        # นำ metadata เดิมทั้งหมด “จาก CSV เดิม” มาใส่คืนก่อน
        # (ค่าที่คำนวณใหม่ของเรา keypoint_coverage/keypoint_ok จะทับคอลัมน์เดิมเอง)
        merged = dict(row)              # เริ่มจากของเดิม (รักษา order ทีหลัง)
        merged.update(rec)              # เติม/ทับด้วยค่าที่เราคำนวณ
        out_rows.append(merged)

        # นับสรุป
        if   rec["error_reason"] == "":               c_ok += 1
        elif rec["error_reason"] == "file_not_found": c_nf += 1
        elif rec["error_reason"] == "imread_failed":  c_rd += 1
        elif rec["error_reason"] == "no_landmarks":   c_nl += 1

    out_df = pd.DataFrame(out_rows)

    # ===== จัดเรียงคอลัมน์ตามที่กำหนด =====
    original_cols = list(pd.read_csv(csv_in, nrows=0).columns)

    preferred_front = [
        "id", "full_path", "img_path", "pose", "risk_level", "view",
        "subject_id", "split", "label_conf",
        "keypoint_coverage", "keypoint_ok", "notes"
    ]
    front_csv_cols = [c for c in preferred_front if c in original_cols]
    other_csv_cols = [c for c in original_cols if c not in front_csv_cols]
    csv_part_ordered = front_csv_cols + other_csv_cols

    x_cols   = [f"x_{j}" for j in range(NUM_JOINTS)] if SAVE_XY else []
    y_cols   = [f"y_{j}" for j in range(NUM_JOINTS)] if SAVE_XY else []

    extras = []
    if "error_reason" in out_df.columns and "error_reason" not in csv_part_ordered:
        extras.append("error_reason")

    ordered_cols = csv_part_ordered + extras + x_cols + y_cols
    ordered_cols = [c for c in ordered_cols if c in out_df.columns]
    missing_cols = [c for c in out_df.columns if c not in ordered_cols]
    final_cols = ordered_cols + missing_cols
    out_df = out_df[final_cols]

    # ===== บันทึก =====
    out_df.to_csv(CSV_OUT, index=False)

    # ===== สรุปตอนท้าย (พิมพ์ทีเดียว) =====
    print("\n=== Summary ===", flush=True)
    print(f"Saved          : {CSV_OUT}  (rows={len(out_df)})", flush=True)
    print(f"  ok_rows        : {c_ok}", flush=True)
    print(f"  file_not_found : {c_nf}", flush=True)
    print(f"  imread_failed  : {c_rd}", flush=True)
    print(f"  no_landmarks   : {c_nl}", flush=True)

    print("\n=== Config ===", flush=True)
    print(f"CSV_IN      : {csv_in}", flush=True)
    print(f"CSV_OUT     : {csv_out}", flush=True)
    print(f"rows        : {total}", flush=True)
    print(f"MIN_DET_CONF= {MIN_DET_CONF} | VIS_PASS_THR= {VIS_PASS_THR} | OK_THR= {OK_THR}", flush=True)
    print(f"SAVE_XY     : {SAVE_XY}", flush=True)

    print("\n=== Split summary (stratified by pose, view, risk_level) ===", flush=True)
    try:
        print(df['split'].value_counts(), flush=True)
        print("\nBy pose:\n", pd.crosstab(df['pose'], df['split']), flush=True)
        print("\nBy view:\n", pd.crosstab(df['view'], df['split']), flush=True)
        print("\nBy risk:\n", pd.crosstab(df['risk_level'], df['split']), flush=True)
    except Exception as e:
        print(f"[WARN] cannot print split summary: {e}", flush=True)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    run_manifest()
