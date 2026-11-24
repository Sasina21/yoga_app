# ----------------------------
# Buckets for sampling
# ----------------------------
from collections import defaultdict
import torch
import torch.nn.functional as F

def build_pose_risk_buckets(df_):
    buckets = defaultdict(lambda: {0:[],1:[],2:[]})
    # ใช้ "row position" idx = 0..len(df_)-1 เสมอ
    for idx, r in enumerate(df_.itertuples(index=False)):
        pose = str(getattr(r, "pose")).upper()
        risk = int(getattr(r, "risk"))
        buckets[pose][risk].append(idx)

    # เก็บเฉพาะ pose ที่มีครบ 0,1,2
    return {p: g for p, g in buckets.items() if all(len(g[k]) > 0 for k in (0,1,2))}

def build_pose_all_buckets(df_):
    buckets = defaultdict(list)
    for idx, r in enumerate(df_.itertuples(index=False)):
        pose = str(getattr(r, "pose")).upper()
        buckets[pose].append(idx)
    return buckets

# ----------------------------
# Sampling
# ----------------------------

def sample_joint_APN(batch_size, rng, pose_risk, pose_list_fine):

    A, P, N = [], [], []

    for _ in range(batch_size):
        pose = rng.choice(pose_list_fine)

        a  = rng.choice(pose_all[pose])
        p = rng.choice(pose_all[pose])
        A.append(a); P.append(p)

        # ---- coarse negative (different pose) ----
        neg_pose = rng.choice([pp for pp in pose_list_all if pp != pose])
        N.append(rng.choice(pose_all[neg_pose]))

    return (np.array(A), np.array(P), np.array(N))

def sample_joint_APN1N2(batch_size, rng, pose_risk, pose_list_fine):
    A, P, N1, N2 = [], [], [], []

    for _ in range(batch_size):
        pose = rng.choice(pose_list_fine)

        a = rng.choice(pose_all[pose])
        p = rng.choice(pose_all[pose])
        A.append(a); P.append(p)

        # ---- coarse negatives (different from pose AND each other) ----
        neg_candidates = [pp for pp in pose_list_all if pp != pose]
        neg_pose1, neg_pose2 = rng.choice(neg_candidates, size=2, replace=False)

        N1.append(rng.choice(pose_all[neg_pose1]))
        N2.append(rng.choice(pose_all[neg_pose2]))

    return np.array(A), np.array(P), np.array(N1), np.array(N2)

def sample_joint_AP1P2N(batch_size, rng, pose_risk, pose_list_fine):
    A, P1, P2, N = [], [], [], []

    for _ in range(batch_size):
        pose = rng.choice(pose_list_fine)

        a  = rng.choice(pose_risk[pose][0])
        p1 = rng.choice(pose_risk[pose][1])
        p2 = rng.choice(pose_risk[pose][2])
        A.append(a); P1.append(p1); P2.append(p2)

        neg_pose = rng.choice([pp for pp in pose_list_all if pp != pose])
        n = rng.choice(pose_all[neg_pose])
        N.append(n)

    return (np.array(A), np.array(P1), np.array(P2), np.array(N))


def sample_joint_HML(batch_size, rng, pose_risk, pose_list_fine):
    H, M, L = [], [], []

    for _ in range(batch_size):
        pose = rng.choice(pose_list_fine)

        h = rng.choice(pose_risk[pose][0])
        m = rng.choice(pose_risk[pose][1])
        l = rng.choice(pose_risk[pose][2])
        H.append(h); M.append(m); L.append(l)


    return (np.array(H), np.array(M), np.array(L))

# ----------------------------
# Losses (with term returns)
# ----------------------------

def coarse_terms_tri(
    z_a, z_p, z_n, *,
    margin_c1
):
    d_ap = torch.sum((z_a - z_p) ** 2, dim=1)
    d_an = torch.sum((z_a - z_n) ** 2, dim=1)
    raw1 = d_ap - d_an + margin_c1
    t1 = F.relu(raw1)

    return (t1, raw1)

def coarse_loss_tri(
    z_a, z_p, z_n, *,
    margin_c1,
):
    t1,raw1 = coarse_terms_tri(
        z_a, z_p, z_n,
        margin_c1=margin_c1
    )
    return t1, (t1, raw1)

def coarse_loss_contrastive(z_a, z_p, z_n, *, margin_c1, eps=1e-9):
    d_pos2 = torch.sum((z_a - z_p) ** 2, dim=1)  # squared dist
    d_neg  = torch.sqrt(torch.sum((z_a - z_n) ** 2, dim=1) + eps)  # euclid

    L_pos  = d_pos2
    L_neg  = F.relu(margin_c1 - d_neg) ** 2

    return (L_pos + L_neg).mean()

def coarse_loss_contrastiv_multi_neg(z_a, z_p, z_n1, z_n2, *, margin_c1, eps=1e-9):

    d_pos2 = torch.sum((z_a - z_p) ** 2, dim=1)  # [B]

    # neg use euclidean distance for hinge comparison
    d_n1 = torch.sqrt(torch.sum((z_a - z_n1) ** 2, dim=1) + eps)
    d_n2 = torch.sqrt(torch.sum((z_a - z_n2) ** 2, dim=1) + eps)

    L_pos = d_pos2
    L_neg1 = F.relu(margin_c1 - d_n1) ** 2
    L_neg2 = F.relu(margin_c1 - d_n2) ** 2

    # pos:neg = 1:2
    Lc = (L_pos + L_neg1 + L_neg2).mean()

    neg_active = torch.cat([(d_n1 < margin_c1).float(), (d_n2 < margin_c1).float()]).mean()

    return Lc, neg_active

def coarse_terms_quad(
    z_a, z_p1, z_p2, z_n, *,
    margin_c1sep, margin_c2sep, margin_c3rank, margin_c4rank
):
    d_ap1 = torch.sum((z_a - z_p1) ** 2, dim=1)         # [B]
    d_ap2 = torch.sum((z_a - z_p2) ** 2, dim=1)         # [B]
    d_an = torch.sum((z_a - z_n) ** 2, dim=1)
    d_p1a  = torch.sum((z_p1 - z_a) ** 2, dim=1)
    d_p1p2 = torch.sum((z_p1 - z_p2) ** 2, dim=1)
    d_p2p1 = torch.sum((z_p2 - z_p1) ** 2, dim=1)
    d_p2a = torch.sum((z_p2 - z_a) ** 2, dim=1)

    raw1 = d_ap1 - d_an + margin_c1sep
    raw2 = d_ap2 - d_an + margin_c2sep

    raw3 = d_ap1 - d_ap2 + margin_c3rank

    # raw4 = d_p1a - d_p1p2 + margin_c4rank
    raw4 = d_ap1 - d_p1p2 + margin_c4rank

    t1 = F.relu(raw1)
    t2 = F.relu(raw2)
    t3 = F.relu(raw3)
    t4 = F.relu(raw4)

    return (t1,t2,t3,t4, raw1,raw2,raw3,raw4)

def coarse_loss_quad(
    z_a, z_p1, z_p2, z_n, *,
    margin_c1sep, margin_c2sep, margin_c3rank, margin_c4rank,
    w_c1sep, w_c2sep, w_c3rank, w_c4rank,
):
    t1,t2,t3,t4, raw1,raw2,raw3,raw4= coarse_terms_quad(
        z_a, z_p1, z_p2, z_n,
        margin_c1sep=margin_c1sep, margin_c2sep=margin_c2sep,
        margin_c3rank=margin_c3rank, margin_c4rank=margin_c4rank
    )
    return (w_c1sep*t1 + w_c2sep*t2 + w_c3rank*t3 + w_c4rank*t4), (t1,t2,t3,t4, raw1,raw2,raw3,raw4)

def fine_terms_tri_dual_view(z_h, z_m, z_l, *, margin_f1rank, margin_f2rank, delta=0.05):
    if margin_f2rank is None:
        margin_f2rank = margin_f1rank
    d_hm = torch.sum((z_h - z_m) ** 2, dim=1)
    d_hl = torch.sum((z_h - z_l) ** 2, dim=1)
    raw1 = d_hm - d_hl + margin_f1rank
    t1 = F.relu(raw1)

    d_ml = torch.sum((z_m - z_l)**2, dim=1)
    # raw2 = d_hm - d_ml + margin_f2rank
    raw2 = (d_hm + delta) - d_hl
    t2 = F.relu(raw2)

    # raw3 = d_ml - d_hl + margin_frank
    # t3 = F.relu(raw3)

    return (t1,t2, raw1,raw2)

def fine_loss_tri_dual_view(z_h, z_m, z_l, *, margin_f1rank, margin_f2rank, w_f1rank=1.0, w_f2rank=1.0, delta=0.05):
    t1,t2, raw1,raw2 = fine_terms_tri_dual_view(z_h, z_m, z_l, margin_f1rank=margin_f1rank, margin_f2rank=margin_f2rank, delta=delta)
    return w_f1rank*t1 + w_f2rank*t2, (t1,t2, raw1,raw2)

def fine_terms_tri_sym(z_h, z_m, z_l, *, margin_frank):
    d_hm = torch.sum((z_h - z_m) ** 2, dim=1)
    d_hl = torch.sum((z_h - z_l) ** 2, dim=1)
    raw1 = d_hm - d_hl + margin_frank
    t1 = F.relu(raw1)

    d_lm = torch.sum((z_l - z_m) ** 2, dim=1)
    d_lh = torch.sum((z_l - z_h) ** 2, dim=1)
    raw2 = d_lm - d_lh + margin_frank
    t2 = F.relu(raw2)

    # d_ml = torch.sum((z_m - z_l)**2, dim=1)
    # raw2 = d_hm - d_ml + margin_frank
    # t2 = F.relu(raw2)

    return (t1,t2, raw1,raw2)

def fine_loss_tri_sym(z_h, z_m, z_l, *, margin_frank, w_f1rank=1.0, w_f2rank=1.0):
    t1,t2, raw1,raw2 = fine_terms_tri_sym(z_h, z_m, z_l, margin_frank=margin_frank)
    return w_f1rank*t1 + w_f2rank*t2, (t1,t2, raw1,raw2)

def fine_terms_tri_vanilla(z_h, z_m, z_l, *, margin_frank):
    d_hm = torch.sum((z_h - z_m) ** 2, dim=1)
    d_hl = torch.sum((z_h - z_l) ** 2, dim=1)
    raw1 = d_hm - d_hl + margin_frank
    t1 = F.relu(raw1)

    return (t1, raw1)

def fine_loss_tri_vanilla(z_h, z_m, z_l, *, margin_frank, w_f1rank=1.0):
    t1, raw1 = fine_terms_tri_vanilla(z_h, z_m, z_l, margin_frank=margin_frank)
    return w_f1rank*t1 , (t1, raw1)

