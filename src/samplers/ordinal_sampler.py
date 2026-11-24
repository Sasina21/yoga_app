import numpy as np

def sample_joint_HML(batch_size, rng, pose_risk, pose_list_fine):
    H, M, L = [], [], []

    for _ in range(batch_size):
        pose = rng.choice(pose_list_fine)

        h = rng.choice(pose_risk[pose][0])
        m = rng.choice(pose_risk[pose][1])
        l = rng.choice(pose_risk[pose][2])
        H.append(h); M.append(m); L.append(l)


    return (np.array(H), np.array(M), np.array(L))

def sample_joint_AP1P2N(batch_size, rng, pose_all, pose_risk, pose_list_fine, pose_list_all):
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