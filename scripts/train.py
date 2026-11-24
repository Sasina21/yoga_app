configs = {

    205: dict(
        margin_f1rank=0.22,
        margin_f3rank=0.10,
        delta=0.05,
        wf1=1.0, wf2=0.5,

        margin_c1sep=0.20, margin_c2sep=0.15, margin_c3rank=0.18, margin_c4rank=0.05,
        w_c1sep=1.0, w_c2sep=1.0, w_c3rank=1.0, w_c4rank=1.0,

        w_fine=3, w_coarse=1,
    ),

}

# ----------------------------
# Global fixed hyperparams
# ----------------------------
_batch = 64
_epoch = 500
patience = 20

DATA_TRAIN          = "data_train.csv"
DATA_TEST           = "data_test.csv"
POSE_DISC_TRAIN_CSV = "m2_pose_disc_train.csv"
POSE_DISC_TEST_CSV  = "m2_pose_disc_test.csv"
GRADING_TEST_CSV    = "m1_grading_test.csv"
GRADING_TEST_CSV_CORRECT    = "m1_grading_test_correct.csv"
POSE_DISC_TEST_CSV_CORRECT  = "m2_pose_disc_test_correct.csv"
SINGLE_REF_POSE_TRAIN = "m1_dir_1ref_per_pose_train.csv"  #med
SINGLE_REF_POSE_TEST  = "m1_dir_1ref_per_pose_test.csv"   #med
MED_SINGLE_REF_POSE_VIEW_TRAIN = "m1_med_dir_1ref_per_pose_view_train.csv"
MED_SINGLE_REF_POSE_VIEW_TEST = "m1_med_dir_1ref_per_pose_view_test.csv"
MIN_SINGLE_REF_POSE_VIEW_TRAIN = "m1_min_dir_1ref_per_pose_view_train.csv"
MIN_SINGLE_REF_POSE_VIEW_TEST = "m1_min_dir_1ref_per_pose_view_test.csv"
MAX_SINGLE_REF_POSE_VIEW_TRAIN = "m1_max_dir_1ref_per_pose_view_train.csv"
MAX_SINGLE_REF_POSE_VIEW_TEST = "m1_max_dir_1ref_per_pose_view_test.csv"
P75_SINGLE_REF_POSE_VIEW_TRAIN = "m1_p75_dir_1ref_per_pose_view_train.csv"
P75_SINGLE_REF_POSE_VIEW_TEST  = "m1_p75_dir_1ref_per_pose_view_test.csv"
SAVE_RAW_RESULT= "raw_results_TEST.csv"

METHOD_NAME = "Proposed"

# ----------------------------
# Load & prep data ONCE
# ----------------------------

train_df = pd.read_csv(DATA_TRAIN)
train_df.columns = [c.strip().lower() for c in train_df.columns]
train_df = train_df.reset_index(drop=True)

test_df = pd.read_csv(DATA_TEST)
test_df.columns = [c.strip().lower() for c in test_df.columns]
test_df = test_df.reset_index(drop=True)

feature_cols = [c for c in train_df.columns if c.startswith(("dir"))]
X_train = train_df[feature_cols].values.astype(np.float32)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Xtr = torch.tensor(X_train_sc, dtype=torch.float32).to(device)
# Xte = torch.tensor(X_test_sc,  dtype=torch.float32).to(device)

pose_risk = build_pose_risk_buckets(train_df)
pose_all  = build_pose_all_buckets(train_df)
pose_list_fine = list(pose_risk.keys())
pose_list_all  = list(pose_all.keys())

N = Xtr.shape[0]
mx_risk = max(max(v) for pose in pose_risk.values() for v in pose.values())
print("Xtr N=", N, "max risk idx=", mx_risk)
assert mx_risk < N

# ----------------------------
# Training (per run)
# ----------------------------
def train_one_run(cfg, run_id, Xtr_train, pose_risk, pose_all,
                  pose_list_fine, pose_list_all,
                  seed=42, seed_offset=0):

    reset_seed(seed + seed_offset)
    rng = np.random.default_rng(seed + seed_offset)

    model = MLP_dual_backbone(in_dim=len(feature_cols)).to(device)
    optimzr = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    def train_epoch():
        model.train()

        # steps = min(50, max(1, Xtr_train.shape[0] // _batch))
        iters_per_epoch = math.ceil(Xtr_train.shape[0] / _batch)

        # N = Xtr_train.shape[0]
        # all_idx = np.arange(N)

        total_L = total_Lc = total_Lf = 0.0

        for _ in range(iters_per_epoch):

            h_idx, m_idx, l_idx = sample_joint_HML(_batch, rng, pose_risk, pose_list_fine)
            a_idx, p1_idx, p2_idx, n_idx = sample_joint_AP1P2N(_batch, rng, pose_risk, pose_list_fine)

            h_idx = to_idx(h_idx, device); m_idx = to_idx(m_idx, device); l_idx = to_idx(l_idx, device)
            a_idx = to_idx(a_idx, device); p1_idx = to_idx(p1_idx, device); p2_idx = to_idx(p2_idx, device); n_idx = to_idx(n_idx, device)

            N = Xtr_train.shape[0]
            for name, idx in [("h",h_idx),("m",m_idx),("l",l_idx),("a",a_idx),("p1",p1_idx),("p2",p2_idx),("n",n_idx)]:
                assert idx.min().item() >= 0 and idx.max().item() < N, f"{name} idx out of range: [{idx.min().item()}, {idx.max().item()}], N={N}"

            # fine embeddings
            _, zh = model(Xtr_train[h_idx])
            _, zm = model(Xtr_train[m_idx])
            _, zl = model(Xtr_train[l_idx])

            # coarse embeddings
            za, _  = model(Xtr_train[a_idx])
            zp1, _ = model(Xtr_train[p1_idx])
            zp2, _ = model(Xtr_train[p2_idx])
            zn, _ = model(Xtr_train[n_idx])

            Lc_vec, aux = coarse_loss_quad(
                za, zp1, zp2, zn,
                margin_c1sep=cfg["margin_c1sep"], margin_c2sep=cfg["margin_c2sep"],
                margin_c3rank=cfg["margin_c3rank"], margin_c4rank=cfg["margin_c4rank"],
                w_c1sep=cfg["w_c1sep"], w_c2sep=cfg["w_c2sep"],
                w_c3rank=cfg["w_c3rank"], w_c4rank=cfg["w_c4rank"]
            )

            Lc = Lc_vec.mean()


            Lf_vec1, _ = fine_loss_tri_dual_view(zh, zm, zl,
                                  margin_f1rank=cfg["margin_f1rank"],
                                  margin_f2rank=cfg.get("margin_f2rank",None),
                                  delta=cfg.get("delta",None)
                                  )

            # neg
            Lf_vec2, _ = fine_loss_tri_dual_view(zl, zm, zh,
                                  margin_f1rank=cfg["margin_f3rank"],
                                  margin_f2rank=cfg.get("margin_f4rank",None),
                                  delta=cfg.get("delta",None)
                                  )

            Lf1 = cfg["wf1"]*Lf_vec1.mean()
            Lf2 = cfg["wf2"]*Lf_vec2.mean()

            Lf  = Lf1 + Lf2

            loss = cfg["w_coarse"]*Lc + cfg["w_fine"]*Lf

            optimzr.zero_grad()
            loss.backward()
            optimzr.step()

            total_L  += loss.item()
            total_Lc += Lc.item()
            total_Lf += Lf.item()

        return total_L/iters_per_epoch, total_Lc/iters_per_epoch, total_Lf/iters_per_epoch

    best_loss = float("inf")
    trigger = 0
    best_epoch = 0

    losses = []
    for ep in range(_epoch):
        L, Lc, Lf = train_epoch()
        losses.append(L)

        if L < best_loss - 1e-4:
            best_loss = L
            best_epoch = ep + 1
            trigger = 0
        else:
            trigger += 1

        if (ep + 1) % 50 == 0:
            print(f"[RUN {run_id:02d}] Epoch {ep+1:03d} | Loss={L:.5f} | Lc={Lc:.5f} | Lf={Lf:.5f} ")

        if trigger >= patience:
            break

    return model, dict(best_loss=float(best_loss), epochs=int(best_epoch), last_epoch=int(ep+1), losses=losses)

# ----------------------------
# Run ALL 10
# ----------------------------
train_id_to_idx = build_id_to_idx(train_df, "id")
test_id_to_idx  = build_id_to_idx(test_df,  "id")
test_id_to_view = build_id_to_view(test_df, "id", "view")
train_id_to_view = build_id_to_view(train_df, "id", "view")

X_train_t = make_feature_tensor(train_df, feature_cols, scaler, device)
X_test_t  = make_feature_tensor(test_df,  feature_cols, scaler, device)

results = []

for run_id in sorted(configs):
    cfg = configs[run_id]
    print("\n" + "="*70)
    print(f"RUN {run_id} config:")
    for k, v in cfg.items():
        print(f"  {k}={v}")
    print("="*70)

    print("len(train_df)=", len(train_df), "Xtr.shape[0]=", Xtr.shape[0])

    # =========================
    # A) Train model
    # =========================
    model, trinfo = train_one_run(
        cfg, run_id,
        Xtr,
        pose_risk, pose_all,
        pose_list_fine, pose_list_all,
        seed=SEED
    )

    # =========================
    # B) Method1 pairwise grading conditions
    # =========================
    m1_pairwise = eval_grading_conditions_d2(
        model, GRADING_TEST_CSV,
        X_tensor=X_test_t,
        id_to_idx=test_id_to_idx,
        batch_size=_batch,
        metrics_fn=metrics_binary,
        include_overall=True,
        method_name=METHOD_NAME,
        result_csv_out=GRADING_TEST_CSV_CORRECT
    )

    # =========================
    # C) Two-stage THR (fit on TRAIN, evaluate on TEST)
    # =========================

    # ====================== MED ref ==========================
    d2_tr, y_tr, df_tr = compute_teacher_student_d2_split(
        model, MED_SINGLE_REF_POSE_VIEW_TRAIN,
        X_student=X_train_t, id_to_idx_student=train_id_to_idx,
        X_teacher=X_train_t, id_to_idx_teacher=train_id_to_idx,
        head="2", which="fine"
    )

    d2_te, y_te, df_te = compute_teacher_student_d2_split(
        model, MED_SINGLE_REF_POSE_VIEW_TEST,
        X_student=X_test_t,  id_to_idx_student=test_id_to_idx,
        X_teacher=X_train_t, id_to_idx_teacher=train_id_to_idx,
        head="2", which="fine"
    )

    t0_SA, t1_DA, s1, s2 = fit_thresholds_3class_two_stage(
        d2_tr, y_tr,
        metric_stage1="f1",
        metric_stage2="f1",
        max_candidates=400
    )

    print(
        "[MED] two-stage thresholds:",
        f"t0_SA={t0_SA}",
        f"t1_DA={t1_DA}",
        f"| stage1_f1(DA)={s1}",
        f"| stage2_f1(SA/CA)={s2}"
    )

    thr2_te_res = eval_two_stage_threshold(
        d2_te, y_te, t1_DA, t0_SA,
        name="TEST 2-stage MED",
        method="MCML"
    )

    yp_thr2_te = thr2_te_res["y_pred"]

    # res = fit_thresholds_3class_two_stage_tiebreak(
    #     d2_tr, y_tr,
    #     metric_stage1="f1",
    #     metric_stage2="f1",
    #     max_candidates=400
    # )

    # print(res)

    # thr_te_res = eval_two_stage_threshold(
    #     d2_te, y_te,
    #     t1_DA=res["t1_DA"],
    #     t0_SA=res["t0_SA"],
    #     name="TEST 2-stage tie-break",
    #     method="Proposed"
    # )

    # ====================== MIN ref ==========================
    d2_min_tr, y_min_tr, df_min_tr = compute_teacher_student_d2_split(
        model, MIN_SINGLE_REF_POSE_VIEW_TRAIN,
        X_student=X_train_t, id_to_idx_student=train_id_to_idx,
        X_teacher=X_train_t, id_to_idx_teacher=train_id_to_idx,
        head="2", which="fine"
    )

    d2_min_te, y_min_te, df_min_te = compute_teacher_student_d2_split(
        model, MIN_SINGLE_REF_POSE_VIEW_TEST,
        X_student=X_test_t, id_to_idx_student=test_id_to_idx,
        X_teacher=X_train_t, id_to_idx_teacher=train_id_to_idx,
        head="2", which="fine"
    )
    t0_SA_min, t1_DA_min, s1_min, s2_min = fit_thresholds_3class_two_stage(
        d2_min_tr, y_min_tr,
        metric_stage1="f1",
        metric_stage2="f1",
        max_candidates=400
    )
    print(
        "[MIN] two-stage thresholds:",
        f"t0_SA={t0_SA_min}",
        f"t1_DA={t1_DA_min}",
        f"| stage1_f1(DA)={s1_min}",
        f"| stage2_f1(SA/CA)={s2_min}"
    )
    thr2_min_te_res = eval_two_stage_threshold(
        d2_min_te, y_min_te,
        t1_DA_min, t0_SA_min,
        name="TEST 2-stage MIN",
        method="MCML"
    )

    yp_min_thr2_te = thr2_min_te_res["y_pred"]

    # ====================== MAX ref ==========================
    d2_max_tr, y_max_tr, df_max_tr = compute_teacher_student_d2_split(
        model, MAX_SINGLE_REF_POSE_VIEW_TRAIN,
        X_student=X_train_t, id_to_idx_student=train_id_to_idx,
        X_teacher=X_train_t, id_to_idx_teacher=train_id_to_idx,
        head="2", which="fine"
    )

    d2_max_te, y_max_te, df_max_te = compute_teacher_student_d2_split(
        model, MAX_SINGLE_REF_POSE_VIEW_TEST,
        X_student=X_test_t, id_to_idx_student=test_id_to_idx,
        X_teacher=X_train_t, id_to_idx_teacher=train_id_to_idx,
        head="2", which="fine"
    )

    t0_SA_max, t1_DA_max, s1_max, s2_max = fit_thresholds_3class_two_stage(
        d2_max_tr, y_max_tr,
        metric_stage1="f1",
        metric_stage2="f1",
        max_candidates=400
    )

    print(
        "[MAX] two-stage thresholds:",
        f"t0_SA={t0_SA_max}",
        f"t1_DA={t1_DA_max}",
        f"| stage1_f1(DA)={s1_max}",
        f"| stage2_f1(SA/CA)={s2_max}"
    )

    thr2_max_te_res = eval_two_stage_threshold(
        d2_max_te, y_max_te,
        t1_DA_max, t0_SA_max,
        name="TEST 2-stage MAX",
        method="MCML"
    )
    yp_max_thr2_te = thr2_max_te_res["y_pred"]

    # ====================== P75 ref ==========================
    d2_p75_tr, y_p75_tr, df_p75_tr = compute_teacher_student_d2_split(
        model, P75_SINGLE_REF_POSE_VIEW_TRAIN,
        X_student=X_train_t, id_to_idx_student=train_id_to_idx,
        X_teacher=X_train_t, id_to_idx_teacher=train_id_to_idx,
        head="2", which="fine"
    )

    d2_p75_te, y_p75_te, df_p75_te = compute_teacher_student_d2_split(
        model, P75_SINGLE_REF_POSE_VIEW_TEST,
        X_student=X_test_t, id_to_idx_student=test_id_to_idx,
        X_teacher=X_train_t, id_to_idx_teacher=train_id_to_idx,
        head="2", which="fine"
    )

    t0_SA_p75, t1_DA_p75, s1_p75, s2_p75 = fit_thresholds_3class_two_stage(
        d2_p75_tr, y_p75_tr,
        metric_stage1="f1",
        metric_stage2="f1",
        max_candidates=400
    )

    print(
        "[P75] two-stage thresholds:",
        f"t0_SA={t0_SA_p75}",
        f"t1_DA={t1_DA_p75}",
        f"| stage1_f1(DA)={s1_p75}",
        f"| stage2_f1(SA/CA)={s2_p75}"
    )

    thr2_p75_te_res = eval_two_stage_threshold(
        d2_p75_te, y_p75_te,
        t1_DA_p75, t0_SA_p75,
        name="TEST 2-stage P75",
        method="MCML"
    )

    yp_p75_thr2_te = thr2_p75_te_res["y_pred"]

    # =========================
    # D) Method2 pose discrimination
    # =========================
    d2_same_pair_tr, d2_diff_pair_tr = compute_pair_d2(
        model, POSE_DISC_TRAIN_CSV,
        X_tensor=X_train_t,
        id_to_idx=train_id_to_idx,
        batch_size=_batch
    )

    best_thr, best_stats = find_best_thr_d2(
        d2_same_pair_tr,
        d2_diff_pair_tr,
        mode="f1"
    )

    res2 = eval_pose_discrimination_d2(
        model, POSE_DISC_TEST_CSV, best_thr,
        X_tensor=X_test_t,
        id_to_idx=test_id_to_idx,
        id_to_view=test_id_to_view,
        batch_size=_batch,
        method_name=METHOD_NAME,
        result_csv_out=POSE_DISC_TEST_CSV_CORRECT
    )

    # =========================
    # E) Supporting metrics / metadata
    # =========================
    eps = choose_eps_from_sa(d2_tr, y_tr, scale=0.1)
    soft = soft_ordering_rates(d2_te, y_te, eps=eps)

    conds = m1_pairwise.get("conditions", {}) or {}
    conds_meta = m1_pairwise.get("conditions_meta")

    if not conds_meta:
        conds_meta = []
        for i, k in enumerate(sorted(conds.keys()), start=1):
            conds_meta.append({"idx": i, "key": k, "note": k})

    # =========================
    # F) Build summary row
    # =========================
    row = dict(
        RUN=run_id,
        train_best_loss=trinfo["best_loss"],
        train_epochs=trinfo["epochs"],

        # ---- MAIN (distance-based) on TEST fine d2 ----
        auc_da_nonda_te=auc_da_vs_nonda(d2_te, y_te),
        spearman_te=spearman_rho(d2_te, y_te),
        auc_ca_sa_te=auc_ca_vs_sa(d2_te, y_te),

        # ---- THR 2-stage (fine) ----
        test_thr2_acc=thr2_te_res["acc"],
        test_thr2_f1m=thr2_te_res["macro_f1"],
        test_thr2_da_recall=thr2_te_res["da_recall"],
        test_thr2_da_precision=thr2_te_res["da_precision"],
        test_thr2_serious_failure=thr2_te_res["serious_failure"],

        # ---- Method2 ----
        m2_thr=float(res2["thr"]),
        m2_acc=float(res2["overall"]["metrics"]["accuracy"]) if res2.get("overall") and res2["overall"].get("metrics") else None,
        m2_f1=float(res2["overall"]["metrics"]["f1"]) if res2.get("overall") and res2["overall"].get("metrics") else None,
    )

    row["soft_p_da_gt_sa_te"] = soft["p_da_gt_sa"]
    row["soft_p_da_gt_ca_te"] = soft["p_da_gt_ca"]
    row["soft_p_ca_gt_sa_te"] = soft["p_ca_gt_sa"]
    row["soft_eps"] = soft["eps"]

    overall_pw = m1_pairwise.get("overall")
    if overall_pw:
        row["m1p_overall_acc"] = _safe_float(overall_pw.get("accuracy"))
        row["m1p_overall_f1"]  = _safe_float(overall_pw.get("f1"))
        row["m1p_overall_n_pairs"] = _safe_int(overall_pw.get("n_pairs"), default=0)
    else:
        row["m1p_overall_acc"] = None
        row["m1p_overall_f1"]  = None
        row["m1p_overall_n_pairs"] = None

    row["m1p_conditions_note"] = " | ".join(
        [f"cond{m['idx']}: {m.get('note', m.get('key'))}" for m in conds_meta]
    )

    for m in conds_meta:
        idx = m["idx"]
        key = m["key"]
        met = conds.get(key, {}) or {}

        row[f"m1p_cond{idx}_acc"] = _safe_float(met.get("accuracy"))
        row[f"m1p_cond{idx}_f1"]  = _safe_float(met.get("f1"))
        row[f"m1p_cond{idx}_n_pairs"] = _safe_int(met.get("n_pairs"), default=None)

    results.append(row)

    # =========================
    # G) Print short summary
    # =========================
    print(f"\n[RUN {run_id:02d}] Summary")
    print("=" * 60)
    print(
        f"  MAIN: AUC(DA vs nonDA)={row['auc_da_nonda_te']:.4f} "
        f"| Spearman={row['spearman_te']:.4f} "
        f"| DA_recall(thr2)={row['test_thr2_da_recall']:.4f}"
    )
    print(
        f"  2-stage THR: acc={row['test_thr2_acc']:.4f} "
        f"| f1m={row['test_thr2_f1m']:.4f} "
        f"| DA_prec={row['test_thr2_da_precision']:.4f} "
        f"| serious_fail={row['test_thr2_serious_failure']:.4f}"
    )

    print("  Method1 Condition:")
    for m in conds_meta:
        idx = m["idx"]
        note = m.get("note", m.get("key"))
        acc = row.get(f"m1p_cond{idx}_acc")
        f1  = row.get(f"m1p_cond{idx}_f1")
        if acc is None or f1 is None:
            print(f"    cond{idx}: {note} | acc=None | f1=None")
        else:
            print(f"    cond{idx}: {note} | acc={acc:.4f} | f1={f1:.4f}")

    print(
        f"  Method2 (Pose Disc): thr={row['m2_thr']:.4f} "
        f"| acc={row['m2_acc']:.4f} | f1={row['m2_f1']:.4f}"
    )

    # =========================
    # Answer Groups
    # =========================
    print("=" * 60)

    g1_med = print_group1_basic_metrics(
        d2_te=d2_te,
        y_te=y_te,
        yp_final=yp_thr2_te,
        title=f"RUN {run_id} | GROUP 1 MED"
    )
    g1_min = print_group1_basic_metrics(
        d2_te=d2_min_te,
        y_te=y_min_te,
        yp_final=yp_min_thr2_te,
        title=f"RUN {run_id} | GROUP 1 MIN"
    )
    g1_max = print_group1_basic_metrics(
        d2_te=d2_max_te,
        y_te=y_max_te,
        yp_final=yp_max_thr2_te,
        title=f"RUN {run_id} | GROUP 1 MAX"
    )
    g1_p75 = print_group1_basic_metrics(
        d2_te=d2_p75_te,
        y_te=y_p75_te,
        yp_final=yp_p75_thr2_te,
        title=f"RUN {run_id} | GROUP 1 P75"
    )

    g2_med = print_group2_ordering(
        d2=d2_te,
        y=y_te,
        title=f"RUN {run_id} | GROUP 2 MED"
    )
    g2_min = print_group2_ordering(
        d2=d2_min_te,
        y=y_min_te,
        title=f"RUN {run_id} | GROUP 2 MIN"
    )
    g2_max = print_group2_ordering(
        d2=d2_max_te,
        y=y_max_te,
        title=f"RUN {run_id} | GROUP 2 MAX"
    )
    g2_p75 = print_group2_ordering(
        d2=d2_p75_te,
        y=y_p75_te,
        title=f"RUN {run_id} | GROUP 2 P75"
    )

    d2_pose_tr, y_pose_tr, df_pose_tr = compute_teacher_student_d2_split(
        model, SINGLE_REF_POSE_TRAIN,
        X_student=X_train_t, id_to_idx_student=train_id_to_idx,
        X_teacher=X_train_t, id_to_idx_teacher=train_id_to_idx,
        head="2", which="fine"
    )

    d2_pose_te, y_pose_te, df_pose_te = compute_teacher_student_d2_split(
        model, SINGLE_REF_POSE_TEST,
        X_student=X_test_t,  id_to_idx_student=test_id_to_idx,
        X_teacher=X_train_t, id_to_idx_teacher=train_id_to_idx,
        head="2", which="fine"
    )

    g3_med = print_group3_per_view_two_sources(
        d2_view_tr=d2_tr, y_view_tr=y_tr,
        d2_view_te=d2_te, y_view_te=y_te, df_view_te=df_te,

        d2_pose_tr=d2_pose_tr, y_pose_tr=y_pose_tr,
        d2_pose_te=d2_pose_te, y_pose_te=y_pose_te, df_pose_te=df_pose_te,

        id_to_view=test_id_to_view,
        metric_stage1="f1",
        metric_stage2="f1",
        title=f"RUN {run_id} | GROUP 3 MED"
    )

    g3_min = print_group3_per_view_two_sources(
        d2_view_tr=d2_min_tr, y_view_tr=y_min_tr,
        d2_view_te=d2_min_te, y_view_te=y_min_te, df_view_te=df_min_te,

        d2_pose_tr=d2_pose_tr, y_pose_tr=y_pose_tr,
        d2_pose_te=d2_pose_te, y_pose_te=y_pose_te, df_pose_te=df_pose_te,

        id_to_view=test_id_to_view,
        metric_stage1="f1",
        metric_stage2="f1",
        title=f"RUN {run_id} | GROUP 3 MIN"
    )

    g3_max = print_group3_per_view_two_sources(
        d2_view_tr=d2_max_tr, y_view_tr=y_max_tr,
        d2_view_te=d2_max_te, y_view_te=y_max_te, df_view_te=df_max_te,

        d2_pose_tr=d2_pose_tr, y_pose_tr=y_pose_tr,
        d2_pose_te=d2_pose_te, y_pose_te=y_pose_te, df_pose_te=df_pose_te,

        id_to_view=test_id_to_view,
        metric_stage1="f1",
        metric_stage2="f1",
        title=f"RUN {run_id} | GROUP 3 MAX"
    )

    g3_p75 = print_group3_per_view_two_sources(
        d2_view_tr=d2_p75_tr, y_view_tr=y_p75_tr,
        d2_view_te=d2_p75_te, y_view_te=y_p75_te, df_view_te=df_p75_te,

        d2_pose_tr=d2_pose_tr, y_pose_tr=y_pose_tr,
        d2_pose_te=d2_pose_te, y_pose_te=y_pose_te, df_pose_te=df_pose_te,

        id_to_view=test_id_to_view,
        metric_stage1="f1",
        metric_stage2="f1",
        title=f"RUN {run_id} | GROUP 3 P75"
    )
    plot_distance_box_all(
        d2=d2_pose_te,
        y=y_pose_te,
        title="Proposed | d2 boxplot (ALL POSES)"
    )


# ========== save file ===========

save_eval_column(
    method_name=METHOD_NAME,
    eval_name="y_true",
    test_csv=MED_SINGLE_REF_POSE_VIEW_TEST,
    result_csv_out=SAVE_RAW_RESULT,
    values=y_te,
    cast="int"
)

save_eval_column(
    method_name=METHOD_NAME,
    eval_name="d2",
    test_csv=MED_SINGLE_REF_POSE_VIEW_TEST,
    result_csv_out=SAVE_RAW_RESULT,
    values=d2_te,
    cast="float"
)

save_eval_column(
    method_name=METHOD_NAME,
    eval_name="pred_thr2",
    test_csv=MED_SINGLE_REF_POSE_VIEW_TEST,
    result_csv_out=SAVE_RAW_RESULT,
    values=yp_thr2_te,
    cast="int"
)

save_eval_column(
    method_name=METHOD_NAME,
    eval_name="correct_thr2",
    test_csv=MED_SINGLE_REF_POSE_VIEW_TEST,
    result_csv_out=SAVE_RAW_RESULT,
    values=(yp_thr2_te == y_te).astype(int),
    cast="int"
)

save_eval_column(
    method_name=METHOD_NAME,
    eval_name="t0_SA",
    test_csv=MED_SINGLE_REF_POSE_VIEW_TEST,
    result_csv_out=SAVE_RAW_RESULT,
    values=np.full(len(y_te), t0_SA),
    cast="float"
)

save_eval_column(
    method_name=METHOD_NAME,
    eval_name="t1_DA",
    test_csv=MED_SINGLE_REF_POSE_VIEW_TEST,
    result_csv_out=SAVE_RAW_RESULT,
    values=np.full(len(y_te), t1_DA),
    cast="float"
)
