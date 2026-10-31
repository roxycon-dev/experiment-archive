"""Build the completed task report from local evidence; no SSH or training calls."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports"
ARCHIVE = ROOT.parent / "_全实验档案_20260920"
CST = timezone(timedelta(hours=8))
CLASSES = ["EX", "HE", "SE", "MA"]
STAGES = ["U1_external_pretrain", "U1_ddr_finetune", "H1_external_pretrain", "H1_ddr_finetune", "A0_ddr_finetune"]
SOURCES = {}


def read(rel):
    p = ROOT / rel
    b = p.read_bytes()
    SOURCES[rel] = {"path": str(p), "sha256": hashlib.sha256(b).hexdigest(), "bytes": len(b)}
    return json.loads(b.decode("utf-8-sig"))


def config(rel):
    p = ROOT / rel
    b = p.read_bytes()
    SOURCES[rel] = {"path": str(p), "sha256": hashlib.sha256(b).hexdigest(), "bytes": len(b)}
    return yaml.safe_load(b.decode("utf-8-sig"))


def link(p, label=None):
    p = Path(p)
    return f"[{label or p.name}](<{p.as_posix()}>)"


def pct(x):
    return f"{100*x:.4f}%"


def pp(x):
    return f"{100*x:+.4f}"


def duration(x):
    n = round(x)
    return f"{n//3600}小时{n%3600//60:02d}分{n%60:02d}秒"


def timestamp(x):
    return datetime.fromtimestamp(x, CST).strftime("%Y-%m-%d %H:%M:%S")


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"] + ["| " + " | ".join(map(str, row)) + " |" for row in rows])


def ious(counts):
    return [tp/(tp+fp+fn) for tp, fp, fn in counts]


def verify_metric(d, expected_n):
    assert d["n_images"] == expected_n
    cs = [[d["per_class"][c][k] for k in ["tp", "fp", "fn"]] for c in CLASSES]
    calc = ious(cs)
    assert all(abs(calc[i] - d["per_class"][c]["iou"]) < 1e-12 for i, c in enumerate(CLASSES))
    assert abs(sum(calc)/4 - d["mIoU_lesion4"]) < 1e-12


def main():
    campaign = read("server_results/work_dirs/campaign_state.json")
    finish = read("server_results/work_dirs/finish_state.json")
    fusion = read("server_results/reports/finish/initial_fusion.json")
    nomination = read("server_results/reports/finish/initial_nomination.json")
    test = read("server_results/reports/finish/final_test.json")
    progress = read("server_results/reports/finish/test_progress.json")
    local = read("reports/final_local_verification.json")
    manifest = read("best_pipeline/local_weight_manifest.json")
    backups = read("reports/verified_backups.json")
    protocol = read("round2_highres/protocol.json")
    r2campaign = read("round2_highres/server_results/work_dirs/campaign_state.json")
    r2finish = read("round2_highres/server_results/work_dirs/finish_state.json")
    r2audit = read("round2_highres/reports/validation_completion_audit.json")
    r2backups = read("round2_highres/reports/verified_backups.json")
    closure = read("round2_highres/reports/task_closure.json")
    cleanup = read("reports/server_weight_cleanup_result.json")
    cleanup_verify = read("reports/server_cleanup_final_verification.json")
    duplicates = read("reports/duplicates/duplicates.json")
    external_partition = read("reports/external_partition.json")
    auxiliary = read("reports/auxiliary_pool_audit.json")
    repair = read("reports/incident_gpu_idle_repair.json")
    repair2 = read("round2_highres/reports/engineering_repair_1000.json")
    read("round2_highres/reports/shutdown_state.json")
    configs = {name: config(rel) for name, rel in {
        "U1": "server_results/configs/resolved/U1_convnext_unet_ext_pretrain.yaml",
        "H1": "server_results/configs/resolved/H1_hrdecoder_ext_pretrain.yaml",
        "A0": "server_results/configs/resolved/A0_hacdr_adapted.yaml",
        "U0_highres": "round2_highres/configs/U0_highres.yaml",
        "U0_continuation2048": "round2_highres/configs/U0_continuation2048.yaml",
    }.items()}

    assert progress["status"] == "complete" and len(progress["images"]) == test["n_images"] == 227
    assert test["lock_sha256"] == manifest["original_lock_sha256"] == local["original_lock_sha256"]
    lock_bytes = (ROOT / "best_pipeline/original_deployment_lock.json").read_bytes()
    assert hashlib.sha256(lock_bytes).hexdigest() == test["lock_sha256"]
    for name, count_key, metric_key, class_key in [
        ("champion", "champion_counts", "champion_miou", "champion_per_class"),
        ("selected_pipeline", "selected_pipeline_counts", "selected_pipeline_miou", "selected_pipeline_per_class"),
    ]:
        summed = np.sum([record[name] for record in progress["images"].values()], axis=0).tolist()
        assert summed == test[count_key]
        calc = ious(summed)
        assert all(abs(a-b) < 1e-12 for a, b in zip(calc, test[class_key]))
        assert abs(sum(calc)/4-test[metric_key]) < 1e-12
    assert abs(test["selected_pipeline_miou"] - test["champion_miou"] - test["delta"]) < 1e-12

    main_nodes = []
    stage_nodes = {}
    for tag in STAGES:
        rows = []
        for p in sorted((ROOT / f"server_results/work_dirs/{tag}").glob("eval_u*.json"), key=lambda p: int(p.stem.split("u")[-1])):
            d = read(p.relative_to(ROOT).as_posix())
            verify_metric(d, 6 if "external" in tag else 76)
            row = {"tag": tag, "update": d["update"], "n_images": d["n_images"], "miou": d["mIoU_lesion4"], "iou": {c: d["per_class"][c]["iou"] for c in CLASSES}, "inference": d.get("eval_inference"), "resize_to": d.get("resize_to")}
            main_nodes.append(row)
            rows.append(row)
        stage_nodes[tag] = rows
        best = campaign["jobs"][tag]["summary"]["best"]
        chosen = next(n for n in rows if n["update"] == best["update"])
        assert abs(chosen["miou"] - best["mIoU_lesion4"]) < 1e-12

    r2nodes = []
    for audited in r2audit["validation_nodes"]:
        tag, update = audited["tag"], audited["update"]
        d = read(f"round2_highres/server_results/work_dirs/{tag}/eval_u{update}.json")
        verify_metric(d, 76)
        assert d["fixed_fusion_images"] == 76
        fused_iou = ious(d["fixed_fusion_counts"])
        assert abs(sum(fused_iou)/4 - d["fixed_fusion_miou"]) < 1e-12
        assert abs(d["mIoU_lesion4"] - audited["solo_miou"]) < 1e-12
        assert abs(d["fixed_fusion_miou"] - audited["fusion_miou"]) < 1e-12
        r2nodes.append({**audited, "solo_per_class": {c: d["per_class"][c]["iou"] for c in CLASSES}, "fusion_per_class": dict(zip(CLASSES, fused_iou))})
    for tag, result in r2finish["results"].items():
        best = max((n for n in r2nodes if n["tag"] == tag), key=lambda n: n["fusion_miou"])
        assert best["update"] == result["best_update"]
        assert abs(best["fusion_miou"] - result["fixed_fusion_miou"]) < 1e-12
        assert SOURCES[f"round2_highres/configs/{tag}.yaml"]["sha256"] == protocol["config_sha256"][tag]

    # Backups were fully SHA-verified at closure. Check existence/size again without
    # rehashing several GB for a reporting-only request.
    weights = []
    for group, items in [("已评价部署", manifest["weights"]), ("主实验 best", backups), ("高分辨率实验 best", r2backups)]:
        for name, info in items.items():
            p = Path(info.get("local", info.get("path")))
            assert p.is_file() and p.stat().st_size == info["bytes"]
            weights.append({"group": group, "name": name, "path": str(p), "bytes": info["bytes"], "sha256": info["sha256"], "sha_verification": "full SHA recorded at completed-task closure; existence and size checked again for this report"})

    main_updates = sum(j["summary"]["updates_done"] for j in campaign["jobs"].values())
    r2_updates = sum(j["summary"]["updates_done"] for j in r2campaign["jobs"].values())
    stage_seconds = sum(j["summary"]["elapsed_seconds"] for j in campaign["jobs"].values())
    main_wall = finish["completed_at_epoch"] - campaign["started_at_epoch"]
    r2_wall = r2campaign["finished_at_epoch"] - r2campaign["started_at_epoch"]
    assert main_updates == 34997 and r2_updates == 6000
    assert abs(main_wall-local["wall_seconds"]) < 1e-6
    assert abs(stage_seconds-local["training_seconds"]) < 1e-6
    assert abs(r2_wall-r2audit["wall_seconds"]) < 1e-6
    baseline = protocol["baseline_val_miou"]
    best2 = max(r2nodes, key=lambda n: n["fusion_miou"])

    figures = OUT / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"], "axes.unicode_minus": False, "font.size": 10, "svg.fonttype": "none"})
    fig, axes = plt.subplots(2, 1, figsize=(9.2, 8.4), constrained_layout=True)
    x = np.arange(5)
    previous = 100*np.array(test["champion_per_class"] + [test["champion_miou"]])
    selected = 100*np.array(test["selected_pipeline_per_class"] + [test["selected_pipeline_miou"]])
    colors = ["#247BA0", "#DB7B20"]
    for values, shift, color, label in [(previous, -.19, colors[0], "原冠军"), (selected, .19, colors[1], "锁定管线：原冠军75%＋历史U0 25%")]:
        bars = axes[0].bar(x+shift, values, .36, color=color, label=label)
        axes[0].bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
    axes[0].set_xticks(x, CLASSES + ["mIoU"])
    axes[0].set_ylim(0, 62)
    axes[0].set_ylabel("IoU（%）")
    axes[0].set_title("A  已锁定管线的测试结果（227张；四类累计计数）", loc="left", pad=13)
    axes[0].legend(loc="upper right", frameon=False, fontsize=9)
    axes[0].text(.01, .93, f"mIoU提升：{pp(test['delta'])}个百分点", transform=axes[0].transAxes)
    for tag, color, marker, label in [("U0_continuation2048", colors[0], "o", "2048续训控制（2k早停）"), ("U0_highres", colors[1], "s", "2048/2560混合（完成4k）")]:
        ns = [n for n in r2nodes if n["tag"] == tag]
        axes[1].plot([n["update"]/1000 for n in ns], [100*n["fusion_miou"] for n in ns], color=color, marker=marker, linewidth=2, label=label)
    axes[1].axhline(100*baseline, color="#5C6670", linestyle="--", linewidth=1.5, label="原锁定融合 val 基线43.4877%")
    axes[1].annotate(f"最佳1k：{pct(best2['fusion_miou'])}\n较基线{pp(best2['fusion_miou']-baseline)}个百分点", xy=(1, 100*best2["fusion_miou"]), xytext=(1.65, 43.69), arrowprops={"arrowstyle": "->", "color": colors[1]}, fontsize=10)
    axes[1].set_xticks([1, 2, 4], ["1000", "2000", "4000"])
    axes[1].set_ylim(43.20, 43.85)
    axes[1].set_xlim(.75, 4.15)
    axes[1].set_xlabel("新增 optimizer updates")
    axes[1].set_ylabel("固定融合 val mIoU（%）")
    axes[1].set_title("B  高分辨率续训完整验证（76张；原冠军75%＋候选25%）", loc="left", pad=13)
    axes[1].legend(loc="lower right", frameon=False, fontsize=9)
    for ax in axes:
        ax.grid(axis="y", color="#E1E6EB", linewidth=.7)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(figures / "HRDecoder_任务结果_20261003.png", dpi=240)
    fig.savefig(figures / "HRDecoder_任务结果_20261003.svg")
    plt.close(fig)

    evidence = {
        "report_type": "completed_task_local_evidence_only", "generated_at_cst": datetime.now(CST).isoformat(),
        "data": {"ddr": {"train": 454, "val": 76, "test": 227}, "external": {"train": len(external_partition["train_ids"]), "dev": len(external_partition["dev_ids"]), "annotation": "user polygon rasterization, not original official pixel masks", "test_used_for_training": False}, "channel_order": CLASSES},
        "main_campaign": {"started_at_cst": timestamp(campaign["started_at_epoch"]), "training_finished_at_cst": timestamp(campaign["finished_at_epoch"]), "completed_at_cst": timestamp(finish["completed_at_epoch"]), "wall_seconds": main_wall, "stage_elapsed_seconds_including_eval_save": stage_seconds, "new_updates": main_updates, "jobs": campaign["jobs"], "validation_nodes": main_nodes, "nomination": nomination, "fusion": {k: v for k, v in fusion.items() if k != "cache_identities"}},
        "test": test,
        "round2": {"protocol": protocol, "validation_nodes": r2nodes, "results": r2finish["results"], "wall_seconds": r2_wall, "job_wall_seconds": r2audit["job_wall_seconds"], "new_updates": r2_updates, "best_candidate": best2},
        "training_budget": {"adjusted_max_updates": 48800, "actual_new_updates": main_updates+r2_updates, "saved_updates": 7803, "historical_updates_excluded": True},
        "cleanup": cleanup, "closure": closure, "weights": weights,
        "report_verification": {"test_image_counts_recomputed": 227, "main_full_metric_nodes_recomputed": len(main_nodes), "round2_full_metric_nodes_recomputed": len(r2nodes), "round2_config_sha_checked": True, "original_test_lock_sha_checked": True, "weight_sizes_checked": len(weights), "new_training": False, "new_test_evaluation": False, "server_accessed": False},
        "source_files": SOURCES,
    }
    json_path = OUT / "HRDecoder_完整实验报告_20261003.json"
    json_path.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    chunks = []
    def add(s):
        chunks.append(s.strip())

    add(f"""
    # HRDecoder 文献驱动实验：完整任务报告

    实验执行日期：2026-10-02至2026-10-03（北京时间）。报告范围：主实验、有限融合、一次锁定测试、高分辨率配对续训，以及权重备份和关机收尾。根据本机原始 JSON、训练完成记录、配置和备份清单整理；本次出报告没有启动训练、访问服务器或重新评价 test。

    **已评价的最佳管线：原冠军75%＋历史U0 25%，227张 test mIoU 为 {pct(test['selected_pipeline_miou'])}，较同一评价器复现的原冠军 {pct(test['champion_miou'])} 提高 {pp(test['delta'])} 个百分点。** 四类 test IoU 均提高。47.20%的挑战目标未达到。

    **新增探索的最佳候选：U0高分辨率续训1000步，76张 val 上固定融合为 {pct(best2['fusion_miou'])}，较既有锁定融合 val {pct(baseline)} 提高 {pp(best2['fusion_miou']-baseline)} 个百分点。** 该候选只做 train/val 探索，没有再次跑 test，也没有替换已评价部署锁。

    全部获准训练和收尾已结束，本机 best 备份已校验，小时模型监控已暂停。关机指令已于 {closure['shutdown_dispatched_at_cst']} 执行，随后 SSH 断开，复查不可达；平台控制台的最终关机状态未独立核验。未释放或删除实例。

    ![锁定测试结果与高分辨率验证轨迹](<{(figures/'HRDecoder_任务结果_20261003.png').as_posix()}>)

    图A是已经完成的锁定测试；图B是后续验证探索。两者数据划分和候选不同，不能把图B的收益加到图A的 test 分数上。
    """)

    add("""
    ## 1. 目标、实际覆盖与方案调整

    原方案希望比较 HRDecoder 与 ConvNeXt-Small＋U-Net 两种结构，以及 DDR-only 与外部病灶预训练→DDR 两条数据路径，并增加 HACDR 异构候选、受约束融合和条件触发的扩展。执行目标是寻找有效路线，保留仍在学习或有互补性的候选，对低希望配置止损，减少机械复跑。

    实际完成：U1外部预训练及DDR适配、H1外部预训练及DDR适配、A0适配试验、历史U0/原冠军严格复现、有限融合及一次锁定test；清理空间后又完成U0的2048控制与2048/2560高分辨率续训。U0复用历史12k完整结果；H0只有历史6k未完成记录且缺少可恢复权重，没有从头重跑。

    因此，这是一轮有完整结果的路线探索，但没有形成四个重新训练、实现和预算完全匹配的主训练臂。历史控制与新臂的比较仅作描述，不能把差异单独归因于外部预训练。
    """)

    add(f"""
    ## 2. 数据与评价规则

    {table(['数据', '实际使用', '边界与限制'], [
        ['用户整理DDR', '454 train / 76 val / 227 test，共757张', '沿用原划分；服务器目录DDR_split_613的后缀不是样本量'],
        ['本机IDRiD版本', '54张官方train来源图，固定拆为48 train / 6 dev', '使用用户多边形栅格化掩膜，不能称为原始官方像素掩膜；27张test未纳入训练'],
        ['MAPLES-DR/MESSIDOR', '未进入训练', '缺少配套MESSIDOR原图'],
        ['FGADR', '未进入训练', '未取得可用资产'],
        ['辅助未标注池', f"审计并保留{auxiliary['kept']}张", '教师质量门槛未通过，未启动半监督训练'],
    ])}

    外部图像级重复审计覆盖DDR全部 {duplicates['ddr_images_compared']} 张，对 {duplicates['external_images']} 张外部图无排除项；这是有限的图像级审计，未知患者ID、任意旋转和形变并未穷举，不能称为完全患者去重。早期合同文件中的“ddr_images_compared=0”不是最终重复审计结果。

    四个独立输出通道依次为 EX（硬性渗出）、HE（出血）、SE（软性渗出）、MA（微动脉瘤）。最终概率严格大于0.5才判阳性；先在完整图像、原始标签坐标累计每类 TP/FP/FN，再计算 IoU=TP/(TP+FP+FN)，最后对四类取均值。不是平均patch IoU，也不是先逐图平均IoU。

    推理几何分别保持记录：新主实验U1/A0采用2048配置缩放后1024窗口、768步长的滑窗推理；H1记录为2048配置的整图推理；历史U0及第二轮U0保持原生尺寸整图、无额外TTA。原冠军保留原有几何、水平翻转TTA和MA头，融合前全部还原到相同原图坐标。因此，“计数口径相同”不表示各候选的推理几何完全相同。

    验证报告中的AUPR来自256-bin近似；本报告不补造最终test AUPR。76张val参与checkpoint选择，融合交叉拟合也不是完全独立外层验证。227张test在历史实验中已被查看，不能充当新的独立外部验证；本任务锁定后只评价一条最终候选管线，随后关闭该test用于新选型。
    """)

    add(f"""
    ## 3. 模型、训练配方与最快路径

    {table(['ID', '结构与初始化', '实际执行'], [
        ['U0', '历史ConvNeXt-Small＋项目U-Net，ImageNet初始化后DDR训练', '复用12k best，按历史原生整图推理复现；后续作为两臂相同起点'],
        ['U1', 'ConvNeXt-Small＋项目U-Net；官方ImageNet权重', '48张外部预训练2400更新→DDR适配12000更新；无额外RGB细节支路'],
        ['H0', '历史HRDecoder-W48控制', '历史6k记录val约36.9226%；未完成，缺恢复权重，不补跑'],
        ['H1', 'HRNet-W48/项目HRDecoder改造；HRNet ImageNet权重', '外部2400→DDR12000；开放主干，冻结BN运行统计、affine可训练'],
        ['A0', '官方HACDR骨干＋项目四logit解码器', '无ImageNet/外部初始化；DDR完成6197更新，在更新边界止损'],
    ])}

    H1采用监督crop训练及来源HRDecoder的global/local特征推理，没有冻结B副本或原论文辅助crop损失；A0也更换了项目解码器和损失。因此，两者都不是完整论文训练配方复现，A0负结果仅对应当前适配实现。A0骨干来源commit为 `93eb3d41e96f5add9027f1a4977e54eb4fdc9bb0`。

    公共设置：4个独立二分类logit；CE＋Dice等权、smooth=1e-5、逐类有效标注掩码；AdamW、weight decay=0.01、5%warmup、cosine降至初始LR的1%；seed42、有效batch8、crop1024、训练基准视野2048、0.5–2.0尺度扰动、随机/前景混合采样、翻转/旋转和轻量光度增强。实际配置按模型保留归一化；AMP、TF32、cuDNN benchmark用于加速。

    {table(['DDR阶段', 'backbone / decoder LR', '微批次×累积', '热身后实测秒/更新'], [
        ['U1/ConvNeXt', '1e-4 / 3e-4', '4×2，workers4', '0.4649（U系列DDR路径校准）'],
        ['H1/HRDecoder', '5e-5 / 2e-4', '8×1，workers4', '1.3452'],
        ['A0/HACDR适配', '6e-5 / 9e-4', '8×1，workers4', '1.3940'],
    ])}

    服务器为单张RTX4090（24564 MiB显存）、约120 GiB RAM；Python3.8.10、PyTorch2.0.1+cu117、mmcv-full1.7.2。H1在实际DDR读取下，8×1比4×2校准均值快约5.7%，在第一次新DDR验证前选定。外部可用数据只有48/6，因此在看新验证前将原8k外部预算统一缩为2.4k。吞吐校准不包含完整验证及checkpoint保存，不能直接当作正式阶段耗时或绝对最快的证明。
    """)

    stage_rows = []
    for tag in STAGES:
        s = campaign['jobs'][tag]['summary']
        stage_rows.append([tag, s['updates_done'], s['best']['update'], pct(s['best']['mIoU_lesion4']), '外部dev 6张' if 'external' in tag else 'DDR val 76张', duration(s['elapsed_seconds'])])
    add(f"""
    ## 4. 主实验结果与止损

    {table(['阶段', '实际更新', '最佳节点', '最佳单模mIoU', '评价数据', '阶段耗时，含验证/保存'], stage_rows)}

    外部dev的27.1605%/42.9998%不能与DDR验证分直接排榜。主DDR完整验证轨迹如下。

    {table(['模型', '3000', '6000', '9000', '12000'], [[tag.split('_')[0]] + [pct(next(n['miou'] for n in stage_nodes[tag] if n['update']==u)) if any(n['update']==u for n in stage_nodes[tag]) else '未执行' for u in [3000,6000,9000,12000]] for tag in ['U1_ddr_finetune','H1_ddr_finetune','A0_ddr_finetune']])}

    U1仍在改善，因此保留到12k；H1最佳在9k，12k轻微回落。H1原始状态为stopped，但实际已经完成全部12000更新，没有因此节省主训练量。A0在6k仅9.6781%，HE真阳性近零、MA为零，无类别优势；经训练图诊断未确认运行故障后，于6197更新边界停止，省去5803更新以及后续验证。这个配置的亮病灶仍有一些学习，结论是当前路线优先级低，不能概括为HACDR论文方法无效。

    ### 与原冠军固定融合的互补性

    {table(['候选', '单模val mIoU', '原冠军75%＋候选25%', '较原冠军42.6691%的增益（百分点）'], [[row['spec']['id'], pct(row['alone_miou']), pct(row['fixed25_miou']), pp(row['fixed25_miou']-nomination['base_miou'])] for row in nomination['rows']])}

    历史U0是最有用的补充。U1单模分较低，但固定融合仍有小收益；H1与A0的固定融合没有收益。A0重新导出概率的浮点结果与训练验证有约0.00001个百分点差异，不构成新改善。新训练的U1/H1/A0均未进入最终test管线。
    """)

    add(f"""
    ## 5. 受约束融合与锁定测试

    每个家族只提名一个候选，最多两个家族进入融合：ConvNeXt家族的历史U0，以及HRDecoder家族的H1。val固定三折，每类最多补充一个家族，候选比例只取0、0.125、0.25、0.5。汇总留出折的TP/FP/FN比较全局与逐类策略，部署参数由固定投票规则产生，不在全部val上重新扫描最高点。

    {table(['融合层评价', 'mIoU'], [
        ['原冠军，汇总76图计数', pct(fusion['oof_base_miou'])],
        ['全局融合的折外计数汇总', pct(fusion['oof_global_miou'])],
        ['按类别融合的折外计数汇总', pct(fusion['oof_class_miou'])],
        ['最终锁定全局配方，在完整val上的描述结果', pct(fusion['locked_val_miou'])],
    ])}

    选择记录为 `global_fallback`，全局策略优于逐类策略。原冠军由历史D/E/B/E4以3:2:1:1权重组合，包含既有H4 MA头及水平翻转TTA。最终EX/HE/SE/MA四通道统一采用：

    `p_final = 0.75 × p_原冠军 + 0.25 × p_历史U0`，最终阈值为严格 `> 0.5`。

    这是此前已有模型的融合。部署共需要D、E、B、E4、H4和U0六份权重；原锁里H1的候选元数据不表示实际最终管线加载H1。

    {table(['227图test', 'EX IoU', 'HE IoU', 'SE IoU', 'MA IoU', 'mIoU'], [
        ['原冠军，同一评价器复现'] + [pct(x) for x in test['champion_per_class']] + [pct(test['champion_miou'])],
        ['锁定管线'] + [pct(x) for x in test['selected_pipeline_per_class']] + [pct(test['selected_pipeline_miou'])],
        ['增益，百分点'] + [pp(a-b) for a,b in zip(test['selected_pipeline_per_class'],test['champion_per_class'])] + [pp(test['delta'])],
    ])}

    原历史报告冠军test为44.2115%；当前同一评价器复现为44.2110073%，本报告的增益采用同评价器两条结果相减。四类IoU均提高，不代表四类召回也都提高。

    原始部署锁SHA-256：`{test['lock_sha256']}`。本次报告独立重新求和227张逐图计数并重算上述四类IoU/mIoU，全部一致；没有重新运行模型。已有描述性bootstrap也受val选型影响，本报告不据此声称独立泛化显著性。
    """)

    add(f"""
    ## 6. 高分辨率配对续训：后续探索结果

    主实验第一次触发高分辨率扩展时，预计需约16.45 GiB，而训练盘仅余14.12 GiB，因此暂缓。用户授权删除未使用权重后，另立协议并实际执行第二轮；最终任务状态不应继续写作“高分辨率未执行”。

    两臂均从历史U0 best开始，fresh optimizer、seed42、backbone/decoder LR=2e-5/6e-5、crop1024、有效batch8（4×2）、相同计划验证节点及最多4000新增更新。控制臂基准长边2048；实验臂随机2048/2560，各保留0.5–2.0缩放扰动。改变的是基准尺度日程，不是每张训练图恒为2048或2560。推理都保持原生整图，无新增TTA。

    best按固定融合val选取：原冠军75%＋当前候选25%，权重不再搜索。1000/2000/4000节点均评价完整76张。达到2k后，若最佳融合收益、近期融合改善和近期单模改善均小于0.1个百分点，则止损；高分辨率1k已有超过门槛的互补收益，因此保留到4k。

    {table(['训练臂', '节点', '单模val mIoU', '固定融合val mIoU', '较43.4877%基线，百分点'], [['2048控制' if n['tag']=='U0_continuation2048' else '2048/2560混合', n['update'], pct(n['solo_miou']), pct(n['fusion_miou']), pp(n['fusion_miou']-baseline)] for n in r2nodes])}

    共有节点比较：在1k，高分辨率融合较控制高0.3665个百分点；在2k，高分辨率较控制低0.0054个百分点。控制2k早停，实际2000更新；高分辨率实际4000更新。两臂均最佳于1k，不能只比较两个最终节点并忽略预算不同。

    高分辨率最佳1k：单模39.6065%，固定融合43.7328%，较原锁定融合val提高0.2451个百分点。2k及4k都未超过1k，继续训练的收益没有出现，保留早期互补checkpoint是合理的。本轮没有第二seed，控制臂恢复后的DataLoader随机流也不是逐位恢复，因此这是一条值得保留的探索路线，不能声称尺度因素已得到严格稳定的因果确认。

    第二轮仅用train/val，旧test已关闭。本轮最佳不是“新的44.7809%测试模型”；如将来检验其泛化，需要另行定义独立评估数据，不能再靠同一227图test挑候选。当前没有排队的后续训练。
    """)

    add("""
    ## 7. 未执行项与正负发现

    | 方向 | 最终状态 | 可以得出的结论 |
    | --- | --- | --- |
    | 历史U0补充原冠军 | 锁定且完成test | 当前协议下提供互补，test增加0.5699个百分点 |
    | U1外部预训练→DDR | 两阶段完整执行 | 12k仍学习，单模及固定融合不及历史U0；不能断言官方像素预训练无效 |
    | H1外部预训练→DDR | 两阶段完整执行 | 9k最佳，12k回落，未带来最终融合增益 |
    | HACDR适配A0 | 6197更新止损 | 当前初始化/解码器/损失配置的红病灶学习失败；不推广到原论文 |
    | U0高分辨率 | 配对探索已完成 | 1k固定融合val收益，2k/4k未持续改善；仅val发现 |
    | 最佳模型第二seed | 未触发 | 单模质量及有用融合门槛未同时达到；避免只为确认重复训练 |
    | 半监督扩容 | 未触发 | 辅助池已有审计资产，但教师单模质量未达到门槛；不是半监督负结果 |
    | 原始官方IDRiD像素监督/MAPLES/FGADR | 资产条件未满足 | 未检验完整官方像素监督路线 |

    最有价值的现象是模型互补性：单模U0明显低于原冠军，却能提高组合结果；而新结构/额外训练并不会自动变成更好的融合成员。当前优先保存已验证的历史U0融合和新的高分辨率1k候选。未到47.20%，也未得到替代原冠军的高分单模型。
    """)

    add(f"""
    ## 8. 实际训练量、时间与空间

    {table(['项目', '实际记录'], [
        ['主实验新增更新', f'{main_updates:,}（2400＋12000＋2400＋12000＋6197）'],
        ['第二轮新增更新', f'{r2_updates:,}（控制2000＋高分辨率4000）'],
        ['合计新增更新', f'{main_updates+r2_updates:,}；不含历史U0/H0和测速'],
        ['调整后最高预算', '48,800更新；A0和第二轮控制止损合计省去7803更新'],
        ['主队列开始', timestamp(campaign['started_at_epoch'])],
        ['主训练结束', timestamp(campaign['finished_at_epoch'])],
        ['主轮融合/test完成', timestamp(finish['completed_at_epoch'])],
        ['主轮墙钟，包含融合/test', duration(main_wall)],
        ['主轮5阶段elapsed之和，含各阶段验证/保存', duration(stage_seconds)],
        ['第二轮开始/训练完成', f"{timestamp(r2campaign['started_at_epoch'])} / {timestamp(r2campaign['finished_at_epoch'])}"],
        ['第二轮墙钟，含验证/修复', duration(r2_wall)],
        ['第二轮控制/高分辨率墙钟', f"{duration(r2audit['job_wall_seconds']['U0_continuation2048'])} / {duration(r2audit['job_wall_seconds']['U0_highres'])}"],
        ['两轮队列墙钟之和', duration(main_wall+r2_wall)],
        ['本机备份/汇总/关机收尾完成', closure['completed_at_cst']],
    ])}

    队列墙钟不等于纯GPU训练时长或计费时长；主阶段elapsed包含验证/保存，第二轮控制的恢复后elapsed只覆盖恢复片段。准备、上传、间歇、清理、权重下载也不都在这些队列时间内。原方案110–200 GPU小时是设计预算，不是实际消耗；没有完整平台账单，不能编造费用。

    用户授权后删除21个未使用服务器checkpoint，名义大小约11.36 GiB，跨两个文件系统实际增加可用空间约11.36 GiB；训练盘从14.12增加至20.09 GiB。10个来自本轮5阶段best/last，5份best已完成本机SHA备份；另11个历史未使用checkpoint按授权删除，没有新增这11份备份。保留部署六权重、必要初始化、日志、配置、数据和概率缓存。清理时保留9个服务器权重，第二轮随后产生新best/last，9不是最终文件总数。
    """)

    add("""
    ## 9. 工程故障与可复现性

    工程故障没有计作算法负结果。主要修复如下：

    - 历史U0初次导出误用了新2048缩放及继承滑窗几何，导致严格复现检查失败。恢复原生整图后，首图四类二值预测逐像素相同，76张完整复现通过。保留原检查标准，没有通过放宽容差换取通过。
    - 大checkpoint下载堵塞单一镜像连接，使健康状态更新滞后、GPU空闲未被及时暴露。随后拆分健康采集与权重传输为独立SSH连接；后台分钟级采集，模型小时巡检，减少额度消耗。零GPU利用率需要结合阶段/进程判断，不能单凭一次读数诊断。
    - 主轮结果已保存后，finisher退出清理的变量捕获冲突留下旧PID锁。修复退出处理及失效锁，没有再跑test。
    - 第二轮控制1000节点概率缓存接口参数错误。使用已保存1000步完整checkpoint、严格相同配置恢复，先补完整76张验证再继续更新；4项回归检查通过，没有重训前1000更新。模型/优化器兼容恢复不等于DataLoader随机流逐位恢复；这个限制保留在配对解释中。
    - AutoDL `/usr/bin/shutdown` 是无shebang文本脚本，直接执行出现Exec format error，未启动关机程序。改由 `/bin/bash /usr/bin/shutdown` 执行，6项guard检查通过；备份和无剩余计算进程的门槛保留。修复后SSH在回执前断开，最终状态记为shutdown_dispatched_unconfirmed，不能据此声称平台状态已独立确认。

    第二轮原protocol配置SHA与修复前source SHA均保留，修复单独记载；部署锁保持原字节。概率缓存采用原图坐标的无损FP32，保留完整val/test计数记录。新报告再次核对全部主训练完整评价节点、第二轮5个完整节点、227图测试计数和第二轮配置SHA；权重完整SHA采用收尾时已核验的清单，本次再次检查13份备份文件存在及大小。
    """)

    weights_rows = []
    for w in weights:
        if w['group'] in ['已评价部署', '高分辨率实验 best']:
            weights_rows.append([w['group'], link(w['path'], w['name']), w['sha256']])
    add(f"""
    ## 10. 本机成果与继续研究所需资产

    {table(['用途', '入口'], [
        ['本报告的机器可读证据、全部完整节点及源文件SHA', link(json_path)],
        ['已评价部署包：6权重、原锁、配置与test记录', link(ROOT/'best_pipeline/README.md', 'best_pipeline/README.md')],
        ['主轮5份完整best备份清单', link(ROOT/'reports/verified_backups.json')],
        ['第二轮2份完整best备份清单', link(ROOT/'round2_highres/reports/verified_backups.json')],
        ['第二轮协议与原始配置SHA', link(ROOT/'round2_highres/protocol.json')],
        ['227图test最终记录', link(ROOT/'server_results/reports/finish/final_test.json')],
        ['高分辨率完整验证审计', link(ROOT/'round2_highres/reports/validation_completion_audit.json')],
        ['收尾与实际关机执行记录', link(ROOT/'round2_highres/reports/收尾与关机记录.md')],
        ['可缩放结果图', link(figures/'HRDecoder_任务结果_20261003.svg')],
    ])}

    {table(['用途', '权重路径', '收尾时完整SHA-256'], weights_rows)}

    最值得继续研究的是 `U0_highres/ckpt_best.pth`（1000步），而不是4k最后权重；已评价部署继续使用历史U0。best是完整训练checkpoint，复用时需要兼容模型/config；已有部署锁包含服务器路径，迁移时记录路径映射，不覆盖原锁或重新按旧test选型。

    本轮全部获准实验已经结束。未触发的条件分支以及缺资产方向没有在队列中继续运行；后续如开展新研究，需重新明确范围和独立评价数据。

    ### 原始依据

    原始方案：{link(Path(r'C:\Users\GuanXingRen\Documents\Codex\2026-09-19\d-dataset-experiment-hrdecoder-fusionaware-20260917\outputs\HRDecoder_文献驱动进取方案_20260920.md'))}。主队列依据：{link(ROOT/'server_results/work_dirs/campaign_state.json')}、{link(ROOT/'server_results/work_dirs/finish_state.json')}；融合依据：{link(ROOT/'server_results/reports/finish/initial_nomination.json')}、{link(ROOT/'server_results/reports/finish/initial_fusion.json')}；第二轮依据：{link(ROOT/'round2_highres/server_results/work_dirs/campaign_state.json')}、{link(ROOT/'round2_highres/reports/task_closure.json')}。全部源文件列表和生成时SHA见本报告配套JSON。
    """)

    # Remove only template indentation, preserving table and paragraph separation.
    report = "\n\n".join("\n".join(line[4:] if line.startswith("    ") else line for line in c.splitlines()) for c in chunks) + "\n"
    md_path = OUT / "HRDecoder_完整实验报告_20261003.md"
    md_path.write_text(report, encoding="utf-8")
    ARCHIVE.mkdir(exist_ok=True)
    index = ARCHIVE / "06_完整任务实验报告_20261003.md"
    index.write_text("# HRDecoder完整任务实验报告\n\n" + link(md_path, "完整实验报告：两轮实验、锁定test、训练量、工程限制与关机收尾") + "\n\n" + link(json_path, "机器可读证据与源文件SHA") + "\n\n此文件索引2026-10-02至10-03的完整任务报告；不修改此前09-20历史快照或旧部署锁。\n", encoding="utf-8")
    print(json.dumps({"report": str(md_path), "evidence": str(json_path), "figure": str(figures/'HRDecoder_任务结果_20261003.png'), "archive_index": str(index), "verification": evidence["report_verification"], "report_bytes": md_path.stat().st_size}, ensure_ascii=True))


if __name__ == "__main__":
    main()
