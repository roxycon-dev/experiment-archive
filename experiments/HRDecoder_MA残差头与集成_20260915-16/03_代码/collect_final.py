#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""汇总 2026-09-15/16 两轮全部实验，生成最终报告。"""
import csv
import json
import os
from datetime import datetime

import paramiko

HOST, PORT, USER = "connect.nmb1.seetacloud.com", 37047, "root"
PWD = os.environ.get("HR_PWD", "")
DIRS = {
    "L_0915": "/root/autodl-tmp/hrplus/results/L_0915",
    "E_0916": "/root/autodl-tmp/hrplus/results/E_0916",
    "E2_0916": "/root/autodl-tmp/hrplus/results/E2_0916",
    "E3_0916": "/root/autodl-tmp/hrplus/results/E3_0916",
}
LOCAL = r"C:\Users\GuanXingRen\WorkBuddy\2026-09-15-22-20-12\hrdecoder_final"
os.makedirs(LOCAL, exist_ok=True)

REF = 42.20      # 历史 baseline
D_SINGLE = 42.575  # D@2000，项目此前最好单模型
BEST_915 = 43.149  # 昨晚最好

LAB = {
    "D_notta_test": "D 单模型（无TTA）",
    "D_flip_test": "D 单模型 + 翻转TTA",
    "L_notta_test": "L 单模型（Dice+0.1Lovasz）",
    "L_flip_test": "L 单模型 + TTA",
    "B_notta_test": "B 单模型",
    "E4_notta_test": "E4 单模型",
    "DL_flip_test": "集成 D+L",
    "DBE4_flip_test": "集成 D+B+E4",
    "D2B1E41_flip_test": "集成 D×2+B+E4",
    "D3B1E41_flip_test": "集成 D×3+B+E4 ← 昨晚最好",
    "DLBE4_flip_test": "集成 D+L+B+E4",
    "TEST_E_head": "★ E（B基准+MA残差头）单模型",
    "TEST_E2_head": "E2（D基准+MA残差头）单模型",
    "DE_flip_test": "集成 D+E",
    "DEB_E4_flip_test": "集成 D+E+B+E4",
    "D2E1B1E41_flip_test": "集成 D×2+E+B+E4",
    "D1E2B1E41_flip_test": "集成 D+E×2+B+E4",
    "D3E2B1E41_flip_test": "集成 D×3+E×2+B+E4",
    "VAL_D_ref": "VAL D 单模型",
    "VAL_D_renorm_stu": "VAL D + BN重估(student)",
    "VAL_D_renorm_all": "VAL D + BN重估(all)",
    "VAL_B_base": "VAL B 起点（E 的真实基线）",
    "VAL_E1500": "VAL E@1500",
    "SCALE_VALID_D": "D 单模型（scales=2048，脚本验证）",
    "SCALE_D_2048_2560": "D 单模型 + 多尺度(2048,2560)",
    "ENS_DE_flip_2scale": "集成 D+E + 翻转 + 多尺度",
    "ENS_D2E1B1E41_2scale": "集成 D×2+E+B+E4 + 翻转 + 多尺度",
    "ENS_DE_2scale": "集成 D+E + 多尺度",
}


def sh(cli, cmd):
    _, out, _ = cli.exec_command(cmd)
    return out.read().decode("utf-8", "replace")


def main():
    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    cli.connect(HOST, port=PORT, username=USER, password=PWD, timeout=30,
                allow_agent=False, look_for_keys=False)
    sftp = cli.open_sftp()

    rows = []
    for tag, rd in DIRS.items():
        for rp in sh(cli, "ls %s/eval_*.json 2>/dev/null" % rd).split():
            base = os.path.basename(rp)
            key = base.replace("eval_", "").replace(".json", "")
            lp = os.path.join(LOCAL, "%s__%s" % (tag, base))
            try:
                sftp.get(rp, lp)
                with open(lp, encoding="utf-8") as f:
                    d = json.load(f)
            except Exception as e:
                print("skip", rp, e)
                continue
            d["_key"] = key
            d["_label"] = LAB.get(key, key)
            d["_tag"] = tag
            rows.append(d)

    for tag, rd in DIRS.items():
        for fn in ["e_train.log", "e_stage.log", "renorm.log", "ens_de.log",
                   "test_heads.log", "smoke.json", "smoke_maond.json"]:
            try:
                sftp.get("%s/%s" % (rd, fn),
                         os.path.join(LOCAL, "%s__%s" % (tag, fn)))
            except Exception:
                pass
    sftp.close()
    cli.close()

    test = [r for r in rows if r.get("split") == "test"]
    val = [r for r in rows if r.get("split") == "val"]
    test.sort(key=lambda d: -d["mIoU"])

    L = []
    A = L.append
    A("# HRDecoder 两轮实验最终报告（2026-09-15 / 09-16）")
    A("")
    A("> 生成：%s" % datetime.now().strftime("%Y-%m-%d %H:%M"))
    A("> 协议：test 227 张 / val 76 张，4 类 IoU 不含背景，sigmoid 阈值 0.5，"
      "train-val-test = 454/76/227 划分不变，未使用任何外部数据。")
    A("> 评估器可信度：D 复现 42.576（官方 42.58）、B 复现 41.918（41.92）、"
      "E4 复现 41.626（41.63），偏差 < 0.005。")
    A("")
    A("**目标 test 45.35 未达成。** 最好成绩 %.3f，比历史 baseline 42.20 高 %+.3f。"
      % (test[0]["mIoU"] * 100 if test else 0,
         (test[0]["mIoU"] * 100 - REF) if test else 0))
    A("")
    A("---")
    A("")
    A("## 一、test 集全部结果（按 mIoU 降序）")
    A("")
    A("| # | 方案 | 模型数 | TTA | test mIoU | vs 42.20 | EX | HE | SE | MA |")
    A("|---:|---|---:|:---:|---:|---:|---:|---:|---:|---:|")
    for i, d in enumerate(test, 1):
        pc = d.get("per_class_iou") or {}
        A("| %d | %s | %d | %s | **%.3f** | %+.3f | %s | %s | %s | %s |" % (
            i, d["_label"], len(d["models"]), "✓" if d["flip_tta"] else "",
            d["mIoU"] * 100, d["mIoU"] * 100 - REF,
            pc.get("EX", "-"), pc.get("HE", "-"),
            pc.get("SE", "-"), pc.get("MA", "-")))
    A("")
    A("## 二、val 集结果（用于诊断，不可用于预测 test）")
    A("")
    A("| 方案 | val mIoU | EX | HE | SE | MA |")
    A("|---|---:|---:|---:|---:|---:|")
    for d in sorted(val, key=lambda x: -x["mIoU"]):
        pc = d.get("per_class_iou") or {}
        A("| %s | %.3f | %s | %s | %s | %s |" % (
            d["_label"], d["mIoU"] * 100, pc.get("EX", "-"), pc.get("HE", "-"),
            pc.get("SE", "-"), pc.get("MA", "-")))
    A("")

    p = os.path.join(LOCAL, "最终报告.md")
    with open(p, "w", encoding="utf-8") as f:
        f.write("\n".join(L))

    with open(os.path.join(LOCAL, "全部结果.csv"), "w", newline="",
              encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["split", "方案", "模型数", "TTA", "mIoU",
                    "EX", "HE", "SE", "MA"])
        for d in sorted(rows, key=lambda x: (x.get("split") != "test", -x["mIoU"])):
            pc = d.get("per_class_iou") or {}
            w.writerow([d.get("split"), d["_label"], len(d["models"]),
                        d["flip_tta"], round(d["mIoU"] * 100, 4),
                        pc.get("EX", ""), pc.get("HE", ""),
                        pc.get("SE", ""), pc.get("MA", "")])

    print("OK ->", p)
    print("--- test ---")
    for d in test:
        print("  %-34s %.3f" % (d["_label"], d["mIoU"] * 100))
    print("--- val ---")
    for d in sorted(val, key=lambda x: -x["mIoU"]):
        print("  %-34s %.3f" % (d["_label"], d["mIoU"] * 100))


if __name__ == "__main__":
    main()
