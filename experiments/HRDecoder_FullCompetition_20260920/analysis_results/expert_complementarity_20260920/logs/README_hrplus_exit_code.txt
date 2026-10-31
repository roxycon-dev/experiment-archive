hrplus_exit_code.txt 说明（不得改写该文件本身——它是历史证据）
============================================================

本目录的 hrplus_exit_code.txt = "2"，时间戳 2026-09-20 21:40。

它记录的是 21:40 那次 export_hrplus_expert_masks.py 运行的真实退出码：
champion / BASE_0 / D_blend 已成功写出 contract 后，在 contract[C2000] 处
被 assert_no_probability_payload() fail closed，因为 pipeline_definition()
把一个纯文字元数据字段命名成了保留键 "probability"。

那次失败是**正确的 fail closed**，不是真的把概率数组写进了契约。

后续处理（见 manifests/REBUILD_REPORT.json，created_at 21:54:50）：
- 修复方式是给该元数据字段改名 probability_source，保留键黑名单未放松，
  并新增"任何位置出现浮点数组/纯浮点列表一律拒绝"的语义级禁令。
- C2000 / MIX_C2000 / S4000 的 contract 用 rebuild_hrplus_contracts.py
  从现有 76 张输出**离线**补写，未重跑 GPU。掩膜仍来自 21:28-21:40 的原 run。
- 回归测试 tests/test_packed_mask_contract.py 服务器实跑 12/12 通过。

因此：exit_code=2 与"契约已完整且三方核对一致"并不矛盾。
本轮的完整复核见 ../EXECUTION_REPORT_20260920.md 第 1 节与第 5 节。
