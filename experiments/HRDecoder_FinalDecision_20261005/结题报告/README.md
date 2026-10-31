# HRDecoder 项目结题报告成果包

主报告已完成逐页检查：29页，正文约1.66万中文字符（含表与图注），11图、11表。成果与证据保持对应。

- [可编辑Word报告](HRDecoder项目结题报告.docx)
- [阅读与打印PDF](HRDecoder项目结题报告.pdf)
- [Markdown全文](HRDecoder项目结题报告.md)
- [内部证据核验清单](内部证据核验清单.md)
- [正文表格证据索引](evidence/正文表格证据索引.md)
- [79条历史成果台账CSV](evidence/historical_experiment_ledger.csv)及[JSON](evidence/historical_experiment_ledger.json)

最佳成果：S20k EMA与P0按固定25%融合，验证45.3218%，锁定测试45.8604%。相对同新DDR划分原HRDecoder报告42.20%，累计约+3.6604个百分点；相对上一系统R本轮+0.1388个百分点。报告完整区分参数父链、历史预测组成、匹配消融及各阶段未采纳的结果。

`figures`含全部图的300dpi PNG、SVG、PDF；`evidence`含图表源数据、116份主要来源SHA、逐图计数派生数据及精确证据映射；`analysis`保留仅离线使用的收集、复算、制图和文档生成脚本。`qa/final`是与当前PDF对应的29页检查图。文件名F编号沿用原分析图身份，正式图1–11与文件名映射见figure_artifact_index.json。

复算已有数据：使用本机Conda Python执行analysis/audit_completion_evidence.py；制图执行analysis/build_scientific_figures.py。文档生成使用bundled Python执行analysis/build_completion_artifacts.py；已有Word导出使用analysis/export_with_word.ps1。生成脚本读取原工作区实际证据路径，需要保留工作区目录结构。不得把离线复算当成新训练或新测试。

最终QA与文件身份见evidence/final_report_qa.json及report_delivery_manifest.json。大体积模型、原始/EMA/优化器/RNG状态继续保存在原规范server_results与保管链，不为报告包另复制一套。

服务器端训练、锁定测试、报告必要取证及两节点全备份均已完成，关机请求已发出；平台实际OFF单独等待确认。报告写作已全部在本地完成。
