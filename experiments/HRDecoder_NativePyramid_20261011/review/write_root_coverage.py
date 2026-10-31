"""Persist the root reviewer's actual reading ledger; no model/data execution."""
import json
from pathlib import Path

here = Path(__file__).resolve().parent
entries = json.loads((here / 'reading_root.json').read_text(encoding='utf-8-sig'))
notes = {
 '00_阅读入口.md': 'Q设计与CPU既有概率诊断入口；非新训练结果。',
 '01_成熟突破方案.md': '旧专家共同漏检多；DINO及外部监督旧方案不满足本轮同W48信息预算。',
 '02_执行与验收规范.md': '微批Dice与全batch不等价；混合监督曝光量、缺标与test口径需单列。',
 '03_最终交叉审查.md': 'Q_MIX后段DDR配比和单模对照公式曾文档冲突。',
 '研究笔记_方案反方审查.md': 'GT oracle不能当部署收益；等步、等图次、等成本是不同约束。',
 '研究笔记_最终方案反方补充.md': '窗口上下文不等整图；早期低分不证明收敛或无效。',
 '研究笔记_真实监督资产.md': '54张IDRiD并非54新增病例；当时MAPLES缺照片，后续完成状态须看新证据。',
 '研究笔记_预训练语义路线.md': '语义预训练候选与本轮限制不相容，不沿用DINO/RETFound路线。',
 '论文创新定位与方法主线.md': 'H4有直接增益；系统45.8604不能归因一个模块，非目标类不变不保证目标IoU。',
 '00_阅读顺序.md': '严格复诊的原始结果/训练实现/外部数据附件入口。',
 '01_严格诊断与下一轮主方案.md': 'H1 train/eval入口不一致，推理几何混杂；HE/SE大病灶内部漏检不能仅用细节解释。',
 '02_外部数据审查_Luna.md': '类别为EX/HE/SE/MA；DDR单通道与外域重叠mask语义不同；SE训练阳性143/454。',
 '03_运行与结果审查_Luna.md': '23节点、微步曝光及原始计数；stage-local warmstart不等精确resume。',
 '04_训练实现审查_Luna.md': 'Dice微批不等价、RNG游标及eval历史恢复缺口，已有H1完整12k被写stopped。',
 '05_文献取舍与新方案依据.md': 'GN包含空间统计；PointRend支持细特征但不证明本项目增益。',
 '01_重新诊断与主方案.md': 'refiner概率clip零局部梯度及GN补零差异有实测；旧误读已修正，不沿用上限论。',
 '02_历史实验台账_审核版.md': '70条含计划/推理非70次训练；H0与T/U旧NOT_RUN被后续运行证据纠正。',
 '02b_集成与尺度候选原表.md': '22test候选是历史探索；BN重估all25.286和多尺度非一致正收益。',
 '03_文献筛选与证据.md': '文献表区分全文和摘要；不得把公开数据集涨幅搬成本项目预测。',
 '04_审查修正与证据链.md': 'OR可部署非oracle，旧委派推理不可靠；resume恒等检查仅适用fresh。',
 'README.md': '审核版优先于delegation未审稿；目录索引不等逐文件核验。',
 'complement_response.md': '已读原始委派全部文字及JSON；精确上界/局部细节归因/跨模型CI判断已被审核版纠正，不采信。',
 'history_response.md': '已读原始委派全部文字及JSON；NOT_RUN、否定路线、初始化含混等按后续审核版修正。',
 '清理结果.md': '运维：42份检查点已删除，旧备份清单不代表当前资产仍存在。',
 '清理记录.md': '运维：1730个文件删除记录；仅报告声明，未重新执行删除或逐SHA核对。',
 '01_全目录实验清单.md': '仅读目录覆盖表/索引结构；427260字符资产清单未逐行精读，不计科学正文已读。',
 '02_HRDecoder逐实验事实档案.md': '旧45条快照含H0未训等过时断言；10/02追加事实与09/26审核版逐时序使用。',
 '03_遗漏与冲突核查.md': '旧3866文件扫描有OR误判、NOT_RUN过时、R1逐点差误述，不能照抄结论。',
 '05_高分辨率配对续训_20261003.md': 'highres4k/control2k只比共同节点；val+0.2451不等新test。',
 '06_完整任务实验报告_20261003.md': '完整报告的导航文件，非独立实验。',
 '实验全景解读报告.md': '已全文；开头EX/SE中文误写、无阳性图不计分母、1024特征尺寸和统一<1pp噪声论均不能采信。',
 '本轮实验报告_20260913.md': 'D20k早峰41.615@2k后40.476，BN已冻结；原本未test后来有D-TEST补充。',
}
out = []
for entry in entries:
    out.append({
        'primary_path': entry['primary_path'], 'paths': entry['paths'],
        'sha256': entry['sha256'],
        'actual_read_status': 'index_only' if entry['is_directory_index'] else 'full_read',
        'finding': notes[Path(entry['primary_path']).name],
        'reviewer': 'root', 'scope': '2026-10-11 historical document review; not all referenced raw files'
    })
(here / 'coverage_root.json').write_text(json.dumps(out, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print({'full_read': sum(x['actual_read_status']=='full_read' for x in out), 'index_only': sum(x['actual_read_status']=='index_only' for x in out)})
