"""P1: 补齐 Codex 原始设计的「语义条件 MA 残差头」——冻结基准主干，只训练这个头。

设计来源（R1 诊断报告 §4 优先级二，原文）：
    MA最终logit = MA基准logit + 修正头(原图细节特征, 对齐的粗语义特征, 基准MA分数)
    「冻结成熟基准网络及其 BN 统计，只训练这个修正头」
    「其余三类不下降是由实现隔离保证的」

与旧的 A 方案（hr100_refinement.SemanticMAResidual）的两处关键差异：
  1. 修正头接收第三个输入「基准 MA 分数」（旧版只有 detail + semantic）
  2. 学生主干全部冻结，只有 ma_refiner 可训练（旧版是全网络低学习率微调）

结构不变量（用于硬检验）：
  - 残差输出层零初始化 → 未训练时 MA logit == 基准 MA logit，模型输出与 D 逐位相同
  - refine_logits 只改写 MA 通道，EX/HE/SE 逐位透传
"""
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.models.builder import SEGMENTORS
from hr100_refinement import HR100Refinement
from hrplus_patch import HRPlusRefinement, _assert_config, _freeze_all_bn


def _block(cin, cout, stride=1):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride, 1, bias=False),
                         nn.GroupNorm(8, cout), nn.ReLU(inplace=True))


class SemanticMAResidualV2(nn.Module):
    """三输入语义条件残差头；输出层零初始化。

    detail:   原图 3 通道 → stride 4 特征（16→32）
    semantic: 主干 720 通道 → 1x1 压到 32（先压缩，避免整张 720 通道上采样）
    base:     基准 MA logit（1 通道）→ 8 通道
    """

    def __init__(self, detail_ch=16, sem_ch=32, sem_in=720, base_ch=8):
        super().__init__()
        self.detail = nn.Sequential(_block(3, detail_ch, 2),
                                    _block(detail_ch, detail_ch * 2, 2))
        self.semantic = nn.Sequential(
            nn.Conv2d(sem_in, sem_ch, 1, bias=False),
            nn.GroupNorm(8, sem_ch), nn.ReLU(inplace=True))
        self.base = nn.Sequential(
            nn.Conv2d(1, base_ch, 3, 1, 1, bias=False),
            nn.GroupNorm(8, base_ch), nn.ReLU(inplace=True))
        cin = detail_ch * 2 + sem_ch + base_ch
        self.fusion = nn.Sequential(_block(cin, 32), _block(32, 32),
                                    nn.Conv2d(32, 1, 1))
        nn.init.zeros_(self.fusion[-1].weight)
        nn.init.zeros_(self.fusion[-1].bias)

    def forward(self, image, semantics, base_ma):
        detail = self.detail(image)                       # 原图 1/4
        hw = detail.shape[-2:]
        semantic = F.interpolate(self.semantic(semantics), size=hw,
                                 mode='bilinear', align_corners=False)
        base = self.base(F.interpolate(base_ma, size=hw,
                                       mode='bilinear', align_corners=False))
        residual = self.fusion(torch.cat([detail, semantic, base], 1))
        return F.interpolate(residual, size=image.shape[-2:], mode='bilinear',
                             align_corners=False)


def _load_allow_head(module, path, allow_prefix='ma_refiner.'):
    """严格加载，但允许 allow_prefix 下的键缺失（新头走零初始化）。

    其余任何缺失/多余都报错，避免静默加载错权重。
    """
    ckpt = torch.load(path, map_location='cpu')
    state = ckpt.get('state_dict', ckpt)
    state = {k[7:] if k.startswith('module.') else k: v for k, v in state.items()}
    model_keys = set(module.state_dict().keys())
    missing = sorted(model_keys - set(state.keys()))
    unexpected = sorted(set(state.keys()) - model_keys)
    bad = [k for k in missing
           if not k.startswith(allow_prefix) and 'num_batches_tracked' not in k]
    assert not bad, 'unexpected missing keys: %s' % bad[:12]
    assert not unexpected, 'unexpected keys in ckpt: %s' % unexpected[:12]
    module.load_state_dict(state, strict=False)
    return [k for k in missing if k.startswith(allow_prefix)]


def _load_student_only(module, path, allow_prefix='reference.'):
    """严格加载 student 权重；允许 ckpt 里多带 reference.*（那是 D 自带的参考分支）。

    用途：把 D@2000（HRPlusRefinement 完整 checkpoint）当作冻结基准，但只用它的
    student 部分；参考分支另用独立的 B 权重构造。
    """
    ckpt = torch.load(path, map_location='cpu')
    state = ckpt.get('state_dict', ckpt)
    state = {k[7:] if k.startswith('module.') else k: v for k, v in state.items()}
    model_keys = set(module.state_dict().keys())
    unexpected = sorted(k for k in set(state) - model_keys
                        if not k.startswith(allow_prefix))
    missing = sorted(k for k in model_keys - set(state)
                     if 'num_batches_tracked' not in k)
    assert not unexpected, 'unexpected keys in ckpt: %s' % unexpected[:12]
    assert not missing, 'missing keys: %s' % missing[:12]
    module.load_state_dict({k: v for k, v in state.items() if k in model_keys},
                           strict=True)
    return {'unexpected_ignored': len(set(state) - model_keys),
            'missing': 0}


@SEGMENTORS.register_module()
class HRPlusMAOnBase(HRPlusRefinement):
    """把「冻结成熟基准 + 只训 MA 残差头」应用到项目最好的模型上。

    student  : source_checkpoint（默认传 D@2000，含 reference.*）→ 冻结 + 可训练 MA 头
    reference: reference_checkpoint（默认传 B）→ 冻结，用于推理时的 0.5 概率融合

    与 HRPlusMAResidual 的区别：后者把 student 和 reference 都绑到同一个 source，
    因此只能以 B 为基准；本类允许两者分开，从而能以 D@2000 为基准。
    """

    def __init__(self, source_checkpoint, reference_checkpoint=None,
                 ma_hard_weight=0.0, detail_ch=16, sem_ch=32, sem_in=720,
                 **kwargs):
        assert source_checkpoint, 'source_checkpoint is REQUIRED'
        kwargs.pop('lovasz_weight', None)
        kwargs['semantic_refinement'] = False
        kwargs['hard_weight'] = 0
        kwargs['freeze_bn'] = True
        ref_kwargs = copy.deepcopy(kwargs)
        ref_ckpt = reference_checkpoint or source_checkpoint

        # 1) 建 student（无 ma_refiner、无 reference），严格加载 source
        HR100Refinement.__init__(self, **kwargs)
        _assert_config(self)
        stu_info = _load_student_only(self, source_checkpoint)
        _freeze_all_bn(self)
        self.lovasz_weight = 0.0
        self.ma_hard_weight = float(ma_hard_weight)
        print('[HRPlusMAOnBase] student <- %s (%s)'
              % (source_checkpoint, stu_info), flush=True)

        # 2) 独立构造并冻结 reference
        reference = HR100Refinement(**ref_kwargs)
        ref_info = _load_student_only(reference, ref_ckpt)
        for p in reference.parameters():
            p.requires_grad_(False)
        _freeze_all_bn(reference)
        reference.eval()
        self.reference = reference
        print('[HRPlusMAOnBase] reference <- %s (%s)' % (ref_ckpt, ref_info),
              flush=True)

        # 3) 装上可训练的 MA 残差头，并冻结其余一切
        self.ma_refiner = SemanticMAResidualV2(detail_ch, sem_ch, sem_in)
        n_tr = n_fr = 0
        for name, p in self.named_parameters():
            keep = name.startswith('ma_refiner.')
            p.requires_grad_(keep)
            if keep:
                n_tr += p.numel()
            else:
                n_fr += p.numel()
        print('[HRPlusMAOnBase] 可训练 %.3fM / 冻结 %.3fM'
              % (n_tr / 1e6, n_fr / 1e6), flush=True)

    def train(self, mode=True):
        super(HRPlusMAOnBase, self).train(mode)
        if hasattr(self, 'reference'):
            self.reference.eval()
        return self

    def frozen_base_fingerprint(self):
        import hashlib
        h = hashlib.sha256()
        for n, p in sorted(self.named_parameters()):
            if n.startswith('ma_refiner.'):
                continue
            h.update(n.encode())
            h.update(p.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()

    def refine_logits(self, logits, img, semantics):
        if self.ma_refiner is None:
            return logits
        base_ma = logits[:, 3:4]
        ma = base_ma + self.ma_refiner(img, semantics, base_ma)
        return torch.cat([logits[:, :3], ma], dim=1)

    def forward_train(self, img, img_metas, gt_semantic_seg,
                      seg_weight=None, return_feat=False, **kwargs):
        """Dice 主目标（R1 设计第一阶段）+ 可选的低权重 MA 难负例项（第二阶段）。

        ma_hard_weight=0 时与第一阶段完全一致，可直接对照。
        """
        logits = self.encode_decode(img, img_metas)
        dice_logits = torch.where(gt_semantic_seg == 255,
                                  logits.new_tensor(-1e4), logits)
        losses = self.decode_head.losses(dice_logits, gt_semantic_seg)
        if getattr(self, 'ma_hard_weight', 0.0) > 0:
            from hr100_refinement import ma_hard_loss
            losses['loss_ma_hard'] = self.ma_hard_weight * ma_hard_loss(
                logits, gt_semantic_seg)
        return losses


@SEGMENTORS.register_module()
class HRPlusMAResidual(HRPlusRefinement):
    """学生 = 冻结的 D 主干 + 可训练语义条件 MA 残差头；参考 = 冻结 B（整图）。

    必须继承 HRPlusRefinement：推理路径（tiled 学生 + 0.5 冻结 B 融合）在
    HRPlusRefinement.inference 里，继承 HR100Refinement 会退化成整图单次前向，
    权重相同但输出完全不同（实测最大差 0.40）。
    """

    def __init__(self, source_checkpoint, ma_hard_weight=0.0,
                 detail_ch=16, sem_ch=32, sem_in=720, **kwargs):
        # 父类配置里带 lovasz_weight，本类固定为 0（L 已被证伪），先摘掉避免重复传参
        kwargs.pop('lovasz_weight', None)
        # 父类会把 semantic_refinement 强制为 False 并严格加载 B 权重、
        # 建好冻结的 reference；此时 ma_refiner 为 None，所以严格加载不会缺键。
        super(HRPlusMAResidual, self).__init__(
            source_checkpoint=source_checkpoint, lovasz_weight=0.0, **kwargs)

        # 装上可训练的 MA 残差头（零初始化 → 初始输出等于 D）
        self.ma_refiner = SemanticMAResidualV2(detail_ch, sem_ch, sem_in)
        self.ma_hard_weight = float(ma_hard_weight)

        # 冻结学生主干：除 ma_refiner 外全部 requires_grad=False
        n_fr = n_tr = 0
        for name, p in self.named_parameters():
            keep = name.startswith('ma_refiner.')
            p.requires_grad_(keep)
            if keep:
                n_tr += p.numel()
            else:
                n_fr += p.numel()
        print('[HRPlusMAResidual] 可训练 %.3fM / 冻结 %.3fM'
              % (n_tr / 1e6, n_fr / 1e6), flush=True)

    # ---- 生命周期守卫 -----------------------------------------------------
    def train(self, mode=True):
        super(HRPlusMAResidual, self).train(mode)
        if hasattr(self, 'reference'):
            self.reference.eval()
        return self

    def frozen_base_fingerprint(self):
        """冻结主干参数的哈希指纹，用于检验「冻结是否真的成立」。"""
        import hashlib
        h = hashlib.sha256()
        for n, p in sorted(self.named_parameters()):
            if n.startswith('ma_refiner.'):
                continue
            h.update(n.encode())
            h.update(p.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()

    # ---- 关键：只改写 MA 通道 --------------------------------------------
    def refine_logits(self, logits, img, semantics):
        if self.ma_refiner is None:
            return logits
        base_ma = logits[:, 3:4]
        ma = base_ma + self.ma_refiner(img, semantics, base_ma)
        return torch.cat([logits[:, :3], ma], dim=1)
