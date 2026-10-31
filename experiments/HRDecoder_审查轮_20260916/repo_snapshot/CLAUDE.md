# HRDecoder 项目

## 沟通约定
- 始终使用简体中文回复
- 回复简洁，避免冗长套话
- 不要解释显而易见的事
- 引用代码时附 `文件:行号` 便于定位

## 项目背景
- HRDecoder：基于 MMsegmentation 的眼底图像病灶分割网络
- 数据集：IDRiD、DDR（详见 README.md）
- 环境：Python 3.8.5 / PyTorch 2.0.1 / mmsegmentation 0.16.0
- 关键路径：`configs/lesion/`、`mmseg/`、`tools/`、`work_dirs/`

## 代码规范
- 优先编辑现有文件，不创建新文件
- 默认不写注释，只在 WHY 不明显时写
- 避免抽象化、为假设场景设计
- 遵循项目现有代码风格（不要无谓重构）
- 错误处理只加在系统边界（用户输入、外部 API）
- 中文回复时，标点用全角；代码内标识符保持英文
