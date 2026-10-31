最终配方为旧冠军75%加历史U0 25%，四类相同，严格阈值>0.5。

六份必要权重位于weights/，完整SHA和路径位于local_weight_manifest.json。original_deployment_lock.json是原服务器锁定配方的逐字节副本，保留原服务器路径及未实际启用的H1候选元数据；实际推理只使用U0。配置副本在configs/。源码与环境依赖在工作目录及清单列明的历史源码归档中。

最终结果见 reports/最终结果.md，227图计数和最终JSON位于server_results/reports/finish/。请保留原锁，不重新按test选择权重。

<!-- SERVER_CLEANUP_20261002 -->
2026-10-02按用户授权清理服务器闲置权重。最终管线D/E/B/E4/H4/U0及初始化权重仍在服务器；本轮U1/H1/A0等5个阶段的服务器best/last已删除，完整best仍在本机server_results/work_dirs/。H1只出现在原锁未启用的候选元数据中；未来需要恢复该候选时，请从本机best上传至新的实验目录，保留本轮原锁及结果。详细路径、SHA与空间变化见reports/server_weight_cleanup_result.json。
