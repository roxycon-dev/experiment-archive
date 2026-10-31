_base_ = './hr100_b.py'
custom_imports = dict(imports=['hr100_ensemble'], allow_failed_imports=False)
model = dict(type='HR100ClasswiseEnsemble', primary_classes=(0, 2),
    reference_checkpoint='/root/HRDecoder/work_dirs/exp_e4_detail_split613/best_mIoU_iter_36000.pth')
data = dict(test=dict(img_dir='images/val', ann_dir='labels/val'))
