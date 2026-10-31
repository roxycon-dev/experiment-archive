_base_ = './hr100_mean_test.py'
model = dict(reference_checkpoint='weights/e4_reference_weights.pth')
data = dict(train=dict(data_root='data/DDR_split_613'), val=dict(data_root='data/DDR_split_613'), test=dict(data_root='data/DDR_split_613'))
