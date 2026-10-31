_base_ = './hr100_a.py'
model = dict(type='HR100Refinement', semantic_refinement=False)
data = dict(test=dict(img_dir='images/val', ann_dir='labels/val'))
