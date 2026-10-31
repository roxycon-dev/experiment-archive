_base_ = './hrplus_patch_dice.py'

# HR+ patch lovasz variant: ONLY the lovasz weight changes; identical init,
# data, schedule and inference as the dice variant.
model = dict(lovasz_weight=0.1)
