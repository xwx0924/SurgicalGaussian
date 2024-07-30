ModelParams = dict(
    dataset_type='endonerf',
    depth_scale=100.0,
    frame_nums=63,
    test_id=[1, 9, 17, 25, 33, 41, 49, 57],
    is_mask=True,
    depth_initial=True,
    accurate_mask=True,
    is_depth=True,
)

OptimizationParams = dict(
    iterations=40_000,
)