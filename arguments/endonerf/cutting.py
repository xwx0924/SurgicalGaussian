ModelParams = dict(
    dataset_type='endonerf',
    depth_scale=100.0,
    frame_nums=156,
    test_id=[1, 9, 17, 25, 33, 41, 49, 57, 65, 73, 81, 89, 97, 105, 113, 121, 129, 137, 145, 153],
    is_mask=True,
    depth_initial=True,
    accurate_mask=False,
    is_depth=True,
)
OptimizationParams = dict(
    iterations=40_000,
    densify_grad_threshold=0.0003, 
    lambda_cov=40,
    lambda_pos=0.2,
)
