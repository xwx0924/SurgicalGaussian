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
    #warm_up=0,
    #densify_until_iter=10_000,
    #lambda_smooth=0.003,  # full=lambda_smooth=0.005  其他0.03
    #lambda_aiap_cov=200,  # full=100,t=10
)