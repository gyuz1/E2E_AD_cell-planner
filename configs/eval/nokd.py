"""평가·제출용 설정 -- configs/nokd.py 의 짝.

학습 설정과 **짝이 맞아야 한다.** 어긋나면 다른 네트워크를 채점하게 되고 아무
경고도 나오지 않는다.

학습 쪽과 다른 점은 데이터 파이프라인뿐이다. 학습은 미리 구운 geometry cache 에서
읽고, 여기서는 원본 이미지를 읽는다 -- 실제 추론이 치르는 비용을 그대로 재기
위해서다. 네트워크는 같다.
"""

point_cloud_range = [-30.0, -15.0, -2.0, 30.0, 15.0, 2.0]
class_names = ['Car', 'Pedestrian', 'Cyclist']
dataset_type = 'LAWVADCustomETRIDataset'
data_root = 'data/etri/.causal_regen_split_301_75/'
input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=True)
file_client_args = dict(backend='disk')
train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='UndistortMultiViewImage'),
    dict(type='CropMultiViewImage'),
    dict(type='PhotoMetricDistortionMultiViewImage'),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True,
        with_attr_label=True),
    dict(
        type='CustomObjectRangeFilter',
        point_cloud_range=[-30.0, -15.0, -2.0, 30.0, 15.0, 2.0]),
    dict(
        type='CustomObjectNameFilter',
        classes=['Car', 'Pedestrian', 'Cyclist']),
    dict(
        type='NormalizeMultiviewImage',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='RandomScaleImageMultiViewImage', scales=[0.4]),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(
        type='CustomDefaultFormatBundle3D',
        class_names=['Car', 'Pedestrian', 'Cyclist'],
        with_ego=True),
    dict(
        type='CustomCollect3D',
        keys=[
            'gt_bboxes_3d', 'gt_labels_3d', 'img', 'ego_his_trajs',
            'ego_fut_trajs', 'ego_fut_masks', 'ego_fut_cmd', 'ego_lcf_feat',
            'ego_target_point', 'ego_long_fut_trajs', 'ego_long_fut_masks',
            'ego_long_fut_valid_flag', 'gt_attr_labels'
        ])
]
test_pipeline = [
    dict(
        type='FastLoadMultiViewImageFromFiles',
        to_float32=False,
        reduced_decode=2),
    dict(type='FastUndistortCropScaleMultiViewImage', scale=0.4),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True,
        with_attr_label=True),
    dict(
        type='CustomObjectRangeFilter',
        point_cloud_range=[-30.0, -15.0, -2.0, 30.0, 15.0, 2.0]),
    dict(
        type='CustomObjectNameFilter',
        classes=['Car', 'Pedestrian', 'Cyclist']),
    dict(
        type='NormalizeMultiviewImage',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1920, 1080),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(type='PadMultiViewImage', size_divisor=32),
            dict(
                type='CustomDefaultFormatBundle3D',
                class_names=['Car', 'Pedestrian', 'Cyclist'],
                with_label=False,
                with_ego=True),
            dict(
                type='CustomCollect3D',
                keys=[
                    'gt_bboxes_3d', 'gt_labels_3d', 'img', 'fut_valid_flag',
                    'ego_his_trajs', 'ego_fut_trajs', 'ego_fut_masks',
                    'ego_fut_cmd', 'ego_lcf_feat', 'ego_target_point',
                    'ego_long_fut_trajs', 'ego_long_fut_masks',
                    'ego_long_fut_valid_flag', 'gt_attr_labels'
                ])
        ])
]
eval_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=dict(backend='disk')),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=10,
        file_client_args=dict(backend='disk')),
    dict(
        type='DefaultFormatBundle3D',
        class_names=[
            'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
            'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
        ],
        with_label=False),
    dict(type='Collect3D', keys=['points'])
]
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
        type='LAWVADCustomETRIDataset',
        data_root='data/etri/.causal_regen_split_301_75/',
        ann_file=
        'data/etri/.causal_regen_split_301_75/vad_etri_infos_temporal_train_split.pkl',
        pipeline=[
            dict(type='LoadMultiViewImageFromFiles', to_float32=True),
            dict(type='UndistortMultiViewImage'),
            dict(type='CropMultiViewImage'),
            dict(type='PhotoMetricDistortionMultiViewImage'),
            dict(
                type='LoadAnnotations3D',
                with_bbox_3d=True,
                with_label_3d=True,
                with_attr_label=True),
            dict(
                type='CustomObjectRangeFilter',
                point_cloud_range=[-30.0, -15.0, -2.0, 30.0, 15.0, 2.0]),
            dict(
                type='CustomObjectNameFilter',
                classes=['Car', 'Pedestrian', 'Cyclist']),
            dict(
                type='NormalizeMultiviewImage',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='RandomScaleImageMultiViewImage', scales=[0.4]),
            dict(type='PadMultiViewImage', size_divisor=32),
            dict(
                type='CustomDefaultFormatBundle3D',
                class_names=['Car', 'Pedestrian', 'Cyclist'],
                with_ego=True),
            dict(
                type='CustomCollect3D',
                keys=[
                    'gt_bboxes_3d', 'gt_labels_3d', 'img', 'ego_his_trajs',
                    'ego_fut_trajs', 'ego_fut_masks', 'ego_fut_cmd',
                    'ego_lcf_feat', 'ego_target_point', 'ego_long_fut_trajs',
                    'ego_long_fut_masks', 'ego_long_fut_valid_flag',
                    'gt_attr_labels'
                ])
        ],
        classes=['Car', 'Pedestrian', 'Cyclist'],
        modality=dict(
            use_lidar=False,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=True),
        test_mode=False,
        box_type_3d='LiDAR',
        use_valid_flag=True,
        bev_size=(100, 100),
        pc_range=[-30.0, -15.0, -2.0, 30.0, 15.0, 2.0],
        queue_length=3,
        target_stride=5,
        map_classes=['divider', 'ped_crossing', 'boundary'],
        map_fixed_ptsnum_per_line=20,
        map_eval_use_same_gt_sample_num_flag=True,
        crop_keep_top=('camera_front_left', 'camera_front_right',
                       'camera_rear_left', 'camera_rear_right',
                       'camera_rear_wide'),
        custom_eval_version='vad_nusc_detection_cvpr_2019'),
    val=dict(
        type='LAWVADCustomETRIDataset',
        ann_file=
        'data/etri/.causal_regen_split_301_75/vad_etri_infos_temporal_val_split.pkl',
        pipeline=[
            dict(type='LoadMultiViewImageFromFiles', to_float32=True),
            dict(type='UndistortMultiViewImage'),
            dict(type='CropMultiViewImage'),
            dict(
                type='LoadAnnotations3D',
                with_bbox_3d=True,
                with_label_3d=True,
                with_attr_label=True),
            dict(
                type='CustomObjectRangeFilter',
                point_cloud_range=[-30.0, -15.0, -2.0, 30.0, 15.0, 2.0]),
            dict(
                type='CustomObjectNameFilter',
                classes=['Car', 'Pedestrian', 'Cyclist']),
            dict(
                type='NormalizeMultiviewImage',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(
                type='MultiScaleFlipAug3D',
                img_scale=(1920, 1080),
                pts_scale_ratio=1,
                flip=False,
                transforms=[
                    dict(type='RandomScaleImageMultiViewImage', scales=[0.4]),
                    dict(type='PadMultiViewImage', size_divisor=32),
                    dict(
                        type='CustomDefaultFormatBundle3D',
                        class_names=['Car', 'Pedestrian', 'Cyclist'],
                        with_label=False,
                        with_ego=True),
                    dict(
                        type='CustomCollect3D',
                        keys=[
                            'gt_bboxes_3d', 'gt_labels_3d', 'img',
                            'fut_valid_flag', 'ego_his_trajs', 'ego_fut_trajs',
                            'ego_fut_masks', 'ego_fut_cmd', 'ego_lcf_feat',
                            'ego_target_point', 'ego_long_fut_trajs',
                            'ego_long_fut_masks', 'ego_long_fut_valid_flag',
                            'gt_attr_labels'
                        ])
                ])
        ],
        classes=['Car', 'Pedestrian', 'Cyclist'],
        modality=dict(
            use_lidar=False,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=True),
        test_mode=True,
        box_type_3d='LiDAR',
        data_root='data/etri/.causal_regen_split_301_75/',
        pc_range=[-30.0, -15.0, -2.0, 30.0, 15.0, 2.0],
        bev_size=(100, 100),
        samples_per_gpu=1,
        map_classes=['divider', 'ped_crossing', 'boundary'],
        map_fixed_ptsnum_per_line=20,
        map_eval_use_same_gt_sample_num_flag=True,
        crop_keep_top=('camera_front_left', 'camera_front_right',
                       'camera_rear_left', 'camera_rear_right',
                       'camera_rear_wide'),
        use_pkl_result=True,
        custom_eval_version='vad_nusc_detection_cvpr_2019'),
    test=dict(
        type='LAWVADCustomETRIDataset',
        data_root='data/etri/.causal_regen_split_301_75/',
        ann_file=
        'data/etri/.causal_regen_split_301_75/vad_etri_infos_temporal_val_split.pkl',
        pipeline=[
            dict(
                type='FastLoadMultiViewImageFromFiles',
                to_float32=False,
                reduced_decode=2),
            dict(type='FastUndistortCropScaleMultiViewImage', scale=0.4),
            dict(
                type='LoadAnnotations3D',
                with_bbox_3d=True,
                with_label_3d=True,
                with_attr_label=True),
            dict(
                type='CustomObjectRangeFilter',
                point_cloud_range=[-30.0, -15.0, -2.0, 30.0, 15.0, 2.0]),
            dict(
                type='CustomObjectNameFilter',
                classes=['Car', 'Pedestrian', 'Cyclist']),
            dict(
                type='NormalizeMultiviewImage',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(
                type='MultiScaleFlipAug3D',
                img_scale=(1920, 1080),
                pts_scale_ratio=1,
                flip=False,
                transforms=[
                    dict(type='PadMultiViewImage', size_divisor=32),
                    dict(
                        type='CustomDefaultFormatBundle3D',
                        class_names=['Car', 'Pedestrian', 'Cyclist'],
                        with_label=False,
                        with_ego=True),
                    dict(
                        type='CustomCollect3D',
                        keys=[
                            'gt_bboxes_3d', 'gt_labels_3d', 'img',
                            'fut_valid_flag', 'ego_his_trajs', 'ego_fut_trajs',
                            'ego_fut_masks', 'ego_fut_cmd', 'ego_lcf_feat',
                            'ego_target_point', 'ego_long_fut_trajs',
                            'ego_long_fut_masks', 'ego_long_fut_valid_flag',
                            'gt_attr_labels'
                        ])
                ])
        ],
        classes=['Car', 'Pedestrian', 'Cyclist'],
        modality=dict(
            use_lidar=False,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=True),
        test_mode=True,
        box_type_3d='LiDAR',
        pc_range=[-30.0, -15.0, -2.0, 30.0, 15.0, 2.0],
        bev_size=(100, 100),
        samples_per_gpu=1,
        map_classes=['divider', 'ped_crossing', 'boundary'],
        map_fixed_ptsnum_per_line=20,
        map_eval_use_same_gt_sample_num_flag=True,
        crop_keep_top=('camera_front_left', 'camera_front_right',
                       'camera_rear_left', 'camera_rear_right',
                       'camera_rear_wide'),
        use_pkl_result=True,
        custom_eval_version='vad_nusc_detection_cvpr_2019'),
    shuffler_sampler=dict(type='DistributedGroupSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler'))
evaluation = dict(
    interval=13,
    pipeline=[
        dict(type='LoadMultiViewImageFromFiles', to_float32=True),
        dict(type='UndistortMultiViewImage'),
        dict(type='CropMultiViewImage'),
        dict(
            type='LoadAnnotations3D',
            with_bbox_3d=True,
            with_label_3d=True,
            with_attr_label=True),
        dict(
            type='CustomObjectRangeFilter',
            point_cloud_range=[-30.0, -15.0, -2.0, 30.0, 15.0, 2.0]),
        dict(
            type='CustomObjectNameFilter',
            classes=['Car', 'Pedestrian', 'Cyclist']),
        dict(
            type='NormalizeMultiviewImage',
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            to_rgb=True),
        dict(
            type='MultiScaleFlipAug3D',
            img_scale=(1920, 1080),
            pts_scale_ratio=1,
            flip=False,
            transforms=[
                dict(type='RandomScaleImageMultiViewImage', scales=[0.4]),
                dict(type='PadMultiViewImage', size_divisor=32),
                dict(
                    type='CustomDefaultFormatBundle3D',
                    class_names=['Car', 'Pedestrian', 'Cyclist'],
                    with_label=False,
                    with_ego=True),
                dict(
                    type='CustomCollect3D',
                    keys=[
                        'gt_bboxes_3d', 'gt_labels_3d', 'img',
                        'fut_valid_flag', 'ego_his_trajs', 'ego_fut_trajs',
                        'ego_fut_masks', 'ego_fut_cmd', 'ego_lcf_feat',
                        'ego_target_point', 'ego_long_fut_trajs',
                        'ego_long_fut_masks', 'ego_long_fut_valid_flag',
                        'gt_attr_labels'
                    ])
            ])
    ],
    metric='bbox',
    map_metric='chamfer')
checkpoint_config = dict(interval=1, max_keep_ckpts=12)
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
        dict(
            type='WandbLoggerHook',
            init_kwargs=dict(
                project='etri-2026-e2e-vad', name='stage2_split_301/75'))
    ])
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = None
load_from = 'work_dirs/stage1_etri_v2/stage2_init_merged.pth'
resume_from = None
workflow = [('train', 1)]
plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'
voxel_size = [0.15, 0.15, 8]
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
num_classes = 3
map_classes = ['divider', 'ped_crossing', 'boundary']
map_num_vec = 100
map_fixed_ptsnum_per_gt_line = 20
map_fixed_ptsnum_per_pred_line = 20
map_eval_use_same_gt_sample_num_flag = True
map_num_classes = 3
_dim_ = 256
_pos_dim_ = 128
_ffn_dim_ = 512
_num_levels_ = 1
bev_h_ = 100
bev_w_ = 100
queue_length = 3
total_epochs = 12
crop_keep_top = ('camera_front_left', 'camera_front_right', 'camera_rear_left',
                 'camera_rear_right', 'camera_rear_wide')
model = dict(
    type='VADLAW',
    use_grid_mask=True,
    video_test_mode=True,
    use_ego_lcf_status=False,
    wm_loss_weight=0.2,
    wm_num_layers=2,
    wm_num_heads=8,
    wm_num_points=4,
    wm_ffn_dims=512,
    wm_dropout=0.1,
    pretrained=dict(img='ckpts/resnet50-19c8e357.pth'),
    img_backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(3, ),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch'),
    img_neck=dict(
        type='FPN',
        in_channels=[2048],
        out_channels=256,
        start_level=0,
        add_extra_convs='on_output',
        num_outs=1,
        relu_before_extra_convs=True),
    pts_bbox_head=dict(
        type='VADHead',
        map_thresh=0.5,
        dis_thresh=0.2,
        pe_normalization=True,
        tot_epoch=12,
        use_traj_lr_warmup=False,
        query_thresh=0.0,
        query_use_fix_pad=False,
        ego_his_encoder=None,
        ego_lcf_feat_idx=None,
        valid_fut_ts=6,
        ego_fut_mode=7,
        prism_latent_supervision=False,
        prism_num_samples=2,
        bev_residual_refine=False,
        ego_agent_decoder=dict(
            type='CustomTransformerDecoder',
            num_layers=1,
            return_intermediate=False,
            transformerlayers=dict(
                type='BaseTransformerLayer',
                attn_cfgs=[
                    dict(
                        type='MultiheadAttention',
                        embed_dims=256,
                        num_heads=8,
                        dropout=0.1)
                ],
                feedforward_channels=512,
                ffn_dropout=0.1,
                operation_order=('cross_attn', 'norm', 'ffn', 'norm'))),
        ego_map_decoder=dict(
            type='CustomTransformerDecoder',
            num_layers=1,
            return_intermediate=False,
            transformerlayers=dict(
                type='BaseTransformerLayer',
                attn_cfgs=[
                    dict(
                        type='MultiheadAttention',
                        embed_dims=256,
                        num_heads=8,
                        dropout=0.1)
                ],
                feedforward_channels=512,
                ffn_dropout=0.1,
                operation_order=('cross_attn', 'norm', 'ffn', 'norm'))),
        motion_decoder=dict(
            type='CustomTransformerDecoder',
            num_layers=1,
            return_intermediate=False,
            transformerlayers=dict(
                type='BaseTransformerLayer',
                attn_cfgs=[
                    dict(
                        type='MultiheadAttention',
                        embed_dims=256,
                        num_heads=8,
                        dropout=0.1)
                ],
                feedforward_channels=512,
                ffn_dropout=0.1,
                operation_order=('cross_attn', 'norm', 'ffn', 'norm'))),
        motion_map_decoder=dict(
            type='CustomTransformerDecoder',
            num_layers=1,
            return_intermediate=False,
            transformerlayers=dict(
                type='BaseTransformerLayer',
                attn_cfgs=[
                    dict(
                        type='MultiheadAttention',
                        embed_dims=256,
                        num_heads=8,
                        dropout=0.1)
                ],
                feedforward_channels=512,
                ffn_dropout=0.1,
                operation_order=('cross_attn', 'norm', 'ffn', 'norm'))),
        use_pe=True,
        bev_h=100,
        bev_w=100,
        num_query=300,
        num_classes=3,
        in_channels=256,
        sync_cls_avg_factor=True,
        with_box_refine=True,
        as_two_stage=False,
        map_num_vec=100,
        map_num_classes=3,
        map_num_pts_per_vec=20,
        map_num_pts_per_gt_vec=20,
        map_query_embed_type='instance_pts',
        map_transform_method='minmax',
        map_gt_shift_pts_pattern='v2',
        map_dir_interval=1,
        map_code_size=2,
        map_code_weights=[1.0, 1.0, 1.0, 1.0],
        transformer=dict(
            type='VADPerceptionTransformer',
            map_num_vec=100,
            map_num_pts_per_vec=20,
            rotate_prev_bev=True,
            use_shift=True,
            use_can_bus=True,
            rotate_center=[50, 50],
            embed_dims=256,
            encoder=dict(
                type='BEVFormerEncoder',
                num_layers=3,
                pc_range=[-30.0, -15.0, -2.0, 30.0, 15.0, 2.0],
                num_points_in_pillar=4,
                return_intermediate=False,
                transformerlayers=dict(
                    type='BEVFormerLayer',
                    attn_cfgs=[
                        dict(
                            type='TemporalSelfAttention',
                            embed_dims=256,
                            num_levels=1),
                        dict(
                            type='SpatialCrossAttention',
                            pc_range=[-30.0, -15.0, -2.0, 30.0, 15.0, 2.0],
                            deformable_attention=dict(
                                type='MSDeformableAttention3D',
                                embed_dims=256,
                                num_points=8,
                                num_levels=1),
                            embed_dims=256)
                    ],
                    feedforward_channels=512,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm'))),
            decoder=dict(
                type='DetectionTransformerDecoder',
                num_layers=3,
                return_intermediate=True,
                transformerlayers=dict(
                    type='DetrTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                        dict(
                            type='CustomMSDeformableAttention',
                            embed_dims=256,
                            num_levels=1)
                    ],
                    feedforward_channels=512,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm'))),
            map_decoder=dict(
                type='MapDetectionTransformerDecoder',
                num_layers=3,
                return_intermediate=True,
                transformerlayers=dict(
                    type='DetrTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                        dict(
                            type='CustomMSDeformableAttention',
                            embed_dims=256,
                            num_levels=1)
                    ],
                    feedforward_channels=512,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')))),
        bbox_coder=dict(
            type='CustomNMSFreeCoder',
            post_center_range=[-35, -20, -10.0, 35, 20, 10.0],
            pc_range=[-30.0, -15.0, -2.0, 30.0, 15.0, 2.0],
            max_num=100,
            voxel_size=[0.15, 0.15, 8],
            num_classes=3),
        map_bbox_coder=dict(
            type='MapNMSFreeCoder',
            post_center_range=[-35, -20, -35, -20, 35, 20, 35, 20],
            pc_range=[-30.0, -15.0, -2.0, 30.0, 15.0, 2.0],
            max_num=50,
            voxel_size=[0.15, 0.15, 8],
            num_classes=3),
        positional_encoding=dict(
            type='LearnedPositionalEncoding',
            num_feats=128,
            row_num_embed=100,
            col_num_embed=100),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=0.25),
        loss_traj=dict(type='L1Loss', loss_weight=0.2),
        loss_traj_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=0.2),
        loss_iou=dict(type='GIoULoss', loss_weight=0.0),
        loss_map_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_map_bbox=dict(type='L1Loss', loss_weight=0.0),
        loss_map_iou=dict(type='GIoULoss', loss_weight=0.0),
        loss_map_pts=dict(type='PtsL1Loss', loss_weight=1.0),
        loss_map_dir=dict(type='PtsDirCosLoss', loss_weight=0.005),
        loss_plan_reg=dict(type='L1Loss', loss_weight=1.0),
        loss_plan_bound=dict(
            type='PlanMapBoundLoss',
            loss_weight=0.0,
            dis_thresh=1.0,
            lane_bound_cls_idx=2,
            point_cloud_range=[-30.0, -15.0, -2.0, 30.0, 15.0, 2.0]),
        loss_plan_col=dict(
            type='PlanCollisionLoss',
            loss_weight=1.0,
            x_dis_thresh=3.0,
            y_dis_thresh=1.5,
            point_cloud_range=[-30.0, -15.0, -2.0, 30.0, 15.0, 2.0]),
        loss_plan_dir=dict(
            type='PlanMapDirectionLoss',
            loss_weight=0.5,
            point_cloud_range=[-30.0, -15.0, -2.0, 30.0, 15.0, 2.0]),
        bev_refine_steps=3,
        prism_posterior_lcf_idx=(0, 1, 4, 7),
        aux_bev_motion=True,
        aux_bev_motion_idx=(0, 1, 2, 3, 4, 7),
        aux_bev_motion_temporal=True,
        ego_status_est_dim=8,
        ego_fut_dec_hidden_dim=512,
        aux_bev_motion_frames=3,
        aux_bev_future_motion=True,
        aux_bev_future_motion_ts=6,
        aux_bev_motion_grid=8,
        aux_long_horizon=True,
        aux_long_horizon_residual=True,
        cell_planner=True,
        cell_layouts=[
            ([
                1.0, 2.0, 3.5, 6.5, 9.0, 12.0, 15.0, 17.0, 19.0, 21.0, 22.5,
                23.5, 25.5, 27.5, 29.5, 31.0, 32.5, 34.0, 35.0, 36.5, 38.0,
                39.5, 40.5, 41.5, 43.5, 45.0, 46.0, 47.5, 49.0, 50.5, 51.5,
                52.5, 53.5, 54.5, 55.5, 56.5, 57.5, 58.5, 59.5, 60.5, 61.5,
                62.5, 63.5, 64.5, 65.5, 67.0, 68.0, 69.0, 70.5, 72.0, 73.5,
                82.0, 95.0, 97.5, 99.0, 100.5, 102.0, 103.0, 104.5, 105.5,
                117.0
            ], [-20.0, 17.0]),
            ([
                1.0, 29.5, 33.5, 40.5, 44.5, 48.5, 53.0, 57.0, 62.5, 71.5,
                116.0
            ], [-16.0, 12.0]),
            ([1.0, 30.0, 38.5, 42.5, 46.5, 50.5, 55.5, 63.0, 75.0,
              115.0], [-12.0, 18.0]),
            ([1.0, 20.5, 24.5, 30.0, 48.0], [-2.0, 8.5, 27.0]),
            ([1.0, 13.5, 17.5, 22.5, 27.5, 33.0, 43.0], [-26.0, -11.5,
                                                         2.0]), None, None
        ],
        cell_pe_dim=256,
        cell_pe_wavelengths=(1.0, 300.0),
        cell_stop_tp_thresh=1.0),
    train_cfg=dict(
        pts=dict(
            grid_size=[512, 512, 1],
            voxel_size=[0.15, 0.15, 8],
            point_cloud_range=[-30.0, -15.0, -2.0, 30.0, 15.0, 2.0],
            out_size_factor=4,
            assigner=dict(
                type='HungarianAssigner3D',
                cls_cost=dict(type='FocalLossCost', weight=2.0),
                reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
                iou_cost=dict(type='IoUCost', weight=0.0),
                pc_range=[-30.0, -15.0, -2.0, 30.0, 15.0, 2.0]),
            map_assigner=dict(
                type='MapHungarianAssigner3D',
                cls_cost=dict(type='FocalLossCost', weight=2.0),
                reg_cost=dict(
                    type='BBoxL1Cost', weight=0.0, box_format='xywh'),
                iou_cost=dict(type='IoUCost', iou_mode='giou', weight=0.0),
                pts_cost=dict(type='OrderedPtsL1Cost', weight=1.0),
                pc_range=[-30.0, -15.0, -2.0, 30.0, 15.0, 2.0]))))
optimizer = dict(
    type='AdamW',
    lr=5e-05,
    paramwise_cfg=dict(custom_keys=dict(img_backbone=dict(lr_mult=0.1))),
    weight_decay=0.01)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.3333333333333333,
    min_lr_ratio=0.001)
runner = dict(type='EpochBasedRunner', max_epochs=12)
custom_hooks = [dict(type='CustomSetEpochInfoHook')]
