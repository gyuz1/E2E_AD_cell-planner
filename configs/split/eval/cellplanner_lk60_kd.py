"""LK60 + 증류, split 최종 평가용.

학습 쪽은 configs/split/cellplanner_lk60_kd.py 다. 평가 설정은 전방 통과 시간을 재기
위해 GT 박스를 일부러 모으지 않는다 -- 충돌 지표는 평가 도구가
주석 레코드에서 점유 격자를 직접 다시 만들어 계산한다.
#
# 이 파일은 해석이 끝난(flattened) 설정이다. 원래 저장소에서는 상속 사슬에
# 흩어져 있었고 파일명이 설계가 아니라 실험 이력이었다. 여기서는 mmcv 가
# 해석한 결과를 그대로 적어 한 파일로 만들었다 -- 다시 읽어도 원본과 같은
# 설정을 준다(검증함).
#
# 이 브랜치는 301/75 분할 계보다. 학습 301 scene, 잔류 검증 75 scene.
# main 브랜치의 fulldata 계보에는 잔류 검증셋이 없어서 ablation 을 바로 재지
# 못한다. 여기서는 잰다. 그래서 설계 판단은 전부 이 계보에서 나왔다.
#
# 공통 골격: VAD/LAW 위의 2단계 학습. 1단계는 월드모델, 2단계가 플래너.
#   - cell planner : 커맨드마다 5초 목표를 셀로 나누고, 하나의 플래너 헤드를
#                    고정 사인파 위치부호(파장 1.0~300.0 m) 위에서 쓸어 후보를
#                    만든다. 셀 수만큼 헤드가 있는 것이 아니다.
#   - ego status   : 플래너는 실제 ego 상태를 읽지 않는다(ego_lcf_feat_idx=None).
#                    BEV 모션 디스크립터에서 추정한 8칸을 scene 512 뒤에 붙인다.
#   - target point : 생성 단계에 들어가지 않는다. 이미 만들어진 후보 중에서
#                    하나를 고르는 데만, 네트워크 밖에서 쓴다.
#   - 3프레임 창   : queue_length=3, target_stride=5 -- 0.5초 간격 세 프레임.
#                    가속도 항이 여기서 나온다. 2프레임에서는 2차 차분이
#                    항등적으로 0 이 되어 정지 구간이 무너진다.
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
                1.0, 3.0, 6.5, 9.0, 12.0, 15.0, 17.0, 19.0, 21.0, 23.5, 25.5,
                27.5, 29.5, 31.0, 32.5, 34.5, 36.5, 38.0, 39.5, 41.5, 43.5,
                45.0, 46.0, 47.5, 49.0, 50.5, 52.0, 53.5, 55.0, 56.5, 58.0,
                59.0, 60.0, 61.5, 62.5, 63.5, 64.5, 65.5, 67.0, 68.5, 70.0,
                72.0, 74.5, 77.0, 79.5, 82.5, 87.0, 89.5, 92.0, 94.5, 96.5,
                98.0, 99.5, 101.5, 103.0, 104.5, 106.0, 107.5, 109.0, 111.0,
                117.0
            ], [-20, 17]),
            ([1, 20, 25, 36, 44, 51, 58, 66, 78, 94, 103, 116], [-16, 12]),
            ([1, 23, 30, 35, 41, 45, 49, 53, 59, 64, 74, 81, 89, 100,
              115], [-12, 18]), ([1, 16, 21, 30, 48], [-2, 10, 27]),
            ([1, 12, 16, 20, 26, 33, 43], [-26, -11.5, 2]), None, None
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
