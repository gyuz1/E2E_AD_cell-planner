import copy
import random
from math import pi, cos, sin

import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
from mmdet.models import HEADS, build_loss 
from mmdet.models.dense_heads import DETRHead
from mmcv.runner import force_fp32, auto_fp16
from mmcv.utils import TORCH_VERSION, digit_version
from mmdet.core import build_assigner, build_sampler
from mmdet3d.core.bbox.coders import build_bbox_coder
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet.core.bbox.transforms import bbox_xyxy_to_cxcywh
from mmcv.cnn import Linear, bias_init_with_prob, xavier_init
from mmdet.core import (multi_apply, multi_apply, reduce_mean)
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence

from projects.mmdet3d_plugin.core.bbox.util import normalize_bbox
from projects.mmdet3d_plugin.VAD.utils.traj_lr_warmup import get_traj_warmup_loss_weight
from projects.mmdet3d_plugin.VAD.utils.map_utils import (
    normalize_2d_pts, normalize_2d_bbox, denormalize_2d_pts, denormalize_2d_bbox
)
from projects.mmdet3d_plugin.VAD.cell_planner_utils import (
    STOP as CELL_STOP, cell_positional_encoding, route_trajectory,
    validate_layouts,
)
from projects.mmdet3d_plugin.VAD.goal_anchor_utils import (
    build_anchor_table, goal_labels_from_targets, goal_points_from_offsets,
    nearest_valid_goal,
)

class MLP(nn.Module):
    def __init__(self, in_channels, hidden_unit, verbose=False):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_unit),
            nn.LayerNorm(hidden_unit),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.mlp(x)
        return x

class LaneNet(nn.Module):
    def __init__(self, in_channels, hidden_unit, num_subgraph_layers):
        super(LaneNet, self).__init__()
        self.num_subgraph_layers = num_subgraph_layers
        self.layer_seq = nn.Sequential()
        for i in range(num_subgraph_layers):
            self.layer_seq.add_module(
                f'lmlp_{i}', MLP(in_channels, hidden_unit))
            in_channels = hidden_unit*2

    def forward(self, pts_lane_feats):
        '''
            Extract lane_feature from vectorized lane representation

        Args:
            pts_lane_feats: [batch size, max_pnum, pts, D]

        Returns:
            inst_lane_feats: [batch size, max_pnum, D]
        '''
        x = pts_lane_feats
        for name, layer in self.layer_seq.named_modules():
            if isinstance(layer, MLP):
                # x [bs,max_lane_num,9,dim]
                x = layer(x)
                x_max = torch.max(x, -2)[0]
                x_max = x_max.unsqueeze(2).repeat(1, 1, x.shape[2], 1)
                x = torch.cat([x, x_max], dim=-1)
        x_max = torch.max(x, -2)[0]
        return x_max


# Seconds between consecutive future waypoints. The GT trajectories are
# per-step DELTAS (the converter builds them with np.diff), so dividing a
# step's displacement by this yields that step's speed in m/s.
FUT_TS_INTERVAL_S = 0.5


@HEADS.register_module()
class VADHead(DETRHead):
    """Head of VAD model.
    Args:
        with_box_refine (bool): Whether to refine the reference points
            in the decoder. Defaults to False.
        as_two_stage (bool) : Whether to generate the proposal from
            the outputs of encoder.
        transformer (obj:`ConfigDict`): ConfigDict is used for building
            the Encoder and Decoder.
        bev_h, bev_w (int): spatial shape of BEV queries.
    """
    def __init__(self,
                 *args,
                 with_box_refine=False,
                 as_two_stage=False,
                 transformer=None,
                 bbox_coder=None,
                 num_cls_fcs=2,
                 code_weights=None,
                 bev_h=30,
                 bev_w=30,
                 fut_ts=6,
                 fut_mode=6,
                 loss_traj=dict(type='L1Loss', loss_weight=0.25),
                 loss_traj_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=0.8),
                 map_bbox_coder=None,
                 map_num_query=900,
                 map_num_classes=3,
                 map_num_vec=20,
                 map_num_pts_per_vec=2,
                 map_num_pts_per_gt_vec=2,
                 map_query_embed_type='all_pts',
                 map_transform_method='minmax',
                 map_gt_shift_pts_pattern='v0',
                 map_dir_interval=1,
                 map_code_size=None,
                 map_code_weights=None,
                loss_map_cls=dict(
                     type='CrossEntropyLoss',
                     bg_cls_weight=0.1,
                     use_sigmoid=False,
                     loss_weight=1.0,
                     class_weight=1.0),
                 loss_map_bbox=dict(type='L1Loss', loss_weight=5.0),
                 loss_map_iou=dict(type='GIoULoss', loss_weight=2.0),
                 loss_map_pts=dict(
                    type='ChamferDistance',loss_src_weight=1.0,loss_dst_weight=1.0
                 ),
                 loss_map_dir=dict(type='PtsDirCosLoss', loss_weight=2.0),
                 tot_epoch=None,
                 use_traj_lr_warmup=False,
                 motion_decoder=None,
                 motion_map_decoder=None,
                 use_pe=False,
                 motion_det_score=None,
                 map_thresh=0.5,
                 dis_thresh=0.2,
                 pe_normalization=True,
                 ego_his_encoder=None,
                 ego_fut_mode=3,
                 loss_plan_reg=dict(type='L1Loss', loss_weight=0.25),
                 loss_plan_bound=dict(type='PlanMapBoundLoss', loss_weight=0.1),
                 loss_plan_col=dict(type='PlanAgentDisLoss', loss_weight=0.1),
                 loss_plan_dir=dict(type='PlanMapThetaLoss', loss_weight=0.1),
                 ego_agent_decoder=None,
                 ego_map_decoder=None,
                 query_thresh=None,
                 query_use_fix_pad=None,
                 ego_lcf_feat_idx=None,
                 valid_fut_ts=6,
                 command_class_weights=None,
                 plan_reg_ts_weight_mode='position',
                 plan_reg_cumulative=False,
                 ego_lcf_embed_dim=None,
                 ego_lcf_embed_hidden=None,
                 ego_lcf_embed_residual=False,
                 ego_status_est_dim=None,
                 ego_status_est_hidden=None,
                 ego_status_est_dropout=0.0,
                 ego_status_distill_idx=None,
                 ego_status_decode=False,
                 ego_status_decode_weight=0.5,
                 bev_residual_refine=False,
                 bev_refine_steps=1,
                 prism_latent_supervision=False,
                 prism_latent_dim=64,
                 prism_kl_weight=0.1,
                 prism_long_fut_ts=10,
                 prism_num_samples=1,
                 prism_posterior_lcf_idx=None,
                 ego_fut_dec_hidden_dim=None,
                 privileged_distill=False,
                 privileged_distill_idx=(0, 1, 2, 3, 4, 7),
                 privileged_distill_weight=1.0,
                 target_point_shortcut=False,
                 target_point_shortcut_mode='both',
                 aux_ego_motion=False,
                 aux_ego_motion_idx=(0, 1, 4, 7),
                 aux_ego_motion_weight=1.0,
                 aux_long_horizon=False,
                 aux_long_horizon_weight=0.5,
                 aux_long_horizon_residual=False,
                 goal_pred=False,
                 goal_anchors=None,
                 goal_bin_edges=None,
                 goal_lat_scale=25.0,
                 goal_scale=None,
                 goal_long_ts=10,
                 goal_cls_weight=0.5,
                 goal_off_weight=0.5,
                 goal_follow_weight=0.1,
                 goal_select_weight=None,
                 goal_expose_candidates=False,
                 goal_head_grad_scale=1.0,
                 cell_planner=False,
                 cell_layouts=None,
                 cell_pe_dim=256,
                 cell_pe_wavelengths=(4.0, 400.0),
                 cell_stop_tp_thresh=1.0,
                 aux_bev_motion=False,
                 aux_bev_motion_idx=(0, 1, 4, 7),
                 aux_bev_motion_weight=0.5,
                 aux_bev_motion_norm=None,
                 aux_bev_motion_frames=2,
                 aux_bev_future_motion=False,
                 aux_bev_future_motion_ts=6,
                 aux_bev_future_motion_weight=0.5,
                 aux_bev_future_motion_norm=5.705,
                 aux_bev_motion_feedback=False,
                 aux_bev_motion_temporal=False,
                 aux_bev_motion_grid=4,
                 aux_bev_motion_proj_dim=32,
                 **kwargs):

        self.bev_h = bev_h
        self.bev_w = bev_w
        self.fp16_enabled = False
        self.fut_ts = fut_ts
        self.fut_mode = fut_mode
        self.tot_epoch = tot_epoch
        self.use_traj_lr_warmup = use_traj_lr_warmup
        self.motion_decoder = motion_decoder
        self.motion_map_decoder = motion_map_decoder
        self.use_pe = use_pe
        self.motion_det_score = motion_det_score
        self.map_thresh = map_thresh
        self.dis_thresh = dis_thresh
        self.pe_normalization = pe_normalization
        self.ego_his_encoder = ego_his_encoder
        self.ego_fut_mode = ego_fut_mode
        self.ego_agent_decoder = ego_agent_decoder
        self.ego_map_decoder = ego_map_decoder
        self.query_thresh = query_thresh
        self.query_use_fix_pad = query_use_fix_pad
        self.ego_lcf_feat_idx = ego_lcf_feat_idx
        self.valid_fut_ts = valid_fut_ts
        # target_point is never wired into this model's forward pass. Goal
        # information may only SELECT among already-generated candidates, never
        # influence generation, and that selection happens outside the network.
        #
        # bev_residual_refine samples bev_embed at each coarse waypoint's own
        # predicted location and predicts a correction from it. Measured worse
        # here, so it is off in every current config.
        self.bev_residual_refine = bool(bev_residual_refine)
        self.bev_refine_steps = max(1, int(bev_refine_steps))

        # Per-command multiplier on loss_plan_reg -- U_TURN is ~0.1% of train
        # frames against LANE_KEEP's ~80%, so its gradient is otherwise drowned
        # out. None (default) = no reweighting.
        #
        # Deliberately NOT hardcoded: per-command frequency differs between data
        # splits, so the pipeline computes it from whichever train pkl it is
        # about to use and passes it via --cfg-options. Length must equal
        # ego_fut_mode.
        self.command_class_weights = (
            list(command_class_weights) if command_class_weights else None)

        # How loss_planning weights each future timestep, to match what the
        # challenge metric actually rewards. 'position' reproduces the
        # metric's weight on the POSITION error at each step; 'cumulative'
        # accounts for ego_fut_preds being per-step DELTAS that the metric
        # cumsums first, so an early delta's error displaces every later
        # position too. See loss_planning for the derivation. Default stays
        # 'position' so existing checkpoints/configs keep their behavior.
        assert plan_reg_ts_weight_mode in ('position', 'cumulative')
        self.plan_reg_ts_weight_mode = plan_reg_ts_weight_mode

        # Score the planning regression on cumulative POSITIONS rather than the
        # per-step deltas the head emits. The metric measures positions, and
        # |sum e| <= sum |e|, so delta L1 is an upper bound that is loose exactly
        # where errors cancel.
        #
        # When this is on, time weights fall back to 'position' whatever
        # plan_reg_ts_weight_mode says: that mode is the same chain rule with the
        # signs discarded, so applying both would count the weighting twice.
        # Only the L1 terms change -- plan_bound/_col/_dir cumsum internally.
        self.plan_reg_cumulative = bool(plan_reg_cumulative)

        # OUTPUT width of the learned ego-status embedding that replaces the raw
        # ego_lcf columns in ego_feats (None = keep the raw columns). Only
        # meaningful with ego_lcf_feat_idx set.
        #
        # This sets ego_fut_dec_in_dim, so it decides whether a stage-1 donor
        # transfers: 8 gives 512+8=520, matching an ego_lcf-ON stage 1, while 64
        # gives 576 and forces zero-padded columns with only stage 2 to grow in.
        self.ego_lcf_embed_dim = (
            int(ego_lcf_embed_dim) if ego_lcf_embed_dim else None)

        # INTERNAL width of that embedding's hidden layer, independent of its
        # output width. Splitting the two lets the output stay narrow
        # (520-compatible) without giving up the transform's expressiveness.
        # Defaults to ego_lcf_embed_dim, reproducing the original square net.
        self.ego_lcf_embed_hidden = (
            int(ego_lcf_embed_hidden) if ego_lcf_embed_hidden
            else self.ego_lcf_embed_dim)

        # Make the embedding a ZERO-INIT RESIDUAL on the raw columns:
        #   ego_status = raw + embed_net(raw),  embed_net's last layer = 0
        # so at step 0 the status block is bit-identical to the raw ego_lcf the
        # donor's decoder was trained against. Without it the donor's well-trained
        # ego columns are handed a fresh MLP's O(1) output in place of physical
        # values they learned the meaning of, and the transfer breaks silently.
        #
        # Requires embed_dim == len(ego_lcf_feat_idx) for the residual to be
        # shape-valid -- the 8-d configuration this is meant for.
        self.ego_lcf_embed_residual = bool(ego_lcf_embed_residual)

        # Scheme-A STUDENT side: width of a VISION-derived ego-status vector that
        # occupies the same ego_feats slot the teacher fills with
        # ego_lcf_embed_net(raw columns). Must equal the teacher's
        # ego_lcf_embed_dim -- the two are aligned by loss_status_distill.
        #
        # The estimate is a function of bev_embed and prev_bev only, never of
        # ego_lcf_feat, so it exists identically at test time. Configs setting
        # this must keep ego_lcf_feat_idx=None (enforced in _init_layers).
        self.ego_status_est_dim = (
            int(ego_status_est_dim) if ego_status_est_dim else None)

        # Hidden width of ego_status_est_net, independent of its output width.
        # Defaults to embed_dims (256).
        #
        # Set it to match the teacher's ego_lcf_embed_hidden: the teacher's slot
        # comes out of Linear(8, H) -> ReLU -> Linear(H, 8), so matching H makes
        # the student's second layer the same shape and the teacher's trained
        # (H -> slot) map can be copied in. With that map shared the student only
        # has to land its hidden vector where the teacher's is, rather than two
        # independently-learned bases being asked to agree through a cosine.
        self.ego_status_est_hidden = (
            int(ego_status_est_hidden) if ego_status_est_hidden else None)

        # Which positions of the status vector take part in loss_status_distill.
        # None means all of them, which the teacher's raw ego_lcf columns make a
        # bad default: ego_length and ego_width have zero variance, so they carry
        # no information yet dominate a cosine target on stopped samples.
        #
        # Restricting the TARGET is enough -- ego_feats keeps all eight columns,
        # so ego_fut_decoder stays 2*D+8 wide and the stage-1 donor still
        # transfers. Teacher and student must set the same value, checked in
        # VAD_LAW before training starts.
        self.ego_status_distill_idx = (
            tuple(ego_status_distill_idx)
            if ego_status_distill_idx is not None else None)

        # Decode the planner's status slot back to physical ego state and
        # supervise that against the real ego_lcf. Train-only; no input path
        # changes.
        #
        # ego_status_est_net feeds the slot the decoder consumes but was
        # supervised only by a cosine to the teacher, and cosine is
        # scale-invariant -- it cannot separate 10.5 m/s from 5.2 m/s, which L2
        # is very sensitive to. This puts the physical value into the slot.
        self.ego_status_decode = bool(ego_status_decode)
        self.ego_status_decode_weight = float(ego_status_decode_weight)
        # Modality dropout on that slot during TRAINING: with this
        # probability the 64 estimated channels are zeroed for the whole
        # batch, forcing ego_fut_decoder to stay able to plan from the 2*D
        # scene half alone. Without it the decoder leans on the estimate the
        # way it leaned on raw ego_lcf, and a noisy estimate then costs more
        # than it gives -- which is how aux_bev_motion_feedback lost 5.5%
        # (see its constructor comment). Never applied at eval.
        self.ego_status_est_dropout = float(ego_status_est_dropout)

        # PRISM-style privileged latent supervision. A posterior encoder sees the
        # GT 0-5s future during TRAINING ONLY and regularizes a vision-only prior
        # via KL; at inference the posterior is never touched and z comes from
        # the prior mean, so no test-time input is added. Injects into ego_feats
        # just before ego_fut_decoder.
        self.prism_latent_supervision = bool(prism_latent_supervision)
        self.prism_latent_dim = int(prism_latent_dim)
        self.prism_kl_weight = float(prism_kl_weight)
        self.prism_long_fut_ts = int(prism_long_fut_ts)
        # Number of posterior samples during training. Only affects the training
        # path with real privileged data; eval and history-frame calls always use
        # exactly 1 (the prior mean), regardless of this setting.
        self.prism_num_samples = int(prism_num_samples)
        # Lets PRISM's posterior additionally condition on current ego
        # status (velocity/accel/yaw-rate), not just the privileged 0-5s
        # future trajectory -- same compliance shape as every other
        # ego_lcf use here: the posterior is discarded at inference (see
        # forward()'s prism_latent_supervision branch), so this only ever
        # shapes the KL pressure on the prior during training, never
        # reaches ego_fut_decoder's inference-time input. None (default)
        # leaves the posterior exactly as before -- future-trajectory-only.
        self.prism_posterior_lcf_idx = (
            list(prism_posterior_lcf_idx) if prism_posterior_lcf_idx
            else None)
        # See _init_layers()'s ego_fut_decoder construction and
        # tools/surgical_ego_fut_decoder_transfer.py.
        self.ego_fut_dec_hidden_dim = ego_fut_dec_hidden_dim

        # Privileged-expert distillation (LEAD-style). A second trajectory head
        # sees ego_feats plus the real ego status and is trained against the same
        # GT; the deployed decoder then also matches that head's detached output.
        #
        # ego_lcf enters only this head, the head runs only under self.training,
        # and the target is detached.
        self.privileged_distill = bool(privileged_distill)
        self.privileged_distill_idx = list(privileged_distill_idx)
        self.privileged_distill_weight = float(privileged_distill_weight)

        # DIAGNOSTIC-ONLY, NEVER SUBMITTABLE. Resurrects the target_point
        # shortcut (goal_xy as attention query-pos conditioning the ego decoders,
        # and/or added into ego_feats) to build the strongest possible teacher.
        #
        # Its TRAJECTORY output is a poor distillation target -- it reflects
        # information the student structurally lacks. It exists for
        # feature-level distillation only.
        self.target_point_shortcut = bool(target_point_shortcut)
        assert target_point_shortcut_mode in ('residual', 'attn', 'both')
        self.target_point_shortcut_mode = target_point_shortcut_mode

        # Auxiliary ego-motion supervision (train-only). Regresses the current ego
        # status (vx, vy, yaw-rate, speed) from the vision-derived planning
        # features, supervised by ego_lcf_feat.
        #
        # ego_lcf_feat enters only as a regression TARGET, through the separate
        # `ego_lcf_target` argument -- never as an input. With
        # ego_lcf_feat_idx=None the input path is dead.
        self.aux_ego_motion = bool(aux_ego_motion)
        self.aux_ego_motion_idx = list(aux_ego_motion_idx)
        self.aux_ego_motion_weight = float(aux_ego_motion_weight)

        # Auxiliary long-horizon trajectory regression (train-only) over the
        # dataset's 5s future. The four steps beyond the scored 3s horizon are the
        # ones short-horizon extrapolation cannot fake, so they force the shared
        # features to carry real scene dynamics -- which matters more once ego
        # status is no longer an input.
        #
        # Deliberately a SEPARATE head rather than widening ego_fut_decoder from 6
        # to 10 steps: that output layer is initialized from the LAW checkpoint at
        # ego_fut_mode*6*2, and changing its shape would discard the pretrained
        # planning weights. This head only shapes ego_feats.
        self.aux_long_horizon = bool(aux_long_horizon)
        self.aux_long_horizon_weight = float(aux_long_horizon_weight)

        # Regress the 5s trajectory as a RESIDUAL over a constant-acceleration
        # extrapolation of the current ego state, rather than as absolute
        # positions.
        #
        # Absolute positions are dominated by the part kinematics already
        # explains, so an L1 on them spends almost all of its gradient
        # reproducing speed times time and leaves what the scene adds as a
        # rounding error. Subtracting a known function of the target does not
        # change what is learnable, but it changes what the loss measures.
        self.aux_long_horizon_residual = bool(aux_long_horizon_residual)

        # Goal-conditioned planning on a PREDICTED 5s goal. Superseded by the
        # cell planner below; kept so existing checkpoints still load.
        # ego_target_point is read only under self.training.
        # Command cell planner (replaces ego_fut_decoder's output when on).
        # One head per command: moving commands share a head across their cells
        # and tell cells apart by a fixed positional code of the cell centre;
        # U_TURN and STOP have plain heads. The head generates every cell's
        # trajectory without the target point; route_trajectory() picks one.
        self.cell_planner = bool(cell_planner)
        self.cell_layouts = None
        self.cell_pe_dim = int(cell_pe_dim)
        self.cell_pe_wavelengths = tuple(float(v) for v in cell_pe_wavelengths)
        self.cell_stop_tp_thresh = float(cell_stop_tp_thresh)
        if self.cell_planner:
            if ego_fut_mode != 7:
                raise ValueError('cell_planner needs the 7-command vocabulary')
            if (goal_pred or prism_latent_supervision or privileged_distill
                    or bev_residual_refine or target_point_shortcut):
                raise ValueError('cell_planner cannot be combined with goal_pred, '
                                 'PRISM, privileged_distill, bev_residual_refine '
                                 'or target_point_shortcut')
            self.cell_layouts = validate_layouts(cell_layouts, ego_fut_mode)

        self.goal_pred = bool(goal_pred)
        self.goal_long_ts = int(goal_long_ts) if self.goal_pred else None
        self.goal_cls_weight = float(goal_cls_weight)
        self.goal_off_weight = float(goal_off_weight)
        self.goal_follow_weight = float(goal_follow_weight)
        # Keep existing 12-bin checkpoints/configs usable. Ragged experiments
        # explicitly enable the extra TP-selected trajectory loss; old runs
        # must not silently acquire a new objective when reloaded.
        self._legacy_goal_bins = self.goal_pred and goal_anchors is None
        self.goal_select_weight = float(
            (0.0 if self._legacy_goal_bins else 1.0)
            if goal_select_weight is None else goal_select_weight)
        self.goal_bin_edges = None
        self.goal_lat_scale = float(goal_lat_scale)
        # Inference-only: also emit one trajectory per anchor so a caller can
        # pick among them. Nothing here reads ego_target_point at inference;
        # who selects is the caller's decision.
        self.goal_expose_candidates = bool(goal_expose_candidates)
        # Gradient the goal heads send back into the shared planner features.
        # Measured at 1.0 (epoch 6): loss_goal_cls's gradient on ego_feats is
        # 8.9x loss_plan_reg's and orthogonal to it, so under Adam it crowds
        # out planning in the shared layers. Forward values are unchanged.
        self.goal_head_grad_scale = float(goal_head_grad_scale)
        if self.goal_pred:
            if self._legacy_goal_bins:
                if goal_bin_edges is None or len(goal_bin_edges) < 3:
                    raise ValueError('goal_pred needs goal_anchors or legacy goal_bin_edges')
                edges = tuple(float(value) for value in goal_bin_edges)
                if not all(np.isfinite(edges)) or any(
                        right <= left for left, right in zip(edges[:-1], edges[1:])):
                    raise ValueError('goal_bin_edges must be finite and strictly increasing')
                self.goal_bin_edges = edges
                row = [[(left + right) * 0.5, right - left, 0.0, self.goal_lat_scale]
                       for left, right in zip(edges[:-1], edges[1:])]
                goal_anchors = [copy.deepcopy(row) for _ in range(self.ego_fut_mode)]
            if self.goal_long_ts < self.fut_ts:
                raise ValueError('goal_long_ts must be >= fut_ts')
            if prism_latent_supervision or privileged_distill or bev_residual_refine:
                raise ValueError('goal_pred cannot be combined with PRISM, '
                                 'privileged_distill or bev_residual_refine')
            tab, mask, counts = build_anchor_table(goal_anchors, self.ego_fut_mode)
            self.goal_k = max(counts)
            self.goal_anchor_table = tab
            self.goal_anchor_mask = mask
            self.goal_anchor_counts = counts
            if goal_scale is None:
                goal_scale = ((self.goal_bin_edges[-1], self.goal_lat_scale)
                              if self._legacy_goal_bins else
                              (max(float(tab[..., 0].abs().max()), 1.0),
                               max(float(tab[..., 2].abs().max()), 1.0)))
            if len(goal_scale) != 2 or not all(
                    np.isfinite(value) and value > 0 for value in goal_scale):
                raise ValueError('goal_scale must contain two finite positive values')
            # Identical explicit scales in a paired experiment prevent a
            # changed lateral layout from also changing embedding/loss units.
            self.goal_coordinate_scale = tuple(float(value) for value in goal_scale)
        else:
            self.goal_anchor_table = None
            self.goal_anchor_mask = None
            self.goal_k = 1
        self.ego_dec_ts = self.goal_long_ts if self.goal_pred else self.fut_ts

        # Same idea as aux_ego_motion, but supervising bev_embed (the BEV
        # encoder's raw output, forward()'s very first shared tensor --
        # detection/map/agent/planning all branch off it) rather than
        # ego_feats (a late, planning-specific feature after the agent/map
        # decoders). Gradient from this loss reaches the BEV encoder itself,
        # not just the planning head, which is a closer match to the
        # organizers' "improves common features across multiple tasks"
        # allowance than aux_ego_motion is. Same compliance shape: ego
        # status is a regression TARGET on a forward-only branch with no
        # injection path back into any decoder output.
        self.aux_bev_motion = bool(aux_bev_motion)
        self.aux_bev_motion_idx = list(aux_bev_motion_idx)
        self.aux_bev_motion_weight = float(aux_bev_motion_weight)
        # Per-component divisor applied to the aux_bev_motion L1, one value per
        # entry of aux_bev_motion_idx. None = raw L1.
        #
        # Raw L1 is dominated by whichever component is physically largest. On
        # the train split with the default idx (0,1,4,7), vx and speed take
        # 98.7% of the gradient while yaw_rate -- what actually matters for
        # turns -- gets 0.1%. Dividing by each component's std turns the loss
        # into relative error so every component pulls comparably.
        # Train-split stds: vx 5.7040, vy 0.1715, ax 0.4625, ay 0.3579,
        # yaw_rate 0.0547, speed 5.7050.
        self.aux_bev_motion_norm = (
            list(aux_bev_motion_norm) if aux_bev_motion_norm else None)
        # How many BEV frames the temporal motion descriptor spans.
        #   2: cat([current, current - prev]) -- one difference, i.e. velocity.
        #   3: cat([current, d1, d1 - d2]) with d1 = current - prev1 and
        #      d2 = prev1 - prev2, so the third block is the second difference
        #      (acceleration), handed over directly rather than left for a
        #      linear layer to subtract.
        # Kinematic oracles on this val split score 0.5965 with perfect velocity
        # and 0.2708 with perfect velocity AND acceleration, so acceleration is
        # the larger half of what ego state can buy. It costs one more history
        # frame at inference.
        self.aux_bev_motion_frames = int(aux_bev_motion_frames)
        if self.aux_bev_motion_frames not in (2, 3):
            raise ValueError(
                'aux_bev_motion_frames must be 2 or 3, got '
                f'{self.aux_bev_motion_frames}.')

        # Regress the FUTURE ego speed profile from the BEV motion descriptor.
        # Train-only: a loss target with no path into any decoder output.
        #
        # aux_bev_motion asks what the ego is doing now; this asks what it is
        # about to do, which no amount of current-state accuracy can supply --
        # that has to come from reading the scene. It supervises the BEV encoder
        # rather than the planner head, so the representation the planner reads
        # carries maneuver cues.
        self.aux_bev_future_motion = bool(aux_bev_future_motion)
        self.aux_bev_future_motion_ts = int(aux_bev_future_motion_ts)
        self.aux_bev_future_motion_weight = float(aux_bev_future_motion_weight)
        # Single scalar, not per-component: every target here is a speed, so
        # one scale (the train-split speed std, 5.705) normalizes them all.
        self.aux_bev_future_motion_norm = float(aux_bev_future_motion_norm)
        if (self.aux_bev_motion_norm is not None
                and len(self.aux_bev_motion_norm)
                != len(self.aux_bev_motion_idx)):
            raise ValueError(
                'aux_bev_motion_norm must have one entry per '
                'aux_bev_motion_idx entry, got '
                f'{len(self.aux_bev_motion_norm)} vs '
                f'{len(self.aux_bev_motion_idx)}.')
        # Feeds aux_bev_motion_head's OWN PREDICTION (never the ground-truth
        # ego_lcf_target) into ego_feats, so ego_fut_decoder uses it rather than
        # only being supervised by it.
        #
        # The prediction is computed unconditionally from bev_embed at both train
        # and eval time -- it never reads ego_lcf_target, so it runs identically
        # whether or not privileged data exists for the sample.
        self.aux_bev_motion_feedback = bool(aux_bev_motion_feedback)

        # Gives aux_bev_motion_head an explicit temporal contrast instead of only
        # the current frame. Speed is a time derivative, and a single BEV snapshot
        # carries it only second-hand.
        #
        # The descriptor keeps spatial structure (a grid x grid regional pool)
        # rather than a global mean: ego motion shows up as a SHIFT of BEV
        # content, and a global mean is nearly shift-invariant, so the delta of
        # two means would be ~0 at any speed. Regional pooling makes the shift
        # move mass between cells.
        #
        # prev_bev is absent on a cold-start frame; the delta half is zero-filled
        # there, so 'no temporal evidence' is a trained state, not a surprise.
        self.aux_bev_motion_temporal = bool(aux_bev_motion_temporal)
        self.aux_bev_motion_grid = int(aux_bev_motion_grid)
        self.aux_bev_motion_proj_dim = int(aux_bev_motion_proj_dim)

        if loss_traj_cls['use_sigmoid'] == True:
            self.traj_num_cls = 1
        else:
          self.traj_num_cls = 2

        self.with_box_refine = with_box_refine
        self.as_two_stage = as_two_stage
        if self.as_two_stage:
            transformer['as_two_stage'] = self.as_two_stage
        if 'code_size' in kwargs:
            self.code_size = kwargs['code_size']
        else:
            self.code_size = 10
        if code_weights is not None:
            self.code_weights = code_weights
        else:
            self.code_weights = [1.0, 1.0, 1.0,
                                 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2]
        if map_code_size is not None:
            self.map_code_size = map_code_size
        else:
            self.map_code_size = 10
        if map_code_weights is not None:
            self.map_code_weights = map_code_weights
        else:
            self.map_code_weights = [1.0, 1.0, 1.0,
                                 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2]

        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.pc_range = self.bbox_coder.pc_range
        self.real_w = self.pc_range[3] - self.pc_range[0]
        self.real_h = self.pc_range[4] - self.pc_range[1]
        self.num_cls_fcs = num_cls_fcs - 1

        self.map_bbox_coder = build_bbox_coder(map_bbox_coder)
        self.map_query_embed_type = map_query_embed_type
        self.map_transform_method = map_transform_method
        self.map_gt_shift_pts_pattern = map_gt_shift_pts_pattern
        map_num_query = map_num_vec * map_num_pts_per_vec
        self.map_num_query = map_num_query
        self.map_num_classes = map_num_classes
        self.map_num_vec = map_num_vec
        self.map_num_pts_per_vec = map_num_pts_per_vec
        self.map_num_pts_per_gt_vec = map_num_pts_per_gt_vec
        self.map_dir_interval = map_dir_interval

        if loss_map_cls['use_sigmoid'] == True:
            self.map_cls_out_channels = map_num_classes
        else:
            self.map_cls_out_channels = map_num_classes + 1

        self.map_bg_cls_weight = 0
        map_class_weight = loss_map_cls.get('class_weight', None)
        if map_class_weight is not None and (self.__class__ is VADHead):
            assert isinstance(map_class_weight, float), 'Expected ' \
                'class_weight to have type float. Found ' \
                f'{type(map_class_weight)}.'
            # NOTE following the official DETR rep0, bg_cls_weight means
            # relative classification weight of the no-object class.
            map_bg_cls_weight = loss_map_cls.get('bg_cls_weight', map_class_weight)
            assert isinstance(map_bg_cls_weight, float), 'Expected ' \
                'bg_cls_weight to have type float. Found ' \
                f'{type(map_bg_cls_weight)}.'
            map_class_weight = torch.ones(map_num_classes + 1) * map_class_weight
            # set background class as the last indice
            map_class_weight[map_num_classes] = map_bg_cls_weight
            loss_map_cls.update({'class_weight': map_class_weight})
            if 'bg_cls_weight' in loss_map_cls:
                loss_map_cls.pop('bg_cls_weight')
            self.map_bg_cls_weight = map_bg_cls_weight
        
        self.traj_bg_cls_weight = 0

        super(VADHead, self).__init__(*args, transformer=transformer, **kwargs)
        self.code_weights = nn.Parameter(torch.tensor(
            self.code_weights, requires_grad=False), requires_grad=False)
        self.map_code_weights = nn.Parameter(torch.tensor(
            self.map_code_weights, requires_grad=False), requires_grad=False)
        
        if kwargs['train_cfg'] is not None:
            assert 'map_assigner' in kwargs['train_cfg'], 'map assigner should be provided '\
                'when train_cfg is set.'
            map_assigner = kwargs['train_cfg']['map_assigner']
            assert loss_map_cls['loss_weight'] == map_assigner['cls_cost']['weight'], \
                'The classification weight for loss and matcher should be' \
                'exactly the same.'
            assert loss_map_bbox['loss_weight'] == map_assigner['reg_cost'][
                'weight'], 'The regression L1 weight for loss and matcher ' \
                'should be exactly the same.'
            assert loss_map_iou['loss_weight'] == map_assigner['iou_cost']['weight'], \
                'The regression iou weight for loss and matcher should be' \
                'exactly the same.'
            assert loss_map_pts['loss_weight'] == map_assigner['pts_cost']['weight'], \
                'The regression l1 weight for map pts loss and matcher should be' \
                'exactly the same.'

            self.map_assigner = build_assigner(map_assigner)
            # DETR sampling=False, so use PseudoSampler
            sampler_cfg = dict(type='PseudoSampler')
            self.map_sampler = build_sampler(sampler_cfg, context=self)
        
        self.loss_traj = build_loss(loss_traj)
        self.loss_traj_cls = build_loss(loss_traj_cls)
        self.loss_map_bbox = build_loss(loss_map_bbox)
        self.loss_map_cls = build_loss(loss_map_cls)
        self.loss_map_iou = build_loss(loss_map_iou)
        self.loss_map_pts = build_loss(loss_map_pts)
        self.loss_map_dir = build_loss(loss_map_dir)
        self.loss_plan_reg = build_loss(loss_plan_reg)
        self.loss_plan_bound = build_loss(loss_plan_bound)
        self.loss_plan_col = build_loss(loss_plan_col)
        self.loss_plan_dir = build_loss(loss_plan_dir)

    def _init_layers(self):
        """Initialize classification branch and regression branch of head."""
        cls_branch = []
        for _ in range(self.num_reg_fcs):
            cls_branch.append(Linear(self.embed_dims, self.embed_dims))
            cls_branch.append(nn.LayerNorm(self.embed_dims))
            cls_branch.append(nn.ReLU(inplace=True))
        cls_branch.append(Linear(self.embed_dims, self.cls_out_channels))
        cls_branch = nn.Sequential(*cls_branch)

        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(Linear(self.embed_dims, self.code_size))
        reg_branch = nn.Sequential(*reg_branch)

        traj_branch = []
        for _ in range(self.num_reg_fcs):
            traj_branch.append(Linear(self.embed_dims*2, self.embed_dims*2))
            traj_branch.append(nn.ReLU())
        traj_branch.append(Linear(self.embed_dims*2, self.fut_ts*2))
        traj_branch = nn.Sequential(*traj_branch)

        traj_cls_branch = []
        for _ in range(self.num_reg_fcs):
            traj_cls_branch.append(Linear(self.embed_dims*2, self.embed_dims*2))
            traj_cls_branch.append(nn.LayerNorm(self.embed_dims*2))
            traj_cls_branch.append(nn.ReLU(inplace=True))
        traj_cls_branch.append(Linear(self.embed_dims*2, self.traj_num_cls))
        traj_cls_branch = nn.Sequential(*traj_cls_branch)

        map_cls_branch = []
        for _ in range(self.num_reg_fcs):
            map_cls_branch.append(Linear(self.embed_dims, self.embed_dims))
            map_cls_branch.append(nn.LayerNorm(self.embed_dims))
            map_cls_branch.append(nn.ReLU(inplace=True))
        map_cls_branch.append(Linear(self.embed_dims, self.map_cls_out_channels))
        map_cls_branch = nn.Sequential(*map_cls_branch)

        map_reg_branch = []
        for _ in range(self.num_reg_fcs):
            map_reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            map_reg_branch.append(nn.ReLU())
        map_reg_branch.append(Linear(self.embed_dims, self.map_code_size))
        map_reg_branch = nn.Sequential(*map_reg_branch)


        def _get_clones(module, N):
            return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

        # last reg_branch is used to generate proposal from
        # encode feature map when as_two_stage is True.
        num_decoder_layers = 1
        num_map_decoder_layers = 1
        if self.transformer.decoder is not None:
            num_decoder_layers = self.transformer.decoder.num_layers
        if self.transformer.map_decoder is not None:
            num_map_decoder_layers = self.transformer.map_decoder.num_layers
        num_motion_decoder_layers = 1
        num_pred = (num_decoder_layers + 1) if \
            self.as_two_stage else num_decoder_layers
        motion_num_pred = (num_motion_decoder_layers + 1) if \
            self.as_two_stage else num_motion_decoder_layers
        map_num_pred = (num_map_decoder_layers + 1) if \
            self.as_two_stage else num_map_decoder_layers

        if self.with_box_refine:
            self.cls_branches = _get_clones(cls_branch, num_pred)
            self.reg_branches = _get_clones(reg_branch, num_pred)
            self.traj_branches = _get_clones(traj_branch, motion_num_pred)
            self.traj_cls_branches = _get_clones(traj_cls_branch, motion_num_pred)
            self.map_cls_branches = _get_clones(map_cls_branch, map_num_pred)
            self.map_reg_branches = _get_clones(map_reg_branch, map_num_pred)
        else:
            self.cls_branches = nn.ModuleList(
                [cls_branch for _ in range(num_pred)])
            self.reg_branches = nn.ModuleList(
                [reg_branch for _ in range(num_pred)])
            self.traj_branches = nn.ModuleList(
                [traj_branch for _ in range(motion_num_pred)])
            self.traj_cls_branches = nn.ModuleList(
                [traj_cls_branch for _ in range(motion_num_pred)])
            self.map_cls_branches = nn.ModuleList(
                [map_cls_branch for _ in range(map_num_pred)])
            self.map_reg_branches = nn.ModuleList(
                [map_reg_branch for _ in range(map_num_pred)])

        if not self.as_two_stage:
            self.bev_embedding = nn.Embedding(
                self.bev_h * self.bev_w, self.embed_dims)
            self.query_embedding = nn.Embedding(self.num_query,
                                                self.embed_dims * 2)
            if self.map_query_embed_type == 'all_pts':
                self.map_query_embedding = nn.Embedding(self.map_num_query,
                                                    self.embed_dims * 2)
            elif self.map_query_embed_type == 'instance_pts':
                self.map_query_embedding = None
                self.map_instance_embedding = nn.Embedding(self.map_num_vec, self.embed_dims * 2)
                self.map_pts_embedding = nn.Embedding(self.map_num_pts_per_vec, self.embed_dims * 2)
        
        if self.motion_decoder is not None:
            self.motion_decoder = build_transformer_layer_sequence(self.motion_decoder)
            self.motion_mode_query = nn.Embedding(self.fut_mode, self.embed_dims)	
            self.motion_mode_query.weight.requires_grad = True
            if self.use_pe:
                self.pos_mlp_sa = nn.Linear(2, self.embed_dims)
        else:
            raise NotImplementedError('Not implement yet')

        if self.motion_map_decoder is not None:
            self.lane_encoder = LaneNet(256, 128, 3)
            self.motion_map_decoder = build_transformer_layer_sequence(self.motion_map_decoder)
            if self.use_pe:
                self.pos_mlp = nn.Linear(2, self.embed_dims)
        
        if self.ego_his_encoder is not None:
            self.ego_his_encoder = LaneNet(2, self.embed_dims//2, 3)
        else:
            self.ego_query = nn.Embedding(1, self.embed_dims)	

        if self.ego_agent_decoder is not None:
            self.ego_agent_decoder = build_transformer_layer_sequence(self.ego_agent_decoder)
            if self.use_pe:
                self.ego_agent_pos_mlp = nn.Linear(2, self.embed_dims)

        if self.ego_map_decoder is not None:
            self.ego_map_decoder = build_transformer_layer_sequence(self.ego_map_decoder)
            if self.use_pe:
                self.ego_map_pos_mlp = nn.Linear(2, self.embed_dims)

        ego_fut_decoder = []
        ego_fut_dec_in_dim = self.embed_dims*2 + len(self.ego_lcf_feat_idx) \
            if self.ego_lcf_feat_idx is not None else self.embed_dims*2
        if self.ego_lcf_embed_dim is not None:
            # Scheme-A teacher: ego status enters as a LEARNED embedding
            # rather than raw columns appended to ego_feats. The point is
            # that the raw form gives the decoder a clean channel to read
            # speed from, so nothing ever pressures the 2*D scene half to
            # encode motion -- and a student distilling that half would
            # learn the same indifference. An embedding trained by the
            # planning loss is a representation of ego status, which a
            # vision-derived estimate can be aligned against.
            if self.ego_lcf_feat_idx is None:
                raise ValueError(
                    'ego_lcf_embed_dim needs ego_lcf_feat_idx: there is '
                    'nothing to embed with ego status off.')
            # Hidden width is independent of output width (see the
            # ego_lcf_embed_hidden constructor comment): expand for the
            # ReLU's expressiveness, emit narrow so ego_fut_dec_in_dim can
            # stay 520 and inherit an ego_lcf-ON stage 1's decoder.
            self.ego_lcf_embed_net = nn.Sequential(
                Linear(len(self.ego_lcf_feat_idx), self.ego_lcf_embed_hidden),
                nn.ReLU(),
                Linear(self.ego_lcf_embed_hidden, self.ego_lcf_embed_dim),
            )
            if self.ego_lcf_embed_residual:
                if self.ego_lcf_embed_dim != len(self.ego_lcf_feat_idx):
                    raise ValueError(
                        'ego_lcf_embed_residual adds the raw columns to the '
                        'embedding output, so ego_lcf_embed_dim must equal '
                        f'len(ego_lcf_feat_idx): got '
                        f'{self.ego_lcf_embed_dim} vs '
                        f'{len(self.ego_lcf_feat_idx)}.')
                # Zero the last layer so the residual starts as an exact
                # no-op and the status block equals the raw columns at
                # step 0. See the ego_lcf_embed_residual comment.
                nn.init.zeros_(self.ego_lcf_embed_net[-1].weight)
                nn.init.zeros_(self.ego_lcf_embed_net[-1].bias)
            ego_fut_dec_in_dim = (self.embed_dims * 2
                                  + self.ego_lcf_embed_dim)
        else:
            self.ego_lcf_embed_net = None
        if self.ego_status_est_dim is not None:
            # Scheme-A student: same ego_feats layout as the Scheme-A teacher
            # (2*D scene + status), except the status slot is estimated from
            # vision instead of read from ego_lcf. Keeping the layout
            # identical is the whole point -- it lets the two halves be
            # distilled separately (2*D <- teacher's 2*D, status <-
            # teacher's embedding) while the student's decoder input stays
            # free of any privileged input.
            if self.ego_lcf_feat_idx is not None:
                raise ValueError(
                    'ego_status_est_dim is the compliant substitute for '
                    'ego_lcf columns; set ego_lcf_feat_idx=None. Enabling '
                    'both would feed real ego status AND an estimate of it.')
            if not self.aux_bev_motion:
                raise ValueError(
                    'ego_status_est_dim needs aux_bev_motion=True: the '
                    'estimate reads the BEV motion descriptor built there.')
            if not self.aux_bev_motion_temporal:
                raise ValueError(
                    'ego_status_est_dim needs aux_bev_motion_temporal=True. '
                    'The single-frame descriptor is a global mean over BEV, '
                    'which is shift-invariant and so carries almost no '
                    'ego-motion signal to estimate status from.')
            ego_fut_dec_in_dim = (self.embed_dims * 2
                                  + self.ego_status_est_dim)
        if self.aux_bev_motion_feedback:
            # ego_feats gets aux_bev_motion_head's own prediction
            # concatenated on too (see forward()) -- widen accordingly.
            ego_fut_dec_in_dim += len(self.aux_bev_motion_idx)
        # Independent of ego_fut_dec_in_dim (default: equal to it) so a checkpoint
        # trained with a wider input can donate its hidden layers unchanged to a
        # narrower-input model -- only the first Linear's input side differs, and
        # dropping input columns from a Linear is exact, not an approximation.
        # See tools/surgery/surgical_ego_fut_decoder_transfer.py.
        hidden_dim = (self.ego_fut_dec_hidden_dim
                      if self.ego_fut_dec_hidden_dim is not None
                      else ego_fut_dec_in_dim)
        prev_dim = ego_fut_dec_in_dim
        for _ in range(self.num_reg_fcs):
            ego_fut_decoder.append(Linear(prev_dim, hidden_dim))
            ego_fut_decoder.append(nn.ReLU())
            prev_dim = hidden_dim
        ego_fut_decoder.append(Linear(
            prev_dim, self.ego_fut_mode * self.ego_dec_ts * 2))
        self.ego_fut_decoder = nn.Sequential(*ego_fut_decoder)

        self.cell_heads = None
        if self.cell_planner:
            # Same depth, hidden width and activation as ego_fut_decoder; a
            # cell head's first Linear is wider by the positional code only,
            # and every head emits one trajectory (fut_ts x 2).
            self.cell_heads = nn.ModuleList()
            for c, layout in enumerate(self.cell_layouts):
                layers = []
                prev = ego_fut_dec_in_dim + (self.cell_pe_dim if layout else 0)
                for _ in range(self.num_reg_fcs):
                    layers += [Linear(prev, hidden_dim), nn.ReLU()]
                    prev = hidden_dim
                layers.append(Linear(prev, self.fut_ts * 2))
                self.cell_heads.append(nn.Sequential(*layers))
                if layout is not None:
                    fe, le = layout
                    self.register_buffer(f'cell_fwd_edges_{c}', torch.tensor(fe))
                    self.register_buffer(f'cell_lat_edges_{c}', torch.tensor(le))
                    self.register_buffer(f'cell_pe_{c}', cell_positional_encoding(
                        fe, le, self.cell_pe_dim, self.cell_pe_wavelengths))

        if self.privileged_distill:
            # Same architecture as ego_fut_decoder, widened by the ego
            # status channels it alone gets to see. Deliberately a separate
            # module rather than a wider ego_fut_decoder: this one must be
            # trivially droppable at inference, and nothing it learns may
            # end up in the deployed weights.
            priv_in = ego_fut_dec_in_dim + len(self.privileged_distill_idx)
            priv_layers = []
            prev = priv_in
            for _ in range(self.num_reg_fcs):
                priv_layers.append(Linear(prev, hidden_dim))
                priv_layers.append(nn.ReLU())
                prev = hidden_dim
            priv_layers.append(
                Linear(prev, self.ego_fut_mode * self.fut_ts * 2))
            self.privileged_head = nn.Sequential(*priv_layers)
        else:
            self.privileged_head = None

        if (self.target_point_shortcut
                and self.target_point_shortcut_mode in ('residual', 'both')):
            self.target_point_encoder = nn.Sequential(
                nn.Linear(2, self.embed_dims), nn.ReLU(),
                nn.Linear(self.embed_dims, ego_fut_dec_in_dim))
        else:
            self.target_point_encoder = None

        if self.bev_residual_refine:
            # Input is the sampled BEV feature only -- NOT concatenated with
            # the coarse (x, y) the sample was taken at. Position already
            # determined *where* to sample; feeding it again here would let
            # this MLP learn a vision-independent coarse_xy -> delta path
            # (the exact bypass this refinement exists to prevent), since
            # nothing forces it to actually use `sampled`.
            self.plan_bev_refine_mlp = nn.Sequential(
                nn.Linear(self.embed_dims, self.embed_dims),
                nn.ReLU(),
                nn.Linear(self.embed_dims, 2),
            )
            nn.init.zeros_(self.plan_bev_refine_mlp[-1].weight)
            nn.init.zeros_(self.plan_bev_refine_mlp[-1].bias)

            # Cascaded refinement: each extra stage re-samples the BEV at the
            # positions the previous stage produced, which is what makes this a
            # cascade rather than a deeper MLP.
            #
            # Separate weights per stage because later stages face a different
            # problem -- small local corrections rather than the initial coarse fix.
            # Every stage is zero-init, so adding them is an exact no-op at
            # initialization.
            #
            # Kept in a separate attribute from plan_bev_refine_mlp so single-stage
            # checkpoints still load their first stage by name; the extra stages are
            # simply absent from their state dict and stay zero.
            self.plan_bev_refine_mlp_extra = nn.ModuleList()
            for _ in range(max(0, self.bev_refine_steps - 1)):
                mlp = nn.Sequential(
                    nn.Linear(self.embed_dims, self.embed_dims),
                    nn.ReLU(),
                    nn.Linear(self.embed_dims, 2),
                )
                nn.init.zeros_(mlp[-1].weight)
                nn.init.zeros_(mlp[-1].bias)
                self.plan_bev_refine_mlp_extra.append(mlp)
        else:
            self.plan_bev_refine_mlp = None
            self.plan_bev_refine_mlp_extra = None

        if self.aux_ego_motion:
            # Reads ego_feats (the same vision-derived tensor ego_fut_decoder
            # consumes) and regresses the current ego status. Train-only:
            # forward() computes it unconditionally but nothing downstream
            # consumes its output, so it contributes no inference cost beyond
            # one small MLP that eval can simply not call.
            self.aux_ego_motion_head = nn.Sequential(
                nn.Linear(ego_fut_dec_in_dim, self.embed_dims),
                nn.ReLU(),
                nn.Linear(self.embed_dims, len(self.aux_ego_motion_idx)),
            )
        else:
            self.aux_ego_motion_head = None

        if self.aux_long_horizon:
            # Per-mode like ego_fut_decoder, because which way the ego goes
            # over 5s is genuinely ambiguous from vision alone -- a single
            # mode-agnostic prediction would be unlearnable at intersections
            # and would inject that ambiguity into the features as noise.
            # The command one-hot selects the supervised mode at loss time,
            # mirroring loss_planning's own masking.
            self.aux_long_horizon_head = nn.Sequential(
                nn.Linear(ego_fut_dec_in_dim, self.embed_dims),
                nn.ReLU(),
                nn.Linear(self.embed_dims,
                          self.ego_fut_mode * self.prism_long_fut_ts * 2),
            )
        else:
            self.aux_long_horizon_head = None

        if self.goal_pred:
            # All three zero-init: offsets start at bin centres, logits
            # uniform, and the goal residual is exactly 0 -- the head
            # reproduces the donor's trajectory at step 0.
            self.goal_cls_head = nn.Linear(
                ego_fut_dec_in_dim, self.ego_fut_mode * self.goal_k)
            self.goal_off_head = nn.Linear(
                ego_fut_dec_in_dim, self.ego_fut_mode * self.goal_k * 2)
            self.goal_embed = nn.Sequential(
                nn.Linear(2, 128), nn.ReLU(),
                nn.Linear(128, ego_fut_dec_in_dim))
            for mod in (self.goal_cls_head, self.goal_off_head,
                        self.goal_embed[-1]):
                nn.init.zeros_(mod.weight)
                nn.init.zeros_(mod.bias)
        else:
            self.goal_cls_head = None
            self.goal_off_head = None
            self.goal_embed = None

        if self.aux_bev_motion:
            # Input is embed_dims (bev_embed's own channel width), not
            # ego_fut_dec_in_dim -- this reads the BEV encoder's raw output
            # directly, before ego_feats (which concatenates agent/map
            # decoder outputs) exists.
            if self.aux_bev_motion_temporal:
                # 1x1 channel reduction before the regional pool, so the
                # descriptor stays small: proj_dim * grid^2 per frame
                # instead of embed_dims * grid^2. Shared between the
                # current and previous frame so their difference is taken
                # in one feature space.
                self.aux_bev_motion_proj = nn.Conv2d(
                    self.embed_dims, self.aux_bev_motion_proj_dim, 1)
                desc_dim = (self.aux_bev_motion_proj_dim
                            * self.aux_bev_motion_grid ** 2)
                # frames=2: [current, d1]           -> velocity only
                # frames=3: [current, d1, d1 - d2]  -> velocity + accel
                # One block per frame either way, so the width is just
                # frames * desc_dim. See aux_bev_motion_frames' comment.
                aux_bev_in_dim = self.aux_bev_motion_frames * desc_dim
            else:
                self.aux_bev_motion_proj = None
                aux_bev_in_dim = self.embed_dims
            self.aux_bev_motion_head = nn.Sequential(
                nn.Linear(aux_bev_in_dim, self.embed_dims),
                nn.ReLU(),
                nn.Linear(self.embed_dims, len(self.aux_bev_motion_idx)),
            )
            if self.ego_status_est_dim is not None:
                # Reads the same descriptor aux_bev_motion_head regresses from. A
                # separate head rather than a reuse of bev_pred's 4 scalars: the target
                # here is the teacher's decoder-input EMBEDDING, not physical
                # quantities, and 4 numbers cannot span it.
                #
                # Hidden width defaults to embed_dims. Setting it to the teacher's
                # ego_lcf_embed_hidden makes the second Linear the same shape on both
                # sides, which lets the teacher's trained (hidden -> slot) map be copied
                # in rather than relearned.
                est_hidden = self.ego_status_est_hidden or self.embed_dims
                self.ego_status_est_net = nn.Sequential(
                    nn.Linear(aux_bev_in_dim, est_hidden),
                    nn.ReLU(),
                    nn.Linear(est_hidden, self.ego_status_est_dim),
                )
            else:
                self.ego_status_est_net = None
            # Reads the status slot (8 numbers), not the descriptor, so the
            # only way to lower this loss is to put the physical state INTO
            # the slot -- which is the point. A single Linear on purpose:
            # anything deeper could hide the state in a form the planner's
            # own first Linear cannot use.
            if self.ego_status_decode:
                slot = (self.ego_status_est_dim
                        if self.ego_status_est_dim is not None
                        else (self.ego_lcf_embed_dim
                              if self.ego_lcf_embed_dim else None))
                if slot is None:
                    raise ValueError(
                        'ego_status_decode needs a status slot: set '
                        'ego_status_est_dim (student) or ego_lcf_embed_dim '
                        '(teacher).')
                if not self.aux_bev_motion_idx:
                    raise ValueError(
                        'ego_status_decode decodes the aux_bev_motion_idx '
                        'columns of ego_lcf, so that must be set.')
                self.ego_status_decode_head = nn.Linear(
                    slot, len(self.aux_bev_motion_idx))
                # Zero-init, the convention plan_bev_refine_mlp, prism_z_proj and
                # ego_lcf_embed_net's last layer already follow here.
                #
                # A random Linear(8 -> 6) reading a slot whose vx and speed are around
                # 10.5 emits O(1..10), and dividing by yaw_rate's small normalizer
                # amplifies it further -- the loss would start orders above
                # loss_plan_reg and spend the gradient budget on an init artifact.
                nn.init.zeros_(self.ego_status_decode_head.weight)
                nn.init.zeros_(self.ego_status_decode_head.bias)
            else:
                self.ego_status_decode_head = None
            if self.aux_bev_future_motion:
                # Reads the same descriptor aux_bev_motion_head does -- that
                # head is the standing evidence this descriptor carries ego
                # motion at all. A separate head rather than extra outputs
                # on the existing one so the two targets (present vs future)
                # keep separate weights and can be enabled independently.
                self.aux_bev_future_motion_head = nn.Sequential(
                    nn.Linear(aux_bev_in_dim, self.embed_dims),
                    nn.ReLU(),
                    nn.Linear(self.embed_dims,
                              self.aux_bev_future_motion_ts),
                )
            else:
                self.aux_bev_future_motion_head = None
        else:
            self.aux_bev_motion_head = None
            self.aux_bev_motion_proj = None
            self.ego_status_est_net = None
            self.aux_bev_future_motion_head = None
            self.ego_status_decode_head = None
            if self.aux_bev_future_motion:
                raise ValueError(
                    'aux_bev_future_motion needs aux_bev_motion=True: it '
                    'reads the BEV motion descriptor built in that block.')

        if self.prism_latent_supervision:
            # Prior: sees only what's already available at inference (the
            # same vision-derived context that feeds ego_fut_decoder).
            self.prism_prior_net = nn.Sequential(
                nn.Linear(ego_fut_dec_in_dim, self.embed_dims),
                nn.ReLU(),
                nn.Linear(self.embed_dims, 2 * self.prism_latent_dim),
            )
            # Posterior: sees the GT 0-5s privileged future trajectory,
            # optionally concatenated with current ego_lcf status (see
            # prism_posterior_lcf_idx above). Train-time only -- never
            # constructed from data that exists at inference. long_fut_trajs
            # is per-step (x,y) deltas.
            posterior_in_dim = self.prism_long_fut_ts * 2 + (
                len(self.prism_posterior_lcf_idx)
                if self.prism_posterior_lcf_idx else 0)
            self.prism_posterior_net = nn.Sequential(
                nn.Linear(posterior_in_dim, self.embed_dims),
                nn.ReLU(),
                nn.Linear(self.embed_dims, 2 * self.prism_latent_dim),
            )
            # Projects a sampled latent back into ego_feats space for
            # injection. Zero-init so this starts as a no-op, matching
            # plan_bev_refine_mlp's convention above -- KL pressure and
            # reconstruction gradient turn it on gradually rather than
            # perturbing ego_feats with noise from step 0.
            self.prism_z_proj = nn.Linear(self.prism_latent_dim, ego_fut_dec_in_dim)
            nn.init.zeros_(self.prism_z_proj.weight)
            nn.init.zeros_(self.prism_z_proj.bias)
        else:
            self.prism_prior_net = None
            self.prism_posterior_net = None
            self.prism_z_proj = None

        self.agent_fus_mlp = nn.Sequential(
            nn.Linear(self.fut_mode*2*self.embed_dims, self.embed_dims, bias=True),
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(),
            nn.Linear(self.embed_dims, self.embed_dims, bias=True))

    def _anchors(self, like):
        """[mode, K, 4] centres/widths and [mode, K] validity, on like's device."""
        tab = self.goal_anchor_table.to(device=like.device, dtype=like.dtype)
        mask = self.goal_anchor_mask.to(device=like.device, dtype=torch.bool)
        return tab, mask

    def _goal_points(self, off, like=None):
        """off [B, mode, K, 2] in anchor widths -> goal points in metres."""
        return goal_points_from_offsets(off, self.goal_anchor_table)

    def _goal_scale(self, like):
        return like.new_tensor(self.goal_coordinate_scale)

    def _goal_label_from_target(self, ego_target_point, batch, dtype, device,
                                cmd):
        """TRAINING LABELS ONLY: nearest anchor index [B] and its offset [B, 2].

        The anchor grid is ragged per command, so the label is the nearest
        anchor of the commanded mode, measured in anchor-width units, and the
        offset is what the regression has to add to reach the real target.
        """
        tp = ego_target_point.to(device=device, dtype=torch.float32).reshape(
            batch, -1)[:, :2]
        if self._legacy_goal_bins:
            # Preserve the exact bucket boundary convention of old models.
            edges = tp.new_tensor(self.goal_bin_edges)
            cell = torch.bucketize(tp[:, 0].contiguous(), edges[1:-1], right=True)
            row = self.goal_anchor_table.to(device=device, dtype=torch.float32)[cmd, cell]
            return cell, (tp - row[:, 0::2]) / row[:, 1::2]
        return goal_labels_from_targets(
            tp, cmd, self.goal_anchor_table, self.goal_anchor_mask)

    def _decode_to_goal(self, feats, goals):
        """feats [B, D], goals [B, mode, 2] -> [B, mode, ego_dec_ts, 2].

        Mode m's trajectory is taken from the decoder pass conditioned on
        mode m's goal; the decoder itself is shared by every goal.
        """
        b, m = goals.shape[0], self.ego_fut_mode
        normalized_goal = (goals.float() / self._goal_scale(goals.float())).to(feats.dtype)
        cond = feats[:, None, :] + self.goal_embed(normalized_goal)   # [B, M, D]
        out = self.ego_fut_decoder(cond).reshape(b, m, m, self.ego_dec_ts, 2)
        idx = torch.arange(m, device=feats.device)
        return out[:, idx, idx]

    def _load_from_state_dict(self, state_dict, prefix, local_metadata,
                              strict, missing_keys, unexpected_keys,
                              error_msgs):
        """Extend a fut_ts-step decoder head to goal_long_ts steps.

        Every donor in this repo has ego_fut_decoder's last layer at
        mode*fut_ts*2. Loading it into a goal_long_ts head would be a size
        mismatch, which mmcv's strict=False turns into a silently random
        planner. Instead the first fut_ts steps copy the donor and each extra
        step repeats the donor's last step (constant velocity), so the
        scored output is the donor's exactly and the 5s tail starts sensible.
        """
        if self.cell_planner:
            self._cell_load_hook(state_dict, prefix)
        if self.goal_pred and self.ego_dec_ts != self.fut_ts:
            last = len(self.ego_fut_decoder) - 1
            for suffix in ('weight', 'bias'):
                key = f'{prefix}ego_fut_decoder.{last}.{suffix}'
                if key not in state_dict:
                    continue
                w = state_dict[key]
                if w.shape[0] != self.ego_fut_mode * self.fut_ts * 2:
                    continue
                tail = w.shape[1:]
                w = w.reshape(self.ego_fut_mode, self.fut_ts, 2, *tail)
                extra = w[:, -1:].expand(
                    self.ego_fut_mode, self.ego_dec_ts - self.fut_ts, 2, *tail)
                state_dict[key] = torch.cat([w, extra], dim=1).reshape(
                    -1, *tail).clone()
        super()._load_from_state_dict(state_dict, prefix, local_metadata,
                                      strict, missing_keys, unexpected_keys,
                                      error_msgs)

    def _cell_load_hook(self, state_dict, prefix):
        """Refuse a checkpoint with another cell layout; seed new heads from
        the donor's ego_fut_decoder.

        A donor has no cell heads. Each head then starts as the donor's
        decoder for its own command: hidden layers copied, the positional-code
        columns of the first Linear zero, and the last Linear's rows for that
        command. Every cell's initial trajectory is the donor's.
        """
        for c, layout in enumerate(self.cell_layouts):
            if layout is None:
                continue
            for name in (f'cell_fwd_edges_{c}', f'cell_lat_edges_{c}', f'cell_pe_{c}'):
                key = prefix + name
                if key in state_dict and not torch.equal(
                        state_dict[key].float().cpu(), getattr(self, name).float().cpu()):
                    raise RuntimeError(f'{key}: the checkpoint was trained with a '
                                       'different cell layout than this config')
        if f'{prefix}cell_heads.0.0.weight' in state_dict:
            return
        linear_ids = [i for i, m in enumerate(self.ego_fut_decoder)
                      if isinstance(m, nn.Linear)]
        src = [state_dict.get(f'{prefix}ego_fut_decoder.{i}.weight') for i in linear_ids]
        if any(w is None for w in src):
            return
        last = linear_ids[-1]
        w_last = state_dict[f'{prefix}ego_fut_decoder.{last}.weight']
        b_last = state_dict[f'{prefix}ego_fut_decoder.{last}.bias']
        per = self.fut_ts * 2
        if w_last.shape[0] != self.ego_fut_mode * per:
            raise RuntimeError(f'donor ego_fut_decoder emits {w_last.shape[0]} values, '
                               f'expected {self.ego_fut_mode * per}')
        for c, layout in enumerate(self.cell_layouts):
            for i in linear_ids:
                w = state_dict[f'{prefix}ego_fut_decoder.{i}.weight']
                b = state_dict[f'{prefix}ego_fut_decoder.{i}.bias']
                if i == linear_ids[0] and layout is not None:
                    w = torch.cat([w, w.new_zeros(w.shape[0], self.cell_pe_dim)], dim=1)
                if i == last:
                    w, b = w[c * per:(c + 1) * per], b[c * per:(c + 1) * per]
                state_dict[f'{prefix}cell_heads.{c}.{i}.weight'] = w.clone()
                state_dict[f'{prefix}cell_heads.{c}.{i}.bias'] = b.clone()

    def cell_generate(self, feats):
        """feats [B, D] -> ({command: [B, Nf, Nl, T, 2]}, U_TURN [B, T, 2],
        STOP [B, T, 2]). Reads no target point."""
        bsz = feats.shape[0]
        cells, plain = {}, {}
        for c, layout in enumerate(self.cell_layouts):
            head = self.cell_heads[c]
            if layout is None:
                plain[c] = head(feats).view(bsz, self.fut_ts, 2)
                continue
            pe = getattr(self, f'cell_pe_{c}').to(feats.dtype)
            nf, nl = pe.shape[:2]
            x = torch.cat([feats[:, None, None, :].expand(bsz, nf, nl, feats.shape[-1]),
                           pe[None].expand(bsz, nf, nl, pe.shape[-1])], dim=-1)
            cells[c] = head(x).view(bsz, nf, nl, self.fut_ts, 2)
        return cells, plain[5], plain[CELL_STOP]

    def cell_edges(self):
        return {c: (getattr(self, f'cell_fwd_edges_{c}'), getattr(self, f'cell_lat_edges_{c}'))
                for c, layout in enumerate(self.cell_layouts) if layout is not None}

    def _distill_status(self, ego_status_est):
        """Select the columns loss_status_distill is allowed to see.

        Applied to the OUTPUT only. ego_feats still carries the full status
        vector, so ego_fut_decoder's width and the stage-1 donor's transfer
        are unaffected -- this narrows the cosine target, nothing else. See
        ego_status_distill_idx in __init__ for why the full vector is a poor
        target (two of its eight columns are constants).
        """
        if ego_status_est is None or self.ego_status_distill_idx is None:
            return ego_status_est
        idx = ego_status_est.new_tensor(self.ego_status_distill_idx,
                                        dtype=torch.long)
        return ego_status_est.index_select(-1, idx)

    def bev_motion_descriptor(self, bev):
        """Shift-sensitive pooled descriptor of a BEV feature map.

        Args:
            bev: [N, B, D] sequence-first BEV (the format bev_embed and
                the stored prev_bev both use -- see
                refine_ego_trajs_with_bev's docstring for why that axis
                order matters). A [B, N, D] tensor is normalized first,
                since prev_bev reaches the head in either layout depending
                on which path produced it (VAD_transformer.get_bev_features
                returns [B, N, D], the full forward returns [N, B, D]).

        Returns:
            [B, proj_dim * grid^2] -- a grid x grid regional average of the
            channel-reduced BEV, flattened. Regional rather than global
            precisely so that an ego-motion-induced shift of the BEV
            content changes it (see aux_bev_motion_temporal's constructor
            comment).
        """
        if bev.shape[1] == self.bev_h * self.bev_w:
            bev = bev.permute(1, 0, 2)
        batch = bev.shape[1]
        # .clone() is load-bearing, not defensive: permute+reshape here
        # returns a VIEW into the caller's storage (splitting the last axis
        # of a [B, D, N] permuted tensor is expressible in strides), and the
        # transformer later rotates prev_bev IN PLACE. The conv saves its
        # input for the weight gradient, so without the copy autograd fails
        # with "variable needed for gradient computation has been modified
        # by an inplace operation".
        bev_map = bev.permute(1, 2, 0).reshape(
            batch, self.embed_dims, self.bev_h, self.bev_w).clone()
        bev_map = self.aux_bev_motion_proj(bev_map.to(
            self.aux_bev_motion_proj.weight.dtype))
        bev_map = F.adaptive_avg_pool2d(bev_map, self.aux_bev_motion_grid)
        return bev_map.flatten(1)

    def refine_ego_trajs_with_bev(self, ego_trajs, bev_embed):
        """ThinkTwice-lite: sample bev_embed at each coarse waypoint's own
        (x, y) location (ego frame, meters) and predict a correction from
        it. `plan_bev_refine_mlp`'s last layer is zero-init, so this is an
        exact no-op (delta==0, and zero gradient into ego_trajs through this
        path) until training moves it away from that.

        ego_trajs (this function's input/output, and everywhere else in
        this codebase ego_fut_preds/outputs_ego_trajs is consumed --
        VAD.py:461, VAD_LAW.py:184, eval_holdout_l2*.py, etri_test_submit.py
        -- all .cumsum(dim=-2) it before treating it as positions) is
        **per-step displacement, not absolute waypoints**. grid_sample needs
        real ego-frame positions, so this cumsums to absolute internally for
        the lookup and the refined output, then converts back to
        displacement before returning -- so the offset-format contract this
        function's caller (forward()) and every downstream consumer expects
        is preserved. Getting this wrong would have meant sampling bev_embed
        near the very first few waypoints' tiny displacements regardless of
        where the trajectory actually goes, for every waypoint.

        Args:
            ego_trajs: [B, ego_fut_mode, fut_ts, 2], coarse plan, per-step
                displacement (ego frame, meters).
            bev_embed: [bev_h*bev_w, B, D], this frame's own BEV feature (the
                same one det/map heads use) -- sequence-first, matching
                VAD_transformer.py's `bev_embed.permute(1, 0, 2)` before
                it's handed to the decoder (VAD_transformer.py:399) and
                returned as-is from there, never permuted back. Flattened
                row-major (h then w) within the leading axis, matching
                self.bev_embedding/grid_length's convention.

                NOT [B, bev_h*bev_w, D] -- an earlier version of this
                docstring assumed that, and .permute(0, 2, 1).reshape(...)
                on the real [N, B, D] tensor doesn't error out (same total
                element count) but silently reads B and N transposed,
                scrambling which channel vector lands at which spatial
                cell. With B=1 this produces a shaped-correctly output that
                is nonetheless the wrong permutation of the same numbers,
                so it never crashed -- found via aux_bev_motion's B-first
                pooling actually raising a shape error on the same tensor.
        Returns:
            [B, ego_fut_mode, fut_ts, 2], refined plan, same per-step
            displacement format as the input.
        """
        B, M, T, _ = ego_trajs.shape
        D = bev_embed.shape[-1]
        bev_map = bev_embed.permute(1, 2, 0).reshape(B, D, self.bev_h, self.bev_w)

        abs_traj = ego_trajs.cumsum(dim=-2)  # displacement -> absolute (x, y)

        # Cascade: stage 0 is plan_bev_refine_mlp, then one pass per module
        # in plan_bev_refine_mlp_extra. Each pass re-runs the grid_sample
        # below against the positions the previous pass produced.
        stages = [self.plan_bev_refine_mlp]
        if self.plan_bev_refine_mlp_extra is not None:
            stages = stages + list(self.plan_bev_refine_mlp_extra)

        for refine_mlp in stages:
            abs_traj = self._bev_refine_step(abs_traj, bev_map, refine_mlp)

        refined_abs = abs_traj

        # absolute -> displacement (inverse of cumsum along the T axis).
        # At init every stage's delta is 0 so refined_abs == the input's
        # cumsum and this recovers ego_trajs exactly -- the no-op property
        # holds end to end across the whole cascade, not just per stage.
        refined_offset = torch.cat([
            refined_abs[..., :1, :],
            refined_abs[..., 1:, :] - refined_abs[..., :-1, :],
        ], dim=-2)
        return refined_offset

    def _bev_refine_step(self, abs_traj, bev_map, refine_mlp):
        """One cascade stage: look up bev_map at abs_traj, add a correction.

        Args:
            abs_traj: [B, M, T, 2] absolute ego-frame positions (meters).
            bev_map: [B, D, bev_h, bev_w] this frame's BEV feature.
            refine_mlp: the stage's zero-init MLP.
        Returns:
            [B, M, T, 2] corrected absolute positions.
        """
        B, M, T, _ = abs_traj.shape
        xs = abs_traj[..., 0]
        ys = abs_traj[..., 1]
        # self.real_w/self.real_h and pc_range[0]/pc_range[1] already used
        # this same way for detection/map box (de)normalization elsewhere in
        # this file -- bev_w spans pc_range's x-extent (real_w), bev_h spans
        # its y-extent (real_h), per grid_length=(real_h/bev_h, real_w/bev_w)
        # at the transformer call above. Cross-checked against encoder.py's
        # get_reference_points/point_sampling (xs normalized by W -> index 0
        # -> pc_range[0]/[3]; ys normalized by H -> index 1 -> pc_range[1]/[4])
        # -- same axis assignment used here.
        grid_x = (xs - self.pc_range[0]) / self.real_w * 2 - 1
        grid_y = (ys - self.pc_range[1]) / self.real_h * 2 - 1
        grid = torch.stack([grid_x, grid_y], dim=-1).reshape(B, M * T, 1, 2)

        sampled = F.grid_sample(
            bev_map.to(grid.dtype), grid, mode='bilinear',
            padding_mode='zeros', align_corners=False)  # [B, D, M*T, 1]
        sampled = sampled.squeeze(-1).permute(0, 2, 1)  # [B, M*T, D]

        # _debug_zero_bev_sampled: eval-only (never set during training) --
        # zeros the sampled BEV feature so the only thing plan_bev_refine_mlp
        # sees is what padding_mode='zeros' would also give an out-of-range
        # waypoint. If the refined output barely changes vs the non-zeroed
        # case, the module isn't actually leaning on BEV content.
        if getattr(self, '_debug_zero_bev_sampled', False):
            sampled = torch.zeros_like(sampled)

        # sampled only -- see plan_bev_refine_mlp's construction comment.
        delta_abs = refine_mlp(sampled).reshape(B, M, T, 2)
        return abs_traj + delta_abs

    def init_weights(self):
        """Initialize weights of the DeformDETR head."""
        self.transformer.init_weights()
        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            for m in self.cls_branches:
                nn.init.constant_(m[-1].bias, bias_init)
        if self.loss_map_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            for m in self.map_cls_branches:
                nn.init.constant_(m[-1].bias, bias_init)
        if self.loss_traj_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            for m in self.traj_cls_branches:
                nn.init.constant_(m[-1].bias, bias_init)
        # for m in self.map_reg_branches:
        #     constant_init(m[-1], 0, bias=0)
        # nn.init.constant_(self.map_reg_branches[0][-1].bias.data[2:], 0.)
        if self.motion_decoder is not None:
            for p in self.motion_decoder.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
            nn.init.orthogonal_(self.motion_mode_query.weight)
            if self.use_pe:
                xavier_init(self.pos_mlp_sa, distribution='uniform', bias=0.)
        if self.motion_map_decoder is not None:
            for p in self.motion_map_decoder.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
            for p in self.lane_encoder.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
            if self.use_pe:
                xavier_init(self.pos_mlp, distribution='uniform', bias=0.)
        if self.ego_his_encoder is not None:
            for p in self.ego_his_encoder.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
        if self.ego_agent_decoder is not None:
            for p in self.ego_agent_decoder.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
        if self.ego_map_decoder is not None:
            for p in self.ego_map_decoder.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

    # @auto_fp16(apply_to=('mlvl_feats'))
    @force_fp32(apply_to=('mlvl_feats', 'prev_bev'))
    def forward(self,
                mlvl_feats,
                img_metas,
                prev_bev=None,
                only_bev=False,
                ego_his_trajs=None,
                ego_lcf_feat=None,
                ego_target_point=None,
                ego_long_fut_trajs=None,
                ego_long_fut_valid_flag=None,
                ego_lcf_target=None,
                ego_fut_cmd=None,
                prev_bev2=None,
            ):
        """Forward function.
        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 5D-tensor with shape
                (B, N, C, H, W).
            prev_bev: previous bev featues
            only_bev: only compute BEV features with encoder. 
        Returns:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, theta, vx, vy). \
                Shape [nb_dec, bs, num_query, 9].
        """
        
        bs, num_cam, _, _, _ = mlvl_feats[0].shape
        dtype = mlvl_feats[0].dtype
        object_query_embeds = self.query_embedding.weight.to(dtype)

        # Snapshot the previous frame's descriptor BEFORE the transformer
        # runs. The encoder rotates prev_bev IN PLACE to yaw-align it
        # (VAD_transformer.py's `prev_bev[:, i] = tmp_prev_bev[:, 0]`,
        # writing through a permuted view into the caller's tensor), so
        # reading prev_bev after the transformer call would compare against
        # an already-yaw-compensated map -- erasing exactly the rotation
        # signal aux_bev_motion is trying to regress yaw rate from.
        prev_bev_desc = None
        prev_bev2_desc = None
        if (self.aux_bev_motion and self.aux_bev_motion_temporal
                and prev_bev is not None):
            prev_bev_desc = self.bev_motion_descriptor(prev_bev)
        # Same in-place hazard as prev_bev: snapshot before the transformer
        # runs. prev_bev2 is never handed to the encoder (only prev_bev is),
        # so it is not rotated -- but it is read here for symmetry with
        # prev_bev's timing and to keep both descriptors from the same
        # pre-transformer state.
        if (self.aux_bev_motion and self.aux_bev_motion_temporal
                and self.aux_bev_motion_frames >= 3
                and prev_bev2 is not None):
            prev_bev2_desc = self.bev_motion_descriptor(prev_bev2)
        
        if self.map_query_embed_type == 'all_pts':
            map_query_embeds = self.map_query_embedding.weight.to(dtype)
        elif self.map_query_embed_type == 'instance_pts':
            map_pts_embeds = self.map_pts_embedding.weight.unsqueeze(0)
            map_instance_embeds = self.map_instance_embedding.weight.unsqueeze(1)
            map_query_embeds = (map_pts_embeds + map_instance_embeds).flatten(0, 1).to(dtype)

        bev_queries = self.bev_embedding.weight.to(dtype)

        bev_mask = torch.zeros((bs, self.bev_h, self.bev_w),
                               device=bev_queries.device).to(dtype)
        bev_pos = self.positional_encoding(bev_mask).to(dtype)
            
        if only_bev:  # only use encoder to obtain BEV features, TODO: refine the workaround
            return self.transformer.get_bev_features(
                mlvl_feats,
                bev_queries,
                self.bev_h,
                self.bev_w,
                grid_length=(self.real_h / self.bev_h,
                             self.real_w / self.bev_w),
                bev_pos=bev_pos,
                img_metas=img_metas,
                prev_bev=prev_bev,
            )
        else:
            outputs = self.transformer(
                mlvl_feats,
                bev_queries,
                object_query_embeds,
                map_query_embeds,
                self.bev_h,
                self.bev_w,
                grid_length=(self.real_h / self.bev_h,
                             self.real_w / self.bev_w),
                bev_pos=bev_pos,
                reg_branches=self.reg_branches if self.with_box_refine else None,  # noqa:E501
                cls_branches=self.cls_branches if self.as_two_stage else None,
                map_reg_branches=self.map_reg_branches if self.with_box_refine else None,  # noqa:E501
                map_cls_branches=self.map_cls_branches if self.as_two_stage else None,
                img_metas=img_metas,
                prev_bev=prev_bev
        )

        bev_embed, hs, init_reference, inter_references, \
            map_hs, map_init_reference, map_inter_references = outputs

        # Auxiliary BEV-level ego-motion regression, placed at the earliest
        # possible point: bev_embed is the shared encoder output every downstream
        # head branches off, so gradient from this loss reaches the BEV encoder
        # itself rather than only a late planning feature.
        #
        # bev_pred is a function of bev_embed alone and is computed at both train
        # and eval time. Only the supervision LOSS needs ego_lcf_target.
        aux_bev_motion_loss = None
        aux_bev_future_motion_loss = None
        bev_pred = None
        bev_pooled = None
        if self.aux_bev_motion:
            if self.aux_bev_motion_temporal:
                # Explicit temporal contrast: [current, current - previous].
                # prev_bev is None on every cold-start frame (each eval
                # window's first frame, and prev_bev_dropout's training
                # steps), in which case the delta half is zeros -- the
                # head is trained on that case too, so it degrades to
                # "estimate from a single frame" rather than breaking.
                cur_desc = self.bev_motion_descriptor(bev_embed)
                if prev_bev_desc is not None:
                    delta = cur_desc - prev_bev_desc
                else:
                    delta = torch.zeros_like(cur_desc)
                blocks = [cur_desc, delta]
                if self.aux_bev_motion_frames >= 3:
                    # Second difference: (cur - prev1) - (prev1 - prev2). That IS the
                    # acceleration, so hand it over directly rather than passing prev2's own
                    # delta and hoping the first Linear learns to subtract two inputs.
                    #
                    # Zeros whenever either history frame is missing. The head trains on that
                    # case too, so it degrades to the 2-frame answer instead of breaking.
                    if prev_bev_desc is not None and prev_bev2_desc is not None:
                        accel = delta - (prev_bev_desc - prev_bev2_desc)
                    else:
                        accel = torch.zeros_like(cur_desc)
                    blocks.append(accel)
                bev_pooled = torch.cat(blocks, dim=-1)
            else:
                # bev_embed is [N, B, D] (sequence-first, see
                # refine_ego_trajs_with_bev's docstring) -- pool over dim=0.
                bev_pooled = bev_embed.mean(dim=0)  # [N, B, D] -> [B, D]
            bev_pred = self.aux_bev_motion_head(bev_pooled)  # [B, K]
            if self.training and ego_lcf_target is not None:
                bev_gt = ego_lcf_target.squeeze(1)[..., self.aux_bev_motion_idx]
                bev_gt = bev_gt.reshape(bev_pred.shape).to(bev_pred.dtype)
                if self.aux_bev_motion_norm is None:
                    aux_bev_motion_loss = (
                        self.aux_bev_motion_weight
                        * F.l1_loss(bev_pred, bev_gt))
                else:
                    # Relative error instead of absolute: divide each
                    # residual by that component's scale so a 10 m/s
                    # velocity and a 0.02 rad/s yaw rate contribute
                    # comparably. See the aux_bev_motion_norm constructor
                    # comment for the measured imbalance this fixes.
                    scale = torch.as_tensor(
                        self.aux_bev_motion_norm,
                        device=bev_pred.device, dtype=bev_pred.dtype)
                    aux_bev_motion_loss = self.aux_bev_motion_weight * (
                        (bev_pred - bev_gt).abs() / scale).mean()

            # Future ego speed profile, from the same descriptor.
            if (self.aux_bev_future_motion_head is not None
                    and self.training
                    and ego_long_fut_trajs is not None):
                fut_pred = self.aux_bev_future_motion_head(bev_pooled)
                T = self.aux_bev_future_motion_ts
                # Per-step deltas (converter builds them with np.diff), so
                # the distance covered in one 0.5s step IS speed * dt. Take
                # the first T steps; ego_long_fut_trajs runs 0-5s while the
                # scored horizon is 3s.
                long_gt = ego_long_fut_trajs.reshape(
                    fut_pred.shape[0], -1, 2)[:, :T].to(fut_pred.dtype)
                speed_gt = long_gt.norm(dim=-1) / FUT_TS_INTERVAL_S
                loss_per_sample = (
                    (fut_pred - speed_gt).abs()
                    / self.aux_bev_future_motion_norm).mean(dim=-1)
                if ego_long_fut_valid_flag is not None:
                    # Same per-sample masking PRISM and aux_long_horizon use:
                    # near a scene's end the future window runs off the data,
                    # and those rows carry garbage rather than a real target.
                    valid = ego_long_fut_valid_flag.reshape(-1).to(
                        fut_pred.dtype)
                    aux_bev_future_motion_loss = (
                        self.aux_bev_future_motion_weight
                        * (loss_per_sample * valid).sum()
                        / valid.sum().clamp(min=1.0))
                else:
                    aux_bev_future_motion_loss = (
                        self.aux_bev_future_motion_weight
                        * loss_per_sample.mean())

        hs = hs.permute(0, 2, 1, 3)
        outputs_classes = []
        outputs_coords = []
        outputs_coords_bev = []
        outputs_trajs = []
        outputs_trajs_classes = []

        map_hs = map_hs.permute(0, 2, 1, 3)
        map_outputs_classes = []
        map_outputs_coords = []
        map_outputs_pts_coords = []
        map_outputs_coords_bev = []

        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.cls_branches[lvl](hs[lvl])
            tmp = self.reg_branches[lvl](hs[lvl])

            # TODO: check the shape of reference
            assert reference.shape[-1] == 3
            tmp[..., 0:2] = tmp[..., 0:2] + reference[..., 0:2]
            tmp[..., 0:2] = tmp[..., 0:2].sigmoid()
            outputs_coords_bev.append(tmp[..., 0:2].clone().detach())
            tmp[..., 4:5] = tmp[..., 4:5] + reference[..., 2:3]
            tmp[..., 4:5] = tmp[..., 4:5].sigmoid()
            tmp[..., 0:1] = (tmp[..., 0:1] * (self.pc_range[3] -
                             self.pc_range[0]) + self.pc_range[0])
            tmp[..., 1:2] = (tmp[..., 1:2] * (self.pc_range[4] -
                             self.pc_range[1]) + self.pc_range[1])
            tmp[..., 4:5] = (tmp[..., 4:5] * (self.pc_range[5] -
                             self.pc_range[2]) + self.pc_range[2])

            # TODO: check if using sigmoid
            outputs_coord = tmp
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
        
        for lvl in range(map_hs.shape[0]):
            if lvl == 0:
                reference = map_init_reference
            else:
                reference = map_inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            map_outputs_class = self.map_cls_branches[lvl](
                map_hs[lvl].view(bs,self.map_num_vec, self.map_num_pts_per_vec,-1).mean(2)
            )
            tmp = self.map_reg_branches[lvl](map_hs[lvl])
            # TODO: check the shape of reference
            assert reference.shape[-1] == 2
            tmp[..., 0:2] += reference[..., 0:2]
            tmp = tmp.sigmoid() # cx,cy,w,h
            map_outputs_coord, map_outputs_pts_coord = self.map_transform_box(tmp)
            map_outputs_coords_bev.append(map_outputs_pts_coord[..., :2].clone().detach())
            map_outputs_classes.append(map_outputs_class)
            map_outputs_coords.append(map_outputs_coord)
            map_outputs_pts_coords.append(map_outputs_pts_coord)
            
        if self.motion_decoder is not None:
            batch_size, num_agent = outputs_coords_bev[-1].shape[:2]
            # motion_query
            motion_query = hs[-1].permute(1, 0, 2)  # [A, B, D]
            mode_query = self.motion_mode_query.weight  # [fut_mode, D]
            # [M, B, D], M=A*fut_mode
            motion_query = (motion_query[:, None, :, :] + mode_query[None, :, None, :]).flatten(0, 1)
            if self.use_pe:
                motion_coords = outputs_coords_bev[-1]  # [B, A, 2]
                motion_pos = self.pos_mlp_sa(motion_coords)  # [B, A, D]
                motion_pos = motion_pos.unsqueeze(2).repeat(1, 1, self.fut_mode, 1).flatten(1, 2)
                motion_pos = motion_pos.permute(1, 0, 2)  # [M, B, D]
            else:
                motion_pos = None

            if self.motion_det_score is not None:
                motion_score = outputs_classes[-1]
                max_motion_score = motion_score.max(dim=-1)[0]
                invalid_motion_idx = max_motion_score < self.motion_det_score  # [B, A]
                invalid_motion_idx = invalid_motion_idx.unsqueeze(2).repeat(1, 1, self.fut_mode).flatten(1, 2)
            else:
                invalid_motion_idx = None

            motion_hs = self.motion_decoder(
                query=motion_query,
                key=motion_query,
                value=motion_query,
                query_pos=motion_pos,
                key_pos=motion_pos,
                key_padding_mask=invalid_motion_idx)

            if self.motion_map_decoder is not None:
                # map preprocess
                motion_coords = outputs_coords_bev[-1]  # [B, A, 2]
                motion_coords = motion_coords.unsqueeze(2).repeat(1, 1, self.fut_mode, 1).flatten(1, 2)
                map_query = map_hs[-1].view(batch_size, self.map_num_vec, self.map_num_pts_per_vec, -1)
                map_query = self.lane_encoder(map_query)  # [B, P, pts, D] -> [B, P, D]
                map_score = map_outputs_classes[-1]
                map_pos = map_outputs_coords_bev[-1]
                map_query, map_pos, key_padding_mask = self.select_and_pad_pred_map(
                    motion_coords, map_query, map_score, map_pos,
                    map_thresh=self.map_thresh, dis_thresh=self.dis_thresh,
                    pe_normalization=self.pe_normalization, use_fix_pad=True)
                map_query = map_query.permute(1, 0, 2)  # [P, B*M, D]
                ca_motion_query = motion_hs.permute(1, 0, 2).flatten(0, 1).unsqueeze(0)

                # position encoding
                if self.use_pe:
                    (num_query, batch) = ca_motion_query.shape[:2] 
                    motion_pos = torch.zeros((num_query, batch, 2), device=motion_hs.device)
                    motion_pos = self.pos_mlp(motion_pos)
                    map_pos = map_pos.permute(1, 0, 2)
                    map_pos = self.pos_mlp(map_pos)
                else:
                    motion_pos, map_pos = None, None
                
                ca_motion_query = self.motion_map_decoder(
                    query=ca_motion_query,
                    key=map_query,
                    value=map_query,
                    query_pos=motion_pos,
                    key_pos=map_pos,
                    key_padding_mask=key_padding_mask)
            else:
                ca_motion_query = motion_hs.permute(1, 0, 2).flatten(0, 1).unsqueeze(0)

            batch_size = outputs_coords_bev[-1].shape[0]
            motion_hs = motion_hs.permute(1, 0, 2).unflatten(
                dim=1, sizes=(num_agent, self.fut_mode)
            )
            ca_motion_query = ca_motion_query.squeeze(0).unflatten(
                dim=0, sizes=(batch_size, num_agent, self.fut_mode)
            )
            motion_hs = torch.cat([motion_hs, ca_motion_query], dim=-1)  # [B, A, fut_mode, 2D]
        else:
            raise NotImplementedError('Not implement yet')

        outputs_traj = self.traj_branches[0](motion_hs)
        outputs_trajs.append(outputs_traj)
        outputs_traj_class = self.traj_cls_branches[0](motion_hs)
        outputs_trajs_classes.append(outputs_traj_class.squeeze(-1))
        (batch, num_agent) = motion_hs.shape[:2]
             
        map_outputs_classes = torch.stack(map_outputs_classes)
        map_outputs_coords = torch.stack(map_outputs_coords)
        map_outputs_pts_coords = torch.stack(map_outputs_pts_coords)

        outputs_classes = torch.stack(outputs_classes)
        outputs_coords = torch.stack(outputs_coords)
        outputs_trajs = torch.stack(outputs_trajs)
        outputs_trajs_classes = torch.stack(outputs_trajs_classes)

        # planning
        (batch, num_agent) = motion_hs.shape[:2]
        if self.ego_his_encoder is not None:
            ego_his_feats = self.ego_his_encoder(ego_his_trajs)  # [B, 1, dim]
        else:
            ego_his_feats = self.ego_query.weight.unsqueeze(0).repeat(batch, 1, 1)

        # Interaction
        ego_query = ego_his_feats
        # ego_pos is uninformative (zeros) in every submittable config --
        # target_point must never influence trajectory generation (organizer
        # ruling: it may only be used to SELECT among already-generated
        # candidates, at inference time, outside the network -- see
        # etri_test_submit.py). target_point_shortcut below is a
        # diagnostic-only escape hatch (default off) for building a
        # deliberately non-compliant teacher; see its constructor comment.
        goal_pos_norm = None
        goal_xy = None
        if self.target_point_shortcut:
            # VADLAW's obtain_history_prediction() reruns this head on
            # every history frame too (world-model objective), never
            # passing ego_target_point -- only the CURRENT frame's own
            # call does (same pattern as ego_lcf_target elsewhere in this
            # function). Fall back to an uninformative goal instead of
            # asserting: those calls' ego_fut_preds is discarded (only
            # bev_embed feeds the world model), so this never masks a
            # real missing-input bug at the frame that actually matters.
            if ego_target_point is None:
                goal_xy = torch.zeros(
                    (batch, 2), device=ego_query.device, dtype=ego_query.dtype)
            else:
                goal_xy = ego_target_point.to(
                    device=ego_query.device, dtype=ego_query.dtype
                ).reshape(batch, -1)
            # ego_agent_pos_mlp/ego_map_pos_mlp are trained on agent_pos/
            # map_pos, i.e. [0,1]-normalized BEV coordinates (see
            # outputs_coords_bev). goal_xy is real meters, so it needs the
            # same pc_range normalization before sharing that MLP.
            goal_pos_norm = goal_xy.clone()
            goal_pos_norm[..., 0] = (goal_xy[..., 0] - self.pc_range[0]) / self.real_w
            goal_pos_norm[..., 1] = (goal_xy[..., 1] - self.pc_range[1]) / self.real_h
        if (self.target_point_shortcut
                and self.target_point_shortcut_mode in ('attn', 'both')):
            ego_pos = goal_pos_norm.unsqueeze(1)
        else:
            ego_pos = torch.zeros((batch, 1, 2), device=ego_query.device)
        ego_pos_emb = self.ego_agent_pos_mlp(ego_pos)
        agent_conf = outputs_classes[-1]
        agent_query = motion_hs.reshape(batch, num_agent, -1)
        agent_query = self.agent_fus_mlp(agent_query) # [B, A, fut_mode, 2*D] -> [B, A, D]
        agent_pos = outputs_coords_bev[-1]
        agent_query, agent_pos, agent_mask = self.select_and_pad_query(
            agent_query, agent_pos, agent_conf,
            score_thresh=self.query_thresh, use_fix_pad=self.query_use_fix_pad
        )
        agent_pos_emb = self.ego_agent_pos_mlp(agent_pos)
        # ego <-> agent interaction
        ego_agent_query = self.ego_agent_decoder(
            query=ego_query.permute(1, 0, 2),
            key=agent_query.permute(1, 0, 2),
            value=agent_query.permute(1, 0, 2),
            query_pos=ego_pos_emb.permute(1, 0, 2),
            key_pos=agent_pos_emb.permute(1, 0, 2),
            key_padding_mask=agent_mask)

        # ego <-> map interaction
        if (self.target_point_shortcut
                and self.target_point_shortcut_mode in ('attn', 'both')):
            ego_pos = goal_pos_norm.unsqueeze(1)
        else:
            ego_pos = torch.zeros((batch, 1, 2), device=agent_query.device)
        ego_pos_emb = self.ego_map_pos_mlp(ego_pos)
        map_query = map_hs[-1].view(batch_size, self.map_num_vec, self.map_num_pts_per_vec, -1)
        map_query = self.lane_encoder(map_query)  # [B, P, pts, D] -> [B, P, D]
        map_conf = map_outputs_classes[-1]
        map_pos = map_outputs_coords_bev[-1]
        # use the most close pts pos in each map inst as the inst's pos
        batch, num_map = map_pos.shape[:2]
        map_dis = torch.sqrt(map_pos[..., 0]**2 + map_pos[..., 1]**2)
        min_map_pos_idx = map_dis.argmin(dim=-1).flatten()  # [B*P]
        min_map_pos = map_pos.flatten(0, 1)  # [B*P, pts, 2]
        min_map_pos = min_map_pos[range(min_map_pos.shape[0]), min_map_pos_idx]  # [B*P, 2]
        min_map_pos = min_map_pos.view(batch, num_map, 2)  # [B, P, 2]
        map_query, map_pos, map_mask = self.select_and_pad_query(
            map_query, min_map_pos, map_conf,
            score_thresh=self.query_thresh, use_fix_pad=self.query_use_fix_pad
        )
        map_pos_emb = self.ego_map_pos_mlp(map_pos)
        ego_map_query = self.ego_map_decoder(
            query=ego_agent_query,
            key=map_query.permute(1, 0, 2),
            value=map_query.permute(1, 0, 2),
            query_pos=ego_pos_emb.permute(1, 0, 2),
            key_pos=map_pos_emb.permute(1, 0, 2),
            key_padding_mask=map_mask)

        # Set by the two ego_his_encoder-free branches below (the ones our
        # configs use); left None on the ego_his_encoder paths, where the
        # 2*D scene half does not exist in that form. Scheme-A distillation
        # asserts on them being present rather than reading a stale value.
        ego_scene_feats = None
        ego_status_est = None
        if self.ego_his_encoder is not None and self.ego_lcf_feat_idx is not None:
            ego_feats = torch.cat(
                [ego_his_feats,
                 ego_map_query.permute(1, 0, 2),
                 ego_lcf_feat.squeeze(1)[..., self.ego_lcf_feat_idx]],
                dim=-1
            )  # [B, 1, 2D+2]
        elif self.ego_his_encoder is not None and self.ego_lcf_feat_idx is None:
            ego_feats = torch.cat(
                [ego_his_feats,
                 ego_map_query.permute(1, 0, 2)],
                dim=-1
            )  # [B, 1, 2D]
        elif self.ego_his_encoder is None and self.ego_lcf_feat_idx is not None:
            ego_status = ego_lcf_feat.squeeze(1)[..., self.ego_lcf_feat_idx]
            if self.ego_lcf_embed_net is not None:
                # Scheme A: a learned representation of ego status, not the
                # raw numbers -- see _init_layers.
                raw_status = ego_status.to(ego_agent_query.dtype)
                ego_status = self.ego_lcf_embed_net(raw_status)
                if self.ego_lcf_embed_residual:
                    # Zero-init residual: identical to raw_status at step 0,
                    # so a donor decoder trained on the raw columns keeps
                    # working while the embedding learns a deviation.
                    ego_status = raw_status + ego_status
            ego_scene_feats = torch.cat(
                [ego_agent_query.permute(1, 0, 2),
                 ego_map_query.permute(1, 0, 2)],
                dim=-1
            )  # [B, 1, 2D]
            # Scheme-A teacher's distillation target for the status half.
            # Only meaningful with ego_lcf_embed_net -- the raw columns are
            # 8 physical numbers, not a representation to align against.
            ego_status_est = (ego_status
                              if self.ego_lcf_embed_net is not None else None)
            ego_feats = torch.cat(
                [ego_scene_feats, ego_status], dim=-1
            )  # [B, 1, 2D + (lcf columns or embed_dim)]
        elif self.ego_his_encoder is None and self.ego_lcf_feat_idx is None:
            ego_scene_feats = torch.cat(
                [ego_agent_query.permute(1, 0, 2),
                 ego_map_query.permute(1, 0, 2)],
                dim=-1
            )  # [B, 1, 2D]
            if self.ego_status_est_net is not None:
                # Scheme-A student: fill the status slot from vision.
                ego_status = self.ego_status_est_net(
                    bev_pooled.to(ego_scene_feats.dtype)).unsqueeze(1)
                ego_status_est = ego_status
                if (self.training and self.ego_status_est_dropout > 0
                        and torch.rand(1).item()
                        < self.ego_status_est_dropout):
                    # Modality dropout -- zero the whole slot for this step
                    # so the decoder cannot come to depend on it. Applied
                    # AFTER ego_status_est is captured, so the distillation
                    # loss still trains the estimator on dropped steps;
                    # only the decoder's view is blanked.
                    ego_status = torch.zeros_like(ego_status)
                ego_feats = torch.cat([ego_scene_feats, ego_status], dim=-1)
            else:
                ego_status_est = None
                ego_feats = ego_scene_feats

        # Physical read-out of the slot the planner consumes. Uses
        # ego_status_est, captured before modality dropout, so the estimator
        # is supervised on dropped steps too -- same reasoning as the
        # distillation loss above it.
        ego_status_decode_loss = None
        if (self.ego_status_decode_head is not None and self.training
                and ego_status_est is not None and ego_lcf_target is not None):
            decoded = self.ego_status_decode_head(
                ego_status_est.to(self.ego_status_decode_head.weight.dtype))
            tgt = ego_lcf_target.reshape(
                decoded.shape[0], 1, -1)[..., self.aux_bev_motion_idx]
            tgt = tgt.reshape(decoded.shape).to(decoded.dtype)
            if self.aux_bev_motion_norm is not None:
                # Same per-component normalization aux_bev_motion uses.
                # Without it vx and speed take 98.7% of the L1 and yaw_rate
                # 0.1%, so turning would be effectively unsupervised here too.
                scale = torch.as_tensor(self.aux_bev_motion_norm,
                                        device=decoded.device,
                                        dtype=decoded.dtype)
                ego_status_decode_loss = self.ego_status_decode_weight * (
                    (decoded - tgt).abs() / scale).mean()
            else:
                ego_status_decode_loss = self.ego_status_decode_weight * (
                    F.l1_loss(decoded, tgt))

        if self.aux_bev_motion_feedback and bev_pred is not None:
            # bev_pred is aux_bev_motion_head's own vision-derived estimate
            # (never the real ego_lcf_target -- see its computation above),
            # concatenated on so ego_fut_decoder actually gets to use it as
            # planning-relevant motion context rather than only training the
            # head via a loss. Compliance basis: forward()'s
            # aux_bev_motion_feedback constructor comment.
            ego_feats = torch.cat(
                [ego_feats, bev_pred.unsqueeze(1).to(ego_feats.dtype)],
                dim=-1
            )

        if (self.target_point_shortcut
                and self.target_point_shortcut_mode in ('residual', 'both')):
            # target_point_encoder is called unconditionally (not gated on
            # any dropout) so the graph is identical across DDP ranks/steps
            # -- see git 3c084f4 for the desync crash this avoids when a
            # module call is conditioned on a per-rank random draw.
            goal_residual = self.target_point_encoder(
                goal_xy.to(device=ego_feats.device, dtype=ego_feats.dtype)
            ).unsqueeze(1)
            ego_feats = ego_feats + goal_residual

        # Auxiliary ego-motion regression (train-only). Computed from
        # ego_feats BEFORE any PRISM injection, so the target it has to
        # explain is purely the vision-derived context -- and consumed only
        # by the loss below, never fed back into ego_feats. See the
        # aux_ego_motion constructor comment for the compliance argument.
        aux_ego_motion_loss = None
        if self.aux_ego_motion and self.training and ego_lcf_target is not None:
            aux_pred = self.aux_ego_motion_head(ego_feats)  # [B, 1, K]
            aux_gt = ego_lcf_target.squeeze(1)[..., self.aux_ego_motion_idx]
            aux_gt = aux_gt.reshape(aux_pred.shape).to(aux_pred.dtype)
            aux_ego_motion_loss = self.aux_ego_motion_weight * F.l1_loss(
                aux_pred, aux_gt)

        # Auxiliary 5s trajectory regression (train-only), same placement and
        # same rules as aux_ego_motion above: reads ego_feats, writes only a
        # loss. See the aux_long_horizon constructor comment.
        aux_long_horizon_loss = None
        if (self.aux_long_horizon and self.training
                and ego_long_fut_trajs is not None
                and ego_long_fut_valid_flag is not None
                and ego_fut_cmd is not None):
            batch_size = ego_feats.shape[0]
            long_pred = self.aux_long_horizon_head(ego_feats).reshape(
                batch_size, self.ego_fut_mode, self.prism_long_fut_ts, 2)
            long_gt = ego_long_fut_trajs.reshape(
                batch_size, 1, self.prism_long_fut_ts, 2).to(long_pred.dtype)
            if self.aux_long_horizon_residual and ego_lcf_target is not None:
                # Absolute positions, then subtract what constant
                # acceleration from the current state already predicts.
                # ego_long_fut_trajs are per-step deltas, so cumsum first --
                # verified against the data: the 10th cumulative step equals
                # gt_ego_target_point to 0.0000m.
                long_gt = long_gt.cumsum(dim=-2)
                lcf = ego_lcf_target.reshape(batch_size, 1, -1).to(
                    long_pred.dtype)
                vel = lcf[..., 0:2].reshape(batch_size, 1, 1, 2)
                acc = lcf[..., 2:4].reshape(batch_size, 1, 1, 2)
                t = torch.arange(
                    1, self.prism_long_fut_ts + 1, device=long_pred.device,
                    dtype=long_pred.dtype).reshape(1, 1, -1, 1) * FUT_TS_INTERVAL_S
                long_gt = long_gt - (vel * t + 0.5 * acc * t * t)
                # The head now predicts cumulative residuals, so its output
                # is compared in the same space.
                long_pred = long_pred.cumsum(dim=-2)
            # [B, mode] one-hot -> supervise only the mode the command names,
            # exactly as loss_planning does for the 3s output.
            cmd = ego_fut_cmd.reshape(batch_size, -1).to(long_pred.dtype)
            valid = ego_long_fut_valid_flag.reshape(batch_size, 1).to(
                long_pred.dtype)
            weight = (cmd * valid)[:, :, None, None].expand_as(long_pred)
            denom = weight.sum().clamp(min=1.0)
            aux_long_horizon_loss = self.aux_long_horizon_weight * (
                (long_pred - long_gt).abs() * weight).sum() / denom

        # PRISM-style privileged latent supervision (train-only signal,
        # zero extra inference cost -- see prism_latent_supervision's
        # constructor comment). Injects a small correction into ego_feats
        # before the planning decoder sees it.
        prism_kl_loss = None
        if self.prism_latent_supervision:
            prior_out = self.prism_prior_net(ego_feats)  # [B, 1, 2*latent_dim]
            prior_mu, prior_logvar = prior_out.chunk(2, dim=-1)
            # Unclamped logvar -> exp() can overflow fp16's ~65504 max
            # within a few hundred steps of an unconstrained linear output,
            # producing inf/nan gradients that grad_clip then spreads to
            # every other parameter (see clip_grad_norm_: one nan input
            # nans the whole clipped batch). +-10 keeps exp() in
            # [4.5e-5, 22026], safely inside fp16 range with headroom for
            # the KL formula's further multiply/divide.
            prior_logvar = prior_logvar.clamp(min=-10.0, max=10.0)

            # VADLAW's obtain_history_prediction() reruns this head on every
            # history frame too (world-model objective), never passing
            # ego_long_fut_trajs -- only the CURRENT frame's own call does.
            # Fall back to the prior mean (no KL term) for those calls
            # instead of asserting, exactly like eval: PRISM only ever
            # supervises the frame whose trajectory is actually the loss
            # target, not history frames used purely for their BEV features.
            if self.training and ego_long_fut_trajs is not None \
                    and ego_long_fut_valid_flag is not None:
                # Posterior sees the privileged 0-5s GT future -- never
                # available at inference, hence train-only. Optionally also
                # sees current ego_lcf status (prism_posterior_lcf_idx):
                # same train-only guarantee, concatenated rather than
                # summed so it can't be confused with the future-trajectory
                # channels it's alongside.
                long_fut_flat = ego_long_fut_trajs.reshape(
                    ego_long_fut_trajs.shape[0], 1, -1)  # [B, 1, T*2]
                if self.prism_posterior_lcf_idx is not None \
                        and ego_lcf_target is not None:
                    lcf_in = ego_lcf_target.squeeze(1)[
                        ..., self.prism_posterior_lcf_idx]
                    lcf_in = lcf_in.reshape(
                        long_fut_flat.shape[0], 1, -1
                    ).to(long_fut_flat.dtype)
                    long_fut_flat = torch.cat([long_fut_flat, lcf_in], dim=-1)
                post_out = self.prism_posterior_net(long_fut_flat)
                post_mu, post_logvar = post_out.chunk(2, dim=-1)
                post_logvar = post_logvar.clamp(min=-10.0, max=10.0)

                # Reparameterize, sampling S times from the POSTERIOR during training.
                # The S decoded trajectory sets are averaged post-hoc rather than
                # averaging the reconstruction loss, which keeps ego_fut_preds a single
                # tensor downstream.
                std = torch.exp(0.5 * post_logvar)
                z = post_mu + torch.randn_like(std) * std
                if self.prism_num_samples > 1:
                    z = [z] + [
                        post_mu + torch.randn_like(std) * std
                        for _ in range(self.prism_num_samples - 1)
                    ]

                # Closed-form KL(q(z|x,y) || p(z|x)) between two diagonal
                # Gaussians.
                kl = 0.5 * (
                    (prior_logvar - post_logvar)
                    + (post_logvar.exp() + (post_mu - prior_mu).pow(2)) / prior_logvar.exp().clamp(min=1e-6)
                    - 1.0
                )
                kl = kl.sum(dim=-1)  # [B, 1]

                # Frames near a scene's end lack a full 0-5s window --
                # long_fut_valid_flag is all-or-nothing (see converter),
                # so masking at this granularity is exact, not approximate.
                valid = ego_long_fut_valid_flag.reshape(-1, 1).float()  # [B, 1]
                denom = valid.sum().clamp(min=1.0)
                prism_kl_loss = self.prism_kl_weight * (kl * valid).sum() / denom
            else:
                # Eval, or a training-time history-frame call with no
                # privileged data available: use the PRIOR MEAN, not a
                # stochastic draw. For eval this is a deliberate deviation
                # from the paper (which samples even at inference) -- a
                # competition submission must be reproducible run-to-run on
                # identical input, and prism_z_proj is zero-init + KL-
                # regularized toward a near-zero correction anyway, so this
                # costs negligible accuracy for guaranteed determinism.
                z = prior_mu

            z_samples = z if isinstance(z, list) else [z]

        # Ego prediction. z_samples has >1 entry only when PRISM is on,
        # training, and prism_num_samples > 1 (S=2 in the paper's terms);
        # otherwise this loop body runs exactly once, identical to the
        # pre-multi-sample code path.
        goal_cls_loss = None
        goal_off_loss = None
        goal_follow_loss = None
        goal_sel = None
        goal_sel_traj = None
        cell_outs = None
        if self.cell_planner:
            bsz = ego_feats.shape[0]
            # --- cell planner: generation (no target point) ---
            cell_trajs, cell_u_turn, cell_stop = self.cell_generate(
                ego_feats.reshape(bsz, -1))
            cell_outs = (cell_trajs, cell_u_turn, cell_stop)
            # --- cell planner: selection ---
            if self.training:
                # Training only: the target point picks which generated
                # trajectory the loss supervises. At inference the caller
                # selects with the same function (cell_planner_utils).
                if ego_target_point is None or ego_fut_cmd is None:
                    raise ValueError('cell_planner training needs ego_target_point '
                                     'and ego_fut_cmd for every planned frame')
                selected, _ = route_trajectory(
                    cell_trajs, cell_u_turn, cell_stop, ego_target_point,
                    ego_fut_cmd.reshape(bsz, -1).argmax(dim=-1),
                    self.cell_edges(), self.cell_stop_tp_thresh)
            else:
                # Placeholder only; callers must select (they refuse otherwise).
                selected = cell_stop
            outputs_ego_trajs = selected[:, None].expand(
                bsz, self.ego_fut_mode, self.fut_ts, 2).contiguous()
        elif self.goal_pred:
            bsz = ego_feats.shape[0]
            m, k = self.ego_fut_mode, self.goal_k
            feats = ego_feats.reshape(bsz, -1)
            if getattr(self, '_diag_capture', None) is not None:
                self._diag_capture.append(feats)   # tools/diag_goal_grad_conflict.py
            goal_cand_trajs = goal_cand_points = None
            _, amask = self._anchors(feats)                        # [M, K]
            neg = torch.finfo(feats.dtype).min
            g = self.goal_head_grad_scale
            goal_in = feats if g == 1.0 else (
                feats.detach() + g * (feats - feats.detach()))
            goal_logits = self.goal_cls_head(goal_in).reshape(bsz, m, k)
            goal_logits = goal_logits.masked_fill(~amask[None], neg)
            goal_off = self.goal_off_head(goal_in).reshape(bsz, m, k, 2)
            goal_all = self._goal_points(goal_off, feats)          # [B, M, K, 2]
            sel = goal_logits.argmax(dim=-1)                       # [B, M]
            goal_sel = goal_all.gather(
                2, sel[:, :, None, None].expand(bsz, m, 1, 2)).squeeze(2)
            traj_long = self._decode_to_goal(feats, goal_sel)      # [B, M, Tl, 2]
            outputs_ego_trajs = traj_long[:, :, :self.fut_ts]
            if self.goal_expose_candidates and not self.training:
                # One trajectory per anchor, each conditioned on that anchor's
                # own predicted goal: K extra decoder passes, no target point.
                goal_cand_trajs = torch.stack(
                    [self._decode_to_goal(feats, goal_all[:, :, c])[:, :, :self.fut_ts]
                     for c in range(k)], dim=2)                    # [B, M, K, T, 2]
                goal_cand_points = goal_all
            # self.training is the compliance gate: the target point builds
            # label targets here and nowhere else, and never at inference.
            if self.training and ego_target_point is not None:
                if ego_fut_cmd is None:
                    raise ValueError('goal_pred training needs ego_fut_cmd, got None')
                ar = torch.arange(bsz, device=feats.device)
                cmd = ego_fut_cmd.reshape(bsz, -1).argmax(dim=-1)
                gt_cell, gt_off = self._goal_label_from_target(
                    ego_target_point, bsz, feats.dtype, feats.device, cmd)
                goal_cls_loss = self.goal_cls_weight * F.cross_entropy(
                    goal_logits[ar, cmd].float(), gt_cell)
                goal_off_loss = self.goal_off_weight * F.smooth_l1_loss(
                    goal_off[ar, cmd, gt_cell].float(), gt_off.float())

                # Supervise the candidate the target point WOULD select --
                # the trajectory inference actually submits under
                # --select-goal-by-tp. Without this only the argmax candidate
                # ever sees the planning loss and the selected one is trained
                # by nothing. The target point picks WHICH candidate is
                # supervised, the role the command already plays for modes;
                # the goal fed to the decoder stays the model's own. The GT
                # trajectory lives in loss(), so the picked trajectory is
                # handed up and scored there with the same mask and weights.
                if self.goal_select_weight > 0:
                    tp_xy = ego_target_point.to(
                        device=feats.device, dtype=feats.dtype
                    ).reshape(bsz, -1)[:, :2]
                    with torch.no_grad():
                        pick = nearest_valid_goal(goal_all[ar, cmd], tp_xy, amask[cmd])
                        # Other command modes have different candidate counts.
                        # Only the commanded mode receives TP-selection loss;
                        # never gather that index from another mode's padding.
                        selected_per_mode = sel.clone()
                        selected_per_mode[ar, cmd] = pick
                    goal_sel_traj = self._decode_to_goal(
                        feats, goal_all.gather(
                            2, selected_per_mode[:, :, None, None].expand(bsz, m, 1, 2)
                        ).squeeze(2))[:, :, :self.fut_ts]          # [B, M, T, 2]

                # Goal following, with goals the network itself proposes: one
                # anchor per sample drawn from its own (detached) distribution,
                # so the alternatives are plausible for this scene.
                if self.goal_follow_weight > 0:
                    with torch.no_grad():
                        probs = goal_logits.float().softmax(dim=-1)
                        alt = torch.multinomial(
                            probs.reshape(-1, k), 1).reshape(bsz, m)
                    alt_goal = goal_all.gather(
                        2, alt[:, :, None, None].expand(bsz, m, 1, 2)
                    ).squeeze(2).detach()
                    alt_traj = self._decode_to_goal(feats, alt_goal)
                    end = alt_traj[ar, cmd].cumsum(dim=-2)[:, -1]  # [B, 2]
                    scale = self._goal_scale(end)
                    goal_follow_loss = self.goal_follow_weight * F.smooth_l1_loss(
                        (end / scale).float(),
                        (alt_goal[ar, cmd] / scale).float())
        elif self.prism_latent_supervision:
            traj_samples = []
            for zi in z_samples:
                feats_i = ego_feats + self.prism_z_proj(zi)
                traj_i = self.ego_fut_decoder(feats_i).reshape(
                    feats_i.shape[0], self.ego_fut_mode, self.fut_ts, 2)
                # _debug_disable_bev_refine: eval-only escape hatch (never
                # set during training) to compare coarse vs BEV-refined
                # output on a trained checkpoint without needing a
                # separate no-refine checkpoint. Unset (getattr default
                # False) is a no-op.
                if self.bev_residual_refine and not getattr(
                        self, '_debug_disable_bev_refine', False):
                    traj_i = self.refine_ego_trajs_with_bev(traj_i, bev_embed)
                traj_samples.append(traj_i)
            # Monte-Carlo average over the S samples -- see the S=2 comment
            # above for how this differs from the paper's loss-averaging.
            outputs_ego_trajs = torch.stack(traj_samples, dim=0).mean(dim=0)
        else:
            outputs_ego_trajs = self.ego_fut_decoder(ego_feats).reshape(
                ego_feats.shape[0], self.ego_fut_mode, self.fut_ts, 2)
            if self.bev_residual_refine and not getattr(
                    self, '_debug_disable_bev_refine', False):
                outputs_ego_trajs = self.refine_ego_trajs_with_bev(
                    outputs_ego_trajs, bev_embed)

        # Privileged expert (train-only). Reads the SAME vision-derived
        # ego_feats the deployed decoder gets, plus the real ego status,
        # and goes through the same refinement -- so the only difference
        # between its trajectory and the deployed one is knowing the ego's
        # own motion. That is exactly the gap distillation should close.
        privileged_ego_fut_preds = None
        if (self.privileged_distill and self.training
                and ego_lcf_target is not None):
            priv_lcf = ego_lcf_target.squeeze(1)[
                ..., self.privileged_distill_idx]
            priv_lcf = priv_lcf.reshape(
                ego_feats.shape[0], 1, -1).to(ego_feats.dtype)
            privileged_ego_fut_preds = self.privileged_head(
                torch.cat([ego_feats, priv_lcf], dim=-1))
            privileged_ego_fut_preds = privileged_ego_fut_preds.reshape(
                privileged_ego_fut_preds.shape[0], self.ego_fut_mode,
                self.fut_ts, 2)
            if self.bev_residual_refine and not getattr(
                    self, '_debug_disable_bev_refine', False):
                privileged_ego_fut_preds = self.refine_ego_trajs_with_bev(
                    privileged_ego_fut_preds, bev_embed)

        outs = {
            'bev_embed': bev_embed,
            'all_cls_scores': outputs_classes,
            'all_bbox_preds': outputs_coords,
            'all_traj_preds': outputs_trajs.repeat(outputs_coords.shape[0], 1, 1, 1, 1),
            'all_traj_cls_scores': outputs_trajs_classes.repeat(outputs_coords.shape[0], 1, 1, 1),
            'map_all_cls_scores': map_outputs_classes,
            'map_all_bbox_preds': map_outputs_coords,
            'map_all_pts_preds': map_outputs_pts_coords,
            'enc_cls_scores': None,
            'enc_bbox_preds': None,
            'map_enc_cls_scores': None,
            'map_enc_bbox_preds': None,
            'map_enc_pts_preds': None,
            'ego_fut_preds': outputs_ego_trajs,
            # Final pre-decoder embedding (post target-point/feedback
            # branches, pre ego_fut_decoder). Exposed unconditionally --
            # harmless for configs that don't read it -- so a feature-level
            # distillation loss can compare it against a frozen teacher's
            # own ego_feats without threading a new flag through forward().
            'ego_feats': ego_feats,
            # ego_fut_decoder's first Linear+ReLU on ego_feats: the post-fusion
            # planning representation, and the distillation target.
            #
            # ego_feats itself is the wrong target -- an ego_lcf-ON teacher keeps ego
            # status in trailing columns that never touched the agent/map attention.
            # One Linear later those columns are mixed into every hidden unit, so a
            # student without them must reconstruct their contribution from vision.
            #
            # Recomputed rather than captured above so the target is deterministic
            # (the forward pass may add PRISM's sampled latent).
            'ego_plan_hidden': self.ego_fut_decoder[:2](ego_feats),
            # Scheme-A distillation targets: the two halves of ego_feats BEFORE they
            # are concatenated, so scene and ego status are aligned by separate
            # losses. Teacher and student share this layout by construction, which is
            # what the fused ego_plan_hidden above gives up in exchange for not
            # needing the student to have a status slot at all.
            'ego_scene_feats': ego_scene_feats,
            'ego_status_feats': self._distill_status(ego_status_est),
        }
        if self.prism_latent_supervision and self.training:
            outs['prism_kl_loss'] = prism_kl_loss
        if aux_ego_motion_loss is not None:
            outs['aux_ego_motion_loss'] = aux_ego_motion_loss
        if aux_long_horizon_loss is not None:
            outs['aux_long_horizon_loss'] = aux_long_horizon_loss
        if aux_bev_motion_loss is not None:
            outs['aux_bev_motion_loss'] = aux_bev_motion_loss
        if aux_bev_future_motion_loss is not None:
            outs['aux_bev_future_motion_loss'] = aux_bev_future_motion_loss
        if ego_status_decode_loss is not None:
            outs['ego_status_decode_loss'] = ego_status_decode_loss
        if goal_cls_loss is not None:
            outs['goal_cls_loss'] = goal_cls_loss
            outs['goal_off_loss'] = goal_off_loss
            outs['goal_follow_loss'] = goal_follow_loss
            if goal_sel_traj is not None:
                outs['goal_sel_fut_preds'] = goal_sel_traj
        if cell_outs is not None and not self.training:
            outs['cell_trajs'] = cell_outs[0]
            outs['cell_u_turn'] = cell_outs[1]
            outs['cell_stop'] = cell_outs[2]
        if goal_sel is not None and not self.training:
            # The network's own predicted 5s goal per mode [B, M, 2], metres.
            # For diagnostics (goal accuracy vs the label); nothing reads it.
            outs['goal_pred'] = goal_sel
            if self.goal_expose_candidates:
                outs['goal_cand_trajs'] = goal_cand_trajs
                outs['goal_cand_points'] = goal_cand_points
                outs['goal_cand_mask'] = amask[None].expand(bsz, -1, -1)
        if bev_pred is not None and not self.training:
            # Vision-derived ego state [len(aux_bev_motion_idx)] in raw units
            # (the L1 divides the error by aux_bev_motion_norm; the prediction
            # itself is never normalized). Exposed so post-processing can
            # choose among the model's own trajectories from the model's own
            # estimate -- see etri_test_submit.py's STOP selection, which used
            # to read the ground-truth target point instead.
            outs['ego_state_pred'] = bev_pred
        if privileged_ego_fut_preds is not None:
            # Threaded to loss() rather than reduced here, because the GT,
            # its validity mask and the per-timestep metric weighting only
            # exist there -- both the expert's own GT loss and the
            # distillation term reuse that exact weighting so all three
            # planning losses stay on one scale.
            outs['privileged_ego_fut_preds'] = privileged_ego_fut_preds

        return outs

    def map_transform_box(self, pts, y_first=False):
        """
        Converting the points set into bounding box.

        Args:
            pts: the input points sets (fields), each points
                set (fields) is represented as 2n scalar.
            y_first: if y_fisrt=True, the point set is represented as
                [y1, x1, y2, x2 ... yn, xn], otherwise the point set is
                represented as [x1, y1, x2, y2 ... xn, yn].
        Returns:
            The bbox [cx, cy, w, h] transformed from points.
        """
        pts_reshape = pts.view(pts.shape[0], self.map_num_vec,
                                self.map_num_pts_per_vec, self.map_code_size)
        pts_y = pts_reshape[:, :, :, 0] if y_first else pts_reshape[:, :, :, 1]
        pts_x = pts_reshape[:, :, :, 1] if y_first else pts_reshape[:, :, :, 0]
        if self.map_transform_method == 'minmax':
            # import pdb;pdb.set_trace()

            xmin = pts_x.min(dim=2, keepdim=True)[0]
            xmax = pts_x.max(dim=2, keepdim=True)[0]
            ymin = pts_y.min(dim=2, keepdim=True)[0]
            ymax = pts_y.max(dim=2, keepdim=True)[0]
            bbox = torch.cat([xmin, ymin, xmax, ymax], dim=2)
            bbox = bbox_xyxy_to_cxcywh(bbox)
        else:
            raise NotImplementedError
        return bbox, pts_reshape

    def _get_target_single(self,
                           cls_score,
                           bbox_pred,
                           gt_labels,
                           gt_bboxes,
                           gt_attr_labels,
                           gt_bboxes_ignore=None):
        """"Compute regression and classification targets for one image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            cls_score (Tensor): Box score logits from a single decoder layer
                for one image. Shape [num_query, cls_out_channels].
            bbox_pred (Tensor): Sigmoid outputs from a single decoder layer
                for one image, with normalized coordinate (cx, cy, w, h) and
                shape [num_query, 10].
            gt_bboxes (Tensor): Ground truth bboxes for one image with
                shape (num_gts, 9) in [x,y,z,w,l,h,yaw,vx,vy] format.
            gt_labels (Tensor): Ground truth class indices for one image
                with shape (num_gts, ).
            gt_bboxes_ignore (Tensor, optional): Bounding boxes
                which can be ignored. Default None.
        Returns:
            tuple[Tensor]: a tuple containing the following for one image.
                - labels (Tensor): Labels of each image.
                - label_weights (Tensor]): Label weights of each image.
                - bbox_targets (Tensor): BBox targets of each image.
                - bbox_weights (Tensor): BBox weights of each image.
                - pos_inds (Tensor): Sampled positive indices for each image.
                - neg_inds (Tensor): Sampled negative indices for each image.
        """

        num_bboxes = bbox_pred.size(0)
        # assigner and sampler
        gt_fut_trajs = gt_attr_labels[:, :self.fut_ts*2]
        gt_fut_masks = gt_attr_labels[:, self.fut_ts*2:self.fut_ts*3]
        gt_bbox_c = gt_bboxes.shape[-1]
        num_gt_bbox, gt_traj_c = gt_fut_trajs.shape

        assign_result = self.assigner.assign(bbox_pred, cls_score, gt_bboxes,
                                             gt_labels, gt_bboxes_ignore)

        sampling_result = self.sampler.sample(assign_result, bbox_pred,
                                              gt_bboxes)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        # label targets
        labels = gt_bboxes.new_full((num_bboxes,),
                                    self.num_classes,
                                    dtype=torch.long)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_bboxes)

        # bbox targets
        bbox_targets = torch.zeros_like(bbox_pred)[..., :gt_bbox_c]
        bbox_weights = torch.zeros_like(bbox_pred)
        bbox_weights[pos_inds] = 1.0

        # trajs targets
        traj_targets = torch.zeros((num_bboxes, gt_traj_c), dtype=torch.float32, device=bbox_pred.device)
        traj_weights = torch.zeros_like(traj_targets)
        traj_targets[pos_inds] = gt_fut_trajs[sampling_result.pos_assigned_gt_inds]
        traj_weights[pos_inds] = 1.0

        # Filter out invalid fut trajs
        traj_masks = torch.zeros_like(traj_targets)  # [num_bboxes, fut_ts*2]
        gt_fut_masks = gt_fut_masks.unsqueeze(-1).repeat(1, 1, 2).view(num_gt_bbox, -1)  # [num_gt_bbox, fut_ts*2]
        traj_masks[pos_inds] = gt_fut_masks[sampling_result.pos_assigned_gt_inds]
        traj_weights = traj_weights * traj_masks

        # Extra future timestamp mask for controlling pred horizon
        fut_ts_mask = torch.zeros((num_bboxes, self.fut_ts, 2),
                                   dtype=torch.float32, device=bbox_pred.device)
        fut_ts_mask[:, :self.valid_fut_ts, :] = 1.0
        fut_ts_mask = fut_ts_mask.view(num_bboxes, -1)
        traj_weights = traj_weights * fut_ts_mask

        # DETR
        bbox_targets[pos_inds] = sampling_result.pos_gt_bboxes

        return (
            labels, label_weights, bbox_targets, bbox_weights, traj_targets,
            traj_weights, traj_masks.view(-1, self.fut_ts, 2)[..., 0],
            pos_inds, neg_inds
        )

    def _map_get_target_single(self,
                           cls_score,
                           bbox_pred,
                           pts_pred,
                           gt_labels,
                           gt_bboxes,
                           gt_shifts_pts,
                           gt_bboxes_ignore=None):
        """"Compute regression and classification targets for one image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            cls_score (Tensor): Box score logits from a single decoder layer
                for one image. Shape [num_query, cls_out_channels].
            bbox_pred (Tensor): Sigmoid outputs from a single decoder layer
                for one image, with normalized coordinate (cx, cy, w, h) and
                shape [num_query, 4].
            gt_bboxes (Tensor): Ground truth bboxes for one image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (Tensor): Ground truth class indices for one image
                with shape (num_gts, ).
            gt_bboxes_ignore (Tensor, optional): Bounding boxes
                which can be ignored. Default None.
        Returns:
            tuple[Tensor]: a tuple containing the following for one image.
                - labels (Tensor): Labels of each image.
                - label_weights (Tensor]): Label weights of each image.
                - bbox_targets (Tensor): BBox targets of each image.
                - bbox_weights (Tensor): BBox weights of each image.
                - pos_inds (Tensor): Sampled positive indices for each image.
                - neg_inds (Tensor): Sampled negative indices for each image.
        """
        num_bboxes = bbox_pred.size(0)
        # assigner and sampler
        gt_c = gt_bboxes.shape[-1]
        assign_result, order_index = self.map_assigner.assign(bbox_pred, cls_score, pts_pred,
                                             gt_bboxes, gt_labels, gt_shifts_pts,
                                             gt_bboxes_ignore)

        sampling_result = self.map_sampler.sample(assign_result, bbox_pred,
                                              gt_bboxes)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        # label targets
        labels = gt_bboxes.new_full((num_bboxes,),
                                    self.map_num_classes,
                                    dtype=torch.long)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_bboxes)
        # bbox targets
        bbox_targets = torch.zeros_like(bbox_pred)[..., :gt_c]
        bbox_weights = torch.zeros_like(bbox_pred)
        bbox_weights[pos_inds] = 1.0
        # pts targets
        if order_index is None:
            assigned_shift = gt_labels[sampling_result.pos_assigned_gt_inds]
        else:
            assigned_shift = order_index[sampling_result.pos_inds, sampling_result.pos_assigned_gt_inds]
        pts_targets = pts_pred.new_zeros((pts_pred.size(0),
                        pts_pred.size(1), pts_pred.size(2)))
        pts_weights = torch.zeros_like(pts_targets)
        pts_weights[pos_inds] = 1.0
        # DETR
        bbox_targets[pos_inds] = sampling_result.pos_gt_bboxes
        pts_targets[pos_inds] = gt_shifts_pts[sampling_result.pos_assigned_gt_inds,assigned_shift,:,:]
        return (labels, label_weights, bbox_targets, bbox_weights,
                pts_targets, pts_weights,
                pos_inds, neg_inds)

    def get_targets(self,
                    cls_scores_list,
                    bbox_preds_list,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_attr_labels_list,
                    gt_bboxes_ignore_list=None):
        """"Compute regression and classification targets for a batch image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            cls_scores_list (list[Tensor]): Box score logits from a single
                decoder layer for each image with shape [num_query,
                cls_out_channels].
            bbox_preds_list (list[Tensor]): Sigmoid outputs from a single
                decoder layer for each image, with normalized coordinate
                (cx, cy, w, h) and shape [num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            tuple: a tuple containing the following targets.
                - labels_list (list[Tensor]): Labels for all images.
                - label_weights_list (list[Tensor]): Label weights for all \
                    images.
                - bbox_targets_list (list[Tensor]): BBox targets for all \
                    images.
                - bbox_weights_list (list[Tensor]): BBox weights for all \
                    images.
                - num_total_pos (int): Number of positive samples in all \
                    images.
                - num_total_neg (int): Number of negative samples in all \
                    images.
        """
        assert gt_bboxes_ignore_list is None, \
            'Only supports for gt_bboxes_ignore setting to None.'
        num_imgs = len(cls_scores_list)
        gt_bboxes_ignore_list = [
            gt_bboxes_ignore_list for _ in range(num_imgs)
        ]

        (labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, traj_targets_list, traj_weights_list,
         gt_fut_masks_list, pos_inds_list, neg_inds_list) = multi_apply(
            self._get_target_single, cls_scores_list, bbox_preds_list,
            gt_labels_list, gt_bboxes_list, gt_attr_labels_list, gt_bboxes_ignore_list
         )
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
                traj_targets_list, traj_weights_list, gt_fut_masks_list, num_total_pos, num_total_neg)

    def map_get_targets(self,
                    cls_scores_list,
                    bbox_preds_list,
                    pts_preds_list,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_shifts_pts_list,
                    gt_bboxes_ignore_list=None):
        """"Compute regression and classification targets for a batch image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            cls_scores_list (list[Tensor]): Box score logits from a single
                decoder layer for each image with shape [num_query,
                cls_out_channels].
            bbox_preds_list (list[Tensor]): Sigmoid outputs from a single
                decoder layer for each image, with normalized coordinate
                (cx, cy, w, h) and shape [num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            tuple: a tuple containing the following targets.
                - labels_list (list[Tensor]): Labels for all images.
                - label_weights_list (list[Tensor]): Label weights for all \
                    images.
                - bbox_targets_list (list[Tensor]): BBox targets for all \
                    images.
                - bbox_weights_list (list[Tensor]): BBox weights for all \
                    images.
                - num_total_pos (int): Number of positive samples in all \
                    images.
                - num_total_neg (int): Number of negative samples in all \
                    images.
        """
        assert gt_bboxes_ignore_list is None, \
            'Only supports for gt_bboxes_ignore setting to None.'
        num_imgs = len(cls_scores_list)
        gt_bboxes_ignore_list = [
            gt_bboxes_ignore_list for _ in range(num_imgs)
        ]

        (labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, pts_targets_list, pts_weights_list,
         pos_inds_list, neg_inds_list) = multi_apply(
            self._map_get_target_single, cls_scores_list, bbox_preds_list,pts_preds_list,
            gt_labels_list, gt_bboxes_list, gt_shifts_pts_list, gt_bboxes_ignore_list)
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, bbox_targets_list,
                bbox_weights_list, pts_targets_list, pts_weights_list,
                num_total_pos, num_total_neg)

    def loss_planning(self,
                      ego_fut_preds,
                      ego_fut_gt,
                      ego_fut_masks,
                      ego_fut_cmd,
                      lane_preds,
                      lane_score_preds,
                      agent_preds,
                      agent_fut_preds,
                      agent_score_preds,
                      agent_fut_cls_preds,
                      privileged_fut_preds=None,
                      goal_sel_fut_preds=None):
        """"Loss function for ego vehicle planning.
        Args:
            ego_fut_preds (Tensor): [B, ego_fut_mode, fut_ts, 2]
            ego_fut_gt (Tensor): [B, fut_ts, 2]
            ego_fut_masks (Tensor): [B, fut_ts]
            ego_fut_cmd (Tensor): [B, ego_fut_mode]
            lane_preds (Tensor): [B, num_vec, num_pts, 2]
            lane_score_preds (Tensor): [B, num_vec, 3]
            agent_preds (Tensor): [B, num_agent, 2]
            agent_fut_preds (Tensor): [B, num_agent, fut_mode, fut_ts, 2]
            agent_score_preds (Tensor): [B, num_agent, 10]
            agent_fut_cls_scores (Tensor): [B, num_agent, fut_mode]
        Returns:
            loss_plan_reg (Tensor): planning reg loss.
            loss_plan_bound (Tensor): planning map boundary constraint loss.
            loss_plan_col (Tensor): planning col constraint loss.
            loss_plan_dir (Tensor): planning directional constraint loss.
        """

        ego_fut_gt = ego_fut_gt.unsqueeze(1).repeat(1, self.ego_fut_mode, 1, 1)
        loss_plan_l1_weight = ego_fut_cmd[..., None, None] * ego_fut_masks[:, None, :, None]
        loss_plan_l1_weight = loss_plan_l1_weight.repeat(1, 1, 1, 2)

        # The challenge metric averages three overlapping windows -- 1s
        # (steps 0:fut_ts/3), 2s (0:2*fut_ts/3), 3s (0:fut_ts) -- then
        # averages those three numbers. Early timesteps land in all three
        # windows and late ones in only the last, so L2_avg implicitly
        # weights step i by (1/3) * sum_{k=1..3} [i < k*step] / (k*step),
        # not uniformly. Scale the L1 target weight per timestep to match,
        # normalized so the mean weight stays 1 (keeps loss_plan_reg's
        # overall scale comparable to before, no other loss_weight retune
        # needed).
        step = self.fut_ts // 3
        pos_weights = [
            sum(1.0 / (k * step) for k in (1, 2, 3) if i < k * step) / 3.0
            for i in range(self.fut_ts)
        ]
        if self.plan_reg_cumulative:
            # The cumsum below applies the metric's chain rule exactly, so the
            # position weights are the right ones regardless of the mode.
            # Applying the delta reweighting on top would count it twice.
            ts_weights = pos_weights
        elif self.plan_reg_ts_weight_mode == 'cumulative':
            # pos_weights above is the metric's weight on the POSITION error
            # at step i. But ego_fut_preds are per-step DELTAS, which the
            # metric cumsums before measuring: an error in delta i shifts
            # every position from i onward, not just position i. So a
            # delta's true influence on the metric is the sum of the
            # position weights it displaces. That turns [.31 .31 .14 .14
            # .06 .06] into [1.00 .69 .39 .25 .11 .06] -- same ordering,
            # much steeper, and it says delta 0 matters 1.4x more than
            # delta 1 where the position form calls them equal.
            ts_weights = [
                sum(pos_weights[j] for j in range(i, self.fut_ts))
                for i in range(self.fut_ts)
            ]
        else:
            ts_weights = pos_weights
        ts_weights = [w * self.fut_ts / sum(ts_weights) for w in ts_weights]
        ts_weight = ego_fut_masks.new_tensor(ts_weights)
        loss_plan_l1_weight = loss_plan_l1_weight * ts_weight[None, None, :, None]

        if self.command_class_weights is not None:
            # ego_fut_cmd is one-hot [B, ego_fut_mode], so this dot
            # product picks out each sample's own command's weight -- cheap
            # way to turn a per-class table into a per-sample [B] scalar
            # without an argmax+gather.
            cmd_w = ego_fut_cmd.new_tensor(self.command_class_weights)
            per_sample_weight = (ego_fut_cmd * cmd_w[None, :]).sum(dim=-1)
            loss_plan_l1_weight = (
                loss_plan_l1_weight * per_sample_weight[:, None, None, None])

        # See plan_reg_cumulative in __init__. Only the L1 regression terms
        # below switch representation; loss_plan_bound/_col/_dir keep taking
        # the deltas they cumsum for themselves.
        if self.plan_reg_cumulative:
            reg_preds = ego_fut_preds.cumsum(dim=-2)
            reg_gt = ego_fut_gt.cumsum(dim=-2)
        else:
            reg_preds, reg_gt = ego_fut_preds, ego_fut_gt

        loss_plan_l1 = self.loss_plan_reg(
            reg_preds,
            reg_gt,
            loss_plan_l1_weight
        )

        loss_goal_select = None
        if goal_sel_fut_preds is not None:
            # Same target, mask and per-step weights as loss_plan_reg, so the
            # two are directly comparable: this is the planning loss applied
            # to the candidate the target point selects.
            loss_goal_select = self.goal_select_weight * self.loss_plan_reg(
                (goal_sel_fut_preds.cumsum(dim=-2)
                 if self.plan_reg_cumulative else goal_sel_fut_preds),
                reg_gt,
                loss_plan_l1_weight
            )

        loss_privileged_reg = None
        loss_plan_distill = None
        if privileged_fut_preds is not None:
            # The expert learns the task with ego status available...
            loss_privileged_reg = self.loss_plan_reg(
                (privileged_fut_preds.cumsum(dim=-2)
                 if self.plan_reg_cumulative else privileged_fut_preds),
                reg_gt,
                loss_plan_l1_weight
            )
            # ...and the deployed decoder is pulled toward what the expert
            # produces. detach() is the compliance-critical part: without
            # it, gradient would run from the deployed decoder's loss back
            # through the expert and into its ego_lcf input, which is the
            # path the ruling forbids.
            loss_plan_distill = self.privileged_distill_weight * (
                self.loss_plan_reg(
                    reg_preds,
                    (privileged_fut_preds.detach().cumsum(dim=-2)
                     if self.plan_reg_cumulative
                     else privileged_fut_preds.detach()),
                    loss_plan_l1_weight
                ))

        loss_plan_bound = self.loss_plan_bound(
            ego_fut_preds[ego_fut_cmd==1],
            lane_preds,
            lane_score_preds,
            weight=ego_fut_masks
        )

        loss_plan_col = self.loss_plan_col(
            ego_fut_preds[ego_fut_cmd==1],
            agent_preds,
            agent_fut_preds,
            agent_score_preds,
            agent_fut_cls_preds,
            weight=ego_fut_masks[:, :, None].repeat(1, 1, 2)
        )

        loss_plan_dir = self.loss_plan_dir(
            ego_fut_preds[ego_fut_cmd==1],
            lane_preds,
            lane_score_preds,
            weight=ego_fut_masks
        )

        if digit_version(TORCH_VERSION) >= digit_version('1.8'):
            loss_plan_l1 = torch.nan_to_num(loss_plan_l1)
            loss_plan_bound = torch.nan_to_num(loss_plan_bound)
            loss_plan_col = torch.nan_to_num(loss_plan_col)
            loss_plan_dir = torch.nan_to_num(loss_plan_dir)
        
        loss_plan_dict = dict()
        loss_plan_dict['loss_plan_reg'] = loss_plan_l1
        if loss_goal_select is not None:
            loss_plan_dict['loss_goal_select'] = loss_goal_select
        if loss_privileged_reg is not None:
            loss_plan_dict['loss_privileged_reg'] = loss_privileged_reg
            loss_plan_dict['loss_plan_distill'] = loss_plan_distill
        loss_plan_dict['loss_plan_bound'] = loss_plan_bound
        loss_plan_dict['loss_plan_col'] = loss_plan_col
        loss_plan_dict['loss_plan_dir'] = loss_plan_dir

        return loss_plan_dict
    
    def loss_single(self,
                    cls_scores,
                    bbox_preds,
                    traj_preds,
                    traj_cls_preds,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_attr_labels_list,
                    gt_bboxes_ignore_list=None):
        """"Loss function for outputs from a single decoder layer of a single
        feature level.
        Args:
            cls_scores (Tensor): Box score logits from a single decoder layer
                for all images. Shape [bs, num_query, cls_out_channels].
            bbox_preds (Tensor): Sigmoid outputs from a single decoder layer
                for all images, with normalized coordinate (cx, cy, w, h) and
                shape [bs, num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components for outputs from
                a single decoder layer.
        """
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        cls_reg_targets = self.get_targets(cls_scores_list, bbox_preds_list,
                                           gt_bboxes_list, gt_labels_list,
                                           gt_attr_labels_list, gt_bboxes_ignore_list)

        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         traj_targets_list, traj_weights_list, gt_fut_masks_list,
         num_total_pos, num_total_neg) = cls_reg_targets

        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)
        traj_targets = torch.cat(traj_targets_list, 0)
        traj_weights = torch.cat(traj_weights_list, 0)
        gt_fut_masks = torch.cat(gt_fut_masks_list, 0)

        # classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))

        cls_avg_factor = max(cls_avg_factor, 1)
        loss_cls = self.loss_cls(cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes accross all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # regression L1 loss
        bbox_preds = bbox_preds.reshape(-1, bbox_preds.size(-1))
        normalized_bbox_targets = normalize_bbox(bbox_targets, self.pc_range)
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)
        bbox_weights = bbox_weights * self.code_weights
        loss_bbox = self.loss_bbox(
            bbox_preds[isnotnan, :10],
            normalized_bbox_targets[isnotnan, :10],
            bbox_weights[isnotnan, :10],
            avg_factor=num_total_pos)

        # traj regression loss
        best_traj_preds = self.get_best_fut_preds(
            traj_preds.reshape(-1, self.fut_mode, self.fut_ts, 2),
            traj_targets.reshape(-1, self.fut_ts, 2), gt_fut_masks)

        neg_inds = (bbox_weights[:, 0] == 0)
        traj_labels = self.get_traj_cls_target(
            traj_preds.reshape(-1, self.fut_mode, self.fut_ts, 2),
            traj_targets.reshape(-1, self.fut_ts, 2),
            gt_fut_masks, neg_inds)

        loss_traj = self.loss_traj(
            best_traj_preds[isnotnan],
            traj_targets[isnotnan],
            traj_weights[isnotnan],
            avg_factor=num_total_pos)

        if self.use_traj_lr_warmup:
            loss_scale_factor = get_traj_warmup_loss_weight(self.epoch, self.tot_epoch)
            loss_traj = loss_scale_factor * loss_traj

        # traj classification loss
        traj_cls_scores = traj_cls_preds.reshape(-1, self.fut_mode)
        # construct weighted avg_factor to match with the official DETR repo
        traj_cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.traj_bg_cls_weight
        if self.sync_cls_avg_factor:
            traj_cls_avg_factor = reduce_mean(
                traj_cls_scores.new_tensor([traj_cls_avg_factor]))

        traj_cls_avg_factor = max(traj_cls_avg_factor, 1)
        loss_traj_cls = self.loss_traj_cls(
            traj_cls_scores, traj_labels, label_weights, avg_factor=traj_cls_avg_factor
        )

        if digit_version(TORCH_VERSION) >= digit_version('1.8'):
            loss_cls = torch.nan_to_num(loss_cls)
            loss_bbox = torch.nan_to_num(loss_bbox)
            loss_traj = torch.nan_to_num(loss_traj)
            loss_traj_cls = torch.nan_to_num(loss_traj_cls)

        return loss_cls, loss_bbox, loss_traj, loss_traj_cls

    def get_best_fut_preds(self,
             traj_preds,
             traj_targets,
             gt_fut_masks):
        """"Choose best preds among all modes.
        Args:
            traj_preds (Tensor): MultiModal traj preds with shape (num_box_preds, fut_mode, fut_ts, 2).
            traj_targets (Tensor): Ground truth traj for each pred box with shape (num_box_preds, fut_ts, 2).
            gt_fut_masks (Tensor): Ground truth traj mask with shape (num_box_preds, fut_ts).
            pred_box_centers (Tensor): Pred box centers with shape (num_box_preds, 2).
            gt_box_centers (Tensor): Ground truth box centers with shape (num_box_preds, 2).

        Returns:
            best_traj_preds (Tensor): best traj preds (min displacement error with gt)
                with shape (num_box_preds, fut_ts*2).
        """

        cum_traj_preds = traj_preds.cumsum(dim=-2)
        cum_traj_targets = traj_targets.cumsum(dim=-2)

        # Get min pred mode indices.
        # (num_box_preds, fut_mode, fut_ts)
        dist = torch.linalg.norm(cum_traj_targets[:, None, :, :] - cum_traj_preds, dim=-1)
        dist = dist * gt_fut_masks[:, None, :]
        dist = dist[..., -1]
        dist[torch.isnan(dist)] = dist[torch.isnan(dist)] * 0
        min_mode_idxs = torch.argmin(dist, dim=-1).tolist()
        box_idxs = torch.arange(traj_preds.shape[0]).tolist()
        best_traj_preds = traj_preds[box_idxs, min_mode_idxs, :, :].reshape(-1, self.fut_ts*2)

        return best_traj_preds

    def get_traj_cls_target(self,
             traj_preds,
             traj_targets,
             gt_fut_masks,
             neg_inds):
        """"Get Trajectory mode classification target.
        Args:
            traj_preds (Tensor): MultiModal traj preds with shape (num_box_preds, fut_mode, fut_ts, 2).
            traj_targets (Tensor): Ground truth traj for each pred box with shape (num_box_preds, fut_ts, 2).
            gt_fut_masks (Tensor): Ground truth traj mask with shape (num_box_preds, fut_ts).
            neg_inds (Tensor): Negtive indices with shape (num_box_preds,)

        Returns:
            traj_labels (Tensor): traj cls labels (num_box_preds,).
        """

        cum_traj_preds = traj_preds.cumsum(dim=-2)
        cum_traj_targets = traj_targets.cumsum(dim=-2)

        # Get min pred mode indices.
        # (num_box_preds, fut_mode, fut_ts)
        dist = torch.linalg.norm(cum_traj_targets[:, None, :, :] - cum_traj_preds, dim=-1)
        dist = dist * gt_fut_masks[:, None, :]
        dist = dist[..., -1]
        dist[torch.isnan(dist)] = dist[torch.isnan(dist)] * 0
        traj_labels = torch.argmin(dist, dim=-1)
        traj_labels[neg_inds] = self.fut_mode

        return traj_labels

    def map_loss_single(self,
                    cls_scores,
                    bbox_preds,
                    pts_preds,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_shifts_pts_list,
                    gt_bboxes_ignore_list=None):
        """"Loss function for outputs from a single decoder layer of a single
        feature level.
        Args:
            cls_scores (Tensor): Box score logits from a single decoder layer
                for all images. Shape [bs, num_query, cls_out_channels].
            bbox_preds (Tensor): Sigmoid outputs from a single decoder layer
                for all images, with normalized coordinate (cx, cy, w, h) and
                shape [bs, num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_pts_list (list[Tensor]): Ground truth pts for each image
                with shape (num_gts, fixed_num, 2) in [x,y] format.
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components for outputs from
                a single decoder layer.
        """
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        pts_preds_list = [pts_preds[i] for i in range(num_imgs)]

        cls_reg_targets = self.map_get_targets(cls_scores_list, bbox_preds_list,pts_preds_list,
                                           gt_bboxes_list, gt_labels_list,gt_shifts_pts_list,
                                           gt_bboxes_ignore_list)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         pts_targets_list, pts_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
 
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)
        pts_targets = torch.cat(pts_targets_list, 0)
        pts_weights = torch.cat(pts_weights_list, 0)

        # classification loss
        cls_scores = cls_scores.reshape(-1, self.map_cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.map_bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))

        cls_avg_factor = max(cls_avg_factor, 1)
        loss_cls = self.loss_map_cls(
            cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes accross all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # regression L1 loss
        bbox_preds = bbox_preds.reshape(-1, bbox_preds.size(-1))
        normalized_bbox_targets = normalize_2d_bbox(bbox_targets, self.pc_range)
        # normalized_bbox_targets = bbox_targets
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)
        bbox_weights = bbox_weights * self.map_code_weights

        loss_bbox = self.loss_map_bbox(
            bbox_preds[isnotnan, :4],
            normalized_bbox_targets[isnotnan,:4],
            bbox_weights[isnotnan, :4],
            avg_factor=num_total_pos)

        # regression pts CD loss
        # num_samples, num_order, num_pts, num_coords
        normalized_pts_targets = normalize_2d_pts(pts_targets, self.pc_range)

        # num_samples, num_pts, num_coords
        pts_preds = pts_preds.reshape(-1, pts_preds.size(-2), pts_preds.size(-1))
        if self.map_num_pts_per_vec != self.map_num_pts_per_gt_vec:
            pts_preds = pts_preds.permute(0,2,1)
            pts_preds = F.interpolate(pts_preds, size=(self.map_num_pts_per_gt_vec), mode='linear',
                                    align_corners=True)
            pts_preds = pts_preds.permute(0,2,1).contiguous()

        loss_pts = self.loss_map_pts(
            pts_preds[isnotnan,:,:],
            normalized_pts_targets[isnotnan,:,:], 
            pts_weights[isnotnan,:,:],
            avg_factor=num_total_pos)

        dir_weights = pts_weights[:, :-self.map_dir_interval,0]
        denormed_pts_preds = denormalize_2d_pts(pts_preds, self.pc_range)
        denormed_pts_preds_dir = denormed_pts_preds[:,self.map_dir_interval:,:] - \
            denormed_pts_preds[:,:-self.map_dir_interval,:]
        pts_targets_dir = pts_targets[:, self.map_dir_interval:,:] - pts_targets[:,:-self.map_dir_interval,:]

        loss_dir = self.loss_map_dir(
            denormed_pts_preds_dir[isnotnan,:,:],
            pts_targets_dir[isnotnan,:,:],
            dir_weights[isnotnan,:],
            avg_factor=num_total_pos)

        bboxes = denormalize_2d_bbox(bbox_preds, self.pc_range)
        # regression IoU loss, defaultly GIoU loss
        loss_iou = self.loss_map_iou(
            bboxes[isnotnan, :4],
            bbox_targets[isnotnan, :4],
            bbox_weights[isnotnan, :4], 
            avg_factor=num_total_pos)

        if digit_version(TORCH_VERSION) >= digit_version('1.8'):
            loss_cls = torch.nan_to_num(loss_cls)
            loss_bbox = torch.nan_to_num(loss_bbox)
            loss_iou = torch.nan_to_num(loss_iou)
            loss_pts = torch.nan_to_num(loss_pts)
            loss_dir = torch.nan_to_num(loss_dir)

        return loss_cls, loss_bbox, loss_iou, loss_pts, loss_dir

    @force_fp32(apply_to=('preds_dicts'))
    def loss(self,
             gt_bboxes_list,
             gt_labels_list,
             map_gt_bboxes_list,
             map_gt_labels_list,
             preds_dicts,
             ego_fut_gt,
             ego_fut_masks,
             ego_fut_cmd,
             gt_attr_labels,
             gt_bboxes_ignore=None,
             map_gt_bboxes_ignore=None,
             img_metas=None):
        """"Loss function.
        Args:

            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            preds_dicts:
                all_cls_scores (Tensor): Classification score of all
                    decoder layers, has shape
                    [nb_dec, bs, num_query, cls_out_channels].
                all_bbox_preds (Tensor): Sigmoid regression
                    outputs of all decode layers. Each is a 4D-tensor with
                    normalized coordinate format (cx, cy, w, h) and shape
                    [nb_dec, bs, num_query, 4].
                enc_cls_scores (Tensor): Classification scores of
                    points on encode feature map , has shape
                    (N, h*w, num_classes). Only be passed when as_two_stage is
                    True, otherwise is None.
                enc_bbox_preds (Tensor): Regression results of each points
                    on the encode feature map, has shape (N, h*w, 4). Only be
                    passed when as_two_stage is True, otherwise is None.
            gt_bboxes_ignore (list[Tensor], optional): Bounding boxes
                which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert gt_bboxes_ignore is None, \
            f'{self.__class__.__name__} only supports ' \
            f'for gt_bboxes_ignore setting to None.'

        map_gt_vecs_list = copy.deepcopy(map_gt_bboxes_list)

        all_cls_scores = preds_dicts['all_cls_scores']
        all_bbox_preds = preds_dicts['all_bbox_preds']
        all_traj_preds = preds_dicts['all_traj_preds']
        all_traj_cls_scores = preds_dicts['all_traj_cls_scores']
        enc_cls_scores = preds_dicts['enc_cls_scores']
        enc_bbox_preds = preds_dicts['enc_bbox_preds']
        map_all_cls_scores = preds_dicts['map_all_cls_scores']
        map_all_bbox_preds = preds_dicts['map_all_bbox_preds']
        map_all_pts_preds = preds_dicts['map_all_pts_preds']
        map_enc_cls_scores = preds_dicts['map_enc_cls_scores']
        map_enc_bbox_preds = preds_dicts['map_enc_bbox_preds']
        map_enc_pts_preds = preds_dicts['map_enc_pts_preds']
        ego_fut_preds = preds_dicts['ego_fut_preds']

        num_dec_layers = len(all_cls_scores)
        device = gt_labels_list[0].device

        gt_bboxes_list = [torch.cat(
            (gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]),
            dim=1).to(device) for gt_bboxes in gt_bboxes_list]

        all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_attr_labels_list = [gt_attr_labels for _ in range(num_dec_layers)]
        all_gt_bboxes_ignore_list = [
            gt_bboxes_ignore for _ in range(num_dec_layers)
        ]

        losses_cls, losses_bbox, loss_traj, loss_traj_cls = multi_apply(
            self.loss_single, all_cls_scores, all_bbox_preds, all_traj_preds,
            all_traj_cls_scores, all_gt_bboxes_list, all_gt_labels_list,
            all_gt_attr_labels_list, all_gt_bboxes_ignore_list)
        

        num_dec_layers = len(map_all_cls_scores)
        device = map_gt_labels_list[0].device

        map_gt_bboxes_list = [
            map_gt_bboxes.bbox.to(device) for map_gt_bboxes in map_gt_vecs_list]
        map_gt_pts_list = [
            map_gt_bboxes.fixed_num_sampled_points.to(device) for map_gt_bboxes in map_gt_vecs_list]
        if self.map_gt_shift_pts_pattern == 'v0':
            map_gt_shifts_pts_list = [
                gt_bboxes.shift_fixed_num_sampled_points.to(device) for gt_bboxes in map_gt_vecs_list]
        elif self.map_gt_shift_pts_pattern == 'v1':
            map_gt_shifts_pts_list = [
                gt_bboxes.shift_fixed_num_sampled_points_v1.to(device) for gt_bboxes in map_gt_vecs_list]
        elif self.map_gt_shift_pts_pattern == 'v2':
            map_gt_shifts_pts_list = [
                gt_bboxes.shift_fixed_num_sampled_points_v2.to(device) for gt_bboxes in map_gt_vecs_list]
        elif self.map_gt_shift_pts_pattern == 'v3':
            map_gt_shifts_pts_list = [
                gt_bboxes.shift_fixed_num_sampled_points_v3.to(device) for gt_bboxes in map_gt_vecs_list]
        elif self.map_gt_shift_pts_pattern == 'v4':
            map_gt_shifts_pts_list = [
                gt_bboxes.shift_fixed_num_sampled_points_v4.to(device) for gt_bboxes in map_gt_vecs_list]
        else:
            raise NotImplementedError
        map_all_gt_bboxes_list = [map_gt_bboxes_list for _ in range(num_dec_layers)]
        map_all_gt_labels_list = [map_gt_labels_list for _ in range(num_dec_layers)]
        map_all_gt_pts_list = [map_gt_pts_list for _ in range(num_dec_layers)]
        map_all_gt_shifts_pts_list = [map_gt_shifts_pts_list for _ in range(num_dec_layers)]
        map_all_gt_bboxes_ignore_list = [
            map_gt_bboxes_ignore for _ in range(num_dec_layers)
        ]

        map_losses_cls, map_losses_bbox, map_losses_iou, \
            map_losses_pts, map_losses_dir = multi_apply(
            self.map_loss_single, map_all_cls_scores, map_all_bbox_preds,
            map_all_pts_preds, map_all_gt_bboxes_list, map_all_gt_labels_list,
            map_all_gt_shifts_pts_list, map_all_gt_bboxes_ignore_list)

        loss_dict = dict()
        # loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_bbox'] = losses_bbox[-1]
        loss_dict['loss_traj'] = loss_traj[-1]
        loss_dict['loss_traj_cls'] = loss_traj_cls[-1]
        # loss from the last decoder layer
        loss_dict['loss_map_cls'] = map_losses_cls[-1]
        loss_dict['loss_map_bbox'] = map_losses_bbox[-1]
        loss_dict['loss_map_iou'] = map_losses_iou[-1]
        loss_dict['loss_map_pts'] = map_losses_pts[-1]
        loss_dict['loss_map_dir'] = map_losses_dir[-1]

        # Planning Loss
        ego_fut_gt = ego_fut_gt.squeeze(1)
        ego_fut_masks = ego_fut_masks.squeeze(1).squeeze(1)
        ego_fut_cmd = ego_fut_cmd.squeeze(1).squeeze(1)

        batch, num_agent = all_traj_preds[-1].shape[:2]
        agent_fut_preds = all_traj_preds[-1].view(batch, num_agent, self.fut_mode, self.fut_ts, 2)
        agent_fut_cls_preds = all_traj_cls_scores[-1].view(batch, num_agent, self.fut_mode)
        loss_plan_input = [ego_fut_preds, ego_fut_gt, ego_fut_masks, ego_fut_cmd,
                           map_all_pts_preds[-1][..., 0:2], map_all_cls_scores[-1].sigmoid(),
                           all_bbox_preds[-1][..., 0:2], agent_fut_preds,
                           all_cls_scores[-1].sigmoid(), agent_fut_cls_preds.sigmoid()]

        loss_planning_dict = self.loss_planning(
            *loss_plan_input,
            privileged_fut_preds=preds_dicts.get('privileged_ego_fut_preds'),
            goal_sel_fut_preds=preds_dicts.get('goal_sel_fut_preds'))
        loss_dict['loss_plan_reg'] = loss_planning_dict['loss_plan_reg']
        if 'loss_privileged_reg' in loss_planning_dict:
            loss_dict['loss_privileged_reg'] = \
                loss_planning_dict['loss_privileged_reg']
            loss_dict['loss_plan_distill'] = \
                loss_planning_dict['loss_plan_distill']
        loss_dict['loss_plan_bound'] = loss_planning_dict['loss_plan_bound']
        loss_dict['loss_plan_col'] = loss_planning_dict['loss_plan_col']
        loss_dict['loss_plan_dir'] = loss_planning_dict['loss_plan_dir']
        if self.prism_latent_supervision:
            # Computed once in forward() (needs the raw, unpadded per-
            # sample long-future trajectory/valid-flag that don't survive
            # into this function's already-squeezed inputs) and threaded
            # through preds_dicts purely as a scalar. Already includes the
            # prism_kl_weight (beta) scaling and per-sample valid-flag
            # masking -- see forward()'s comment at the injection site.
            loss_dict['loss_prism_kl'] = preds_dicts['prism_kl_loss']
        if 'aux_ego_motion_loss' in preds_dicts:
            # Same threading pattern as loss_prism_kl above: computed in
            # forward() (where ego_feats and the raw ego_lcf target are both
            # in scope) and carried here as an already-weighted scalar.
            loss_dict['loss_aux_ego_motion'] = preds_dicts['aux_ego_motion_loss']
        if 'aux_long_horizon_loss' in preds_dicts:
            loss_dict['loss_aux_long_horizon'] = preds_dicts[
                'aux_long_horizon_loss']
        if 'aux_bev_future_motion_loss' in preds_dicts:
            loss_dict['loss_aux_bev_future_motion'] = preds_dicts[
                'aux_bev_future_motion_loss']
        if 'aux_bev_motion_loss' in preds_dicts:
            loss_dict['loss_aux_bev_motion'] = preds_dicts[
                'aux_bev_motion_loss']
        if 'ego_status_decode_loss' in preds_dicts:
            loss_dict['loss_ego_status_decode'] = preds_dicts[
                'ego_status_decode_loss']
        if 'goal_cls_loss' in preds_dicts:
            loss_dict['loss_goal_cls'] = preds_dicts['goal_cls_loss']
            loss_dict['loss_goal_off'] = preds_dicts['goal_off_loss']
            if preds_dicts.get('goal_follow_loss') is not None:
                loss_dict['loss_goal_follow'] = preds_dicts['goal_follow_loss']
            if 'loss_goal_select' in loss_planning_dict:
                loss_dict['loss_goal_select'] = \
                    loss_planning_dict['loss_goal_select']

        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_bbox_i in zip(losses_cls[:-1], losses_bbox[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_bbox'] = loss_bbox_i
            num_dec_layer += 1
        # loss from other decoder layers
        num_dec_layer = 0
        for map_loss_cls_i, map_loss_bbox_i, map_loss_iou_i, map_loss_pts_i, map_loss_dir_i in zip(
            map_losses_cls[:-1],
            map_losses_bbox[:-1],
            map_losses_iou[:-1],
            map_losses_pts[:-1],
            map_losses_dir[:-1]
        ):
            loss_dict[f'd{num_dec_layer}.loss_map_cls'] = map_loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_map_bbox'] = map_loss_bbox_i
            loss_dict[f'd{num_dec_layer}.loss_map_iou'] = map_loss_iou_i
            loss_dict[f'd{num_dec_layer}.loss_map_pts'] = map_loss_pts_i
            loss_dict[f'd{num_dec_layer}.loss_map_dir'] = map_loss_dir_i
            num_dec_layer += 1

        # loss of proposal generated from encode feature map.
        if enc_cls_scores is not None:
            binary_labels_list = [
                torch.zeros_like(gt_labels_list[i])
                for i in range(len(all_gt_labels_list))
            ]
            enc_loss_cls, enc_losses_bbox = \
                self.loss_single(enc_cls_scores, enc_bbox_preds,
                                 gt_bboxes_list, binary_labels_list,
                                 gt_bboxes_ignore)
            loss_dict['enc_loss_cls'] = enc_loss_cls
            loss_dict['enc_loss_bbox'] = enc_losses_bbox

        if map_enc_cls_scores is not None:
            map_binary_labels_list = [
                torch.zeros_like(map_gt_labels_list[i])
                for i in range(len(map_all_gt_labels_list))
            ]
            # TODO bug here, but we dont care enc_loss now
            map_enc_loss_cls, map_enc_loss_bbox, map_enc_loss_iou, \
                 map_enc_loss_pts, map_enc_loss_dir = \
                self.map_loss_single(
                    map_enc_cls_scores, map_enc_bbox_preds,
                    map_enc_pts_preds, map_gt_bboxes_list,
                    map_binary_labels_list, map_gt_pts_list,
                    map_gt_bboxes_ignore
                )
            loss_dict['enc_loss_map_cls'] = map_enc_loss_cls
            loss_dict['enc_loss_map_bbox'] = map_enc_loss_bbox
            loss_dict['enc_loss_map_iou'] = map_enc_loss_iou
            loss_dict['enc_loss_map_pts'] = map_enc_loss_pts
            loss_dict['enc_loss_map_dir'] = map_enc_loss_dir

        return loss_dict

    @force_fp32(apply_to=('preds_dicts'))
    def get_bboxes(self, preds_dicts, img_metas, rescale=False):
        """Generate bboxes from bbox head predictions.
        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
            img_metas (list[dict]): Point cloud and image's meta info.
        Returns:
            list[dict]: Decoded bbox, scores and labels after nms.
        """

        det_preds_dicts = self.bbox_coder.decode(preds_dicts)
        # map_bboxes: xmin, ymin, xmax, ymax
        map_preds_dicts = self.map_bbox_coder.decode(preds_dicts)

        num_samples = len(det_preds_dicts)
        assert len(det_preds_dicts) == len(map_preds_dicts), \
             'len(preds_dict) should be equal to len(map_preds_dicts)'
        ret_list = []
        for i in range(num_samples):
            preds = det_preds_dicts[i]
            bboxes = preds['bboxes']
            bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5
            code_size = bboxes.shape[-1]
            bboxes = img_metas[i]['box_type_3d'](bboxes, code_size)
            scores = preds['scores']
            labels = preds['labels']
            trajs = preds['trajs']

            map_preds = map_preds_dicts[i]
            map_bboxes = map_preds['map_bboxes']
            map_scores = map_preds['map_scores']
            map_labels = map_preds['map_labels']
            map_pts = map_preds['map_pts']

            ret_list.append([bboxes, scores, labels, trajs, map_bboxes,
                             map_scores, map_labels, map_pts])

        return ret_list

    def select_and_pad_pred_map(
        self,
        motion_pos,
        map_query,
        map_score,
        map_pos,
        map_thresh=0.5,
        dis_thresh=None,
        pe_normalization=True,
        use_fix_pad=False
    ):
        """select_and_pad_pred_map.
        Args:
            motion_pos: [B, A, 2]
            map_query: [B, P, D].
            map_score: [B, P, 3].
            map_pos: [B, P, pts, 2].
            map_thresh: map confidence threshold for filtering low-confidence preds
            dis_thresh: distance threshold for masking far maps for each agent in cross-attn
            use_fix_pad: always pad one lane instance for each batch
        Returns:
            selected_map_query: [B*A, P1(+1), D], P1 is the max inst num after filter and pad.
            selected_map_pos: [B*A, P1(+1), 2]
            selected_padding_mask: [B*A, P1(+1)]
        """
        
        if dis_thresh is None:
            raise NotImplementedError('Not implement yet')

        # use the most close pts pos in each map inst as the inst's pos
        batch, num_map = map_pos.shape[:2]
        map_dis = torch.sqrt(map_pos[..., 0]**2 + map_pos[..., 1]**2)
        min_map_pos_idx = map_dis.argmin(dim=-1).flatten()  # [B*P]
        min_map_pos = map_pos.flatten(0, 1)  # [B*P, pts, 2]
        min_map_pos = min_map_pos[range(min_map_pos.shape[0]), min_map_pos_idx]  # [B*P, 2]
        min_map_pos = min_map_pos.view(batch, num_map, 2)  # [B, P, 2]

        # select & pad map vectors for different batch using map_thresh
        map_score = map_score.sigmoid()
        map_max_score = map_score.max(dim=-1)[0]
        map_idx = map_max_score > map_thresh
        batch_max_pnum = 0
        for i in range(map_score.shape[0]):
            pnum = map_idx[i].sum()
            if pnum > batch_max_pnum:
                batch_max_pnum = pnum

        selected_map_query, selected_map_pos, selected_padding_mask = [], [], []
        for i in range(map_score.shape[0]):
            dim = map_query.shape[-1]
            valid_pnum = map_idx[i].sum()
            valid_map_query = map_query[i, map_idx[i]]
            valid_map_pos = min_map_pos[i, map_idx[i]]
            pad_pnum = batch_max_pnum - valid_pnum
            padding_mask = torch.tensor([False], device=map_score.device).repeat(batch_max_pnum)
            if pad_pnum != 0:
                valid_map_query = torch.cat([valid_map_query, torch.zeros((pad_pnum, dim), device=map_score.device)], dim=0)
                valid_map_pos = torch.cat([valid_map_pos, torch.zeros((pad_pnum, 2), device=map_score.device)], dim=0)
                padding_mask[valid_pnum:] = True
            selected_map_query.append(valid_map_query)
            selected_map_pos.append(valid_map_pos)
            selected_padding_mask.append(padding_mask)

        selected_map_query = torch.stack(selected_map_query, dim=0)
        selected_map_pos = torch.stack(selected_map_pos, dim=0)
        selected_padding_mask = torch.stack(selected_padding_mask, dim=0)

        # generate different pe for map vectors for each agent
        num_agent = motion_pos.shape[1]
        selected_map_query = selected_map_query.unsqueeze(1).repeat(1, num_agent, 1, 1)  # [B, A, max_P, D]
        selected_map_pos = selected_map_pos.unsqueeze(1).repeat(1, num_agent, 1, 1)  # [B, A, max_P, 2]
        selected_padding_mask = selected_padding_mask.unsqueeze(1).repeat(1, num_agent, 1)  # [B, A, max_P]
        # move lane to per-car coords system
        selected_map_dist = selected_map_pos - motion_pos[:, :, None, :]  # [B, A, max_P, 2]
        if pe_normalization:
            selected_map_pos = selected_map_pos - motion_pos[:, :, None, :]  # [B, A, max_P, 2]

        # filter far map inst for each agent
        map_dis = torch.sqrt(selected_map_dist[..., 0]**2 + selected_map_dist[..., 1]**2)
        valid_map_inst = (map_dis <= dis_thresh)  # [B, A, max_P]
        invalid_map_inst = (valid_map_inst == False)
        selected_padding_mask = selected_padding_mask + invalid_map_inst

        selected_map_query = selected_map_query.flatten(0, 1)
        selected_map_pos = selected_map_pos.flatten(0, 1)
        selected_padding_mask = selected_padding_mask.flatten(0, 1)

        num_batch = selected_padding_mask.shape[0]
        feat_dim = selected_map_query.shape[-1]
        if use_fix_pad:
            pad_map_query = torch.zeros((num_batch, 1, feat_dim), device=selected_map_query.device)
            pad_map_pos = torch.ones((num_batch, 1, 2), device=selected_map_pos.device)
            pad_lane_mask = torch.tensor([False], device=selected_padding_mask.device).unsqueeze(0).repeat(num_batch, 1)
            selected_map_query = torch.cat([selected_map_query, pad_map_query], dim=1)
            selected_map_pos = torch.cat([selected_map_pos, pad_map_pos], dim=1)
            selected_padding_mask = torch.cat([selected_padding_mask, pad_lane_mask], dim=1)

        return selected_map_query, selected_map_pos, selected_padding_mask


    def select_and_pad_query(
        self,
        query,
        query_pos,
        query_score,
        score_thresh=0.5,
        use_fix_pad=True
    ):
        """select_and_pad_query.
        Args:
            query: [B, Q, D].
            query_pos: [B, Q, 2]
            query_score: [B, Q, C].
            score_thresh: confidence threshold for filtering low-confidence query
            use_fix_pad: always pad one query instance for each batch
        Returns:
            selected_query: [B, Q', D]
            selected_query_pos: [B, Q', 2]
            selected_padding_mask: [B, Q']
        """

        # select & pad query for different batch using score_thresh
        query_score = query_score.sigmoid()
        query_score = query_score.max(dim=-1)[0]
        query_idx = query_score > score_thresh
        batch_max_qnum = 0
        for i in range(query_score.shape[0]):
            qnum = query_idx[i].sum()
            if qnum > batch_max_qnum:
                batch_max_qnum = qnum

        selected_query, selected_query_pos, selected_padding_mask = [], [], []
        for i in range(query_score.shape[0]):
            dim = query.shape[-1]
            valid_qnum = query_idx[i].sum()
            valid_query = query[i, query_idx[i]]
            valid_query_pos = query_pos[i, query_idx[i]]
            pad_qnum = batch_max_qnum - valid_qnum
            padding_mask = torch.tensor([False], device=query_score.device).repeat(batch_max_qnum)
            if pad_qnum != 0:
                valid_query = torch.cat([valid_query, torch.zeros((pad_qnum, dim), device=query_score.device)], dim=0)
                valid_query_pos = torch.cat([valid_query_pos, torch.zeros((pad_qnum, 2), device=query_score.device)], dim=0)
                padding_mask[valid_qnum:] = True
            selected_query.append(valid_query)
            selected_query_pos.append(valid_query_pos)
            selected_padding_mask.append(padding_mask)

        selected_query = torch.stack(selected_query, dim=0)
        selected_query_pos = torch.stack(selected_query_pos, dim=0)
        selected_padding_mask = torch.stack(selected_padding_mask, dim=0)

        num_batch = selected_padding_mask.shape[0]
        feat_dim = selected_query.shape[-1]
        if use_fix_pad:
            pad_query = torch.zeros((num_batch, 1, feat_dim), device=selected_query.device)
            pad_query_pos = torch.ones((num_batch, 1, 2), device=selected_query_pos.device)
            pad_mask = torch.tensor([False], device=selected_padding_mask.device).unsqueeze(0).repeat(num_batch, 1)
            selected_query = torch.cat([selected_query, pad_query], dim=1)
            selected_query_pos = torch.cat([selected_query_pos, pad_query_pos], dim=1)
            selected_padding_mask = torch.cat([selected_padding_mask, pad_mask], dim=1)

        return selected_query, selected_query_pos, selected_padding_mask
