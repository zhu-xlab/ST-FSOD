# Copyright (c) OpenMMLab. All rights reserved.
import copy
import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.ops import batched_nms
from mmcv.parallel import MMDistributedDataParallel

from rsidet.models.builder import HEADS
from rsidet.models import AnchorHead
from rsidet.core import (anchor_inside_flags, build_assigner, build_bbox_coder,
                        build_prior_generator, build_sampler, images_to_levels,
                        multi_apply, unmap)

def get_module(module):

    if isinstance(module, MMDistributedDataParallel):
        return module.module

    return module

@HEADS.register_module()
class STRPNHeadV4(AnchorHead):
    """RPN head.

    Args:
        in_channels (int): Number of channels in the input feature map.
        init_cfg (dict or list[dict], optional): Initialization config dict.
        num_convs (int): Number of convolution layers in the head. Default 1.
    """  # noqa: W605

    def __init__(self,
                 in_channels,
                 init_cfg=dict(type='Normal', layer='Conv2d', std=0.01),
                 num_convs=1,
                 st_thre=1.,
                 bg_thre=0.8,
                 alpha=0.999,
                 drop_rate=0.3,
                 init_cls=True,
                 novel_class_inds=None,
                 neg_nms_type='reverse',
                 apply_loss_base=False,
                 **kwargs):
        self.num_convs = num_convs
        self.st_thre = st_thre
        self.bg_thre = bg_thre
        self.alpha = alpha
        self.local_iter = 0
        self.drop_rate = drop_rate
        self.neg_nms_type = neg_nms_type
        self.novel_class_inds = novel_class_inds
        self.init_cls = init_cls
        self.apply_loss_base = apply_loss_base
        super(STRPNHeadV4, self).__init__(
            1, in_channels, init_cfg=init_cfg, **kwargs)

    def _init_layers(self):
        """Initialize layers of the head."""
        if self.num_convs > 1:
            rpn_convs = []
            for i in range(self.num_convs):
                if i == 0:
                    in_channels = self.in_channels
                else:
                    in_channels = self.feat_channels
                # use ``inplace=False`` to avoid error: one of the variables
                # needed for gradient computation has been modified by an
                # inplace operation.
                rpn_convs.append(
                    ConvModule(
                        in_channels,
                        self.feat_channels,
                        3,
                        padding=1,
                        inplace=False))
            self.rpn_conv = nn.Sequential(*rpn_convs)
        else:
            self.rpn_conv = nn.Conv2d(
                self.in_channels, self.feat_channels, 3, padding=1)

        self.rpn_cls = nn.Conv2d(self.feat_channels, self.num_base_priors * self.cls_out_channels, 1)
        self.rpn_reg = nn.Conv2d(self.feat_channels, self.num_base_priors * 4, 1)

        self.rpn_conv_stu = copy.deepcopy(self.rpn_conv)
        self.rpn_cls_stu = nn.Conv2d(self.feat_channels, self.num_base_priors * self.cls_out_channels, 1)
        self.rpn_reg_stu = nn.Conv2d(self.feat_channels, self.num_base_priors * 4, 1)
        self.rpn_conv_tch = copy.deepcopy(self.rpn_conv)
        self.rpn_cls_tch = nn.Conv2d(self.feat_channels, self.num_base_priors * self.cls_out_channels, 1)
        self.rpn_reg_tch = nn.Conv2d(self.feat_channels, self.num_base_priors * 4, 1)
        self.dropout = torch.nn.Dropout(p=self.drop_rate)

    def _init_stu_tch_weights(self):
        param_list = ['conv', 'reg', 'cls']
        # param_list = ['conv', 'reg']
        # if self.init_cls:
            # param_list.append('cls')

        for param_name in param_list:
            # for param in get_module(eval(f'self.rpn_{param_name}')).parameters():
            #     param.detach_()

            mp = list(get_module(eval(f'self.rpn_{param_name}')).parameters())
            mcp_stu = list(get_module(eval(f'self.rpn_{param_name}_stu')).parameters())
            mcp_tch = list(get_module(eval(f'self.rpn_{param_name}_tch')).parameters())

            for i in range(0, len(mp)):
                if not mcp_stu[i].data.shape:  # scalar tensor
                    mcp_stu[i].data = mp[i].data.clone()
                    mcp_tch[i].data = mp[i].data.clone()
                    if self.init_cls and param_name == 'cls':
                        mcp_stu[i].data.fill_(0)
                        mcp_tch[i].data.fill_(0)
                else:
                    mcp_stu[i].data[:] = mp[i].data[:].clone()
                    mcp_tch[i].data[:] = mp[i].data[:].clone()
                    if self.init_cls and param_name == 'cls':
                        torch.nn.init.normal_(mcp_stu[i].data, 0, 0.01)
                        torch.nn.init.normal_(mcp_tch[i].data, 0, 0.01)

    def _update_tch(self, iter):
        alpha_teacher = min(1 - 1 / (iter + 1), self.alpha)
        for param_name in ['conv', 'cls', 'reg']:
            for ema_param, param in zip(get_module(eval(f'self.rpn_{param_name}_tch')).parameters(),
                                        get_module(eval(f'self.rpn_{param_name}_stu')).parameters()):
                if not param.data.shape:  # scalar tensor
                    ema_param.data = \
                        alpha_teacher * ema_param.data + \
                        (1 - alpha_teacher) * param.data
                else:
                    ema_param.data[:] = \
                        alpha_teacher * ema_param[:].data[:] + \
                        (1 - alpha_teacher) * param[:].data[:]

    def forward_single(self, ori_x):
        """Forward feature map of a single scale level."""
        x_stu = self.dropout(ori_x)
        x_stu = self.rpn_conv_stu(x_stu)
        x_stu = F.relu(x_stu, inplace=False)

        x_tch = self.rpn_conv_tch(ori_x)
        x_tch = F.relu(x_tch, inplace=False)

        x = self.rpn_conv(ori_x)
        x = F.relu(x, inplace=False)

        rpn_cls_score = self.rpn_cls(x)
        rpn_bbox_pred = self.rpn_reg(x)

        rpn_cls_score_tch = self.rpn_cls_tch(x_tch)
        rpn_bbox_pred_tch = self.rpn_reg_tch(x_tch)

        rpn_cls_score_stu = self.rpn_cls_stu(x_stu)
        rpn_bbox_pred_stu = self.rpn_reg_stu(x_stu)

        return rpn_cls_score, rpn_bbox_pred, rpn_cls_score_stu, \
                rpn_bbox_pred_stu, rpn_cls_score_tch, rpn_bbox_pred_tch

    def simple_test_rpn(self, x, img_metas, proposal_cfg=None):
        """Test without augmentation, only for ``RPNHead`` and its variants,
        e.g., ``GARPNHead``, etc.

        Args:
            x (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
            img_metas (list[dict]): Meta info of each image.

        Returns:
            list[Tensor]: Proposals of each image, each item has shape (n, 5),
                where 5 represent (tl_x, tl_y, br_x, br_y, score).
        """
        base_cls_score, base_bbox_pred, novel_cls_score, novel_bbox_pred, _, _ = self(x)
        # all_cls_score = []
        # all_bbox_pred = []
        # for cur_base_cls_score, cur_novel_cls_score in zip(base_cls_score, novel_cls_score):
        #     all_cls_score.append(torch.cat([cur_base_cls_score, cur_novel_cls_score], dim=0))

        # for cur_base_bbox_pred, cur_novel_bbox_pred in zip(base_bbox_pred, novel_bbox_pred):
        #     all_bbox_pred.append(torch.cat([cur_base_bbox_pred, cur_novel_bbox_pred], dim=1))

        proposal_list_base = self.get_bboxes(base_cls_score, base_bbox_pred, img_metas=img_metas)
        proposal_list_novel = self.get_bboxes(novel_cls_score, novel_bbox_pred, img_metas=img_metas)
        proposal_list_all = []
        for proposal_base, proposal_novel in zip(proposal_list_base, proposal_list_novel):
            merged_proposal = torch.cat([proposal_base, proposal_novel], dim=0)
            sorted_inds = merged_proposal[:, -1].sort(dim=0, descending=True)
            proposal_list_all.append(merged_proposal[sorted_inds[1]])

        return proposal_list_all

    def loss(self,
             cls_scores,
             bbox_preds,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute losses of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W)
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss. Default: None

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.prior_generator.num_levels

        device = cls_scores[0].device

        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas, device=device)
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1

        cls_reg_targets = self.get_targets(
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels)
        if cls_reg_targets is None:
            return None
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        num_total_samples = (
            num_total_pos + num_total_neg if self.sampling else num_total_pos)

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        # concat all level anchors and flags to a single tensor
        concat_anchor_list = []
        for i in range(len(anchor_list)):
            concat_anchor_list.append(torch.cat(anchor_list[i]))
        all_anchor_list = images_to_levels(concat_anchor_list, num_level_anchors)

        losses_cls, losses_bbox = multi_apply(
            self.loss_single,
            cls_scores,
            bbox_preds,
            all_anchor_list,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            num_total_samples=num_total_samples)

        # return dict(loss_cls=losses_cls, loss_bbox=losses_bbox)
        return dict(
            loss_rpn_cls=losses_cls,
            loss_rpn_bbox=losses_bbox
        )


    def forward_train(self,
                      x,
                      img_metas,
                      gt_bboxes,
                      gt_labels=None,
                      gt_bboxes_ignore=None,
                      proposal_cfg=None,
                      **kwargs):
        """
        Args:
            x (list[Tensor]): Features from FPN.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            proposal_cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used

        Returns:
            tuple:
                losses: (dict[str, Tensor]): A dictionary of loss components.
                proposal_list (list[Tensor]): Proposals of each image.
        """

        if self.local_iter == 0:
            self._init_stu_tch_weights()
        else:
            self._update_tch(self.local_iter)

        rpn_cls_score, rpn_bbox_pred, rpn_cls_score_stu, rpn_bbox_pred_stu,\
                rpn_cls_score_tch, rpn_bbox_pred_tch = self(x)

        # assert gt_labels is not None
        # outs = (rpn_cls_score_stu, rpn_bbox_pred_stu)

        proposal_list_tch = self.get_bboxes(
            rpn_cls_score_tch, rpn_bbox_pred_tch, img_metas=img_metas, cfg=proposal_cfg
        )

        novel_pseudo_bboxes = []
        novel_ignore_bboxes = []
        for cur_proposal_tch, cur_gt_bboxes, cur_gt_labels in zip(proposal_list_tch, gt_bboxes, gt_labels):
            novel_inds = torch.tensor(self.novel_class_inds).to(cur_gt_bboxes.device)
            novel_inds = (cur_gt_labels.unsqueeze(1) == novel_inds.unsqueeze(0)).sum(dim=1)
            novel_inds = novel_inds > 0

            cur_novel_pseudo_bboxes = cur_proposal_tch[cur_proposal_tch[:, -1] > self.st_thre, :-1]
            cur_novel_pseudo_bboxes = torch.cat([cur_gt_bboxes[novel_inds], cur_novel_pseudo_bboxes], dim=0)

            ignore_inds = (cur_proposal_tch[:, -1] < self.st_thre) \
                    & (cur_proposal_tch[:, -1] > self.bg_thre)
            cur_novel_ignore_bboxes = cur_proposal_tch[ignore_inds, :-1]
            # gt_labels_batch = torch.cat([gt_labels_batch[novel_inds], 2])

            novel_pseudo_bboxes.append(cur_novel_pseudo_bboxes)
            novel_ignore_bboxes.append(cur_novel_ignore_bboxes)

        # will need to merge the novel_ignore_bboxes with the gt_bboxes_ignore if gt_bboxes_ignore is not None
        assert gt_bboxes_ignore is None

        losses = self.loss(
            rpn_cls_score_stu, rpn_bbox_pred_stu,
            novel_pseudo_bboxes, None, img_metas,
            gt_bboxes_ignore=novel_ignore_bboxes,
        )

        if self.apply_loss_base:
            losses_base = self.loss(
                rpn_cls_score, rpn_bbox_pred,
                gt_bboxes, None, img_metas,
                gt_bboxes_ignore=None,
            )
            for key, loss_item in losses_base.items():
                losses[key + '_base'] = loss_item

        self.local_iter += 1

        if proposal_cfg is None:
            return losses
        else:
            proposal_list_base = self.get_bboxes(
                rpn_cls_score, rpn_bbox_pred,
                img_metas=img_metas, cfg=proposal_cfg
            )

            proposal_list_novel = self.get_bboxes(
                rpn_cls_score_stu, rpn_bbox_pred_stu,
                img_metas=img_metas, cfg=proposal_cfg,
                nms_type=self.neg_nms_type
            )
            proposal_list = []
            for proposal_base, proposal_novel in zip(proposal_list_base, proposal_list_novel):
                proposal_list.append(torch.cat([proposal_base, proposal_novel], dim=0))

            return losses, proposal_list

    def loss_single(self, cls_score, bbox_pred, anchors, labels, label_weights,
                    bbox_targets, bbox_weights, num_total_samples,
                    cls_score_stu=None, bbox_pred_stu=None):
        """Compute loss of a single scale level.

        Args:
            cls_score (Tensor): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W).
            bbox_pred (Tensor): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W).
            anchors (Tensor): Box reference for each scale level with shape
                (N, num_total_anchors, 4).
            labels (Tensor): Labels of each anchors with shape
                (N, num_total_anchors).
            label_weights (Tensor): Label weights of each anchor with shape
                (N, num_total_anchors)
            bbox_targets (Tensor): BBox regression targets of each anchor
                weight shape (N, num_total_anchors, 4).
            bbox_weights (Tensor): BBox regression loss weights of each anchor
                with shape (N, num_total_anchors, 4).
            num_total_samples (int): If sampling, num total samples equal to
                the number of total anchors; Otherwise, it is the number of
                positive anchors.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        # classification loss
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        cls_score = cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)

        # In few-shot scenario, filter out the possibly true negative samples
        # label_weights[F.sigmoid(cls_score).sum(dim=1) > self.st_thre] = 0

        loss_cls = self.loss_cls(
            cls_score, labels, label_weights, avg_factor=num_total_samples)

        # regression loss
        bbox_targets = bbox_targets.reshape(-1, 4)
        bbox_weights = bbox_weights.reshape(-1, 4)
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
        if self.reg_decoded_bbox:
            # When the regression loss (e.g. `IouLoss`, `GIouLoss`)
            # is applied directly on the decoded bounding boxes, it
            # decodes the already encoded coordinates to absolute format.
            anchors = anchors.reshape(-1, 4)
            bbox_pred = self.bbox_coder.decode(anchors, bbox_pred)
        loss_bbox = self.loss_bbox(
            bbox_pred,
            bbox_targets,
            bbox_weights,
            avg_factor=num_total_samples)
        return loss_cls, loss_bbox

    def _get_bboxes_single(self,
                           cls_score_list,
                           bbox_pred_list,
                           score_factor_list,
                           mlvl_anchors,
                           img_meta,
                           cfg,
                           rescale=False,
                           with_nms=True,
                           nms_type='normal',
                           **kwargs):
        """Transform outputs of a single image into bbox predictions.

        Args:
            cls_score_list (list[Tensor]): Box scores from all scale
                levels of a single image, each item has shape
                (num_anchors * num_classes, H, W).
            bbox_pred_list (list[Tensor]): Box energies / deltas from
                all scale levels of a single image, each item has
                shape (num_anchors * 4, H, W).
            score_factor_list (list[Tensor]): Score factor from all scale
                levels of a single image. RPN head does not need this value.
            mlvl_anchors (list[Tensor]): Anchors of all scale level
                each item has shape (num_anchors, 4).
            img_meta (dict): Image meta info.
            cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.

        Returns:
            Tensor: Labeled boxes in shape (n, 5), where the first 4 columns
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the
                5-th column is a score between 0 and 1.
        """
        cfg = self.test_cfg if cfg is None else cfg
        cfg = copy.deepcopy(cfg)
        img_shape = img_meta['img_shape']

        # bboxes from different level should be independent during NMS,
        # level_ids are used as labels for batched NMS to separate them
        level_ids = []
        mlvl_scores = []
        mlvl_bbox_preds = []
        mlvl_valid_anchors = []
        nms_pre = cfg.get('nms_pre', -1)
        for level_idx in range(len(cls_score_list)):
            rpn_cls_score = cls_score_list[level_idx]
            rpn_bbox_pred = bbox_pred_list[level_idx]
            assert rpn_cls_score.size()[-2:] == rpn_bbox_pred.size()[-2:]
            rpn_cls_score = rpn_cls_score.permute(1, 2, 0)
            if self.use_sigmoid_cls:
                rpn_cls_score = rpn_cls_score.reshape(-1)
                scores = rpn_cls_score.sigmoid()
            else:
                rpn_cls_score = rpn_cls_score.reshape(-1, 2)
                # We set FG labels to [0, num_class-1] and BG label to
                # num_class in RPN head since rsidet v2.5, which is unified to
                # be consistent with other head since rsidet v2.0. In rsidet v2.0
                # to v2.4 we keep BG label as 0 and FG label as 1 in rpn head.
                scores = rpn_cls_score.softmax(dim=1)[:, 0]
            rpn_bbox_pred = rpn_bbox_pred.permute(1, 2, 0).reshape(-1, 4)

            anchors = mlvl_anchors[level_idx]
            if 0 < nms_pre < scores.shape[0]:
                # sort is faster than topk
                # _, topk_inds = scores.topk(cfg.nms_pre)
                ranked_scores, rank_inds = scores.sort(descending=True)
                topk_inds = rank_inds[:nms_pre]
                scores = ranked_scores[:nms_pre]
                rpn_bbox_pred = rpn_bbox_pred[topk_inds, :]
                anchors = anchors[topk_inds, :]

            mlvl_scores.append(scores)
            mlvl_bbox_preds.append(rpn_bbox_pred)
            mlvl_valid_anchors.append(anchors)
            level_ids.append(
                scores.new_full((scores.size(0), ),
                                level_idx,
                                dtype=torch.long))

        return self._bbox_post_process(mlvl_scores, mlvl_bbox_preds,
                                       mlvl_valid_anchors, level_ids, cfg,
                                       img_shape, nms_type=nms_type)

    def _bbox_post_process(self, mlvl_scores, mlvl_bboxes, mlvl_valid_anchors,
                           level_ids, cfg, img_shape, nms_type='normal', **kwargs):
        """bbox post-processing method.

        Do the nms operation for bboxes in same level.

        Args:
            mlvl_scores (list[Tensor]): Box scores from all scale
                levels of a single image, each item has shape
                (num_bboxes, ).
            mlvl_bboxes (list[Tensor]): Decoded bboxes from all scale
                levels of a single image, each item has shape (num_bboxes, 4).
            mlvl_valid_anchors (list[Tensor]): Anchors of all scale level
                each item has shape (num_bboxes, 4).
            level_ids (list[Tensor]): Indexes from all scale levels of a
                single image, each item has shape (num_bboxes, ).
            cfg (mmcv.Config): Test / postprocessing configuration,
                if None, `self.test_cfg` would be used.
            img_shape (tuple(int)): The shape of model's input image.

        Returns:
            Tensor: Labeled boxes in shape (n, 5), where the first 4 columns
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the
                5-th column is a score between 0 and 1.
        """
        scores = torch.cat(mlvl_scores)
        anchors = torch.cat(mlvl_valid_anchors)
        rpn_bbox_pred = torch.cat(mlvl_bboxes)
        proposals = self.bbox_coder.decode(
            anchors, rpn_bbox_pred, max_shape=img_shape)
        ids = torch.cat(level_ids)

        if cfg.min_bbox_size >= 0:
            w = proposals[:, 2] - proposals[:, 0]
            h = proposals[:, 3] - proposals[:, 1]
            valid_mask = (w > cfg.min_bbox_size) & (h > cfg.min_bbox_size)
            if not valid_mask.all():
                proposals = proposals[valid_mask]
                scores = scores[valid_mask]
                ids = ids[valid_mask]

        if proposals.numel() > 0:
            if nms_type == 'normal':
                dets, _ = batched_nms(proposals, scores, ids, cfg.nms)
            elif nms_type == 'reverse':
                dets, _ = batched_nms(proposals, 1 - scores, ids, cfg.nms)
                mask = ((1 - dets)[:, -1:] < cfg.neg_filter_thre).repeat(1, 5)
                dets = dets[mask].view(-1, 5)
                # dets = torch.stack([det for det in dets if 1 - det[-1] < cfg.neg_filter_thre])
            elif nms_type == 'random':
                random_scores = torch.rand(scores.shape).to(scores.device)
                dets, _ = batched_nms(proposals, random_scores, ids, cfg.nms)
                # mask = ((1 - dets)[:, -1:] < cfg.neg_filter_thre).repeat(1, 5)
                # dets = dets[mask].view(-1, 5)
            else:
                raise ValueError('No such a nms_type')

        else:
            return proposals.new_zeros(0, 5)

        return dets[:cfg.max_per_img]

    def onnx_export(self, x, img_metas):
        """Test without augmentation.

        Args:
            x (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
            img_metas (list[dict]): Meta info of each image.
        Returns:
            Tensor: dets of shape [N, num_det, 5].
        """
        cls_scores, bbox_preds = self(x)

        assert len(cls_scores) == len(bbox_preds)

        batch_bboxes, batch_scores = super(RPNHead, self).onnx_export(
            cls_scores, bbox_preds, img_metas=img_metas, with_nms=False)
        # Use ONNX::NonMaxSuppression in deployment
        from rsidet.core.export import add_dummy_nms_for_onnx
        cfg = copy.deepcopy(self.test_cfg)
        score_threshold = cfg.nms.get('score_thr', 0.0)
        nms_pre = cfg.get('deploy_nms_pre', -1)
        # Different from the normal forward doing NMS level by level,
        # we do NMS across all levels when exporting ONNX.
        dets, _ = add_dummy_nms_for_onnx(batch_bboxes, batch_scores,
                                         cfg.max_per_img,
                                         cfg.nms.iou_threshold,
                                         score_threshold, nms_pre,
                                         cfg.max_per_img)
        return dets
