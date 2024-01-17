# Copyright (c) OpenMMLab. All rights reserved.
import torch
import pdb
import torch.nn.functional as F

from mmcv.parallel import MMDistributedDataParallel
from rsidet.core import bbox2result, bbox2roi, build_assigner, build_sampler
from rsidet.models.builder import HEADS, build_head, build_roi_extractor
# from .base_roi_head import BaseRoIHead
# from .test_mixins import BBoxTestMixin, MaskTestMixin
from rsidet.models.roi_heads import StandardRoIHead

def get_module(module):

    if isinstance(module, MMDistributedDataParallel):
        return module.module

    return module

@HEADS.register_module()
class STRoIHeadV2(StandardRoIHead):

    def __init__(self, stu_bbox_head=None, alpha=0.999, local_iter=0,
                 novel_class_inds=None, num_classes=None, st_thre=0.9,
                 bg_thre=0.0, **kwargs):
        self.stu_bbox_head = stu_bbox_head
        self.alpha = alpha
        self.local_iter = local_iter
        self.novel_class_inds = novel_class_inds
        self.num_classes = num_classes
        self.st_thre = st_thre
        self.bg_thre = bg_thre
        super(STRoIHeadV2, self).__init__(**kwargs)

    def init_assigner_sampler(self):
        """Initialize assigner and sampler."""
        self.bbox_assigner = None
        self.bbox_sampler_pos = None
        self.bbox_sampler_neg = None
        if self.train_cfg:
            self.bbox_assigner = build_assigner(self.train_cfg.assigner)
            self.bbox_sampler_pos = build_sampler(
                self.train_cfg.sampler_pos, context=self)
            self.bbox_sampler_neg = build_sampler(
                self.train_cfg.sampler_neg, context=self)

    def _init_stu_bbox_head_weights(self):
        for param in get_module(self.bbox_head).parameters():
            param.detach_()
        mp = list(get_module(self.bbox_head).parameters())
        mcp = list(get_module(self.stu_bbox_head).parameters())

        for i in range(0, len(mp)):
            if not mcp[i].data.shape:  # scalar tensor
                mcp[i].data = mp[i].data.clone()
            else:
                mcp[i].data[:] = mp[i].data[:].clone()

    def _update_tch(self, iter):
        alpha_teacher = min(1 - 1 / (iter + 1), self.alpha)
        for ema_param, param in zip(get_module(self.bbox_head).parameters(),
                                    get_module(self.stu_bbox_head).parameters()):
            if not param.data.shape:  # scalar tensor
                ema_param.data = \
                    alpha_teacher * ema_param.data + \
                    (1 - alpha_teacher) * param.data
            else:
                ema_param.data[:] = \
                    alpha_teacher * ema_param[:].data[:] + \
                    (1 - alpha_teacher) * param[:].data[:]

    def init_bbox_head(self, bbox_roi_extractor, bbox_head):
        """Initialize ``bbox_head``"""
        self.bbox_roi_extractor = build_roi_extractor(bbox_roi_extractor)
        self.bbox_head = build_head(bbox_head)

        if self.stu_bbox_head is not None:
            drop_rate = self.stu_bbox_head.pop('drop_rate')
            self.stu_bbox_head = build_head(self.stu_bbox_head)
            self.dropout = torch.nn.Dropout(p=drop_rate)

    def _bbox_forward(self, x, rois, type='tch'):
        """Box head forward function used in both training and testing."""
        # TODO: a more flexible way to decide which feature maps to use
        bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)

        if self.type == 'tch':
            cls_score, bbox_pred = self.bbox_head(bbox_feats)
        else:
            dropped_bbox_feats = self.dropout(bbox_feats)
            cls_score, bbox_pred = self.stu_bbox_head(dropped_bbox_feats)

        bbox_results = dict(
            cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats)
        return bbox_results

    def _bbox_forward_train(self, x, sampling_results, gt_bboxes, gt_labels,
                            img_metas):
        """Run forward function and calculate loss for box head in training."""

        rois = bbox2roi([res.bboxes for res in sampling_results])
        bbox_results = self._bbox_forward(x, rois)
        stu_bbox_results = self._bbox_forward(x, rois, type='stu')

        bbox_targets = self.bbox_head.get_targets(sampling_results, gt_bboxes,
                                                  gt_labels, self.train_cfg)

        num_bboxes_per_img = [res.bboxes.shape[0] for res in sampling_results]
        num_pos_bboxes = [res.pos_bboxes.shape[0] for res in sampling_results]

        det_bboxes, det_labels = self.decode_train_bbox_results(img_metas, sampling_results,
                                                                bbox_results, None, False)
        # pseudo_labels, pseudo_bboxes = self._get_pseudo_bboxes(
        #     bbox_results['cls_score'], bbox_results['bbox_pred'], num_bboxes_per_img,
        #     num_pos_bboxes
        # )
        pseudo_labels, pseudo_bboxes = self._get_pseudo_bboxes(det_bboxes, det_labels,
                                                               num_pos_bboxes)
        # if pseudo_bboxes[0].shape[0] > 0:
        #     pdb.set_trace()
        #     pass

        new_gt_bboxes = []
        new_gt_labels = []
        for gt_bbox, pseudo_bbox, gt_label, pseudo_label in zip(gt_bboxes, pseudo_bboxes, gt_labels, pseudo_labels):
            new_gt_bbox = torch.cat([gt_bbox, pseudo_bbox], dim=0)
            new_gt_label = torch.cat([gt_label, pseudo_label])
            new_gt_bboxes.append(new_gt_bbox)
            new_gt_labels.append(new_gt_label)


        # bbox_targets_pseudo = self.bbox_head.get_targets(sampling_results, gt_bboxes, gt_labels, self.train_cfg)
        bbox_targets_pseudo = self.bbox_head.get_targets(sampling_results, new_gt_bboxes, new_gt_labels, self.train_cfg)

        loss_bbox = self.stu_bbox_head.loss(
            stu_bbox_results['cls_score'],
            stu_bbox_results['bbox_pred'],
            bbox_results['cls_score'],
            bbox_results['bbox_pred'],
            rois, *bbox_targets_pseudo
        )

        bbox_results.update(loss_bbox=loss_bbox)
        return bbox_results

    def _get_pseudo_bboxes(self, det_bboxes_list, det_labels_list, num_pos_bboxes_list):

        pseudo_label_list, pseudo_bbox_list = [], []
        for cls_score, bbox_pred, num_pos_bboxes in zip(det_labels_list, det_bboxes_list, num_pos_bboxes_list):
            bg_class_ind = self.num_classes
            # 0~self.num_classes-1 are FG, self.num_classes is BG
            # pos_inds = (label >= 0) & (label < bg_class_ind)
            # neg_inds = (label < 0) | (label == bg_class_ind)
            pos_inds = (torch.arange(0, cls_score.shape[0]) < num_pos_bboxes).to(cls_score.device)
            neg_inds = ~pos_inds

            # cls_score_probs = F.softmax(cls_score, dim=1).detach()
            probs, pseudo_labels = cls_score.max(dim=1)

            # if self.novel_class_inds is not None:
            assert self.novel_class_inds is not None

            novel_inds = torch.tensor(self.novel_class_inds).to(pseudo_labels.device)
            novel_inds = (pseudo_labels.unsqueeze(1) == novel_inds.unsqueeze(0)).sum(dim=1)
            novel_inds = novel_inds > 0

            pseudo_inds = neg_inds & novel_inds & (probs >= self.st_thre)
            ignore_inds = neg_inds & novel_inds & (probs >= self.bg_thre) & (probs < self.st_thre)

            pseudo_label_list.append(pseudo_labels[pseudo_inds])
            pseudo_bboxes = bbox_pred[pseudo_inds].view(-1, self.num_classes, 4)
            pseudo_bboxes_inds = pseudo_labels.view(-1, 1, 1).repeat(1, 1, 4)[pseudo_inds, :, :]
            pseudo_bboxes = torch.gather(pseudo_bboxes, 1, pseudo_bboxes_inds)
            pseudo_bbox_list.append(pseudo_bboxes[:, 0, :])

            # res_labels = pseudo_labels[pseudo_inds].split(num_bboxes_per_img)

        return pseudo_label_list, pseudo_bbox_list



    # def _get_pseudo_bboxes(self, ori_cls_score, ori_bbox_pred, num_bboxes_per_img,
    #                        num_pos_bboxes_list):

    #     cls_score_list = ori_cls_score.split(num_bboxes_per_img)
    #     bbox_pred_list = ori_bbox_pred.split(num_bboxes_per_img)

    #     pseudo_label_list, pseudo_bbox_list = [], []
    #     for cls_score, bbox_pred, num_pos_bboxes in zip(cls_score_list, bbox_pred_list,
    #                                                     num_pos_bboxes_list):
    #         bg_class_ind = self.num_classes
    #         # 0~self.num_classes-1 are FG, self.num_classes is BG
    #         # pos_inds = (label >= 0) & (label < bg_class_ind)
    #         # neg_inds = (label < 0) | (label == bg_class_ind)
    #         pos_inds = (torch.arange(0, cls_score.shape[0]) < num_pos_bboxes).to(cls_score.device)
    #         neg_inds = ~pos_inds

    #         cls_score_probs = F.softmax(cls_score, dim=1).detach()
    #         probs, pseudo_labels = cls_score_probs.max(dim=1)

    #         # if self.novel_class_inds is not None:
    #         assert self.novel_class_inds is not None

    #         novel_inds = torch.tensor(self.novel_class_inds).to(pseudo_labels.device)
    #         novel_inds = (pseudo_labels.unsqueeze(1) == novel_inds.unsqueeze(0)).sum(dim=1)
    #         novel_inds = novel_inds > 0

    #         pseudo_inds = neg_inds & novel_inds & (probs >= self.st_thre)
    #         ignore_inds = neg_inds & novel_inds & (probs >= self.bg_thre) & (probs < self.st_thre)

    #         pseudo_label_list.append(pseudo_labels[pseudo_inds])
    #         pseudo_bboxes = bbox_pred[pseudo_inds].view(-1, self.num_classes, 4)
    #         pseudo_bboxes_inds = pseudo_labels.view(-1, 1, 1).repeat(1, 1, 4)[pseudo_inds, :, :]
    #         pseudo_bboxes = torch.gather(pseudo_bboxes, 1, pseudo_bboxes_inds)
    #         pseudo_bbox_list.append(pseudo_bboxes[:, 0, :])

    #         # res_labels = pseudo_labels[pseudo_inds].split(num_bboxes_per_img)

    #     return pseudo_label_list, pseudo_bbox_list


        # # For matched foreground boxes, use the ground truth directly
        # pseudo_labels[pos_inds] = labels[pos_inds]

        # # For background boxes, if the output probability is larger than st_thre, use them as the pseudo labels
        # # if self.fix_neg_inds:
        # # pseudo_labels[neg_inds] = labels[neg_inds]
        # pseudo_labels[neg_inds & (probs < self.st_thre)] = labels[neg_inds & (probs < self.st_thre)]
        # pseudo_labels[~novel_inds] = labels[~novel_inds]

        # # For background boxes, if the output probablity is smaller than st_thre, but larger
        # # than bg_thre, ignore them. They could be foreground as well.
        # # For base classes, just use the normal label weights
        # weight_mask = neg_inds & (probs > self.bg_thre) & (probs < self.st_thre)
        # weight_mask_bg = neg_inds & (probs < self.st_thre)
        # weight_mask = weight_mask & novel_inds

        # label_weights[weight_mask] = 0



    def forward_train(self,
                      x,
                      img_metas,
                      proposal_list_pos,
                      proposal_list_neg,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      **kwargs):
        """
        Args:
            x (list[Tensor]): list of multi-level img features.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `rsidet/datasets/pipelines/formatting.py:Collect`.
            proposals (list[Tensors]): list of region proposals.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        if self.local_iter == 0:
            self._init_stu_bbox_head_weights()
        else:
            self._update_tch(self.local_iter)

        # assign gts and sample proposals
        if self.with_bbox or self.with_mask:
            num_imgs = len(img_metas)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            sampling_results_pos = []
            sampling_results_neg = []
            for i in range(num_imgs):
                assign_result = self.bbox_assigner.assign(
                    proposal_list_pos[i], gt_bboxes[i], gt_bboxes_ignore[i],
                    gt_labels[i])
                sampling_result_pos = self.bbox_sampler_pos.sample(
                    assign_result,
                    proposal_list_pos[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x])

                assign_result = self.bbox_assigner.assign(
                    proposal_list_neg[i], gt_bboxes[i], gt_bboxes_ignore[i],
                    gt_labels[i])
                sampling_result_neg = self.bbox_sampler_neg.sample(
                    assign_result,
                    proposal_list_neg[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x]
                )

                sampling_result = self._merge_sampling_results(sampling_result_pos, sampling_result_neg)
                sampling_results.append(sampling_result)
                sampling_results_pos.append(sampling_result_pos)
                sampling_results_neg.append(sampling_result_neg)


        losses = dict()

        # bbox head forward and loss
        if self.with_bbox:
            bbox_results = self._bbox_forward_train(x, sampling_results,
                                                    gt_bboxes, gt_labels,
                                                    img_metas)
            losses.update(bbox_results['loss_bbox'])

        # mask head forward and loss
        if self.with_mask:
            mask_results = self._mask_forward_train(x, sampling_results,
                                                    bbox_results['bbox_feats'],
                                                    gt_masks, img_metas)
            losses.update(mask_results['loss_mask'])

        states = self._prepare_states_rcnn(
            img_metas, sampling_results, bbox_results, self.test_cfg,
        )

        states_proposals_pos = self._prepare_states_proposals(
            img_metas, sampling_results_pos, self.test_cfg, postfix='pos'
        )

        states_proposals_neg = self._prepare_states_proposals(
            img_metas, sampling_results_neg, self.test_cfg, postfix='neg'
        )

        states.update(states_proposals_pos)
        states.update(states_proposals_neg)

        losses['states'] = states

        self.local_iter += 1

        return losses

    def _prepare_states_rcnn(self, img_metas, sampling_results, bbox_results,
                        vis_rcnn_cfg=None, rescale=False):

        det_bboxes, det_labels = self.decode_train_bbox_results(
            img_metas, sampling_results, bbox_results,
            vis_rcnn_cfg,
            rescale=rescale,
        )

        bbox_results = [
            bbox2result(det_bboxes[i], det_labels[i],
                        self.bbox_head.num_classes)
            for i in range(len(det_bboxes))
        ]


        # if not self.with_mask:
        #     return bbox_results
        # else:
        #     segm_results = self.simple_test_mask(
        #         x, img_metas, det_bboxes, det_labels, rescale=rescale)
        #     return list(zip(bbox_results, segm_results))

        states = {
            'else|bbox_results': bbox_results,
        }
        return states


    def _prepare_states_proposals(self, img_metas, proposal_list, vis_rcnn_cfg=None,
                                  rescale=False, postfix=''):

        det_bboxes, det_labels = self.decode_sampling_results(
            img_metas, proposal_list, None, rescale=rescale,
        )

        bbox_results = [
            bbox2result(det_bboxes[i], det_labels[i],
                        self.bbox_head.num_classes)
            for i in range(len(det_bboxes))
        ]


        # if not self.with_mask:
        #     return bbox_results
        # else:
        #     segm_results = self.simple_test_mask(
        #         x, img_metas, det_bboxes, det_labels, rescale=rescale)
        #     return list(zip(bbox_results, segm_results))

        states = {
            f'else|proposals_{postfix}': bbox_results,
        }
        return states



    def _merge_sampling_results(self, result1, result2):
        import copy
        result = copy.deepcopy(result1)
        # result.pos_bboxes = torch.cat([result1.pos_bboxes, result2.pos_bboxes])
        result.neg_bboxes = torch.cat([result1.neg_bboxes, result2.neg_bboxes])
        # result.pos_inds = torch.cat([result1.pos_inds, result2.pos_inds])
        result.neg_inds = torch.cat([result1.neg_inds, result2.neg_inds])
        # result.pos_assigned_gt_inds = torch.cat([result1.pos_assigned_gt_inds, result2.pos_assigned_gt_inds])
        # result.pos_is_gt = torch.cat([result1.pos_is_gt, result2.pos_is_gt])
        # result.num_gts = result1.num_gts + result2.num_gts

        return result

    def simple_test(self,
                    x,
                    proposal_list,
                    img_metas,
                    proposals=None,
                    rescale=False):
        """Test without augmentation.

        Args:
            x (tuple[Tensor]): Features from upstream network. Each
                has shape (batch_size, c, h, w).
            proposal_list (list(Tensor)): Proposals from rpn head.
                Each has shape (num_proposals, 5), last dimension
                5 represent (x1, y1, x2, y2, score).
            img_metas (list[dict]): Meta information of images.
            rescale (bool): Whether to rescale the results to
                the original image. Default: True.

        Returns:
            list[list[np.ndarray]] or list[tuple]: When no mask branch,
            it is bbox results of each image and classes with type
            `list[list[np.ndarray]]`. The outer list
            corresponds to each image. The inner list
            corresponds to each class. When the model has mask branch,
            it contains bbox results and mask results.
            The outer list corresponds to each image, and first element
            of tuple is bbox results, second element is mask results.
        """
        assert self.with_bbox, 'Bbox head must be implemented.'

        det_bboxes, det_labels = self.simple_test_bboxes(
            x, img_metas, proposal_list, self.test_cfg, rescale=rescale)

        bbox_results = [
            bbox2result(det_bboxes[i], det_labels[i],
                        self.bbox_head.num_classes)
            for i in range(len(det_bboxes))
        ]

        if not self.with_mask:
            return bbox_results
        else:
            segm_results = self.simple_test_mask(
                x, img_metas, det_bboxes, det_labels, rescale=rescale)
            return list(zip(bbox_results, segm_results))
