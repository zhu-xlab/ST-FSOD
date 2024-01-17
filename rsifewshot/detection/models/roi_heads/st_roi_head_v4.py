# Copyright (c) OpenMMLab. All rights reserved.
import torch
import pdb

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
class STRoIHeadV4(StandardRoIHead):

    def __init__(self, stu_bbox_head=None, alpha=0.999,
                 local_iter=0, num_display_proposals=64, **kwargs):
        self.stu_bbox_head = stu_bbox_head
        self.alpha = alpha
        self.local_iter = local_iter
        self.num_display_proposals = num_display_proposals
        super(STRoIHeadV4, self).__init__(**kwargs)

    def init_assigner_sampler(self):
        """Initialize assigner and sampler."""
        self.bbox_assigner = None
        self.bbox_sampler = None
        if self.train_cfg:
            self.bbox_assigner = build_assigner(self.train_cfg.assigner)
            self.bbox_sampler = build_sampler(
                self.train_cfg.sampler, context=self)

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
        loss_bbox = self.stu_bbox_head.loss(
            stu_bbox_results['cls_score'],
            stu_bbox_results['bbox_pred'],
            bbox_results['cls_score'],
            bbox_results['bbox_pred'],
            rois, *bbox_targets
        )

        bbox_results.update(loss_bbox=loss_bbox)
        return bbox_results

    def _get_pseudo_bboxes(self, cls_score, bbox_pred):
        pass

    def forward_train(self,
                      x,
                      img_metas,
                      proposal_list,
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
            for i in range(num_imgs):
                assign_result = self.bbox_assigner.assign(
                    proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i],
                    gt_labels[i])
                sampling_result = self.bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x])

                sampling_results.append(sampling_result)


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

        states_proposals = self._prepare_states_proposals(
            img_metas, sampling_results, self.test_cfg, postfix='pos'
        )

        states.update(states_proposals)

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

        display_proposal_list = [proposal[:self.num_display_proposals] for proposal in proposal_list]
        proposal_bboxes, proposal_labels = self.decode_proposals(
            img_metas, display_proposal_list, None, rescale=rescale,
        )

        proposal_results = [
            bbox2result(proposal_bboxes[i], proposal_labels[i], 2)
            for i in range(len(proposal_bboxes))
        ]

        det_bboxes, det_labels = self.simple_test_bboxes(
            x, img_metas, proposal_list, self.test_cfg, rescale=rescale)


        bbox_results = [
            bbox2result(det_bboxes[i], det_labels[i],
                        self.bbox_head.num_classes)
            for i in range(len(det_bboxes))
        ]

        result_list = []
        for bbox, proposal in zip(bbox_results, proposal_results):
            results = {}
            results['bbox_results'] = bbox
            results['proposal_results'] = proposal
            result_list.append(results)

        return result_list


        # if self.with_mask:
        #     segm_results = self.simple_test_mask(
        #         x, img_metas, det_bboxes, det_labels, rescale=rescale)
        #     results['segm_results'] = segm_results
        #     # return list(zip(bbox_results, segm_results))



    # def aug_test(self, x, proposal_list, img_metas, rescale=False):
    #     """Test with augmentations.

    #     If rescale is False, then returned bboxes and masks will fit the scale
    #     of imgs[0].
    #     """

    #     det_bboxes, det_labels = self.aug_test_bboxes(x, img_metas,
    #                                                   proposal_list,
    #                                                   self.test_cfg)
    #     if rescale:
    #         _det_bboxes = det_bboxes
    #     else:
    #         _det_bboxes = det_bboxes.clone()
    #         _det_bboxes[:, :4] *= det_bboxes.new_tensor(
    #             img_metas[0][0]['scale_factor'])
    #     bbox_results = bbox2result(_det_bboxes, det_labels,
    #                                self.bbox_head.num_classes)

    #     results = {}
    #     results['bbox_results'] = bbox_results
    #     # det_bboxes always keep the original scale
    #     if self.with_mask:
    #         segm_results = self.aug_test_mask(x, img_metas, det_bboxes, det_labels)
    #         results['segm_results'] = segm_results


    #     return results
