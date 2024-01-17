# Copyright (c) OpenMMLab. All rights reserved.
import itertools
import logging
import os.path as osp
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence, Union
import pdb

import mmcv
import numpy as np
from mmcv.utils import print_log
from rsidet.datasets.api_wrappers import COCO, COCOeval
from rsidet.datasets.builder import DATASETS
from rsidet.datasets.coco import CocoDataset
from terminaltables import AsciiTable
from rsidet.core import eval_recalls
from rsifewshot.detection.core import eval_map

from .base import BaseFewShotDataset

# pre-defined classes split for few shot setting
ISAID_SPLIT = dict(
    ALL_CLASSES_SPLIT1 = (
        'Small_Vehicle', 'storage_tank', 'Swimming_pool', 'Harbor',
        'tennis_court', 'Ground_Track_Field', 'Soccer_ball_field',
        'baseball_diamond', 'Bridge', 'basketball_court', 'Roundabout',
        'Helicopter', 'ship', 'plane', 'Large_Vehicle'
    ),
    BASE_CLASSES_SPLIT1 = (
        'Small_Vehicle', 'storage_tank', 'Swimming_pool', 'Harbor',
        'tennis_court', 'Ground_Track_Field', 'Soccer_ball_field',
        'baseball_diamond', 'Bridge', 'basketball_court', 'Roundabout',
    ),
    NOVEL_CLASSES_SPLIT1 = (
        'Helicopter', 'ship', 'plane', 'Large_Vehicle'
    ),
    ALL_CLASSES_SPLIT2 = (
        'Small_Vehicle', 'Large_Vehicle', 'plane', 'storage_tank', 'ship',
        'Swimming_pool', 'Harbor', 'tennis_court', 'Ground_Track_Field',
        'Bridge', 'basketball_court', 'Helicopter',
        'baseball_diamond', 'Soccer_ball_field', 'Roundabout'
    ),
    BASE_CLASSES_SPLIT2 = (
        'Small_Vehicle', 'Large_Vehicle', 'plane', 'storage_tank', 'ship',
        'Swimming_pool', 'Harbor', 'tennis_court', 'Ground_Track_Field',
        'Bridge', 'basketball_court', 'Helicopter'
    ),
    NOVEL_CLASSES_SPLIT2 = (
        'baseball_diamond', 'Soccer_ball_field', 'Roundabout'
    ),
    ALL_CLASSES_SPLIT3 = (
        'Small_Vehicle', 'Large_Vehicle', 'plane', 'storage_tank', 'ship',
        'Swimming_pool', 'Harbor', 'tennis_court', 'Bridge',
        'Ground_Track_Field', 'Helicopter', 'Roundabout',
        'Soccer_ball_field', 'basketball_court', 'baseball_diamond'
    ),
    BASE_CLASSES_SPLIT3 = (
        'Small_Vehicle', 'Large_Vehicle', 'plane', 'storage_tank', 'ship',
        'Swimming_pool', 'Harbor', 'tennis_court', 'Bridge',
    ),
    NOVEL_CLASSES_SPLIT3 = (
        'Ground_Track_Field', 'Helicopter', 'Roundabout',
        'Soccer_ball_field', 'basketball_court', 'baseball_diamond'
    )
)


@DATASETS.register_module()
class FewShotISAIDDataset(BaseFewShotDataset, CocoDataset):
    """ISAID dataset for few shot detection.

    Args:
        classes (str | Sequence[str] | None): Classes for model training and
            provide fixed label for each class. When classes is string,
            it will load pre-defined classes in :obj:`FewShotCocoDataset`.
            For example: 'BASE_CLASSES', 'NOVEL_CLASSES` or `ALL_CLASSES`.
        num_novel_shots (int | None): Max number of instances used for each
            novel class. If is None, all annotation will be used.
            Default: None.
        num_base_shots (int | None): Max number of instances used for each base
            class. If is None, all annotation will be used. Default: None.
        ann_shot_filter (dict | None): Used to specify the class and the
            corresponding maximum number of instances when loading
            the annotation file. For example: {'dog': 10, 'person': 5}.
            If set it as None, `ann_shot_filter` will be
            created according to `num_novel_shots` and `num_base_shots`.
        min_bbox_area (int | float | None):  Filter images with bbox whose
            area smaller `min_bbox_area`. If set to None, skip
            this filter. Default: None.
        dataset_name (str | None): Name of dataset to display. For example:
            'train dataset' or 'query dataset'. Default: None.
        test_mode (bool): If set True, annotation will not be loaded.
            Default: False.
    """

    def __init__(self,
                 classes: Optional[Union[str, Sequence[str]]] = None,
                 num_novel_shots: Optional[int] = None,
                 num_base_shots: Optional[int] = None,
                 ann_shot_filter: Optional[Dict[str, int]] = None,
                 min_bbox_area: Optional[Union[int, float]] = None,
                 dataset_name: Optional[str] = None,
                 test_mode: bool = False,
                 coordinate_offset: List[int] = [-1, -1, 0, 0],
                 **kwargs) -> None:
        if dataset_name is None:
            self.dataset_name = 'Test dataset' \
                if test_mode else 'Train dataset'
        else:
            self.dataset_name = dataset_name
        self.coordinate_offset = coordinate_offset
        self.SPLIT = ISAID_SPLIT
        self.split_id = None
        assert classes is not None, f'{self.dataset_name}: classes in ' \
                                    f'`FewShotISAIDDataset` can not be None.'
        # `ann_shot_filter` will be used to filter out excess annotations
        # for few shot setting. It can be configured manually or generated
        # by the `num_novel_shots` and `num_base_shots`
        self.num_novel_shots = num_novel_shots
        self.num_base_shots = num_base_shots
        self.min_bbox_area = min_bbox_area
        self.CLASSES = self.get_classes(classes)
        if ann_shot_filter is None:
            if num_novel_shots is not None or num_base_shots is not None:
                ann_shot_filter = self._create_ann_shot_filter()
        else:
            assert num_novel_shots is None and num_base_shots is None, \
                f'{self.dataset_name}: can not config ann_shot_filter and ' \
                f'num_novel_shots/num_base_shots at the same time.'

        # these values would be set in `self.load_annotations_coco`
        self.cat_ids = []
        self.cat2label = {}
        self.coco = None
        self.img_ids = None

        super().__init__(
            classes=None,
            ann_shot_filter=ann_shot_filter,
            dataset_name=dataset_name,
            test_mode=test_mode,
            **kwargs)

    def get_classes(self, classes: Union[str, Sequence[str]]) -> List[str]:
        """Get class names.

        It supports to load pre-defined classes splits.
        The pre-defined classes splits are:
        ['ALL_CLASSES', 'NOVEL_CLASSES', 'BASE_CLASSES']

        Args:
            classes (str | Sequence[str]): Classes for model training and
                provide fixed label for each class. When classes is string,
                it will load pre-defined classes in `FewShotCocoDataset`.
                For example: 'NOVEL_CLASSES'.

        Returns:
            list[str]: list of class names.
        """
        # configure few shot classes setting
        if isinstance(classes, str):
            assert classes in self.SPLIT.keys(
            ), f'{self.dataset_name} : not a pre-defined classes or split ' \
               f'in ISAID_SPLIT.'
            class_names = self.SPLIT[classes]
            if 'BASE_CLASSES' in classes:
                assert self.num_novel_shots is None, \
                    f'{self.dataset_name}: BASE_CLASSES do not have ' \
                    f'novel instances.'
            elif 'NOVEL_CLASSES' in classes:
                assert self.num_base_shots is None, \
                    f'{self.dataset_name}: NOVEL_CLASSES do not have ' \
                    f'base instances.'
            self.split_id = int(classes[-1])

        elif isinstance(classes, (tuple, list)):
            class_names = classes
        else:
            raise ValueError(f'Unsupported type {type(classes)} of classes.')
        return class_names

    def _create_ann_shot_filter(self) -> Dict:
        """Generate `ann_shot_filter` for novel and base classes.

        Returns:
            dict[str, int]: The number of shots to keep for each class.
        """
        ann_shot_filter = {}
        # generate annotation filter for novel classes
        if self.num_novel_shots is not None:
            # for class_name in self.SPLIT['NOVEL_CLASSES']:
            #     ann_shot_filter[class_name] = self.num_novel_shots
            for class_name in self.SPLIT[
                    f'NOVEL_CLASSES_SPLIT{self.split_id}']:
                ann_shot_filter[class_name] = self.num_novel_shots

        # generate annotation filter for base classes
        if self.num_base_shots is not None:
            for class_name in self.SPLIT[f'BASE_CLASSES_SPLIT{self.split_id}']:
                ann_shot_filter[class_name] = self.num_base_shots
            # for class_name in self.SPLIT['BASE_CLASSES']:
            #     ann_shot_filter[class_name] = self.num_base_shots
        return ann_shot_filter

    def load_annotations(self, ann_cfg: List[Dict]) -> List[Dict]:
        """Support to Load annotation from two type of ann_cfg.

            - type of 'ann_file': COCO-style annotation file.
            - type of 'saved_dataset': Saved COCO dataset json.

        Args:
            ann_cfg (list[dict]): Config of annotations.

        Returns:
            list[dict]: Annotation infos.
        """
        data_infos = []
        for ann_cfg_ in ann_cfg:
            if ann_cfg_['type'] == 'saved_dataset':
                data_infos += self.load_annotations_saved(ann_cfg_['ann_file'])
            elif ann_cfg_['type'] == 'ann_file':
                data_infos += self.load_annotations_coco(ann_cfg_['ann_file'])
            else:
                raise ValueError(f'not support annotation type '
                                 f'{ann_cfg_["type"]} in ann_cfg.')
        return data_infos

    def load_annotations_coco(self, ann_file: str) -> List[Dict]:
        """Load annotation from COCO style annotation file.

        Args:
            ann_file (str): Path of annotation file.

        Returns:
            list[dict]: Annotation info from COCO api.
        """
        self.coco = COCO(ann_file)
        # to keep the label order equal to the order in CLASSES
        if len(self.cat_ids) == 0:
            for i, class_name in enumerate(self.CLASSES):
                cat_id = self.coco.get_cat_ids(cat_names=[class_name])[0]
                self.cat_ids.append(cat_id)
                self.cat2label[cat_id] = i
        else:
            # check categories id consistency between different files
            for i, class_name in enumerate(self.CLASSES):
                cat_id = self.coco.get_cat_ids(cat_names=[class_name])[0]
                assert self.cat2label[cat_id] == i, \
                    'please make sure all the json files use same ' \
                    'categories id for same class'
        self.img_ids = self.coco.get_img_ids()

        data_infos = []
        total_ann_ids = []
        for i in self.img_ids:
            info = self.coco.load_imgs([i])[0]
            info['filename'] = info['file_name']
            info['ann'] = self._get_ann_info(info)

            data_infos.append(info)
            ann_ids = self.coco.get_ann_ids(img_ids=[i])
            total_ann_ids.extend(ann_ids)

        assert len(set(total_ann_ids)) == len(
            total_ann_ids
        ), f'{self.dataset_name}: Annotation ids in {ann_file} are not unique!'

        return data_infos

    def _get_ann_info(self, data_info: Dict) -> Dict:
        """Get COCO annotation by index.

        Args:
            data_info(dict): Data info.

        Returns:
            dict: Annotation info of specified index.
        """

        img_id = data_info['id']
        ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
        ann_info = self.coco.load_anns(ann_ids)
        return self._parse_ann_info(data_info, ann_info)

    def get_cat_ids(self, idx: int) -> List[int]:
        """Get category ids by index.

        Overwrite the function in CocoDataset.

        Args:
            idx (int): Index of data.

        Returns:
            list[int]: All categories in the image of specified index.
        """

        return self.data_infos[idx]['ann']['labels'].astype(np.int64).tolist()

    def _filter_imgs(self,
                     min_size: int = 32,
                     min_bbox_area: Optional[int] = None) -> List[int]:
        """Filter images that do not meet the requirements.

        Args:
            min_size (int): Filter images with length or width
                smaller than `min_size`. Default: 32.
            min_bbox_area (int | None): Filter images with bbox whose
                area smaller `min_bbox_area`. If set to None, skip
                this filter. Default: None.

        Returns:
            list[int]: valid indices of data_infos.
        """
        valid_inds = []
        valid_img_ids = []
        if min_bbox_area is None:
            min_bbox_area = self.min_bbox_area
        for i, img_info in enumerate(self.data_infos):
            # filter empty image
            if self.filter_empty_gt and img_info['ann']['labels'].size == 0:
                continue
            # filter images smaller than `min_size`
            if min(img_info['width'], img_info['height']) < min_size:
                continue
            # filter image with bbox smaller than min_bbox_area
            # it is usually used in Attention RPN
            if min_bbox_area is not None:
                skip_flag = False
                for bbox in img_info['ann']['bboxes']:
                    bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                    if bbox_area < min_bbox_area:
                        skip_flag = True
                if skip_flag:
                    continue
            valid_inds.append(i)
            valid_img_ids.append(img_info['id'])
        # update coco img_ids
        self.img_ids = valid_img_ids
        return valid_inds

    def evaluate(self,
                 results: List[Sequence],
                 metric: Union[str, List[str]] = 'mAP',
                 logger: Optional[object] = None,
                 proposal_nums: Sequence[int] = (100, 300, 1000),
                 iou_thr: Optional[Union[float, Sequence[float]]] = 0.5,
                 class_splits: Optional[List[str]] = None) -> Dict:
        """Evaluation in VOC protocol and summary results of different splits
        of classes.

        Args:
            results (list[list | tuple]): Predictions of the model.
            metric (str | list[str]): Metrics to be evaluated. Options are
                'mAP', 'recall'. Default: mAP.
            logger (logging.Logger | None): Logger used for printing
                related information during evaluation. Default: None.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thr (float | list[float]): IoU threshold. Default: 0.5.
            class_splits: (list[str] | None): Calculate metric of classes
                split  defined in VOC_SPLIT. For example:
                ['BASE_CLASSES_SPLIT1', 'NOVEL_CLASSES_SPLIT1'].
                Default: None.

        Returns:
            dict[str, float]: AP/recall metrics.
        """
        # It should be noted that in the original rsidet implementation,
        # the four coordinates are reduced by 1 when the annotation
        # is parsed. Here we following detectron2, only xmin and ymin
        # will be reduced by 1 during training. The groundtruth used for
        # evaluation or testing keep consistent with original xml
        # annotation file and the xmin and ymin of prediction results
        # will add 1 for inverse of data loading logic.
        for i in range(len(results)):
            for j in range(len(results[i])):
                for k in range(4):
                    results[i][j][:, k] -= self.coordinate_offset[k]

        if not isinstance(metric, str):
            assert len(metric) == 1
            metric = metric[0]
        allowed_metrics = ['mAP', 'recall']
        if metric not in allowed_metrics:
            raise KeyError(f'metric {metric} is not supported')
        if class_splits is not None:
            for k in class_splits:
                assert k in self.SPLIT.keys(), 'undefiend classes split.'
            class_splits = {k: self.SPLIT[k] for k in class_splits}
            class_splits_mean_aps = {k: [] for k in class_splits.keys()}

        annotations = [self.get_ann_info(i) for i in range(len(self))]
        eval_results = OrderedDict()
        iou_thrs = [iou_thr] if isinstance(iou_thr, float) else iou_thr
        if metric == 'mAP':
            assert isinstance(iou_thrs, list)
            mean_aps = []
            for iou_thr in iou_thrs:
                print_log(f'\n{"-" * 15}iou_thr: {iou_thr}{"-" * 15}')
                mean_ap, ap_results = eval_map(
                    results,
                    annotations,
                    classes=self.CLASSES,
                    scale_ranges=None,
                    iou_thr=iou_thr,
                    dataset='voc07',
                    logger=logger,
                    use_legacy_coordinate=True)
                mean_aps.append(mean_ap)
                eval_results[f'AP{int(iou_thr * 100):02d}'] = round(mean_ap, 3)

                # calculate evaluate results of different class splits
                if class_splits is not None:
                    for k in class_splits.keys():
                        aps = [
                            cls_results['ap']
                            for i, cls_results in enumerate(ap_results)
                            if self.CLASSES[i] in class_splits[k]
                        ]
                        thres = [
                            cls_results['thre']
                            for i, cls_results in enumerate(ap_results)
                            if self.CLASSES[i] in class_splits[k]
                        ]
                        # for i, cls_results in enumerate(ap_results):
                        #     if self.CLASSES[i] in class_splits[k]:
                        #         cls_results[]

                        class_splits_mean_ap = np.array(aps).mean().item()
                        class_splits_thres = np.array(thres)

                        class_splits_mean_aps[k].append(class_splits_mean_ap)
                        eval_results[
                            f'{k}: AP{int(iou_thr * 100):02d}'] = round(
                                class_splits_mean_ap, 3)
                        eval_results[f'{k}: thres'] = class_splits_thres


            eval_results['mAP'] = sum(mean_aps) / len(mean_aps)
            if class_splits is not None:
                for k in class_splits.keys():
                    mAP = sum(class_splits_mean_aps[k]) / len(
                        class_splits_mean_aps[k])
                    print_log(f'{k} mAP: {mAP}', logger=logger)
        elif metric == 'recall':
            gt_bboxes = [ann['bboxes'] for ann in annotations]
            recalls = eval_recalls(
                gt_bboxes, results, proposal_nums, iou_thr, logger=logger)
            for i, num in enumerate(proposal_nums):
                for j, iou in enumerate(iou_thr):
                    eval_results[f'recall@{num}@{iou}'] = recalls[i, j]
            if recalls.shape[1] > 1:
                ar = recalls.mean(axis=1)
                for i, num in enumerate(proposal_nums):
                    eval_results[f'AR@{num}'] = ar[i]
        return eval_results

    # def evaluate(self,
    #              results: List[Sequence],
    #              metric: Union[str, List[str]] = 'bbox',
    #              logger: Optional[object] = None,
    #              jsonfile_prefix: Optional[str] = None,
    #              classwise: bool = False,
    #              proposal_nums: Sequence[int] = (100, 300, 1000),
    #              iou_thrs: Optional[Union[float, Sequence[float]]] = None,
    #              metric_items: Optional[Union[List[str], str]] = None,
    #              class_splits: Optional[List[str]] = None) -> Dict:
    #     """Evaluation in COCO protocol and summary results of different splits
    #     of classes.

    #     Args:
    #         results (list[list | tuple]): Testing results of the dataset.
    #         metric (str | list[str]): Metrics to be evaluated. Options are
    #             'bbox', 'proposal', 'proposal_fast'. Default: 'bbox'
    #         logger (logging.Logger | None): Logger used for printing
    #             related information during evaluation. Default: None.
    #         jsonfile_prefix (str | None): The prefix of json files. It includes
    #             the file path and the prefix of filename, e.g., "a/b/prefix".
    #             If not specified, a temp file will be created. Default: None.
    #         classwise (bool): Whether to evaluating the AP for each class.
    #         proposal_nums (Sequence[int]): Proposal number used for evaluating
    #             recalls, such as recall@100, recall@1000.
    #             Default: (100, 300, 1000).
    #         iou_thrs (Sequence[float] | float | None): IoU threshold used for
    #             evaluating recalls/mAPs. If set to a list, the average of all
    #             IoUs will also be computed. If not specified, [0.50, 0.55,
    #             0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95] will be used.
    #             Default: None.
    #         metric_items (list[str] | str | None): Metric items that will
    #             be returned. If not specified, ``['AR@100', 'AR@300',
    #             'AR@1000', 'AR_s@1000', 'AR_m@1000', 'AR_l@1000' ]`` will be
    #             used when ``metric=='proposal'``, ``['mAP', 'mAP_50', 'mAP_75',
    #             'mAP_s', 'mAP_m', 'mAP_l']`` will be used when
    #             ``metric=='bbox'``.
    #         class_splits: (list[str] | None): Calculate metric of classes split
    #             in ISAID_SPLIT. For example: ['BASE_CLASSES', 'NOVEL_CLASSES'].
    #             Default: None.

    #     Returns:
    #         dict[str, float]: COCO style evaluation metric.
    #     """
    #     if class_splits is not None:
    #         for k in class_splits:
    #             assert k in self.SPLIT.keys(), 'please define classes split.'
    #     metrics = metric if isinstance(metric, list) else [metric]
    #     allowed_metrics = ['bbox', 'proposal', 'proposal_fast']
    #     for metric in metrics:
    #         if metric not in allowed_metrics:
    #             raise KeyError(f'metric {metric} is not supported')
    #     if iou_thrs is None:
    #         iou_thrs = np.linspace(
    #             .5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
    #     if metric_items is not None:
    #         if not isinstance(metric_items, list):
    #             metric_items = [metric_items]

    #     if type(results[0]) == dict:
    #         results = [res['bbox_results'] for res in results]
    #     result_files, tmp_dir = self.format_results(results, jsonfile_prefix)

    #     eval_results = OrderedDict()
    #     cocoGt = self.coco
    #     for metric in metrics:
    #         msg = f'Evaluating {metric}...'
    #         if logger is None:
    #             msg = '\n' + msg
    #         print_log(msg, logger=logger)

    #         iou_type = 'bbox' if metric == 'proposal' else metric
    #         if metric not in result_files:
    #             raise KeyError(f'{metric} is not in results')
    #         try:
    #             predictions = mmcv.load(result_files[metric])
    #             cocoDt = cocoGt.loadRes(predictions)
    #         except IndexError:
    #             print_log(
    #                 'The testing results of the whole dataset is empty.',
    #                 logger=logger,
    #                 level=logging.ERROR)
    #             break

    #         # eval each class splits
    #         if class_splits is not None:
    #             class_splits = {k: ISAID_SPLIT[k] for k in class_splits}
    #             for split_name in class_splits.keys():
    #                 split_cat_ids = [
    #                     self.cat_ids[i] for i in range(len(self.CLASSES))
    #                     if self.CLASSES[i] in class_splits[split_name]
    #                 ]
    #                 self._evaluate_by_class_split(
    #                     cocoGt,
    #                     cocoDt,
    #                     iou_type,
    #                     proposal_nums,
    #                     iou_thrs,
    #                     split_cat_ids,
    #                     metric,
    #                     metric_items,
    #                     eval_results,
    #                     False,
    #                     logger,
    #                     split_name=split_name + ' ')
    #         # eval all classes
    #         self._evaluate_by_class_split(cocoGt, cocoDt, iou_type,
    #                                       proposal_nums, iou_thrs,
    #                                       self.cat_ids, metric, metric_items,
    #                                       eval_results, classwise, logger)

    #     if tmp_dir is not None:
    #         tmp_dir.cleanup()
    #     return eval_results

    def _evaluate_by_class_split(self,
                                 cocoGt: object,
                                 cocoDt: object,
                                 iou_type: str,
                                 proposal_nums: Sequence[int],
                                 iou_thrs: Union[float, Sequence[float]],
                                 cat_ids: List[int],
                                 metric: str,
                                 metric_items: Union[str, List[str]],
                                 eval_results: Dict,
                                 classwise: bool,
                                 logger: object,
                                 split_name: str = '') -> Dict:
        """Evaluation a split of classes in COCO protocol.

        Args:
            cocoGt (object): coco object with ground truth annotations.
            cocoDt (object): coco object with detection results.
            iou_type (str): Type of IOU.
            proposal_nums (Sequence[int]): Number of proposals.
            iou_thrs (float | Sequence[float]): Thresholds of IoU.
            cat_ids (list[int]): Class ids of classes to be evaluated.
            metric (str): Metrics to be evaluated.
            metric_items (str | list[str]): Metric items that will
                be returned. If not specified, ``['AR@100', 'AR@300',
                'AR@1000', 'AR_s@1000', 'AR_m@1000', 'AR_l@1000' ]`` will be
                used when ``metric=='proposal'``, ``['mAP', 'mAP_50', 'mAP_75',
                'mAP_s', 'mAP_m', 'mAP_l']`` will be used when
                ``metric=='bbox'``.
            eval_results (dict[str, float]): COCO style evaluation metric.
            classwise (bool): Whether to evaluating the AP for each class.
            split_name (str): Name of split. Default:''.

        Returns:
            dict[str, float]: COCO style evaluation metric.
        """
        cocoEval = COCOeval(cocoGt, cocoDt, iou_type)
        cocoEval.params.imgIds = self.img_ids
        cocoEval.params.maxDets = list(proposal_nums)
        cocoEval.params.iouThrs = iou_thrs

        cocoEval.params.catIds = cat_ids
        # mapping of cocoEval.stats
        coco_metric_names = {
            'mAP': 0,
            'mAP_50': 1,
            'mAP_75': 2,
            'mAP_s': 3,
            'mAP_m': 4,
            'mAP_l': 5,
            'AR@100': 6,
            'AR@300': 7,
            'AR@1000': 8,
            'AR_s@1000': 9,
            'AR_m@1000': 10,
            'AR_l@1000': 11
        }
        if metric_items is not None:
            for metric_item in metric_items:
                if metric_item not in coco_metric_names:
                    raise KeyError(
                        f'metric item {metric_item} is not supported')
        if split_name is not None:
            print_log(f'\n evaluation of {split_name} class', logger=logger)
        if metric == 'proposal':
            cocoEval.params.useCats = 0
            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()
            if metric_items is None:
                metric_items = [
                    'AR@100', 'AR@300', 'AR@1000', 'AR_s@1000', 'AR_m@1000',
                    'AR_l@1000'
                ]

            for item in metric_items:
                val = float(f'{cocoEval.stats[coco_metric_names[item]]:.3f}')
                eval_results[split_name + item] = val
        else:
            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()
            if classwise:  # Compute per-category AP
                # Compute per-category AP
                # from https://github.com/facebookresearch/detectron2/
                precisions = cocoEval.eval['precision']
                scores = cocoEval.eval['scores']
                # precision: (iou, recall, cls, area range, max dets)
                assert len(self.cat_ids) == precisions.shape[2], \
                    f'{self.cat_ids},{precisions.shape}'

                results_per_category = []
                sorted_cat_ids = np.sort(cat_ids)
                coco_cat2label = {v:k for k, v in enumerate(sorted_cat_ids)}

                for idx, catId in enumerate(self.cat_ids):
                    # area range index 0: all area ranges
                    # max dets index -1: typically 100 per image
                    nm = self.coco.loadCats(catId)[0]
                    # precision = precisions[:, :, coco_cat2label[catId], 0, -1]
                    precision = precisions[0, :, coco_cat2label[catId], 0, -1] # use per class AP@0.5
                    score = scores[0, :, coco_cat2label[catId], 0, -1]
                    # precision = precisions[0:1, :, catId - 1, 0, -1] 
                    det_thre_idx = (precision * (np.arange(101) / 100)).argmax()
                    det_thre = score[det_thre_idx]

                    precision = precision[precision > -1]
                    if precision.size:
                        ap = np.mean(precision)
                    else:
                        ap = float('nan')
                    results_per_category.append(
                        (f'{nm["name"]}', f'{float(ap):0.3f}', f'{det_thre:0.3f}'))

                num_columns = min(9, len(results_per_category) * 3)
                results_flatten = list(itertools.chain(*results_per_category))
                headers = [split_name + 'category', split_name + 'AP', split_name + 'thre'] * (
                    num_columns // 3)
                results_2d = itertools.zip_longest(*[
                    results_flatten[i::num_columns] for i in range(num_columns)
                ])
                table_data = [headers]
                table_data += [result for result in results_2d]
                table = AsciiTable(table_data)
                print_log('\n' + table.table, logger=logger)

            if metric_items is None:
                metric_items = [
                    'mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l'
                ]

            for metric_item in metric_items:
                key = f'{metric}_{metric_item}'
                val = float(
                    f'{cocoEval.stats[coco_metric_names[metric_item]]:.3f}')
                eval_results[split_name + key] = val
            ap = cocoEval.stats[:6]
            eval_results[split_name + f'{metric}_mAP_copypaste'] = (
                f'{ap[0]:.3f} {ap[1]:.3f} {ap[2]:.3f} {ap[3]:.3f} '
                f'{ap[4]:.3f} {ap[5]:.3f}')

            return eval_results


@DATASETS.register_module()
class FewShotISAIDCopyDataset(FewShotISAIDDataset):
    """Copy other COCO few shot datasets' `data_infos` directly.

    This dataset is mainly used for model initialization in some meta-learning
    detectors. In their cases, the support data are randomly sampled
    during training phase and they also need to be used in model
    initialization before evaluation. To copy the random sampling results,
    this dataset supports to load `data_infos` of other datasets via `ann_cfg`

    Args:
        ann_cfg (list[dict] | dict): contain `data_infos` from other
            dataset. Example: [dict(data_infos=FewShotCocoDataset.data_infos)]
    """

    def __init__(self, ann_cfg: Union[List[Dict], Dict], **kwargs) -> None:
        super().__init__(ann_cfg=ann_cfg, **kwargs)

    def ann_cfg_parser(self, ann_cfg: Union[List[Dict], Dict]) -> List[Dict]:
        """Parse annotation config from a copy of other dataset's `data_infos`.

        Args:
            ann_cfg (list[dict] | dict): contain `data_infos` from other
                dataset. Example:
                [dict(data_infos=FewShotCocoDataset.data_infos)]

        Returns:
            list[dict]: Annotation information.
        """
        data_infos = []
        if isinstance(ann_cfg, dict):
            assert ann_cfg.get('data_infos', None) is not None, \
                'ann_cfg of FewShotCocoCopyDataset require data_infos.'
            # directly copy data_info
            data_infos = ann_cfg['data_infos']
        elif isinstance(ann_cfg, list):
            for ann_cfg_ in ann_cfg:
                assert ann_cfg_.get('data_infos', None) is not None, \
                    'ann_cfg of FewShotCocoCopyDataset require data_infos.'
                # directly copy data_info
                data_infos += ann_cfg_['data_infos']
        return data_infos


@DATASETS.register_module()
class FewShotISAIDDefaultDataset(FewShotISAIDDataset):
    """FewShot COCO Dataset with some pre-defined annotation paths.

    :obj:`FewShotCocoDefaultDataset` provides pre-defined annotation files
    to ensure the reproducibility. The pre-defined annotation files provide
    fixed training data to avoid random sampling. The usage of `ann_cfg' is
    different from :obj:`FewShotCocoDataset`. The `ann_cfg' should contain
    two filed: `method` and `setting`.

    Args:
        ann_cfg (list[dict]): Each dict should contain
            `method` and `setting` to get corresponding
            annotation from `DEFAULT_ANN_CONFIG`.
            For example: [dict(method='TFA', setting='1shot')].
    """
    isaid_benchmark = {
        f'{shot}SHOT': [
            dict(
                type='ann_file',
                ann_file=f'data/few_shot_ann/coco/benchmark_{shot}shot/'
                f'full_box_{shot}shot_{class_name}_trainval.json')
            for class_name in ISAID_SPLIT['ALL_CLASSES_SPLIT1']
        ]
        for shot in [10, 30]
    }

    # pre-defined annotation config for model reproducibility
    DEFAULT_ANN_CONFIG = dict(
        TFA=isaid_benchmark,
        FSCE=isaid_benchmark,
        Attention_RPN={
            **isaid_benchmark, 'Official_10SHOT': [
                dict(
                    type='ann_file',
                    ann_file='data/few_shot_ann/coco/attention_rpn_10shot/'
                    'official_10_shot_from_instances_train2017.json')
            ]
        },
        MPSR=isaid_benchmark,
        MetaRCNN=isaid_benchmark,
        FSDetView=isaid_benchmark)

    def __init__(self, ann_cfg: List[Dict], **kwargs) -> None:
        super().__init__(ann_cfg=ann_cfg, **kwargs)

    def ann_cfg_parser(self, ann_cfg: List[Dict]) -> List[Dict]:
        """Parse pre-defined annotation config to annotation information.

        Args:
            ann_cfg (list[dict]): Each dict should contain
                `method` and `setting` to get corresponding
                annotation from `DEFAULT_ANN_CONFIG`.
                For example: [dict(method='TFA', setting='1shot')]

        Returns:
            list[dict]: Annotation information.
        """
        new_ann_cfg = []
        for ann_cfg_ in ann_cfg:
            assert isinstance(ann_cfg_, dict), \
                f'{self.dataset_name} : ann_cfg should be list of dict.'
            method = ann_cfg_['method']
            setting = ann_cfg_['setting']
            default_ann_cfg = self.DEFAULT_ANN_CONFIG[method][setting]
            ann_root = ann_cfg_.get('ann_root', None)
            if ann_root is not None:
                for i in range(len(default_ann_cfg)):
                    default_ann_cfg[i]['ann_file'] = \
                        osp.join(ann_root, default_ann_cfg[i]['ann_file'])
            new_ann_cfg += default_ann_cfg
        return super(FewShotCocoDataset, self).ann_cfg_parser(new_ann_cfg)
