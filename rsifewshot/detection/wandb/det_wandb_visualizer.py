# Copyright (c) OpenMMLab. All rights reserved.
import importlib
import os.path as osp
import sys
import warnings
import pdb
from PIL import Image
import wandb
import torch
import os

import numpy as np

def img_loader(path, retry=5):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    ri = 0
    while ri < retry:
        try:
            with open(path, 'rb') as f:
                img = Image.open(f)
                return img.convert('RGB')
        except:
            ri += 1

class DetWandbVisualizer():

    def __init__(self,
                 val_dataset,
                 init_cfg,
                 **kwargs):

        self.val_dataset = val_dataset
        self.init_cfg = init_cfg
        self.num_eval_images = init_cfg.get('num_eval_images', 100)
        self.bbox_score_thr = init_cfg['bbox_score_thr']
        self.ckpt_hook: CheckpointHook = None
        self.eval_hook: EvalHook = None
        self.BGR2RGB = init_cfg.get('BGR2RGB', True)
        self.without_mask = init_cfg.get('without_mask', True)
        self.wandb = wandb
        self.interval = init_cfg.get('interval', 1)
        self.max_vis_boxes = init_cfg.get('max_vis_boxes', 100)
        self.wandb.init(
            project=init_cfg['project'],
            name=init_cfg['name'],
            dir=init_cfg['root']
        )

        # Initialize data table
        self._init_data_table()
        # Add data to the data table
        self._add_ground_truth()
        # Log ground truth data
        self._log_data_table()

    def process_vis_data(self, state):
        if state is not None:
            for key, value in state.items():
                if key.startswith('vis|'):
                    if key.split('|')[1].startswith('seg_mask'):
                        self._vis_seg_mask(key.split('|')[1], value)

                    elif key.split('|')[1].startswith('density'):
                        self._vis_density(key.split('|')[1], value)

                    elif key.split('|')[1].startswith('hist'):
                        self._vis_hist(key.split('|')[1], value[0], value[1])

                    elif key.split('|')[1].startswith('dets'):
                        self._vis_dets(key.split('|')[1], **value)

    def _result2wandb_image(self, img, bbox_result, segm_result, class_id_to_label=None):

        class_set = self.class_set
        if class_id_to_label is not None:
            class_set = self.wandb.Classes([{
                'id': id,
                'name': name
            } for id, name in class_id_to_label.items()])

        # Get labels
        bboxes = np.vstack(bbox_result)
        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(bbox_result)
        ]
        labels = np.concatenate(labels)

        # Get segmentation mask if available.
        segms = None
        if segm_result is not None and len(labels) > 0:
            segms = mmcv.concat_list(segm_result)
            segms = mask_util.decode(segms)
            segms = segms.transpose(2, 0, 1)
            assert len(segms) == len(labels)
        # TODO: Panoramic segmentation visualization.

        # Remove bounding boxes and masks with score lower than threshold.
        if type(self.bbox_score_thr) == list or self.bbox_score_thr > 0:
            # assert bboxes is not None and bboxes.shape[1] == 5
            scores = bboxes[:, -1]
            inds = scores > np.array(self.bbox_score_thr)[labels]
            bboxes = bboxes[inds, :]
            labels = labels[inds]
            if segms is not None:
                segms = segms[inds, ...]

        # Get dict of bounding boxes to be logged.
        if self.max_vis_boxes is not None:
            bboxes = bboxes[:self.max_vis_boxes]
            labels = labels[:self.max_vis_boxes]
        wandb_boxes = self._get_wandb_bboxes(bboxes, labels, log_gt=False, class_id_to_label=class_id_to_label)
        # Get dict of masks to be logged.
        if segms is not None:
            wandb_masks = self._get_wandb_masks(segms, labels)
        else:
            wandb_masks = None

        wandb_img = self.wandb.Image(
            img,
            boxes=wandb_boxes,
            masks=wandb_masks,
            classes=class_set
        )
        return wandb_img

    def _vis_dets(self, prefix, imgs, bbox_results, segm_results=None):

        img_log = {}
        img_log[prefix] = []

        for i in range(len(imgs)):

            cur_img = imgs[i]
            if type(cur_img) == torch.Tensor:
                cur_img = cur_img.permute(1,2,0).cpu().numpy()

            if self.BGR2RGB:
                cur_img = cur_img[:, :, [2,1,0]]

            class_id_to_label = None
            if 'proposal' in prefix:
                class_id_to_label = {1: 'proposal', 2: 'background'}

            wandb_img = self._result2wandb_image(
                cur_img, bbox_results[i],
                segm_results[i] if segm_results is not None else None,
                class_id_to_label=class_id_to_label
            )

            # wandb_img = self.wandb.Image(
            #     cur_img,
            #     masks={'mask': vis_mask} if mask is not None else None
            # )
            img_log[prefix].append(wandb_img)

        self.wandb.log(img_log)

    def after_train_iter(self, runner):
        if self.get_mode(runner) == 'train':
            # An ugly patch. The iter-based eval hook will call the
            # `after_train_iter` method of all logger hooks before evaluation.
            # Use this trick to skip that call.
            # Don't call super method at first, it will clear the log_buffer
            return super(RSIDetWandbHook, self).after_train_iter(runner)
        else:
            super(RSIDetWandbHook, self).after_train_iter(runner)

        if self.by_epoch:
            return

        if self.every_n_iters(runner, self.interval):
            results = runner.outputs
            if 'states' in results:
                self.process_vis_data(runner, results['states'])

        # Save checkpoint and metadata
        if (self.log_checkpoint
                and self.every_n_iters(runner, self.ckpt_interval)
                or (self.ckpt_hook.save_last and self.is_last_iter(runner))):
            if self.log_checkpoint_metadata and self.eval_hook:
                metadata = {
                    'iter': runner.iter + 1,
                    **self._get_eval_results()
                }
            else:
                metadata = None
            aliases = [f'iter_{runner.iter + 1}', 'latest']
            model_path = osp.join(self.ckpt_hook.out_dir,
                                  f'iter_{runner.iter + 1}.pth')
            self._log_ckpt_as_artifact(model_path, aliases, metadata)

        # Save prediction table
        if self.log_evaluation and self.eval_hook._should_evaluate(runner):
            results = self.eval_hook.latest_results
            if type(results[0]) != dict:
                # Initialize evaluation table
                self._init_pred_table()
                # Log predictions
                self._log_predictions(results)
                # Log the table
                self._log_eval_table(runner.iter + 1)
            else:
                for key in results[0].keys():
                    self._init_pred_table()
                    class_id_to_label = None
                    if key.startswith('proposal'):
                        class_id_to_label = {1: 'proposal', 2: 'background'}

                    cur_results = [result[key] for result in results]
                    self._log_predictions(cur_results, class_id_to_label=class_id_to_label)
                    self._log_eval_table(runner.iter + 1, table_key=key, with_artifact=False)

    def _update_wandb_config(self, runner):
        """Update wandb config."""
        # Import the config file.
        sys.path.append(runner.work_dir)
        config_filename = runner.meta['exp_name'][:-3]
        configs = importlib.import_module(config_filename)
        # Prepare a nested dict of config variables.
        config_keys = [key for key in dir(configs) if not key.startswith('__')]
        config_dict = {key: getattr(configs, key) for key in config_keys}
        # Update the W&B config.
        self.wandb.config.update(config_dict)

    def _log_ckpt_as_artifact(self, model_path, aliases, metadata=None):
        """Log model checkpoint as  W&B Artifact.

        Args:
            model_path (str): Path of the checkpoint to log.
            aliases (list): List of the aliases associated with this artifact.
            metadata (dict, optional): Metadata associated with this artifact.
        """
        model_artifact = self.wandb.Artifact(
            f'run_{self.wandb.run.id}_model', type='model', metadata=metadata)
        model_artifact.add_file(model_path)
        self.wandb.log_artifact(model_artifact, aliases=aliases)

    def _get_eval_results(self):
        """Get model evaluation results."""
        results = self.eval_hook.latest_results
        eval_results = self.val_dataset.evaluate(
            results, logger='silent', **self.eval_hook.eval_kwargs)
        return eval_results

    def _init_data_table(self):
        """Initialize the W&B Tables for validation data."""
        columns = ['image_name', 'image']
        self.data_table = self.wandb.Table(columns=columns)

    def _init_pred_table(self):
        """Initialize the W&B Tables for model evaluation."""
        columns = ['image_name', 'ground_truth', 'prediction']
        self.eval_table = self.wandb.Table(columns=columns)


    def _add_ground_truth(self):

        # Select the images to be logged.
        self.eval_image_indexs = np.arange(len(self.val_dataset))
        # Set seed so that same validation set is logged each time.
        np.random.seed(42)
        np.random.shuffle(self.eval_image_indexs)
        self.eval_image_indexs = self.eval_image_indexs[:self.num_eval_images]

        CLASSES = self.val_dataset.CLASSES
        self.class_id_to_label = {
            id + 1: name
            for id, name in enumerate(CLASSES)
        }
        self.class_set = self.wandb.Classes([{
            'id': id,
            'name': name
        } for id, name in self.class_id_to_label.items()])

        for idx in self.eval_image_indexs:
            img_info = self.val_dataset.data_infos[idx]
            image_name = img_info.get('filename', f'img_{idx}')
            img_height, img_width = img_info['height'], img_info['width']

            image = img_loader(os.path.join(self.val_dataset.img_prefix, image_name))

            # Get image and convert from BGR to RGB
            # image = mmcv.bgr2rgb(img_meta['img'])

            data_ann = self.val_dataset.get_ann_info(idx)
            bboxes = data_ann['bboxes']
            labels = data_ann['labels']
            masks = data_ann.get('masks', None)
            if self.without_mask:
                masks = None

            # Get dict of bounding boxes to be logged.
            assert len(bboxes) == len(labels)
            wandb_boxes = self._get_wandb_bboxes(bboxes, labels)

            # Get dict of masks to be logged.
            if masks is not None:
                wandb_masks = self._get_wandb_masks(
                    masks,
                    labels,
                    is_poly_mask=True,
                    height=img_height,
                    width=img_width)
            else:
                wandb_masks = None
            # TODO: Panoramic segmentation visualization.

            # Log a row to the data table.
            self.data_table.add_data(
                image_name,
                self.wandb.Image(
                    image,
                    boxes=wandb_boxes,
                    masks=wandb_masks,
                    classes=self.class_set))

    def _log_predictions(self, results, class_id_to_label=None):
        if class_id_to_label is None:
            class_id_to_label = self.class_id_to_label

        table_idxs = self.data_table_ref.get_index()
        assert len(table_idxs) == len(self.eval_image_indexs)

        for ndx, eval_image_index in enumerate(self.eval_image_indexs):
            # Get the result
            result = results[eval_image_index]
            if isinstance(result, tuple):
                bbox_result, segm_result = result
                if isinstance(segm_result, tuple):
                    segm_result = segm_result[0]  # ms rcnn
            else:
                bbox_result, segm_result = result, None

            assert len(bbox_result) == len(class_id_to_label)

            # Get labels
            bboxes = np.vstack(bbox_result)
            labels = [
                np.full(bbox.shape[0], i, dtype=np.int32)
                for i, bbox in enumerate(bbox_result)
            ]
            labels = np.concatenate(labels)

            # Get segmentation mask if available.
            segms = None
            if segm_result is not None and len(labels) > 0:
                segms = mmcv.concat_list(segm_result)
                segms = mask_util.decode(segms)
                segms = segms.transpose(2, 0, 1)
                assert len(segms) == len(labels)
            # TODO: Panoramic segmentation visualization.

            # Remove bounding boxes and masks with score lower than threshold.
            if type(self.bbox_score_thr) == list or self.bbox_score_thr > 0:
                assert bboxes is not None and bboxes.shape[1] == 5
                scores = bboxes[:, -1]
                inds = scores > np.array(self.bbox_score_thr)[labels]
                bboxes = bboxes[inds, :]
                labels = labels[inds]
                if segms is not None:
                    segms = segms[inds, ...]

            # Get dict of bounding boxes to be logged.
            wandb_boxes = self._get_wandb_bboxes(bboxes, labels, log_gt=False,
                                                 class_id_to_label=class_id_to_label)
            # Get dict of masks to be logged.
            if segms is not None:
                wandb_masks = self._get_wandb_masks(segms, labels)
            else:
                wandb_masks = None

            class_set = self.wandb.Classes([{
                'id': id,
                'name': name
            } for id, name in class_id_to_label.items()])
            # Log a row to the eval table.
            self.eval_table.add_data(
                self.data_table_ref.data[ndx][0],
                self.data_table_ref.data[ndx][1],
                self.wandb.Image(
                    self.data_table_ref.data[ndx][1],
                    boxes=wandb_boxes,
                    masks=wandb_masks,
                    classes=class_set))

    def _get_wandb_bboxes(self, bboxes, labels, log_gt=True, class_id_to_label=None):
        """Get list of structured dict for logging bounding boxes to W&B.

        Args:
            bboxes (list): List of bounding box coordinates in
                        (minX, minY, maxX, maxY) format.
            labels (int): List of label ids.
            log_gt (bool): Whether to log ground truth or prediction boxes.

        Returns:
            Dictionary of bounding boxes to be logged.
        """
        if class_id_to_label is None:
            class_id_to_label = self.class_id_to_label

        wandb_boxes = {}

        box_data = []
        for bbox, label in zip(bboxes, labels):
            if not isinstance(label, int):
                label = int(label)
            label = label + 1

            if len(bbox) == 5:
                confidence = float(bbox[4])
                class_name = class_id_to_label[label]
                box_caption = f'{class_name} {confidence:.2f}'
            else:
                box_caption = str(class_id_to_label[label])

            position = dict(
                minX=int(bbox[0]),
                minY=int(bbox[1]),
                maxX=int(bbox[2]),
                maxY=int(bbox[3]))

            box_data.append({
                'position': position,
                'class_id': label,
                'box_caption': box_caption,
                'domain': 'pixel'
            })

        wandb_bbox_dict = {
            'box_data': box_data,
            'class_labels': class_id_to_label
        }

        if log_gt:
            wandb_boxes['ground_truth'] = wandb_bbox_dict
        else:
            wandb_boxes['predictions'] = wandb_bbox_dict

        return wandb_boxes

    def _get_wandb_masks(self,
                         masks,
                         labels,
                         is_poly_mask=False,
                         height=None,
                         width=None):
        """Get list of structured dict for logging masks to W&B.

        Args:
            masks (list): List of masks.
            labels (int): List of label ids.
            is_poly_mask (bool): Whether the mask is polygonal or not.
                This is true for CocoDataset.
            height (int): Height of the image.
            width (int): Width of the image.

        Returns:
            Dictionary of masks to be logged.
        """
        mask_label_dict = dict()
        for mask, label in zip(masks, labels):
            label = label + 1
            # Get bitmap mask from polygon.
            if is_poly_mask:
                if height is not None and width is not None:
                    from rsidet.core.mask.structures import polygon_to_bitmap
                    mask = polygon_to_bitmap(mask, height, width)
            # Create composite masks for each class.
            if label not in mask_label_dict.keys():
                mask_label_dict[label] = mask
            else:
                mask_label_dict[label] = np.logical_or(mask_label_dict[label],
                                                       mask)

        wandb_masks = dict()
        for key, value in mask_label_dict.items():
            # Create mask for that class.
            value = value.astype(np.uint8)
            value[value > 0] = key

            # Create dict of masks for logging.
            class_name = self.class_id_to_label[key]
            wandb_masks[class_name] = {
                'mask_data': value,
                'class_labels': self.class_id_to_label
            }

        return wandb_masks

    def _log_data_table(self):
        """Log the W&B Tables for validation data as artifact and calls
        `use_artifact` on it so that the evaluation table can use the reference
        of already uploaded images.

        This allows the data to be uploaded just once.
        """
        data_artifact = self.wandb.Artifact('val', type='dataset')
        data_artifact.add(self.data_table, 'val_data')

        self.wandb.run.use_artifact(data_artifact)
        data_artifact.wait()

        self.data_table_ref = data_artifact.get('val_data')

    def _log_eval_table(self, idx, with_artifact=True, table_key='table'):
        """Log the W&B Tables for model evaluation.

        The table will be logged multiple times creating new version. Use this
        to compare models at different intervals interactively.
        """
        if with_artifact:
            pred_artifact = self.wandb.Artifact(
                f'run_{self.wandb.run.id}_pred', type='evaluation')
            pred_artifact.add(self.eval_table, 'eval_data')
            if self.by_epoch:
                aliases = ['latest', f'epoch_{idx}']
            else:
                aliases = ['latest', f'iter_{idx}']
            self.wandb.run.log_artifact(pred_artifact, aliases=aliases)
        else:
            self.wandb.log({table_key: self.eval_table})
