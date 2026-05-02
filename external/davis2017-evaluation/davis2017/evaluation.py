import sys
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import numpy as np
from davis2017.davis import DAVIS
from davis2017.metrics import db_eval_boundary, db_eval_iou
from davis2017 import utils
from davis2017.results import Results
from scipy.optimize import linear_sum_assignment


class DAVISEvaluation(object):
    def __init__(self, davis_root, task, gt_set, sequences='all', codalab=False):
        """
        Class to evaluate DAVIS sequences from a certain set and for a certain task
        :param davis_root: Path to the DAVIS folder that contains JPEGImages, Annotations, etc. folders.
        :param task: Task to compute the evaluation, chose between semi-supervised or unsupervised.
        :param gt_set: Set to compute the evaluation
        :param sequences: Sequences to consider for the evaluation, 'all' to use all the sequences in a set.
        """
        self.davis_root = davis_root
        self.task = task
        self.dataset = DAVIS(root=davis_root, task=task, subset=gt_set, sequences=sequences, codalab=codalab)

    @staticmethod
    def _evaluate_semisupervised(all_gt_masks, all_res_masks, all_void_masks, metric, merge_gt_objects=True):
        """Evaluate semi-supervised task.
        
        Args:
            merge_gt_objects: If True, merge all GT object masks into a single binary mask.
                This treats all objects as one unified mask.
        """
        if merge_gt_objects:
            # Merge all GT object masks: any pixel belonging to any object is foreground
            merged_gt = np.any(all_gt_masks, axis=0)  # (T, H, W)
            all_gt_masks = merged_gt[np.newaxis, ...]  # Add object dimension back -> (1, T, H, W)
            # Adjust res_masks to match: use first prediction or handle appropriately
            if all_res_masks.shape[0] > 1:
                # Multiple predictions - just use the first one since GT is now single mask
                all_res_masks = all_res_masks[:1]
        
        if all_res_masks.shape[0] > all_gt_masks.shape[0]:
            sys.stdout.write("\nIn your PNG files there is an index higher than the number of objects in the sequence!")
            sys.exit()
        elif all_res_masks.shape[0] < all_gt_masks.shape[0]:
            zero_padding = np.zeros((all_gt_masks.shape[0] - all_res_masks.shape[0], *all_res_masks.shape[1:]))
            all_res_masks = np.concatenate([all_res_masks, zero_padding], axis=0)
        j_metrics_res, f_metrics_res = np.zeros(all_gt_masks.shape[:2]), np.zeros(all_gt_masks.shape[:2])
        for ii in range(all_gt_masks.shape[0]):
            if 'J' in metric:
                j_metrics_res[ii, :] = db_eval_iou(all_gt_masks[ii, ...], all_res_masks[ii, ...], all_void_masks)
            if 'F' in metric:
                f_metrics_res[ii, :] = db_eval_boundary(all_gt_masks[ii, ...], all_res_masks[ii, ...], all_void_masks)
        return j_metrics_res, f_metrics_res

    @staticmethod
    def _evaluate_unsupervised(all_gt_masks, all_res_masks, all_void_masks, metric, max_n_proposals=20, merge_gt_objects=True):
        """Evaluate unsupervised task.
        
        Args:
            merge_gt_objects: If True, merge all GT object masks into a single binary mask.
                This treats all objects as one unified mask and simplifies matching.
        """
        if merge_gt_objects:
            # Merge all GT object masks: any pixel belonging to any object is foreground
            merged_gt = np.any(all_gt_masks, axis=0)  # (T, H, W)
            all_gt_masks = merged_gt[np.newaxis, ...]  # (1, T, H, W)
            # For unsupervised, we need to handle the matching differently
            # Since GT is now single mask, we just evaluate each proposal against it
            j_metrics_res = np.zeros((all_res_masks.shape[0], 1, all_gt_masks.shape[2]))
            f_metrics_res = np.zeros((all_res_masks.shape[0], 1, all_gt_masks.shape[2]))
            for jj in range(all_res_masks.shape[0]):
                if 'J' in metric:
                    j_metrics_res[jj, 0, :] = db_eval_iou(all_gt_masks[0, ...], all_res_masks[jj, ...], all_void_masks)
                if 'F' in metric:
                    f_metrics_res[jj, 0, :] = db_eval_boundary(all_gt_masks[0, ...], all_res_masks[jj, ...], all_void_masks)
            # Return the best matching proposal (or average if needed)
            # Here we just return all results and let the caller handle it
            return j_metrics_res[:, 0, :], f_metrics_res[:, 0, :]
        
        # Original code for when not merging
        if all_res_masks.shape[0] > max_n_proposals:
            sys.stdout.write(f"\nIn your PNG files there is an index higher than the maximum number ({max_n_proposals}) of proposals allowed!")
            sys.exit()
        elif all_res_masks.shape[0] < all_gt_masks.shape[0]:
            zero_padding = np.zeros((all_gt_masks.shape[0] - all_res_masks.shape[0], *all_res_masks.shape[1:]))
            all_res_masks = np.concatenate([all_res_masks, zero_padding], axis=0)
        j_metrics_res = np.zeros((all_res_masks.shape[0], all_gt_masks.shape[0], all_gt_masks.shape[1]))
        f_metrics_res = np.zeros((all_res_masks.shape[0], all_gt_masks.shape[0], all_gt_masks.shape[1]))
        for ii in range(all_gt_masks.shape[0]):
            for jj in range(all_res_masks.shape[0]):
                if 'J' in metric:
                    j_metrics_res[jj, ii, :] = db_eval_iou(all_gt_masks[ii, ...], all_res_masks[jj, ...], all_void_masks)
                if 'F' in metric:
                    f_metrics_res[jj, ii, :] = db_eval_boundary(all_gt_masks[ii, ...], all_res_masks[jj, ...], all_void_masks)
        if 'J' in metric and 'F' in metric:
            all_metrics = (np.mean(j_metrics_res, axis=2) + np.mean(f_metrics_res, axis=2)) / 2
        else:
            all_metrics = np.mean(j_metrics_res, axis=2) if 'J' in metric else np.mean(f_metrics_res, axis=2)
        row_ind, col_ind = linear_sum_assignment(-all_metrics)
        return j_metrics_res[row_ind, col_ind, :], f_metrics_res[row_ind, col_ind, :]

    def evaluate(self, res_path, metric=('J', 'F'), debug=False, merge_gt_objects=True):
        """Evaluate DAVIS sequences.
        
        Args:
            res_path: Path to prediction results
            metric: Metrics to compute ('J', 'F', or both)
            debug: Whether to print debug info
            merge_gt_objects: If True, merge all GT object masks into a single binary mask.
                This treats all objects as one unified mask for evaluation.
        """
        metric = metric if isinstance(metric, tuple) or isinstance(metric, list) else [metric]
        if 'T' in metric:
            raise ValueError('Temporal metric not supported!')
        if 'J' not in metric and 'F' not in metric:
            raise ValueError('Metric possible values are J for IoU or F for Boundary')

        # Containers
        metrics_res = {}
        if 'J' in metric:
            metrics_res['J'] = {"M": [], "R": [], "D": [], "M_per_object": {}}
        if 'F' in metric:
            metrics_res['F'] = {"M": [], "R": [], "D": [], "M_per_object": {}}

        # Sweep all sequences
        results = Results(root_dir=res_path)
        for seq in tqdm(list(self.dataset.get_sequences())):
            all_gt_masks, all_void_masks, all_masks_id = self.dataset.get_all_masks(seq, True)
            if self.task == 'semi-supervised':
                all_gt_masks, all_masks_id = all_gt_masks[:, 1:-1, :, :], all_masks_id[1:-1]
            all_res_masks = results.read_masks(seq, all_masks_id)
            
            if merge_gt_objects:
                # Merge all GT object masks into a single binary mask
                merged_gt = np.any(all_gt_masks, axis=0)  # (T, H, W)
                all_gt_masks = merged_gt[np.newaxis, ...]  # (1, T, H, W)
                # For predictions, use the first one or average if multiple
                if all_res_masks.shape[0] > 1:
                    all_res_masks = all_res_masks[:1]
                # Evaluate without Hungarian matching since GT is now single mask
                j_metrics_res = np.zeros((1, all_gt_masks.shape[1]))
                f_metrics_res = np.zeros((1, all_gt_masks.shape[1]))
                for ii in range(all_gt_masks.shape[1]):  # Iterate over frames
                    if 'J' in metric:
                        j_metrics_res[0, ii] = db_eval_iou(all_gt_masks[0, ii, ...], all_res_masks[0, ii, ...], all_void_masks)
                    if 'F' in metric:
                        f_metrics_res[0, ii] = db_eval_boundary(all_gt_masks[0, ii, ...], all_res_masks[0, ii, ...], all_void_masks)
            else:
                # Original evaluation with Hungarian matching
                if self.task == 'unsupervised':
                    j_metrics_res, f_metrics_res = self._evaluate_unsupervised(all_gt_masks, all_res_masks, all_void_masks, metric)
                elif self.task == 'semi-supervised':
                    j_metrics_res, f_metrics_res = self._evaluate_semisupervised(all_gt_masks, all_res_masks, None, metric, merge_gt_objects=False)
            
            n_objects = all_gt_masks.shape[0]
            for ii in range(n_objects):
                seq_name = f'{seq}_{ii+1}' if not merge_gt_objects else f'{seq}_merged'
                if 'J' in metric:
                    [JM, JR, JD] = utils.db_statistics(j_metrics_res[ii])
                    metrics_res['J']["M"].append(JM)
                    metrics_res['J']["R"].append(JR)
                    metrics_res['J']["D"].append(JD)
                    metrics_res['J']["M_per_object"][seq_name] = JM
                if 'F' in metric:
                    [FM, FR, FD] = utils.db_statistics(f_metrics_res[ii])
                    metrics_res['F']["M"].append(FM)
                    metrics_res['F']["R"].append(FR)
                    metrics_res['F']["D"].append(FD)
                    metrics_res['F']["M_per_object"][seq_name] = FM

            # Show progress
            if debug:
                sys.stdout.write(seq + '\n')
                sys.stdout.flush()
        return metrics_res
