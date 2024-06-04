from __future__ import annotations

import warnings
from collections.abc import Sequence
import copy
import fastremap
import numpy as np
import pandas as pd
import pingouin
from scipy import ndimage
from sklearn.metrics import cohen_kappa_score
from tqdm.autonotebook import tqdm

import torch
from einops import rearrange

from monai.metrics.utils import do_metric_reduction, ignore_background
from monai.utils import MetricReduction, ensure_tuple
from monai.metrics import CumulativeIterationMetric
from monai.metrics.confusion_matrix import compute_confusion_matrix_metric, get_confusion_matrix, ConfusionMatrixMetric
from monai.networks.utils import one_hot
from monai.utils.type_conversion import convert_data_type, convert_to_dst_type, convert_to_tensor, get_equivalent_dtype

from monai.transforms import KeepLargestConnectedComponent
from monai.transforms.utils_pytorch_numpy_unification import (
    argwhere,
    concatenate,
    cumsum,
    stack,
    unique,
    where,
)

class ProbConfusionMatrixMetric(ConfusionMatrixMetric):
    """
    nonbinarized predictions. otherwise, same as monai.metrics.ConfusionMatrixMetrics
    """
    def __init__(
        self,
        include_background: bool = True,
        metric_name: Sequence[str] | str = "hit_rate",
        compute_sample: bool = False,
        reduction: MetricReduction | str = MetricReduction.MEAN,
        get_not_nans: bool = False,
    ) -> None:
        super().__init__()
        self.include_background = include_background
        self.metric_name = ensure_tuple(metric_name)
        self.compute_sample = compute_sample
        self.reduction = reduction
        self.get_not_nans = get_not_nans
    
    def _compute_tensor(self, y_pred: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:  # type: ignore[override]
        dims = y_pred.ndimension()
        if dims < 2:
            raise ValueError("y_pred should have at least two dimensions.")
        if dims == 2 or (dims == 3 and y_pred.shape[-1] == 1):
            if self.compute_sample:
                warnings.warn("As for classification task, compute_sample should be False.")
                self.compute_sample = False

        y_pred_bin = one_hot(y_pred.argmax(1, keepdim=True), y_pred.shape[1])

        return get_confusion_matrix(y_pred=y_pred_bin, y=y, include_background=self.include_background)

        
def get_longest_consecutive_positive(x, axis=-1):
    _np = isinstance(x, (np.ndarray, list))
    nd = x.ndim
    assert nd <= 2
    
    osh = list(x.shape)
    osh[-1] = 1
    osh = tuple(osh)
    
    slice0 = [slice(None)] * nd
    slice1 = [slice(None)] * nd
    slice2 = [slice(None)] * nd
    slice0[axis] = slice(0,1)
    slice1[axis] = slice(None, -1)
    slice2[axis] = slice(1, None)
    slice0 = tuple(slice0)
    slice1 = tuple(slice1)
    slice2 = tuple(slice2)    

    x1 = concatenate((x[slice0], x[slice1] != x[slice2], x[slice0]==x[slice0]), axis=axis)
    wx = argwhere(x1)

    idxlist = list(np.arange(nd))
    
    if nd == 1:
        cnt_wx = []
    else:
        if _np:
            uni_wx, cnt_wx = unique(wx[:,np.array([z for z in idxlist if z != idxlist[axis]])], return_counts=True, axis=0)
        else:
            uni_wx, cnt_wx = unique(wx[:,np.array([z for z in idxlist if z != idxlist[axis]])], return_counts=True, dim=0)
        cnt_wx = cumsum(cnt_wx)[:-1]

    max_wx = []
    for z in np.split(wx[:,np.array([idxlist[axis]])], cnt_wx):
        if len(z) > 1:
            max_wx.append((z[1:] - z[:-1])[::2].max())
        else:
            max_wx.append(0 if _np else torch.tensor(0))
    max_wx = np.array(max_wx) if _np else torch.tensor(max_wx)
    max_wx = max_wx.reshape(osh)
    return max_wx
        
def psg_event_analysis(tensor_preds, tensor_label, threshold_consecutive=80, num_points=51, progress=False):
    tps = np.zeros((num_points, 1))
    fps = np.zeros((num_points, 1))
    pos = np.zeros((num_points, 1))
    neg = np.zeros((num_points, 1))
    
    roc_thrs = np.linspace(0, 1, num_points)

    for i in tqdm(range(len(roc_thrs)), disable=not progress):
        roc_thr = roc_thrs[i]    
        cpreds = 1 - tensor_preds[:,0]
        cpreds = get_longest_consecutive_positive(cpreds >= roc_thr)
        cpreds = cpreds >= threshold_consecutive
        #cpreds = keeplargeCC(cpreds >= roc_thr)
        #cpreds = cpreds.sum(1) >= threshold_consecutive
        clabel = 1 - tensor_label[:,0]
        clabel = get_longest_consecutive_positive(clabel)
        clabel = clabel >= threshold_consecutive
        #clabel = keeplargeCC(clabel)
        #clabel = clabel.sum(1) >= threshold_consecutive
        
        preds_onehot = cpreds.detach().cpu().numpy()
        label_onehot = clabel.detach().cpu().numpy()
        tps[i] += (preds_onehot * label_onehot).sum()
        fps[i] += (preds_onehot * (1 - label_onehot)).sum()
        pos[i] += label_onehot.sum()
        neg[i] += (1 - label_onehot).sum()
        
    return tps, fps, pos, neg, roc_thrs

def channelwise_get_longest_consecutive_positive(x):
    """input shape (batch, channel, 1dsize)"""
    x = rearrange(x, "b c h -> c b h")
    x = torch.stack([get_longest_consecutive_positive(xx) for xx in x])
    x = rearrange(x, "c b h -> b c h")
    return x

class PSGConfusionMatrixMetric(CumulativeIterationMetric):
    """
    one-hot format (non-binarized) y_pred and y's.
    """
    def __init__(
        self,
        include_background: bool = True,
        metric_name: Sequence[str] | str = "hit_rate",
        compute_sample: bool = False,
        reduction: MetricReduction | str = MetricReduction.MEAN,
        get_not_nans: bool = False,
        num_thresholds: int = 11,
        threshold_consecutive: int = 80,
        best_threshold_type: str = 'f1',
    ) -> None:
        super().__init__()
        self.include_background = include_background
        self.metric_name = ensure_tuple(metric_name)
        self.compute_sample = compute_sample
        self.reduction = reduction
        self.get_not_nans = get_not_nans
        self.num_thresholds = num_thresholds
        self.list_thresholds = np.linspace(0, 1, num_thresholds)
        self.threshold_consecutive = threshold_consecutive
        self.best_threshold_type = best_threshold_type
    
    def _compute_tensor(self, y_pred: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:  # type: ignore[override]
        # check dimension
        dims = y_pred.ndimension()
        if dims < 2:
            raise ValueError("y_pred should have at least two dimensions.")
        if dims == 2 or (dims == 3 and y_pred.shape[-1] == 1):
            if self.compute_sample:
                warnings.warn("As for classification task, compute_sample should be False.")
                self.compute_sample = False
        
        if not self.include_background:
            y_pred, y = ignore_background(y_pred=y_pred, y=y)
        
        psg_preds = torch.cat([channelwise_get_longest_consecutive_positive(y_pred >= t) >= self.threshold_consecutive for t in self.list_thresholds], -1)
        psg_label = torch.cat([channelwise_get_longest_consecutive_positive(y >= t) >= self.threshold_consecutive for t in self.list_thresholds], -1)
        return psg_preds, psg_label
    
    def aggregate(
        self, compute_sample: bool = False, reduction: MetricReduction | str | None = None
    ) -> list[torch.Tensor | tuple[torch.Tensor, torch.Tensor]]:
        y_pred, y = self.get_buffer()
        y_pred = rearrange(y_pred.float(), "b c t -> t b c")
        y = rearrange(y.float(), "b c t -> t b c")
        
        tps = y_pred * y
        fps = y_pred * (1 - y)
        tns = (1 - y_pred) * (1 - y)
        fns = (1 - y_pred) * y
        
        thr_confusion_matrix = torch.stack([tps, fps, tns, fns], dim=-1)
        best_thr_search = []
        for x in thr_confusion_matrix:
            f, not_nans = do_metric_reduction(x, self.reduction)
            f = compute_confusion_matrix_metric(self.best_threshold_type, f)
            best_thr_search.append(f.item())
        best_thr_idx = np.argmax(best_thr_search)
        
        data = thr_confusion_matrix[best_thr_idx]
        if not isinstance(data, torch.Tensor):
            raise ValueError("the data to aggregate must be PyTorch Tensor.")

        results: list[torch.Tensor | tuple[torch.Tensor, torch.Tensor]] = []
        for metric_name in self.metric_name:
            if compute_sample or self.compute_sample:
                sub_confusion_matrix = compute_confusion_matrix_metric(metric_name, data)
                f, not_nans = do_metric_reduction(sub_confusion_matrix, reduction or self.reduction)
            else:
                f, not_nans = do_metric_reduction(data, reduction or self.reduction)
                f = compute_confusion_matrix_metric(metric_name, f)
            if self.get_not_nans:
                results.append((f, not_nans))
            else:
                results.append(f)
        return results
    
    
def apply_valid_consecutive(x, threshold_consecutive=5):
    """input shape: (batch, channel, size)
    """
    b, c = x.shape[:2]
    for i in range(b):
        for j in range(c):
            label_im, nb_labels = ndimage.label(x[i,j])
            remap_dict = {}
            for k in range(nb_labels):
                if fastremap.foreground(label_im == k+1) < threshold_consecutive:
                    remap_dict[k+1] = 0
            label_im = fastremap.remap(label_im, remap_dict, preserve_missing_labels=True, in_place=True)
            x[i,j][label_im==0] = 0
    return x

def _get_tp_fp_fn(y_pred, y, iou_thr: Sequence[float] | float = 0.5):
    iou_thr = ensure_tuple(iou_thr)
    
    y_pred_im, nb_labels_pred = ndimage.label(y_pred)
    y_im, nb_labels = ndimage.label(y)    

    if nb_labels_pred == 0 or nb_labels == 0:
        tp = 0
        fp = nb_labels_pred
        fn = nb_labels
        result = np.array([[tp, fp, fn],]*len(iou_thr))        
        return convert_to_dst_type(result, y_pred, dtype=float)[0]
    
    Y1 = np.array([y_pred_im == i for i in range(1, nb_labels_pred+1)], dtype=float)
    Y2 = np.array([y_im == i for i in range(1, nb_labels+1)], dtype=float)
    YDOT = np.matmul(Y1, Y2.T)
    YSUM = np.stack([Y1.sum(1),]*Y2.shape[0], -1) + np.stack([Y2.sum(1),]*Y1.shape[0], 0)
    YOR = YSUM - YDOT
    iou_grid = YDOT / (YOR + 1e-5)

    N = (iou_grid > 0).sum()
    if N == 0:
        tp = 0
        fp = nb_labels_pred
        fn = nb_labels
        result = np.array([[tp, fp, fn],]*len(iou_thr))   
        return convert_to_dst_type(result, y_pred, dtype=float)[0]
    
    max_ious = np.sort(np.partition(np.asarray(iou_grid), iou_grid.size - N, axis=None)[-N:])[::-1]
    for max_iou in max_ious:
        _idx = np.argwhere(iou_grid == max_iou)
        for i,j in _idx:
            iou_grid[i,np.arange(iou_grid.shape[1])] = 0
            iou_grid[np.arange(iou_grid.shape[0]),j] = 0
            iou_grid[i,j] = max_iou

    result = []
    for _thr in iou_thr:
        tp = (iou_grid >= _thr).sum()
        fp = nb_labels_pred - tp
        fn = nb_labels - tp
        result.append([tp, fp, fn])
    result = np.array(result)
    return convert_to_dst_type(result, y_pred, dtype=float)[0]

def get_detect_matrix(y_pred, y, iou_thr: Sequence[float] | float = 0.5):
    """y_pred, y shape: (batch, channel, size) binarized"""
    iou_thr = ensure_tuple(iou_thr)
    batch, channel = y_pred.shape[:2]
    mat = torch.zeros((batch, channel, len(iou_thr), 3))
    for i in range(batch):
        for j in range(channel):
            _mini_mat = _get_tp_fp_fn(y_pred[i,j], y[i,j], iou_thr = iou_thr)
            mat[i,j] = _mini_mat
    return mat

def _get_tp_fp_fn_V2(y_pred, y, iou_thr: Sequence[float] | float = 0.5, consecutive_thr: float = 0.):
    iou_thr = ensure_tuple(iou_thr)
    result = []
    for _iou_thr in iou_thr:
        y_im, nb_labels = ndimage.label(y)    
        gtlabel_set, gtlabel_cnt = np.unique(y_im, return_counts=True)
        if 0 in gtlabel_set:
            idx0 = np.argwhere(gtlabel_set==0)[0]
            gtlabel_set = np.array([x for i,x in enumerate(gtlabel_set) if i != idx0])
            gtlabel_cnt = np.array([x for i,x in enumerate(gtlabel_cnt) if i != idx0])
        gtlabel_sorted = [gtlabel_set[i] for i in gtlabel_cnt.argsort()[::-1]]
        nb_labels_consecutive = (gtlabel_cnt >= consecutive_thr).sum()

        y_pred_im, nb_labels_pred = ndimage.label(y_pred)
        pdlabel_set, pdlabel_cnt = np.unique(y_pred_im, return_counts=True)
        if 0 in pdlabel_set:
            idx0 = np.argwhere(pdlabel_set==0)[0]
            pdlabel_cnt = np.array([x for i,x in enumerate(pdlabel_cnt) if i != idx0])
        nb_labels_pred_consecutive = (pdlabel_cnt >= consecutive_thr).sum()

        tp_match = []
        for i in gtlabel_sorted:
            intersect = (y_im == i) * (y_pred_im > 0)
            intersect_pred_labels = y_pred_im[intersect]
            plabel_set, plabel_cnt = np.unique(intersect_pred_labels, return_counts=True)
            
            area_y = (y_im == i).sum()
            if area_y < consecutive_thr:
                tp_match.append(np.nan)
                continue
            
            this_ious = []
            for j, cnt in zip(plabel_set, plabel_cnt):
                area_p = (y_pred_im == j).sum()
                area_inter = cnt
                if area_p < consecutive_thr:
                    this_ious.append(0.)
                else:
                    this_ious.append(area_inter/(area_y + area_p - area_inter + 1e-5))
            
            if len(this_ious) == 0:
                tp_match.append(np.nan)
                continue
            
            _idx = np.argmax(this_ious)
            max_iou = this_ious[_idx]
            cnt_max = len(np.argwhere(np.array(this_ious)==max_iou))
            if cnt_max > 1: print(i, cnt_max)
            if max_iou >= _iou_thr:
                match_j = plabel_set[_idx]
                tp_match.append(match_j)
                y_pred_im[y_pred_im == match_j] = 0
            else:
                tp_match.append(np.nan)
                
        tp = (~np.isnan(tp_match)).sum()
        fp = nb_labels_pred_consecutive - tp
        fn = nb_labels_consecutive - tp
        result.append([tp, fp, fn])
    result = np.array(result)
    return convert_to_dst_type(result, y_pred, dtype=float)[0]

def get_detect_matrix_V2(y_pred, y, iou_thr: Sequence[float] | float = 0.5, consecutive_thr: float = 0.):
    """y_pred, y shape: (batch, channel, size) binarized"""
    iou_thr = ensure_tuple(iou_thr)
    batch, channel = y_pred.shape[:2]
    mat = torch.zeros((batch, channel, len(iou_thr), 3))
    for i in range(batch):
        for j in range(channel):
            _mini_mat = _get_tp_fp_fn_V2(y_pred[i,j], y[i,j], iou_thr = iou_thr, consecutive_thr=consecutive_thr)
            mat[i,j] = _mini_mat
    return mat

class PSGDetectionMatrixMetric(CumulativeIterationMetric):
    """
    one-hot format (non-binarized) y_pred and y's.
    """
    def __init__(
        self,
        include_background: bool = True,
        metric_name: Sequence[str] | str = "f1",
        compute_sample: bool = False,
        reduction: MetricReduction | str = MetricReduction.MEAN,
        get_not_nans: bool = False,
        pred_thresholds: Sequence[float] | None = None,
        iou_threshold: Sequence[float] | float = 0.5,
        threshold_consecutive_seconds: float = 10,
        sampling_frequency: float = 8,
        best_threshold_type: str = 'f1',
    ) -> None:
        super().__init__()
        self.include_background = include_background
        self.metric_name = ensure_tuple(metric_name)
        self.compute_sample = compute_sample
        self.reduction = reduction
        self.get_not_nans = get_not_nans
        
        self.pred_thresholds = pred_thresholds
        if pred_thresholds is None:
            self.pred_thresholds = np.linspace(0, 1, 31)
        self.iou_threshold = ensure_tuple(iou_threshold)
        self.threshold_consecutive = threshold_consecutive_seconds * sampling_frequency
        self.sampling_frequency = sampling_frequency
        self.best_threshold_type = best_threshold_type
    
    def _compute_tensor(self, y_pred: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:  # type: ignore[override]
        """return matrix shape (batch, channel, pred_thr, 3)
        3 -> tp, fp, fn
        """
        # check dimension
        dims = y_pred.ndimension()
        if dims < 2:
            raise ValueError("y_pred should have at least two dimensions.")
        if dims == 2 or (dims == 3 and y_pred.shape[-1] == 1):
            if self.compute_sample:
                warnings.warn("As for classification task, compute_sample should be False.")
                self.compute_sample = False
        
        if not self.include_background:
            y_pred, y = ignore_background(y_pred=y_pred, y=y)
        
        y_pred = y_pred.detach().cpu()
        y = y.detach().cpu()

        psg_matrix = torch.stack([get_detect_matrix(apply_valid_consecutive(y_pred >= t, threshold_consecutive=self.threshold_consecutive), apply_valid_consecutive(y > 0, threshold_consecutive=self.threshold_consecutive), iou_thr=self.iou_threshold) for t in self.pred_thresholds], dim=-2)
        
        y_sh = y_pred.flatten(start_dim=2).shape[2]
        
        return psg_matrix, torch.tensor([y_sh,]*y_pred.shape[0])
    
    def aggregate(
        self, compute_sample: bool = False, reduction: MetricReduction | str | None = None
    ) -> list[torch.Tensor | tuple[torch.Tensor, torch.Tensor]]:
        mat, im_size = self.get_buffer()
        mat = mat.float()
        
        tps = mat[...,0]
        fps = mat[...,1]
        fns = mat[...,2]

        im_size_hours = im_size / self.sampling_frequency / 3600
        im_size_hours = im_size_hours[(..., ) + (None,)*3]
        
        pr = (tps.sum(0) + 1e-5) / (tps.sum(0) + fps.sum(0) + 1e-5)
        rc = (tps.sum(0) + 1e-5) / (tps.sum(0) + fns.sum(0) + 1e-5)
        f1score = 2*pr*rc/(pr+rc)
                
        ahi = (tps + fns) / im_size_hours
        est_ahi = (tps + fps) / im_size_hours
        
        if self.best_threshold_type == 'ahimae':
            ahi_dif = -torch.abs(ahi - est_ahi)
            m_ahi_dif = ahi_dif.mean((0,1,2))
            _idx = m_ahi_dif.argmax().item()
            self.best_pred_threshold = self.pred_thresholds[_idx]
        else:
            m_f1 = f1score.mean((0,1))
            _idx = m_f1.argmax().item()
            self.best_pred_threshold = self.pred_thresholds[_idx]
        
        results: list[torch.Tensor | tuple[torch.Tensor, torch.Tensor]] = []
        for metric_name in self.metric_name:
            if metric_name == 'f1':
                f = f1score[...,_idx].mean((0,1))
            elif metric_name == 'precision':
                f = pr[...,_idx].mean((0,1))
            elif metric_name == 'recall':
                f = rc[...,_idx].mean((0,1))
            elif metric_name == 'ahicor':
                best_ahi = ahi[...,_idx].mean((1,2))
                best_est_ahi = est_ahi[...,_idx].mean((1,2))
                best_ahi_corrcoef = torch.corrcoef(torch.stack([best_ahi, best_est_ahi]))[0,1]
                f =  best_ahi_corrcoef
            elif metric_name == 'ahimae':
                best_ahi = ahi[...,_idx].mean((1,2))
                best_est_ahi = est_ahi[...,_idx].mean((1,2))
                best_ahi_diff = best_est_ahi - best_ahi
                f = -torch.abs(best_ahi_diff).mean()
            elif metric_name == 'ahiicc':
                best_ahi = ahi[...,_idx].mean((1,2))
                best_est_ahi = est_ahi[...,_idx].mean((1,2))                
                n_samples = best_ahi.size(0)
                if n_samples < 5:
                    f = torch.tensor(0.)
                else:
                    icc_df = pd.DataFrame(data={
                        'targets': [*np.arange(n_samples),]*2, 
                        'ratings': best_ahi.cpu().numpy().tolist() + best_est_ahi.cpu().numpy().tolist(), 
                        'raters': [1]*n_samples + [2]*n_samples,
                    })
                    icc_stat = pingouin.intraclass_corr(data=icc_df, targets='targets', raters='raters', ratings='ratings')
                    f = icc_stat[icc_stat['Type']=='ICC2']['ICC'].item()
                    f = torch.tensor(f)
            elif metric_name == 'osakappa':
                best_ahi = ahi[...,_idx].mean((1,2)).cpu().numpy()
                best_est_ahi = est_ahi[...,_idx].mean((1,2)).cpu().numpy()
                def ahi_to_osa(x):
                    if x >= 30:
                        return 3
                    elif x >= 15:
                        return 2
                    elif x >= 5:
                        return 1
                    else:
                        return 0
                osa_ahi = np.array(list(map(ahi_to_osa, best_ahi)))
                osa_est_ahi = np.array(list(map(ahi_to_osa, best_est_ahi)))
                f = cohen_kappa_score(osa_est_ahi, osa_ahi)
                f = torch.tensor(f)

            results.append(f)
        return results
    
        """
        if not isinstance(data, torch.Tensor):
            raise ValueError("the data to aggregate must be PyTorch Tensor.")

        results: list[torch.Tensor | tuple[torch.Tensor, torch.Tensor]] = []
        for metric_name in self.metric_name:
            if compute_sample or self.compute_sample:
                sub_confusion_matrix = compute_confusion_matrix_metric(metric_name, data)
                f, not_nans = do_metric_reduction(sub_confusion_matrix, reduction or self.reduction)
            else:
                f, not_nans = do_metric_reduction(data, reduction or self.reduction)
                f = compute_confusion_matrix_metric(metric_name, f)
            if self.get_not_nans:
                results.append((f, not_nans))
            else:
                results.append(f)
        return results
        """
        
    def reset(self):
        super().reset()
        self.best_pred_threshold = None

class PSGDetectionMatrixMetricV2(PSGDetectionMatrixMetric):    
    def _compute_tensor(self, y_pred: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:  # type: ignore[override]
        """return matrix shape (batch, channel, pred_thr, 3)
        3 -> tp, fp, fn
        """
        # check dimension
        dims = y_pred.ndimension()
        if dims < 2:
            raise ValueError("y_pred should have at least two dimensions.")
        if dims == 2 or (dims == 3 and y_pred.shape[-1] == 1):
            if self.compute_sample:
                warnings.warn("As for classification task, compute_sample should be False.")
                self.compute_sample = False
        
        if not self.include_background:
            y_pred, y = ignore_background(y_pred=y_pred, y=y)
        
        y_pred = y_pred.detach().cpu()
        y = y.detach().cpu()

        psg_matrix = torch.stack(
            [get_detect_matrix_V2(y_pred >= t, y > 0, iou_thr=self.iou_threshold, consecutive_thr=self.threshold_consecutive) for t in self.pred_thresholds], 
            dim=-2
        )
        
        y_sh = y_pred.flatten(start_dim=2).shape[2]
        
        return psg_matrix, torch.tensor([y_sh,]*y_pred.shape[0])

        
def count_valid_consecutive(x, threshold_consecutive=0):
    """x: (batch, channel, size)"""
    counts = np.zeros((x.shape[0], x.shape[1], 1))
    
    for b in range(x.shape[0]):
        for c in range(x.shape[1]):
            y_im, nb_labels = ndimage.label(x[b,c])
            set_labels, cnt_labels = np.unique(y_im, return_counts=True)
            counts[b,c] = (cnt_labels[1:] >= threshold_consecutive).sum()
    
    return convert_to_dst_type(counts, x, dtype=float)[0]
        
def count_valid_consecutive_stage(x, stage, threshold_consecutive=0):
    """
    x: (batch, channel, size)
    stage: (batch, 1, size)
    """
    counts = np.zeros((x.shape[0], x.shape[1], 1))
    
    for b in range(x.shape[0]):
        for c in range(x.shape[1]):
            y_im, nb_labels = ndimage.label(x[b,c])
            set_labels, cnt_labels = np.unique(y_im, return_counts=True)
            cnt = 0
            for xset, xcnt in zip(set_labels[1:], cnt_labels[1:]):
                if xcnt >= threshold_consecutive and stage[b,0][np.argwhere(y_im == xset).min()] > 0:
                    cnt += 1
            counts[b,c] = cnt
    
    return convert_to_dst_type(counts, x, dtype=float)[0]
        
def ahi_to_osa(x):
    if x >= 30:
        return 3
    elif x >= 15:
        return 2
    elif x >= 5:
        return 1
    else:
        return 0
        
class PSGAHIMetric(CumulativeIterationMetric):
    """
    one-hot format (non-binarized) y_pred and y's.
    use stage mask for postprocessing.
    """
    def __init__(
        self,
        include_background: bool = True,
        metric_name: Sequence[str] | str = "ahimae",
        compute_sample: bool = False,
        reduction: MetricReduction | str = MetricReduction.MEAN,
        get_not_nans: bool = False,
        pred_thresholds: Sequence[float] | None = None,
        threshold_consecutive_seconds: float = 10,
        sampling_frequency: float = 8,
        best_threshold_type: str = 'ahimae',
        postprocess: bool = True,
    ) -> None:
        super().__init__()
        self.include_background = include_background
        self.metric_name = ensure_tuple(metric_name)
        self.compute_sample = compute_sample
        self.reduction = reduction
        self.get_not_nans = get_not_nans
        
        self.pred_thresholds = pred_thresholds
        if pred_thresholds is None:
            self.pred_thresholds = np.linspace(0, 1, 31)
        self.threshold_consecutive = threshold_consecutive_seconds * sampling_frequency
        self.sampling_frequency = sampling_frequency
        self.best_threshold_type = best_threshold_type
        self.postprocess = postprocess
    
    def _compute_tensor(self, y_pred: torch.Tensor, y: torch.Tensor, mask: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:  # type: ignore[override]
        """
        y_pred, y : (batch, channel, size). y should be one-hot.
        mask: (batch, 1, size). mask should be binarized.
        return matrix shape 
        nevent_pred: (batch, channel, pred_thr)
        nevent_gt, tst: (batch, channel, 1)
        """
        if self.postprocess:
            ym = torch.concat([1-mask]+[mask]*(y_pred.shape[1]-1), 1)
            y_pred = y_pred * ym
        
        # check dimension
        dims = y_pred.ndimension()
        if dims < 2:
            raise ValueError("y_pred should have at least two dimensions.")
        if dims == 2 or (dims == 3 and y_pred.shape[-1] == 1):
            if self.compute_sample:
                warnings.warn("As for classification task, compute_sample should be False.")
                self.compute_sample = False
        
        if not self.include_background:
            y_pred, y = ignore_background(y_pred=y_pred, y=y)
        
        y_pred = y_pred.detach().cpu()
        y = y.detach().cpu()
        mask = mask.detach().cpu()

        cnt_pred = torch.concat([count_valid_consecutive(y_pred >= t, threshold_consecutive=self.threshold_consecutive) for t in self.pred_thresholds], dim=-1)
        cnt = count_valid_consecutive(y)        
        
        tst = mask.flatten(start_dim=2).sum(-1, keepdim=True)
                
        return cnt_pred, cnt, tst
    
    def aggregate(
        self, compute_sample: bool = False, reduction: MetricReduction | str | None = None
    ) -> list[torch.Tensor | tuple[torch.Tensor, torch.Tensor]]:
        est_nev, nev, tst = self.get_buffer()
        tst = tst / self.sampling_frequency / 3600
        est_ahi = est_nev / tst
        ahi = nev / tst        
        
        if self.best_threshold_type == 'ahimae':
            ahi_dif = -torch.abs(est_ahi - ahi)
            m_ahi_dif = ahi_dif.mean((0,1))
            _idx = m_ahi_dif.argmax().item()
            self.best_pred_threshold = self.pred_thresholds[_idx]
        elif self.best_threshold_type == 'osakappalinear':
            losa_est_ahi = est_ahi.cpu().clone().detach().mean(1).apply_(ahi_to_osa)
            losa_ahi = ahi.cpu().clone().detach().mean(1).apply_(ahi_to_osa)
            losa_kappalinear = [cohen_kappa_score(losa_est_ahi[...,i], losa_ahi[...,0], weights='linear') for i in range(losa_est_ahi.shape[-1])]
            _idx = np.argmax(losa_kappalinear)
            self.best_pred_threshold = self.pred_thresholds[_idx]
        elif self.best_threshold_type == 'osakappalinear_ahiicc':
            losa_est_ahi = est_ahi.cpu().clone().detach().mean(1).apply_(ahi_to_osa)
            losa_ahi = ahi.cpu().clone().detach().mean(1).apply_(ahi_to_osa)
            losa_kappalinear = [cohen_kappa_score(losa_est_ahi[...,i], losa_ahi[...,0], weights='linear') for i in range(losa_est_ahi.shape[-1])]
            _idx_can = np.argwhere(losa_kappalinear == max(losa_kappalinear)).ravel()
            
            list_ahiicc = []
            best_ahi = ahi.mean((1,2))
            for i in _idx_can:
                _can_est_ahi = est_ahi[..., i].mean(1)    
                n_samples = best_ahi.size(0)
                if n_samples < 5:
                    f = 0
                else:
                    icc_df = pd.DataFrame(data={
                        'targets': [*np.arange(n_samples),]*2, 
                        'ratings': best_ahi.cpu().numpy().tolist() + _can_est_ahi.cpu().numpy().tolist(), 
                        'raters': [1]*n_samples + [2]*n_samples,
                    })
                    icc_stat = pingouin.intraclass_corr(data=icc_df, targets='targets', raters='raters', ratings='ratings')
                    f = icc_stat[icc_stat['Type']=='ICC2']['ICC'].item()
                list_ahiicc.append(f)
            _idx = _idx_can[np.argmax(list_ahiicc)]
            
            self.best_pred_threshold = self.pred_thresholds[_idx]    
        else: # ahimae
            ahi_dif = -torch.abs(est_ahi - ahi)
            m_ahi_dif = ahi_dif.mean((0,1))
            _idx = m_ahi_dif.argmax().item()
            self.best_pred_threshold = self.pred_thresholds[_idx]
             
        results: list[torch.Tensor | tuple[torch.Tensor, torch.Tensor]] = []
        for metric_name in self.metric_name:
            f = None
            if metric_name == 'ahimae':
                best_est_ahi = est_ahi[..., _idx].mean(1)
                best_ahi = ahi.mean((1,2))
                f = -torch.abs(best_est_ahi - best_ahi).mean()
            elif metric_name == 'eventmae':
                best_est_nev = est_nev[..., _idx].mean(1)
                best_nev = nev.mean((1,2))
                f = -torch.abs(best_est_nev - best_nev).mean()
            elif metric_name == 'ahimape':
                best_est_ahi = est_ahi[..., _idx].mean(1)
                best_ahi = ahi.mean((1,2))
                mape = (best_est_ahi - best_ahi)/(best_ahi + 1)
                f = -torch.abs(mape).mean()
            elif metric_name == 'ahicor':
                best_est_ahi = est_ahi[..., _idx].mean(1)
                best_ahi = ahi.mean((1,2))
                f = torch.corrcoef(torch.stack([best_est_ahi, best_ahi]))[0,1]    
            elif metric_name == 'ahiicc':
                best_est_ahi = est_ahi[..., _idx].mean(1)
                best_ahi = ahi.mean((1,2))
                n_samples = best_ahi.size(0)
                if n_samples < 5:
                    f = torch.tensor(0.)
                else:
                    icc_df = pd.DataFrame(data={
                        'targets': [*np.arange(n_samples),]*2, 
                        'ratings': best_ahi.cpu().numpy().tolist() + best_est_ahi.cpu().numpy().tolist(), 
                        'raters': [1]*n_samples + [2]*n_samples,
                    })
                    icc_stat = pingouin.intraclass_corr(data=icc_df, targets='targets', raters='raters', ratings='ratings')
                    f = icc_stat[icc_stat['Type']=='ICC2']['ICC'].item()
                    f = torch.tensor(f)
            elif metric_name == 'osakappa':
                best_est_ahi = est_ahi[..., _idx].mean(1).cpu().numpy()
                best_ahi = ahi.mean((1,2)).cpu().numpy()
                
                osa_ahi = np.array(list(map(ahi_to_osa, best_ahi)))
                osa_est_ahi = np.array(list(map(ahi_to_osa, best_est_ahi)))
                f = cohen_kappa_score(osa_est_ahi, osa_ahi)
                f = torch.tensor(f)
            elif metric_name == 'osakappalinear':
                best_est_ahi = est_ahi[..., _idx].mean(1).cpu().numpy()
                best_ahi = ahi.mean((1,2)).cpu().numpy()
                
                osa_ahi = np.array(list(map(ahi_to_osa, best_ahi)))
                osa_est_ahi = np.array(list(map(ahi_to_osa, best_est_ahi)))
                f = cohen_kappa_score(osa_est_ahi, osa_ahi, weights='linear')
                f = torch.tensor(f)
            elif metric_name == 'osakappaquadratic':
                best_est_ahi = est_ahi[..., _idx].mean(1).cpu().numpy()
                best_ahi = ahi.mean((1,2)).cpu().numpy()
                
                osa_ahi = np.array(list(map(ahi_to_osa, best_ahi)))
                osa_est_ahi = np.array(list(map(ahi_to_osa, best_est_ahi)))
                f = cohen_kappa_score(osa_est_ahi, osa_ahi, weights='quadratic')
                f = torch.tensor(f)
            elif metric_name == 'osakappalinear_ahiicc0.1':
                best_est_ahi = est_ahi[..., _idx].mean(1).cpu().numpy()
                best_ahi = ahi.mean((1,2)).cpu().numpy()
                
                osa_ahi = np.array(list(map(ahi_to_osa, best_ahi)))
                osa_est_ahi = np.array(list(map(ahi_to_osa, best_est_ahi)))
                f1 = cohen_kappa_score(osa_est_ahi, osa_ahi, weights='linear')
                
                best_est_ahi = est_ahi[..., _idx].mean(1)
                best_ahi = ahi.mean((1,2))
                n_samples = best_ahi.size(0)
                if n_samples < 5:
                    f2 = 0.
                else:
                    icc_df = pd.DataFrame(data={
                        'targets': [*np.arange(n_samples),]*2, 
                        'ratings': best_ahi.cpu().numpy().tolist() + best_est_ahi.cpu().numpy().tolist(), 
                        'raters': [1]*n_samples + [2]*n_samples,
                    })
                    icc_stat = pingouin.intraclass_corr(data=icc_df, targets='targets', raters='raters', ratings='ratings')
                    f2 = icc_stat[icc_stat['Type']=='ICC2']['ICC'].item()
                    
                f = torch.tensor(f1 + 0.1*f2)
            else:
                raise NotImplementedError(f"metric {metric_name} is not implemented")
                
            results.append(f)
        return results
                                
    def reset(self):
        super().reset()
        self.best_pred_threshold = None
        
class PSGAHIMetricV2(CumulativeIterationMetric):
    """
    one-hot format (non-binarized) y_pred and y's.
    use stage mask for postprocessing V2.
        - W->W : X
        - W->sleep stage : X
        - sleep stage?->sleep stage? : O
        - sleep stage?->W : O
    sum events
    """
    def __init__(
        self,
        include_background: bool = False,
        metric_name: Sequence[str] | str = "ahimae",
        compute_sample: bool = False,
        reduction: MetricReduction | str = MetricReduction.MEAN,
        get_not_nans: bool = False,
        pred_thresholds: Sequence[Sequence[float]] | Sequence[float] | None = None,
        threshold_consecutive_seconds: float = 10,
        sampling_frequency: float = 8,
        best_threshold_type: str = 'ahimae',
        postprocess: bool = True,
    ) -> None:
        super().__init__()
        self.include_background = include_background
        self.metric_name = ensure_tuple(metric_name)
        self.compute_sample = compute_sample
        self.reduction = reduction
        self.get_not_nans = get_not_nans
        
        self.pred_thresholds = pred_thresholds
        if pred_thresholds is None:
            self.pred_thresholds = np.linspace(0, 1, 31)
        self.threshold_consecutive = threshold_consecutive_seconds * sampling_frequency
        self.sampling_frequency = sampling_frequency
        self.best_threshold_type = best_threshold_type
        self.postprocess = postprocess
    
    def _compute_tensor(self, y_pred: torch.Tensor, y: torch.Tensor, mask: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:  # type: ignore[override]
        """
        y_pred, y : (batch, channel, size). y should be one-hot.
        mask: (batch, 1, size). mask should be binarized.
        return matrix shape 
        nevent_pred: (batch, channel, pred_thr)
        nevent_gt, tst: (batch, channel, 1)
        """
        # check dimension
        dims = y_pred.ndimension()
        if dims < 3:
            raise ValueError("y_pred should have at least three dimensions.")
        
        if not self.include_background:
            y_pred, y = ignore_background(y_pred=y_pred, y=y)

        # pred thresholds for each channel
        nch = y_pred.shape[1]
        if hasattr(self.pred_thresholds[0], '__iter__'):
            pred_thresholds = self.pred_thresholds
            assert len(pred_thresholds) == nch, "number of pred thresholds should match number of (foreground) channels"
        else:
            pred_thresholds = [self.pred_thresholds] * nch
                
        y_pred = y_pred.detach().cpu()
        y = y.detach().cpu()
        if isinstance(mask, torch.Tensor):
            mask = mask.detach().cpu()

        if self.postprocess:
            cnt_pred = torch.concat([
                torch.concat([
                    count_valid_consecutive_stage(y_pred[:,i:i+1] >= t, mask, threshold_consecutive=self.threshold_consecutive) for t in pred_thresholds[i]
                ], dim=-1)
            for i in range(nch)], dim=-2)
        else:
            cnt_pred = torch.concat([
                torch.concat([
                    count_valid_consecutive(y_pred[:,i:i+1] >= t, threshold_consecutive=self.threshold_consecutive) for t in pred_thresholds[i]
                ], dim=-1)
            for i in range(nch)], dim=-2)
        cnt = count_valid_consecutive(y)        
        
        tst = mask.flatten(start_dim=2).sum(-1, keepdim=True)
                
        return cnt_pred, cnt, tst
    
    def aggregate(
        self, compute_sample: bool = False, reduction: MetricReduction | str | None = None, report_reduction: bool = True,
    ) -> list[torch.Tensor | tuple[torch.Tensor, torch.Tensor]]:
        est_nev, nev, tst = self.get_buffer()
        tst = tst / self.sampling_frequency / 3600
        est_ahi = est_nev / tst
        ahi = nev / tst        
        
        # pred thresholds for each channel
        nch = est_nev.shape[1]
        if hasattr(self.pred_thresholds[0], '__iter__'):
            pred_thresholds = self.pred_thresholds
            assert len(pred_thresholds) == nch, "number of pred thresholds should match number of (foreground) channels"
        else:
            pred_thresholds = [self.pred_thresholds] * nch
        
        if self.best_threshold_type == 'ahimae':
            ahi_dif = -torch.abs(est_ahi - ahi)
            m_ahi_dif = ahi_dif.mean(0)
            _idx = m_ahi_dif.argmax(1).cpu().numpy()
        elif self.best_threshold_type == 'osakappalinear':
            losa_est_ahi = est_ahi.cpu().clone().detach().apply_(ahi_to_osa)
            losa_ahi = ahi.cpu().clone().detach().apply_(ahi_to_osa)
            losa_kappalinear = np.array([[cohen_kappa_score(losa_est_ahi[:,j,i], losa_ahi[:,j], weights='linear') for i in range(losa_est_ahi.shape[-1])] for j in range(losa_est_ahi.shape[-2])])
            _idx = losa_kappalinear.argmax(1).cpu().numpy()  
        else: # ahimae
            ahi_dif = -torch.abs(est_ahi - ahi)
            m_ahi_dif = ahi_dif.mean(0)
            _idx = m_ahi_dif.argmax(1).cpu().numpy()
        self.best_pred_threshold = [pred_thresholds[i][_id] for i,_id in enumerate(_idx)]
        
        results: list[torch.Tensor | tuple[torch.Tensor, torch.Tensor]] = []
        for metric_name in self.metric_name:
            f = None
            if metric_name == 'ahimae':
                best_est_ahi = torch.stack([est_ahi[:,i,_id] for i, _id in enumerate(_idx)], dim=1).sum(1)
                best_ahi = ahi[...,0].sum(1)
                f = -torch.abs(best_est_ahi - best_ahi).mean(0)
            elif metric_name == 'eventmae':
                best_est_nev = torch.stack([est_nev[:,i,_id] for i, _id in enumerate(_idx)], dim=1).sum(1)
                best_nev = nev[...,0].sum(1)
                f = -torch.abs(best_est_nev - best_nev).mean(0)
            elif metric_name == 'ahimape':
                best_est_ahi = torch.stack([est_ahi[:,i,_id] for i, _id in enumerate(_idx)], dim=1).sum(1)
                best_ahi = ahi[...,0].sum(1)
                mape = (best_est_ahi - best_ahi)/(best_ahi + 1)
                f = -torch.abs(mape).mean(0)
            elif metric_name == 'ahicor':
                best_est_ahi = torch.stack([est_ahi[:,i,_id] for i, _id in enumerate(_idx)], dim=1).sum(1)
                best_ahi = ahi[...,0].sum(1)
                f = torch.corrcoef(torch.stack([best_est_ahi, best_ahi]))[0,1]
                #f = torch.stack([torch.corrcoef(torch.stack([best_est_ahi[...,i], best_ahi[...,i]]))[0,1] for i in range(len(_idx))])
            elif metric_name == 'ahiicc':
                best_est_ahi = torch.stack([est_ahi[:,i,_id] for i, _id in enumerate(_idx)], dim=1).sum(1)
                best_ahi = ahi[...,0].sum(1)
                n_samples = best_ahi.size(0)
                if n_samples < 5:
                    #f = torch.tensor([0.]*len(_idx))
                    f = torch.tensor(0.)
                else:
                    #f = []
                    #for i in range(len(_idx)):
                    #    icc_df = pd.DataFrame(data={
                    #        'targets': [*np.arange(n_samples),]*2, 
                    #        'ratings': best_ahi[...,i].cpu().numpy().tolist() + best_est_ahi[...,i].cpu().numpy().tolist(), 
                    #        'raters': [1]*n_samples + [2]*n_samples,
                    #    })
                    #    icc_stat = pingouin.intraclass_corr(data=icc_df, targets='targets', raters='raters', ratings='ratings')
                    #    sf = icc_stat[icc_stat['Type']=='ICC2']['ICC'].item()
                    #    f.append(sf)
                    icc_df = pd.DataFrame(data={
                        'targets': [*np.arange(n_samples),]*2, 
                        'ratings': best_ahi.cpu().numpy().tolist() + best_est_ahi.cpu().numpy().tolist(), 
                        'raters': [1]*n_samples + [2]*n_samples,
                    })
                    icc_stat = pingouin.intraclass_corr(data=icc_df, targets='targets', raters='raters', ratings='ratings')
                    f = icc_stat[icc_stat['Type']=='ICC2']['ICC'].item()
                    f = torch.tensor(f)  
            elif metric_name == 'osakappa':
                best_est_ahi = torch.stack([est_ahi[:,i,_id] for i, _id in enumerate(_idx)], dim=1).sum(1)
                best_ahi = ahi[...,0].sum(1)
                
                osa_ahi = best_ahi.cpu().clone().detach().apply_(ahi_to_osa)
                osa_est_ahi = best_est_ahi.cpu().clone().detach().apply_(ahi_to_osa)
                
                #f = np.array([cohen_kappa_score(osa_est_ahi[...,i], osa_ahi[...,i]) for i in range(osa_est_ahi.shape[-1])])
                f = cohen_kappa_score(osa_est_ahi, osa_ahi)
                
                f = torch.tensor(f)
            elif metric_name == 'osakappalinear':
                best_est_ahi = torch.stack([est_ahi[:,i,_id] for i, _id in enumerate(_idx)], dim=1).sum(1)
                best_ahi = ahi[...,0].sum(1)
                
                osa_ahi = best_ahi.cpu().clone().detach().apply_(ahi_to_osa)
                osa_est_ahi = best_est_ahi.cpu().clone().detach().apply_(ahi_to_osa)
                
                #f = np.array([cohen_kappa_score(osa_est_ahi[...,i], osa_ahi[...,i], weights='linear') for i in range(osa_est_ahi.shape[-1])])
                f = cohen_kappa_score(osa_est_ahi, osa_ahi, weights='linear')
                f = torch.tensor(f)
            elif metric_name == 'osakappaquadratic':
                best_est_ahi = torch.stack([est_ahi[:,i,_id] for i, _id in enumerate(_idx)], dim=1).sum(1)
                best_ahi = ahi[...,0].sum(1)
                
                osa_ahi = best_ahi.cpu().clone().detach().apply_(ahi_to_osa)
                osa_est_ahi = best_est_ahi.cpu().clone().detach().apply_(ahi_to_osa)
                
                #f = np.array([cohen_kappa_score(osa_est_ahi[...,i], osa_ahi[...,i], weights='quadratic') for i in range(osa_est_ahi.shape[-1])])
                f = cohen_kappa_score(osa_est_ahi, osa_ahi, weights='quadratic')
                f = torch.tensor(f)
            else:
                raise NotImplementedError(f"metric {metric_name} is not implemented")
            
            """
            if report_reduction:
                this_reduct = self.reduction if reduction is None else reduction
                if this_reduct == "mean":
                    f = f.mean()
                elif this_reduct == "sum":
                    f = f.sum()  
            """          
            results.append(f)
        return results
                                
    def reset(self):
        super().reset()
        self.best_pred_threshold = None
        
        
def calculate_icc(labels: Sequence[float], preds: Sequence[float]):
    n_samples = len(labels)
    if n_samples < 5:
        return 0
    else:
        icc_df = pd.DataFrame(data={
            'targets': [*np.arange(n_samples),]*2, 
            'ratings': labels + preds, 
            'raters': [1]*n_samples + [2]*n_samples,
        })
        icc_stat = pingouin.intraclass_corr(data=icc_df, targets='targets', raters='raters', ratings='ratings')
        f = icc_stat[icc_stat['Type']=='ICC2']['ICC'].item()
        return f
        
class PSGAHIMetricV3(CumulativeIterationMetric):
    """
    one-hot format (non-binarized) y_pred and y's.
    use stage mask for postprocessing V2.
        - W->W : X
        - W->sleep stage : X
        - sleep stage?->sleep stage? : O
        - sleep stage?->W : O
    sum events
    use single set of pred_thresholds for normal vs abnormal
    """
    def __init__(
        self,
        include_background: bool = False,
        metric_name: Sequence[str] | str = "ahimae",
        compute_sample: bool = False,
        reduction: MetricReduction | str = MetricReduction.MEAN,
        get_not_nans: bool = False,
        pred_thresholds: Sequence[float] | None = None,
        threshold_consecutive_seconds: float = 10,
        sampling_frequency: float = 8,
        best_threshold_type: str = 'ahimae',
        postprocess: bool = True,
    ) -> None:
        super().__init__()
        self.include_background = include_background
        self.metric_name = ensure_tuple(metric_name)
        self.compute_sample = compute_sample
        self.reduction = reduction
        self.get_not_nans = get_not_nans
        
        self.pred_thresholds = pred_thresholds
        if pred_thresholds is None:
            self.pred_thresholds = np.linspace(0, 1, 31)
        self.threshold_consecutive = threshold_consecutive_seconds * sampling_frequency
        self.sampling_frequency = sampling_frequency
        self.best_threshold_type = best_threshold_type
        self.postprocess = postprocess
    
    def _compute_tensor(self, y_pred: torch.Tensor, y: torch.Tensor, mask: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:  # type: ignore[override]
        """
        y_pred, y : (batch, channel, size). y should be one-hot.
        mask: (batch, 1, size). mask should be binarized.
        return matrix shape 
        nevent_pred: (batch, 1, pred_thr)
        nevent_gt, tst: (batch, 1, 1)
        """
        # check dimension
        dims = y_pred.ndimension()
        if dims < 3:
            raise ValueError("y_pred should have at least three dimensions.")
        
        if not self.include_background:
            y_pred, y = ignore_background(y_pred=y_pred, y=y)
                
        y_pred = y_pred.sum(1, keepdim=True).detach().cpu()
        y = y.sum(1, keepdim=True).detach().cpu()
        if isinstance(mask, torch.Tensor):
            mask = mask.detach().cpu()

        if self.postprocess:
            cnt_pred = torch.concat([count_valid_consecutive_stage(y_pred >= t, mask, threshold_consecutive=self.threshold_consecutive) for t in self.pred_thresholds], dim=-1)
        else:
            cnt_pred = torch.concat([count_valid_consecutive(y_pred >= t, threshold_consecutive=self.threshold_consecutive) for t in self.pred_thresholds], dim=-1)
        cnt = count_valid_consecutive(y)        
        
        tst = mask.flatten(start_dim=2).sum(-1, keepdim=True)
                
        return cnt_pred, cnt, tst
    
    def aggregate(
        self, compute_sample: bool = False, reduction: MetricReduction | str | None = None, report_reduction: bool = True,
    ) -> list[torch.Tensor | tuple[torch.Tensor, torch.Tensor]]:
        est_nev, nev, tst = self.get_buffer()
        tst = tst / self.sampling_frequency / 3600
        est_ahi = est_nev / tst
        ahi = nev / tst        
                
        if self.best_threshold_type == 'ahimae':
            ahi_dif = -torch.abs(est_ahi - ahi)
            m_ahi_dif = ahi_dif.mean(0)
            _idx = m_ahi_dif.argmax(1).cpu().numpy()
        elif self.best_threshold_type == 'osakappalinear':
            losa_est_ahi = est_ahi.cpu().clone().detach().apply_(ahi_to_osa)
            losa_ahi = ahi.cpu().clone().detach().apply_(ahi_to_osa)
            losa_kappalinear = np.array([[cohen_kappa_score(losa_est_ahi[:,j,i], losa_ahi[:,j], weights='linear') for i in range(losa_est_ahi.shape[-1])] for j in range(losa_est_ahi.shape[-2])])
            _idx = losa_kappalinear.argmax(1).cpu().numpy()  
        elif self.best_threshold_type == 'ahiicc':
            ahiiccs = []
            for i in range(est_ahi.shape[-1]):
                _check_est_ahi = est_ahi[...,i].sum(1).cpu().numpy().tolist()
                _check_ahi = ahi[...,0].sum(1).cpu().numpy().tolist()
                _check_icc = calculate_icc(_check_ahi, _check_est_ahi)
                ahiiccs.append(_check_icc)
            ahiiccs = np.array([ahiiccs])
            _idx = ahiiccs.argmax(1)
        else: # ahimae
            ahi_dif = -torch.abs(est_ahi - ahi)
            m_ahi_dif = ahi_dif.mean(0)
            _idx = m_ahi_dif.argmax(1).cpu().numpy()
        self.best_pred_threshold = [self.pred_thresholds[_id] for _id in _idx]
        
        results: list[torch.Tensor | tuple[torch.Tensor, torch.Tensor]] = []
        for metric_name in self.metric_name:
            f = None
            if metric_name == 'ahimae':
                best_est_ahi = torch.stack([est_ahi[:,i,_id] for i, _id in enumerate(_idx)], dim=1).sum(1)
                best_ahi = ahi[...,0].sum(1)
                f = -torch.abs(best_est_ahi - best_ahi).mean(0)
            elif metric_name == 'eventmae':
                best_est_nev = torch.stack([est_nev[:,i,_id] for i, _id in enumerate(_idx)], dim=1).sum(1)
                best_nev = nev[...,0].sum(1)
                f = -torch.abs(best_est_nev - best_nev).mean(0)
            elif metric_name == 'ahimape':
                best_est_ahi = torch.stack([est_ahi[:,i,_id] for i, _id in enumerate(_idx)], dim=1).sum(1)
                best_ahi = ahi[...,0].sum(1)
                mape = (best_est_ahi - best_ahi)/(best_ahi + 1)
                f = -torch.abs(mape).mean(0)
            elif metric_name == 'ahicor':
                best_est_ahi = torch.stack([est_ahi[:,i,_id] for i, _id in enumerate(_idx)], dim=1).sum(1)
                best_ahi = ahi[...,0].sum(1)
                f = torch.corrcoef(torch.stack([best_est_ahi, best_ahi]))[0,1]
                #f = torch.stack([torch.corrcoef(torch.stack([best_est_ahi[...,i], best_ahi[...,i]]))[0,1] for i in range(len(_idx))])
            elif metric_name == 'ahiicc':
                best_est_ahi = torch.stack([est_ahi[:,i,_id] for i, _id in enumerate(_idx)], dim=1).sum(1)
                best_ahi = ahi[...,0].sum(1)
                n_samples = best_ahi.size(0)
                if n_samples < 5:
                    #f = torch.tensor([0.]*len(_idx))
                    f = torch.tensor(0.)
                else:
                    #f = []
                    #for i in range(len(_idx)):
                    #    icc_df = pd.DataFrame(data={
                    #        'targets': [*np.arange(n_samples),]*2, 
                    #        'ratings': best_ahi[...,i].cpu().numpy().tolist() + best_est_ahi[...,i].cpu().numpy().tolist(), 
                    #        'raters': [1]*n_samples + [2]*n_samples,
                    #    })
                    #    icc_stat = pingouin.intraclass_corr(data=icc_df, targets='targets', raters='raters', ratings='ratings')
                    #    sf = icc_stat[icc_stat['Type']=='ICC2']['ICC'].item()
                    #    f.append(sf)
                    icc_df = pd.DataFrame(data={
                        'targets': [*np.arange(n_samples),]*2, 
                        'ratings': best_ahi.cpu().numpy().tolist() + best_est_ahi.cpu().numpy().tolist(), 
                        'raters': [1]*n_samples + [2]*n_samples,
                    })
                    icc_stat = pingouin.intraclass_corr(data=icc_df, targets='targets', raters='raters', ratings='ratings')
                    f = icc_stat[icc_stat['Type']=='ICC2']['ICC'].item()
                    f = torch.tensor(f)  
            elif metric_name == 'osakappa':
                best_est_ahi = torch.stack([est_ahi[:,i,_id] for i, _id in enumerate(_idx)], dim=1).sum(1)
                best_ahi = ahi[...,0].sum(1)
                
                osa_ahi = best_ahi.cpu().clone().detach().apply_(ahi_to_osa)
                osa_est_ahi = best_est_ahi.cpu().clone().detach().apply_(ahi_to_osa)
                
                #f = np.array([cohen_kappa_score(osa_est_ahi[...,i], osa_ahi[...,i]) for i in range(osa_est_ahi.shape[-1])])
                f = cohen_kappa_score(osa_est_ahi, osa_ahi)
                
                f = torch.tensor(f)
            elif metric_name == 'osakappalinear':
                best_est_ahi = torch.stack([est_ahi[:,i,_id] for i, _id in enumerate(_idx)], dim=1).sum(1)
                best_ahi = ahi[...,0].sum(1)
                
                osa_ahi = best_ahi.cpu().clone().detach().apply_(ahi_to_osa)
                osa_est_ahi = best_est_ahi.cpu().clone().detach().apply_(ahi_to_osa)
                
                #f = np.array([cohen_kappa_score(osa_est_ahi[...,i], osa_ahi[...,i], weights='linear') for i in range(osa_est_ahi.shape[-1])])
                f = cohen_kappa_score(osa_est_ahi, osa_ahi, weights='linear')
                f = torch.tensor(f)
            elif metric_name == 'osakappaquadratic':
                best_est_ahi = torch.stack([est_ahi[:,i,_id] for i, _id in enumerate(_idx)], dim=1).sum(1)
                best_ahi = ahi[...,0].sum(1)
                
                osa_ahi = best_ahi.cpu().clone().detach().apply_(ahi_to_osa)
                osa_est_ahi = best_est_ahi.cpu().clone().detach().apply_(ahi_to_osa)
                
                #f = np.array([cohen_kappa_score(osa_est_ahi[...,i], osa_ahi[...,i], weights='quadratic') for i in range(osa_est_ahi.shape[-1])])
                f = cohen_kappa_score(osa_est_ahi, osa_ahi, weights='quadratic')
                f = torch.tensor(f)
            else:
                raise NotImplementedError(f"metric {metric_name} is not implemented")
            
            """
            if report_reduction:
                this_reduct = self.reduction if reduction is None else reduction
                if this_reduct == "mean":
                    f = f.mean()
                elif this_reduct == "sum":
                    f = f.sum()  
            """          
            results.append(f)
        return results
                                
    def reset(self):
        super().reset()
        self.best_pred_threshold = None
        
        
"""Simple Detection/ROC Matrix"""

def apply_valid_sleep_event(y_pred, stage, threshold_consecutive = 80):
    """y_pred, stage: 1D binarized
    """
    y_pred_im, nb_labels_pred = ndimage.label(y_pred)
    new_y_pred = copy.deepcopy(y_pred)
    for i in range(1, nb_labels_pred+1):
        if stage[np.argwhere(y_pred_im == i).min()] <= 0 or (y_pred_im == i).sum() < threshold_consecutive:
            new_y_pred[y_pred_im == i] = 0
    return new_y_pred

def get_matrix_sleep_event(y_pred, y, stage, threshold_consecutive=80, iou_thr=1e-5, max_label=6):
    """
    y_pred, stage: 1D binarized
    y: 1D integer label
    Return:
        [[fp,1]] + [[tp,fn] for each label i in range(1, max_label+1)]
    """
    y_pred = apply_valid_sleep_event(y_pred, stage, threshold_consecutive=threshold_consecutive)
    
    y_pred_im, nb_labels_pred = ndimage.label(y_pred)
    y_im, nb_labels = ndimage.label(y > 0) 
    total_labels = np.array([y[y_im==(x+1)].max() for x in range(nb_labels)])

    if nb_labels_pred == 0 or nb_labels == 0:
        tpfn_each = []
        for i in range(1,max_label+1):
            _idx = np.argwhere(total_labels == i).ravel()
            tp_i = 0
            fn_i = len(_idx)
            tpfn_each.append([tp_i, fn_i])
        tpfn_each = np.array(tpfn_each)
    else:
        Y1 = np.array([y_pred_im == i for i in range(1, nb_labels_pred+1)], dtype=float)
        Y2 = np.array([y_im == i for i in range(1, nb_labels+1)], dtype=float)
        YDOT = np.matmul(Y1, Y2.T)
        YSUM = np.stack([Y1.sum(1),]*Y2.shape[0], -1) + np.stack([Y2.sum(1),]*Y1.shape[0], 0)
        YOR = YSUM - YDOT
        iou_grid = YDOT / (YOR + 1e-5)

        N = (iou_grid > 0).sum()
        if N == 0:
            tpfn_each = []
            for i in range(1,max_label+1):
                _idx = np.argwhere(total_labels == i).ravel()
                tp_i = 0
                fn_i = len(_idx)
                tpfn_each.append([tp_i, fn_i])
            tpfn_each = np.array(tpfn_each)
        else:
            max_ious = np.sort(np.partition(np.asarray(iou_grid), iou_grid.size - N, axis=None)[-N:])[::-1]
            for max_iou in max_ious:
                _idx = np.argwhere(iou_grid == max_iou)
                for i,j in _idx:
                    iou_grid[i,np.arange(iou_grid.shape[1])] = 0
                    iou_grid[np.arange(iou_grid.shape[0]),j] = 0
                    iou_grid[i,j] = max_iou
            iou_grid[iou_grid<iou_thr] = 0
            
            tp_preddict = np.argwhere(iou_grid > 0)
            
            tpfn_each = []
            for i in range(1,max_label+1):
                _idx = np.argwhere(total_labels == i).ravel()
                nb_i = len(_idx)
                tp_i = len(set(tp_preddict[:,1]).intersection(set(_idx)))
                fn_i = nb_i - tp_i
                tpfn_each.append([tp_i, fn_i])
            tpfn_each = np.array(tpfn_each)

    tp, fn = tpfn_each.sum(0)
    fp = nb_labels_pred - tp
    det_matrix = np.concatenate([np.array([[fp, 1]]), tpfn_each], axis=0)
    return det_matrix


def get_matrix_sleep_event_V2(y_pred, y, stage, threshold_consecutive=80, iou_thr=1e-5, max_label=6):
    """
    y_pred: (channel, size) one-hot binarized preds (including background)
    stage, y: 1D integer label
    Return:
        [[0,   fp_0,  fp_1, ...],
        [fn0, tp0_0, tp0_1, ...],
        .. for each label i in range(1, max_label+1)
        ]
        
    """
    nch = y_pred.shape[0] - 1
    y_pred_abnormal = 1 - y_pred[0]
    y_pred_abnormal = apply_valid_sleep_event(y_pred_abnormal, stage, threshold_consecutive=threshold_consecutive)
        
    y_pred_im, nb_labels_pred = ndimage.label(y_pred_abnormal)
    y_im, nb_labels = ndimage.label(y > 0) 
    total_labels = np.array([y[y_im==(x+1)].max() for x in range(nb_labels)])
    total_preds = np.array([(y_pred.argmax(0))[y_pred_im==(x+1)].max() for x in range(nb_labels_pred)])

    if nb_labels_pred == 0 or nb_labels == 0:
        det_matrix = []
        cnt_i = [0]
        cnt_i += [(total_preds==(j+1)).sum() for j in range(nch)]
        det_matrix.append(cnt_i)
        for i in range(1,max_label+1):
            _idx = np.argwhere(total_labels == i).ravel()
            fn_i = len(_idx)
            det_matrix.append([fn_i] + [0,]*nch)                
        det_matrix = np.array(det_matrix)
    else:
        Y1 = np.array([y_pred_im == i for i in range(1, nb_labels_pred+1)], dtype=float)
        Y2 = np.array([y_im == i for i in range(1, nb_labels+1)], dtype=float)
        YDOT = np.matmul(Y1, Y2.T)
        YSUM = np.stack([Y1.sum(1),]*Y2.shape[0], -1) + np.stack([Y2.sum(1),]*Y1.shape[0], 0)
        YOR = YSUM - YDOT
        iou_grid = YDOT / (YOR + 1e-5)

        N = (iou_grid > 0).sum()
        if N == 0:
            det_matrix = []
            cnt_i = [0]
            cnt_i += [(total_preds==(j+1)).sum() for j in range(nch)]
            det_matrix.append(cnt_i)
            for i in range(1,max_label+1):
                _idx = np.argwhere(total_labels == i).ravel()
                fn_i = len(_idx)
                det_matrix.append([fn_i] + [0,]*nch)                
            det_matrix = np.array(det_matrix)
        else:
            max_ious = np.sort(np.partition(np.asarray(iou_grid), iou_grid.size - N, axis=None)[-N:])[::-1]
            for max_iou in max_ious:
                _idx = np.argwhere(iou_grid == max_iou)
                for i,j in _idx:
                    iou_grid[i,np.arange(iou_grid.shape[1])] = 0
                    iou_grid[np.arange(iou_grid.shape[0]),j] = 0
                    iou_grid[i,j] = max_iou
            iou_grid[iou_grid<iou_thr] = 0
            
            tp_preddict = np.argwhere(iou_grid > 0)
            
            det_matrix = []
            # FP
            fp_idx = np.array(list(set(np.arange(nb_labels_pred)) - set(tp_preddict[:,0])))
            fp_preds = total_preds[fp_idx] if len(fp_idx)>0 else fp_idx
            cnt_i = [0]
            cnt_i += [(fp_preds==(j+1)).sum() for j in range(nch)]
            det_matrix.append(cnt_i)
            # TP, FN
            for i in range(1,max_label+1):
                _idx = np.argwhere(total_labels == i).ravel()
                nb_i = len(_idx)
                set_tp_i = set(tp_preddict[:,1]).intersection(set(_idx))
                tp_i = len(set_tp_i)
                fn_i = nb_i - tp_i
                cnt_i = [fn_i]
                for j in range(nch):
                    _pidx = np.argwhere(total_preds == (j+1)).ravel()
                    gt_pidx = [tp_preddict[list(tp_preddict[:,0]).index(x), 1] for x in _pidx if x in tp_preddict[:,0]]
                    cnt_i.append(len(set(gt_pidx).intersection(set_tp_i)))
                det_matrix.append(cnt_i)
            det_matrix = np.array(det_matrix)

    #det_matrix[0,0] = 1
    return det_matrix