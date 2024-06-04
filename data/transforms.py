from __future__ import annotations

import copy
import itertools
from itertools import chain
from math import ceil
import numbers
import numpy as np
import random
from typing import Any, Callable, Dict, Hashable, List, Mapping, Optional, Sequence, Tuple, Union
import numpy as np

import torch

from monai.utils.type_conversion import convert_to_dst_type, convert_to_tensor

from monai.config import DtypeLike, IndexSelection, KeysCollection, SequenceStr
from monai.config.type_definitions import NdarrayOrTensor
from monai.data.meta_obj import get_track_meta
from monai.networks.utils import one_hot
from monai.transforms import (
    Transform,
    MapTransform,
    RandCoarseDropout,
    RandCoarseShuffle,
    RandomizableTransform,
)
from monai.utils.enums import TransformBackends
from monai.utils.type_conversion import convert_data_type, convert_to_dst_type, convert_to_tensor, get_equivalent_dtype

from monai.transforms import (
    BorderPad,
    CenterSpatialCrop,
    Compose,
    Crop,
    Cropd,
    Transform,
    MapTransform,
    AsDiscreted,
    EnsureChannelFirstd,
    EnsureTyped,
    GridPatchd,
    Flip,
    InvertibleTransform,
    Lambdad,
    LoadImaged,
    NormalizeIntensity,
    NormalizeIntensityd,
    Pad,
    RandCropd,
    RandFlipd,
    RandGridPatchd,
    RandCropd,
    Randomizable,
    RandRotate90d,
    RandSpatialCrop,
    Resize,
    ScaleIntensityRange,
    ScaleIntensityRanged,
    SpatialPad,
    SplitDimd,
    ToTensord,
    
)

from monai.transforms.utils_pytorch_numpy_unification import (
    argwhere,
    concatenate,
    cumsum,
    percentile,
    stack,
    unique,
    where,
)


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


class LongestConsecutivePositiveLabel(Transform):
    def __init__(self, threshold: int, dim: int=-1) -> None:
        self.threshold = threshold
        self.dim = dim
        
    def __call__(self, img: NdarrayOrTensor) -> NdarrayOrTensor:
        long_pos = get_longest_consecutive_positive(img, self.dim)
        long_pos = long_pos.max()

        new_label = 1
        if long_pos < self.threshold:
            new_label = 0        
        return new_label
    
class LongestConsecutivePositiveLabeld(MapTransform):
    def __init__(self, keys: KeysCollection, *args, **kwargs) -> None:
        super().__init__(keys)
        self.transform = LongestConsecutivePositiveLabel(*args, **kwargs)
        
    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Mapping[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.keys:
            d[key] = self.transform.__call__(d[key])
        return d 
    
    
class ThresholdSumLabel(Transform):
    def __init__(self, threshold: int) -> None:
        self.threshold = threshold
        
    def __call__(self, img: NdarrayOrTensor) -> NdarrayOrTensor:
        label = img.sum() >= self.threshold
        return np.array([label], dtype=float) if isinstance(img, np.ndarray) else torch.tensor([label.float().item()])

class ThresholdSumLabeld(MapTransform):
    def __init__(self, keys: KeysCollection, *args, **kwargs) -> None:
        super().__init__(keys)
        self.transform = ThresholdSumLabel(*args, **kwargs)
        
    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Mapping[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.keys:
            d[key] = self.transform.__call__(d[key])
        return d 
    
    
class BatchwiseNormalizeIntensity(NormalizeIntensity):
    """
    copied from monai.transforms.NormalizeIntensity
    for batched inputs
    """

    def __call__(self, img: NdarrayOrTensor) -> NdarrayOrTensor:
        """
        Apply the transform to `img`, assuming `img` is a batched channel-first array if `self.channel_wise` is True,
        """
        img = convert_to_tensor(img, track_meta=get_track_meta())
        dtype = self.dtype or img.dtype
        if self.channel_wise:
            if self.subtrahend is not None and len(self.subtrahend) != len(img):
                raise ValueError(f"img has {len(img)} channels, but subtrahend has {len(self.subtrahend)} components.")
            if self.divisor is not None and len(self.divisor) != len(img):
                raise ValueError(f"img has {len(img)} channels, but divisor has {len(self.divisor)} components.")

            for j, b in enumerate(img):
                for i, d in enumerate(b):
                    img[j,i] = self._normalize(  # type: ignore
                        d,
                        sub=self.subtrahend[i] if self.subtrahend is not None else None,
                        div=self.divisor[i] if self.divisor is not None else None,
                    )
        else:
            for i, d in enumerate(img):
                img[i] = self._normalize(d, self.subtrahend, self.divisor)

        out = convert_to_dst_type(img, img, dtype=dtype)[0]
        return out
    
class BatchwiseNormalizeIntensityd(MapTransform):
    backend = BatchwiseNormalizeIntensity.backend

    def __init__(self, keys: KeysCollection, *args, **kwargs) -> None:
        super().__init__(keys)
        self.transform = BatchwiseNormalizeIntensity(*args, **kwargs)

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.transform.__call__(d[key])
        return d
    
    
class RandCoarseDropoutChannelwise(RandCoarseDropout):
    """
    MONAI's RandCoarseDropout + Channelwise holes

    """

    def __init__(
        self,
        holes: int,
        spatial_size: Sequence[int] | int,
        dropout_holes: bool = True,
        fill_value: tuple[float, float] | float | None = None,
        max_holes: int | None = None,
        max_spatial_size: Sequence[int] | int | None = None,
        prob: float = 0.1,
        prob_channelwise: float = 0.5,
    ) -> None:
        super().__init__(
            holes=holes, spatial_size=spatial_size, max_holes=max_holes, max_spatial_size=max_spatial_size, prob=prob
        )
        self.dropout_holes = dropout_holes
        if isinstance(fill_value, (tuple, list)):
            if len(fill_value) != 2:
                raise ValueError("fill value should contain 2 numbers if providing the `min` and `max`.")
        self.fill_value = fill_value
        self.prob_channelwise = prob_channelwise

    def _transform_holes(self, img: np.ndarray):
        """
        Fill the randomly selected `self.hole_coords` in input images.
        Please note that we usually only use `self.R` in `randomize()` method, here is a special case.

        """
        fill_value = (img.min(), img.max()) if self.fill_value is None else self.fill_value

        if self.dropout_holes:
            for h in self.hole_coords:
                for i, _ in enumerate(img[h]):
                    if self.R.uniform() < self.prob_channelwise:
                        if isinstance(fill_value, (tuple, list)):
                            img[h][i] = self.R.uniform(fill_value[0], fill_value[1], size=img[h][i].shape)
                        else:
                            img[h][i] = fill_value
            ret = img
        else:
            if isinstance(fill_value, (tuple, list)):
                ret = self.R.uniform(fill_value[0], fill_value[1], size=img.shape).astype(img.dtype, copy=False)
            else:
                ret = np.full_like(img, fill_value)
            for h in self.hole_coords:
                for i, _ in enumerate(img[h]):
                    if self.R.uniform() < self.prob_channelwise:
                        ret[h][i] = img[h][i]
        return ret
    
class RandCoarseDropoutChannelwised(RandomizableTransform, MapTransform):

    backend = RandCoarseDropoutChannelwise.backend

    def __init__(
        self,
        keys: KeysCollection,
        holes: int,
        spatial_size: Sequence[int] | int,
        dropout_holes: bool = True,
        fill_value: tuple[float, float] | float | None = None,
        max_holes: int | None = None,
        max_spatial_size: Sequence[int] | int | None = None,
        prob: float = 0.1,
        prob_channelwise: float = 0.5,
        allow_missing_keys: bool = False,
    ):
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob=prob)
        self.dropper = RandCoarseDropoutChannelwise(
            holes=holes,
            spatial_size=spatial_size,
            dropout_holes=dropout_holes,
            fill_value=fill_value,
            max_holes=max_holes,
            max_spatial_size=max_spatial_size,
            prob=1.0,
            prob_channelwise=prob_channelwise,
        )

    def set_random_state(
        self, seed: int | None = None, state: np.random.RandomState | None = None
    ) -> RandCoarseDropoutChannelwised:
        super().set_random_state(seed, state)
        self.dropper.set_random_state(seed, state)
        return self


    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        self.randomize(None)
        if not self._do_transform:
            for key in self.key_iterator(d):
                d[key] = convert_to_tensor(d[key], track_meta=get_track_meta())
            return d

        # expect all the specified keys have same spatial shape and share same random holes
        first_key: Hashable = self.first_key(d)
        if first_key == ():
            for key in self.key_iterator(d):
                d[key] = convert_to_tensor(d[key], track_meta=get_track_meta())
            return d

        self.dropper.randomize(d[first_key].shape[1:])
        for key in self.key_iterator(d):
            d[key] = self.dropper(img=d[key], randomize=False)

        return d
    
class RandCoarseShuffleChannelwise(RandCoarseShuffle):
    def __init__(
        self,
        holes: int,
        spatial_size: Sequence[int] | int,
        max_holes: int | None = None,
        max_spatial_size: Sequence[int] | int | None = None,
        prob: float = 0.1,
        prob_channelwise: float = 0.5,
    ) -> None:
        super().__init__(
            holes=holes, spatial_size=spatial_size, max_holes=max_holes, max_spatial_size=max_spatial_size, prob=prob
        )
        self.prob_channelwise = prob_channelwise

    def _transform_holes(self, img: np.ndarray):
        """
        Shuffle the content of randomly selected `self.hole_coords` in input images.
        Please note that we usually only use `self.R` in `randomize()` method, here is a special case.

        """
        for h in self.hole_coords:
            # shuffle every channel separately
            for i, c in enumerate(img[h]):
                if self.R.uniform() < self.prob_channelwise:
                    patch_channel = c.flatten()
                    self.R.shuffle(patch_channel)
                    img[h][i] = patch_channel.reshape(c.shape)
        return img
    
class RandCoarseShuffleChannelwised(RandomizableTransform, MapTransform):

    backend = RandCoarseShuffleChannelwise.backend

    def __init__(
        self,
        keys: KeysCollection,
        holes: int,
        spatial_size: Sequence[int] | int,
        max_holes: int | None = None,
        max_spatial_size: Sequence[int] | int | None = None,
        prob: float = 0.1,
        prob_channelwise: float = 0.5,
        allow_missing_keys: bool = False,
    ):
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob=prob)
        self.shuffle = RandCoarseShuffleChannelwise(
            holes=holes, spatial_size=spatial_size, max_holes=max_holes, max_spatial_size=max_spatial_size, prob=1.0, prob_channelwise=prob_channelwise,
        )

    def set_random_state(
        self, seed: int | None = None, state: np.random.RandomState | None = None
    ) -> RandCoarseShuffleChannelwised:
        super().set_random_state(seed, state)
        self.shuffle.set_random_state(seed, state)
        return self


    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        self.randomize(None)
        if not self._do_transform:
            for key in self.key_iterator(d):
                d[key] = convert_to_tensor(d[key], track_meta=get_track_meta())
            return d

        # expect all the specified keys have same spatial shape and share same random holes
        first_key: Hashable = self.first_key(d)
        if first_key == ():
            for key in self.key_iterator(d):
                d[key] = convert_to_tensor(d[key], track_meta=get_track_meta())
            return d

        self.shuffle.randomize(d[first_key].shape[1:])
        for key in self.key_iterator(d):
            d[key] = self.shuffle(img=d[key], randomize=False)

        return d
    
    
class ClipIntensityRangePercentiles(Transform):
    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]
    def __init__(
        self,
        lower: float,
        upper: float,
        channel_wise: bool = False,
        dtype: DtypeLike = np.float32,
    ) -> None:
        if lower < 0.0 or lower > 100.0:
            raise ValueError("Percentiles must be in the range [0, 100]")
        if upper < 0.0 or upper > 100.0:
            raise ValueError("Percentiles must be in the range [0, 100]")
        self.lower = lower
        self.upper = upper
        self.channel_wise = channel_wise
        self.dtype = dtype

    def _normalize(self, img: NdarrayOrTensor) -> NdarrayOrTensor:
        a_min: float = percentile(img, self.lower)  # type: ignore
        a_max: float = percentile(img, self.upper)  # type: ignore
        b_min = a_min
        b_max = a_max

        img[img<a_min] = a_min
        img[img>a_max] = a_max

        #scalar = ScaleIntensityRange(
        #    a_min=a_min, a_max=a_max, b_min=b_min, b_max=b_max, clip=True, #dtype=self.dtype
        #)
        #img = scalar(img)
        img = convert_to_tensor(img, track_meta=False)
        return img

    def __call__(self, img: NdarrayOrTensor) -> NdarrayOrTensor:
        """
        Apply the transform to `img`.
        """
        img = convert_to_tensor(img, track_meta=get_track_meta())
        img_t = convert_to_tensor(img, track_meta=False)
        if self.channel_wise:
            img_t = torch.stack([self._normalize(img=d) for d in img_t])  # type: ignore
        else:
            img_t = self._normalize(img=img_t)

        return convert_to_dst_type(img_t, dst=img)[0]
    
class ClipIntensityRangePercentilesd(MapTransform):
    backend = ClipIntensityRangePercentiles.backend

    def __init__(self, keys: KeysCollection, *args, **kwargs) -> None:
        super().__init__(keys)
        self.transform = ClipIntensityRangePercentiles(*args, **kwargs)

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.transform.__call__(d[key])
        return d
    

    
class ConvertLabel(Transform):
    def __init__(self, convert_dict: dict = {}) -> None:
        self.convert_dict = convert_dict
        
    def __call__(self, img: NdarrayOrTensor) -> NdarrayOrTensor:
        if len(self.convert_dict.keys()) > 0:
            select_k = []
            select_v = []
            for k,v in self.convert_dict.items():
                select_k.append(img==k)
                select_v.append(v)
            img1 = np.select(select_k, select_v, img)
            img = convert_to_dst_type(img1, img, dtype=img.dtype)[0]
            return img
        else:
            return img
        
class ConvertLabeld(MapTransform):
    def __init__(self, keys: KeysCollection, *args, **kwargs) -> None:
        super().__init__(keys)
        self.transform = ConvertLabel(*args, **kwargs)
        
    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Mapping[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.keys:
            d[key] = self.transform.__call__(d[key])
        return d 
    
    
class OneHotLabel(Transform):
    def __init__(self, num_classes: int, dim: int=1) -> None:
        self.num_classes = num_classes
        self.dim = dim
        
    def __call__(self, img: NdarrayOrTensor) -> NdarrayOrTensor:
        img = one_hot(img, num_classes=self.num_classes, dim=self.dim)
        return img

class OneHotLabeld(MapTransform):
    def __init__(self, keys: KeysCollection, *args, **kwargs) -> None:
        super().__init__(keys)
        self.transform = OneHotLabel(*args, **kwargs)
        
    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Mapping[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.keys:
            d[key] = self.transform.__call__(d[key])
        return d 