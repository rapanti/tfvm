"""Transforms Factory
Factory methods for building image transforms for use with TIMM (PyTorch Image Models)

Hacked together by / Copyright 2019, Ross Wightman
"""

from typing import Optional, Tuple, Union

import torch
from torchvision import transforms

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data.auto_augment import (
    rand_augment_transform,
    augment_and_mix_transform,
    auto_augment_transform,
)
from timm.data.transforms import (
    str_to_pil_interp,
    RandomResizedCropAndInterpolation,
    ResizeKeepRatio,
    CenterCropOrPad,
    RandomCropOrPad,
    ToNumpy,
    MaybeToTensor,
    MaybePILToTensor,
)
from timm.data.random_erasing import RandomErasing
from timm.data.transforms_factory import (
    transforms_imagenet_eval,
    transforms_noaug_train,
)

def transforms_imagenet_train(
    img_size: Union[int, Tuple[int, int]] = 224,
    scale: Optional[Tuple[float, float]] = None,
    ratio: Optional[Tuple[float, float]] = None,
    train_crop_mode: Optional[str] = None,
    hflip: float = 0.5,
    vflip: float = 0.0,
    color_jitter: Union[float, Tuple[float, ...]] = 0.4,
    color_jitter_prob: Optional[float] = None,
    force_color_jitter: bool = False,
    grayscale_prob: float = 0.0,
    gaussian_blur_prob: float = 0.0,
    auto_augment: Optional[str] = None,
    trivial_augment: bool = False,
    rand_augment: bool = False,
    ra_num_ops: int = 2,
    ra_magnitude: int = 8,
    ra_magnitude_bins: int = 31,
    interpolation: str = "random",
    mean: Tuple[float, ...] = IMAGENET_DEFAULT_MEAN,
    std: Tuple[float, ...] = IMAGENET_DEFAULT_STD,
    re_prob: float = 0.0,
    re_mode: str = "const",
    re_count: int = 1,
    re_num_splits: int = 0,
    use_prefetcher: bool = False,
    normalize: bool = True,
    separate: bool = False,
):
    """ImageNet-oriented image transforms for training.

    Args:
        img_size: Target image size.
        train_crop_mode: Training random crop mode ('rrc', 'rkrc', 'rkrr').
        scale: Random resize scale range (crop area, < 1.0 => zoom in).
        ratio: Random aspect ratio range (crop ratio for RRC, ratio adjustment factor for RKR).
        hflip: Horizontal flip probability.
        vflip: Vertical flip probability.
        color_jitter: Random color jitter component factors (brightness, contrast, saturation, hue).
            Scalar is applied as (scalar,) * 3 (no hue).
        color_jitter_prob: Apply color jitter with this probability if not None (for SimlCLR-like aug).
        force_color_jitter: Force color jitter where it is normally disabled (ie with RandAugment on).
        grayscale_prob: Probability of converting image to grayscale (for SimCLR-like aug).
        gaussian_blur_prob: Probability of applying gaussian blur (for SimCLR-like aug).
        auto_augment: Auto augment configuration string (see auto_augment.py).
        trivial_augment: Add Trivial augment.
        rand_augment: Add RandAugment augmentations.
        ra_num_ops: Number of augmentation transformations in RandAugment.
        ra_magnitude: Magnitude of transformations in RandAugment.
        interpolation: Image interpolation mode.
        mean: Image normalization mean.
        std: Image normalization standard deviation.
        re_prob: Random erasing probability.
        re_mode: Random erasing fill mode.
        re_count: Number of random erasing regions.
        re_num_splits: Control split of random erasing across batch size.
        use_prefetcher: Prefetcher enabled. Do not convert image to tensor or normalize.
        normalize: Normalize tensor output w/ provided mean/std (if prefetcher not used).
        separate: Output transforms in 3-stage tuple.

    Returns:
        If separate==True, the transforms are returned as a tuple of 3 separate transforms
        for use in a mixing dataset that passes
         * all data through the first (primary) transform, called the 'clean' data
         * a portion of the data through the secondary transform
         * normalizes and converts the branches above with the third, final transform
    """
    train_crop_mode = train_crop_mode or "rrc"
    assert train_crop_mode in {"rrc", "rkrc", "rkrr"}
    if train_crop_mode in ("rkrc", "rkrr"):
        # FIXME integration of RKR is a WIP
        scale = tuple(scale or (0.8, 1.00))  # type: ignore
        ratio = tuple(ratio or (0.9, 1 / 0.9))  # type: ignore
        primary_tfl = [
            ResizeKeepRatio(
                img_size,
                interpolation=interpolation,
                random_scale_prob=0.5,
                random_scale_range=scale,
                random_scale_area=True,  # scale compatible with RRC
                random_aspect_prob=0.5,
                random_aspect_range=ratio,
            ),
            CenterCropOrPad(img_size, padding_mode="reflect")  # type: ignore
            if train_crop_mode == "rkrc"
            else RandomCropOrPad(img_size, padding_mode="reflect"),  # type: ignore
        ]
    else:
        scale = tuple(scale or (0.08, 1.0))  # type: ignore
        ratio = tuple(ratio or (3.0 / 4.0, 4.0 / 3.0))  # type: ignore
        primary_tfl = [
            RandomResizedCropAndInterpolation(
                img_size,
                scale=scale,
                ratio=ratio,
                interpolation=interpolation,
            )
        ]
    if hflip > 0.0:
        primary_tfl += [transforms.RandomHorizontalFlip(p=hflip)]
    if vflip > 0.0:
        primary_tfl += [transforms.RandomVerticalFlip(p=vflip)]

    secondary_tfl = []
    disable_color_jitter = False
    if auto_augment:
        assert isinstance(auto_augment, str)
        # color jitter is typically disabled if AA/RA on,
        # this allows override without breaking old hparm cfgs
        disable_color_jitter = not (force_color_jitter or "3a" in auto_augment)
        if isinstance(img_size, (tuple, list)):
            img_size_min = min(img_size)
        else:
            img_size_min = img_size
        aa_params = dict(
            translate_const=int(img_size_min * 0.45),
            img_mean=tuple([min(255, round(255 * x)) for x in mean]),
        )
        if interpolation and interpolation != "random":
            aa_params["interpolation"] = str_to_pil_interp(interpolation)
        if auto_augment.startswith("rand"):
            secondary_tfl += [rand_augment_transform(auto_augment, aa_params)]
        elif auto_augment.startswith("augmix"):
            aa_params["translate_pct"] = 0.3  # type: ignore
            secondary_tfl += [augment_and_mix_transform(auto_augment, aa_params)]
        else:
            secondary_tfl += [auto_augment_transform(auto_augment, aa_params)]

    if color_jitter is not None and not disable_color_jitter:
        # color jitter is enabled when not using AA or when forced
        if isinstance(color_jitter, (list, tuple)):
            # color jitter should be a 3-tuple/list if spec brightness/contrast/saturation
            # or 4 if also augmenting hue
            assert len(color_jitter) in (3, 4)
        else:
            # if it's a scalar, duplicate for brightness, contrast, and saturation, no hue
            color_jitter = (float(color_jitter),) * 3
        if color_jitter_prob is not None:
            secondary_tfl += [
                transforms.RandomApply(
                    [
                        transforms.ColorJitter(*color_jitter),
                    ],
                    p=color_jitter_prob,
                )
            ]
        else:
            secondary_tfl += [transforms.ColorJitter(*color_jitter)]

    if grayscale_prob:
        secondary_tfl += [transforms.RandomGrayscale(p=grayscale_prob)]

    if gaussian_blur_prob:
        secondary_tfl += [
            transforms.RandomApply(
                [
                    transforms.GaussianBlur(kernel_size=23),  # hardcoded for now
                ],
                p=gaussian_blur_prob,
            )
        ]

    if trivial_augment:
        secondary_tfl += [transforms.TrivialAugmentWide()]

    if rand_augment:
        secondary_tfl += [transforms.RandAugment(ra_num_ops, ra_magnitude, ra_magnitude_bins)]

    final_tfl = []
    if use_prefetcher:
        # prefetcher and collate will handle tensor conversion and norm
        final_tfl += [ToNumpy()]
    elif not normalize:
        # when normalize disable, converted to tensor without scaling, keeps original dtype
        final_tfl += [MaybePILToTensor()]
    else:
        final_tfl += [
            MaybeToTensor(),
            transforms.Normalize(
                mean=torch.tensor(mean),
                std=torch.tensor(std),
            ),
        ]
        if re_prob > 0.0:
            final_tfl += [
                RandomErasing(
                    re_prob,
                    mode=re_mode,
                    max_count=re_count,
                    num_splits=re_num_splits,
                    device="cpu",
                )
            ]

    if separate:
        return (
            transforms.Compose(primary_tfl),
            transforms.Compose(secondary_tfl),
            transforms.Compose(final_tfl),
        )
    else:
        return transforms.Compose(primary_tfl + secondary_tfl + final_tfl)


def create_transform(
    input_size: Union[int, Tuple[int, int], Tuple[int, int, int]] = 224,
    is_training: bool = False,
    no_aug: bool = False,
    train_crop_mode: Optional[str] = None,
    scale: Optional[Tuple[float, float]] = None,
    ratio: Optional[Tuple[float, float]] = None,
    hflip: float = 0.5,
    vflip: float = 0.0,
    color_jitter: Union[float, Tuple[float, ...]] = 0.4,
    color_jitter_prob: Optional[float] = None,
    grayscale_prob: float = 0.0,
    gaussian_blur_prob: float = 0.0,
    auto_augment: Optional[str] = None,
    trivial_augment: bool = False,
    rand_augment: bool = False,
    ra_num_ops: int = 2,
    ra_magnitude: int = 9,
    ra_magnitude_bins: int = 31,
    interpolation: str = "bilinear",
    mean: Tuple[float, ...] = IMAGENET_DEFAULT_MEAN,
    std: Tuple[float, ...] = IMAGENET_DEFAULT_STD,
    re_prob: float = 0.0,
    re_mode: str = "const",
    re_count: int = 1,
    re_num_splits: int = 0,
    crop_pct: Optional[float] = None,
    crop_mode: Optional[str] = None,
    crop_border_pixels: Optional[int] = None,
    tf_preprocessing: bool = False,
    use_prefetcher: bool = False,
    normalize: bool = True,
    separate: bool = False,
):
    """

    Args:
        input_size: Target input size (channels, height, width) tuple or size scalar.
        is_training: Return training (random) transforms.
        no_aug: Disable augmentation for training (useful for debug).
        train_crop_mode: Training random crop mode ('rrc', 'rkrc', 'rkrr').
        scale: Random resize scale range (crop area, < 1.0 => zoom in).
        ratio: Random aspect ratio range (crop ratio for RRC, ratio adjustment factor for RKR).
        hflip: Horizontal flip probability.
        vflip: Vertical flip probability.
        color_jitter: Random color jitter component factors (brightness, contrast, saturation, hue).
            Scalar is applied as (scalar,) * 3 (no hue).
        color_jitter_prob: Apply color jitter with this probability if not None (for SimlCLR-like aug).
        grayscale_prob: Probability of converting image to grayscale (for SimCLR-like aug).
        gaussian_blur_prob: Probability of applying gaussian blur (for SimCLR-like aug).
        auto_augment: Auto augment configuration string (see auto_augment.py).
        interpolation: Image interpolation mode.
        mean: Image normalization mean.
        std: Image normalization standard deviation.
        re_prob: Random erasing probability.
        re_mode: Random erasing fill mode.
        re_count: Number of random erasing regions.
        re_num_splits: Control split of random erasing across batch size.
        crop_pct: Inference crop percentage (output size / resize size).
        crop_mode: Inference crop mode. One of ['squash', 'border', 'center']. Defaults to 'center' when None.
        crop_border_pixels: Inference crop border of specified # pixels around edge of original image.
        tf_preprocessing: Use TF 1.0 inference preprocessing for testing model ports
        use_prefetcher: Pre-fetcher enabled. Do not convert image to tensor or normalize.
        normalize: Normalization tensor output w/ provided mean/std (if prefetcher not used).
        separate: Output transforms in 3-stage tuple.

    Returns:
        Composed transforms or tuple thereof
    """
    if isinstance(input_size, (tuple, list)):
        img_size = input_size[-2:]
    else:
        img_size = input_size

    if tf_preprocessing and use_prefetcher:
        assert not separate, "Separate transforms not supported for TF preprocessing"
        from timm.data.tf_preprocessing import TfPreprocessTransform

        transform = TfPreprocessTransform(
            is_training=is_training,
            size=img_size,  # type: ignore
            interpolation=interpolation,
        )
    else:
        if is_training and no_aug:
            assert not separate, "Cannot perform split augmentation with no_aug"
            transform = transforms_noaug_train(
                img_size,
                interpolation=interpolation,
                mean=mean,
                std=std,
                use_prefetcher=use_prefetcher,
                normalize=normalize,
            )
        elif is_training:
            transform = transforms_imagenet_train(
                img_size,
                train_crop_mode=train_crop_mode,
                scale=scale,
                ratio=ratio,
                hflip=hflip,
                vflip=vflip,
                color_jitter=color_jitter,
                color_jitter_prob=color_jitter_prob,
                grayscale_prob=grayscale_prob,
                gaussian_blur_prob=gaussian_blur_prob,
                auto_augment=auto_augment,
                trivial_augment=trivial_augment,
                rand_augment=rand_augment,
                ra_num_ops=ra_num_ops,
                ra_magnitude=ra_magnitude,
                ra_magnitude_bins=ra_magnitude_bins,
                interpolation=interpolation,
                mean=mean,
                std=std,
                re_prob=re_prob,
                re_mode=re_mode,
                re_count=re_count,
                re_num_splits=re_num_splits,
                use_prefetcher=use_prefetcher,
                normalize=normalize,
                separate=separate,
            )
        else:
            assert not separate, "Separate transforms not supported for validation preprocessing"
            transform = transforms_imagenet_eval(
                img_size,
                interpolation=interpolation,
                mean=mean,
                std=std,
                crop_pct=crop_pct,
                crop_mode=crop_mode,
                crop_border_pixels=crop_border_pixels,
                use_prefetcher=use_prefetcher,
                normalize=normalize,
            )

    return transform
