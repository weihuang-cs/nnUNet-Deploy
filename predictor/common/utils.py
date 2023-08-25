#    Copyright 2021 HIP Applied Computer Vision Lab, Division of Medical Image Computing, German Cancer Research Center
#    (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
import os.path
import re

import numpy as np
from predictor.common.file_and_folder_operations import *
from copy import deepcopy
from typing import List, Tuple, Union

import torch
from skimage.measure import label
from skimage.morphology import ball
from torch.nn import functional as F

import numpy as np
from skimage.transform import resize

from typing import Callable

import predictor
from predictor.common.file_and_folder_operations import join

import importlib
import pkgutil

import torch


def softmax_helper_dim0(x: torch.Tensor) -> torch.Tensor:
    return torch.softmax(x, 0)


def softmax_helper_dim1(x: torch.Tensor) -> torch.Tensor:
    return torch.softmax(x, 1)


def empty_cache(device: torch.device):
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        from torch import mps

        mps.empty_cache()
    else:
        pass


class dummy_context(object):
    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


def recursive_find_python_class(folder: str, class_name: str, current_module: str):
    tr = None
    for importer, modname, ispkg in pkgutil.iter_modules([folder]):
        # print(modname, ispkg)
        if not ispkg:
            m = importlib.import_module(current_module + "." + modname)
            if hasattr(m, class_name):
                tr = getattr(m, class_name)
                break

    if tr is None:
        for importer, modname, ispkg in pkgutil.iter_modules([folder]):
            if ispkg:
                next_current_module = current_module + "." + modname
                tr = recursive_find_python_class(
                    join(folder, modname),
                    class_name,
                    current_module=next_current_module,
                )
            if tr is not None:
                break
    return tr


def recursive_find_resampling_fn_by_name(resampling_fn: str) -> Callable:
    ret = recursive_find_python_class(
        join(predictor.__path__[0], "data_ops"), resampling_fn, "predictor.data_ops"
    )
    if ret is None:
        raise RuntimeError(
            "Unable to find resampling function named '%s'. Please make sure this fn is located in the "
            "predictor.data_ops module." % resampling_fn
        )
    else:
        return ret


def label_with_component_sizes(
    binary_image: np.ndarray, connectivity: int = None
) -> Tuple[np.ndarray, dict]:
    if not binary_image.dtype == bool:
        print("Warning: it would be way faster if your binary image had dtype bool")
    labeled_image, num_components = label(
        binary_image, return_num=True, connectivity=connectivity
    )
    component_sizes = {
        i + 1: j for i, j in enumerate(np.bincount(labeled_image.ravel())[1:])
    }
    return labeled_image, component_sizes


def generate_ball(
    radius: Union[Tuple, List], spacing: Union[Tuple, List] = (1, 1, 1), dtype=np.uint8
) -> np.ndarray:
    """
    Returns a ball/ellipsoid corresponding to the specified size (radius = list/tuple of len 3 with one radius per axis)
    If you use spacing, both radius and spacing will be interpreted relative to each other, so a radius of 10 with a
    spacing of 5 will result in a ball with radius 2 pixels.
    """
    radius_in_voxels = np.array([round(i) for i in radius / np.array(spacing)])
    n = 2 * radius_in_voxels + 1
    ball_iso = ball(max(n) * 2, dtype=np.float64)
    ball_resampled = resize(
        ball_iso,
        n,
        1,
        "constant",
        0,
        clip=True,
        anti_aliasing=False,
        preserve_range=True,
    )
    ball_resampled[ball_resampled > 0.5] = 1
    ball_resampled[ball_resampled <= 0.5] = 0
    return ball_resampled.astype(dtype)


def pad_bbox(
    bounding_box: Union[List[List[int]], Tuple[Tuple[int, int]]],
    pad_amount: Union[int, List[int]],
    array_shape: Tuple[int, ...] = None,
) -> List[List[int]]:
    """ """
    if isinstance(bounding_box, tuple):
        # convert to list
        bounding_box = [list(i) for i in bounding_box]
    else:
        # needed because otherwise we overwrite input which could have unforseen consequences
        bounding_box = deepcopy(bounding_box)

    if isinstance(pad_amount, int):
        pad_amount = [pad_amount] * len(bounding_box)

    for i in range(len(bounding_box)):
        new_values = [
            max(0, bounding_box[i][0] - pad_amount[i]),
            bounding_box[i][1] + pad_amount[i],
        ]
        if array_shape is not None:
            new_values[1] = min(array_shape[i], new_values[1])
        bounding_box[i] = new_values

    return bounding_box


def regionprops_bbox_to_proper_bbox(
    regionprops_bbox: Tuple[int, ...]
) -> List[List[int]]:
    """
    regionprops_bbox is what you get from `from skimage.measure import regionprops`
    """
    dim = len(regionprops_bbox) // 2
    return [[regionprops_bbox[i], regionprops_bbox[i + dim]] for i in range(dim)]


def bounding_box_to_slice(bounding_box: List[List[int]]):
    return tuple([slice(*i) for i in bounding_box])


def crop_to_bbox(array: np.ndarray, bounding_box: List[List[int]]):
    assert len(bounding_box) == len(array.shape), (
        f"Dimensionality of bbox and array do not match. bbox has length "
        f"{len(bounding_box)} while array has dimension {len(array.shape)}"
    )
    slicer = bounding_box_to_slice(bounding_box)
    return array[slicer]


def get_bbox_from_mask(mask: np.ndarray) -> List[List[int]]:
    """
    this implementation uses less ram than the np.where one and is faster as well IF we expect the bounding box to
    be close to the image size. If it's not it's likely slower!

    :param mask:
    :param outside_value:
    :return:
    """
    Z, X, Y = mask.shape
    minzidx, maxzidx, minxidx, maxxidx, minyidx, maxyidx = 0, Z, 0, X, 0, Y
    zidx = list(range(Z))
    for z in zidx:
        if np.any(mask[z]):
            minzidx = z
            break
    for z in zidx[::-1]:
        if np.any(mask[z]):
            maxzidx = z + 1
            break

    xidx = list(range(X))
    for x in xidx:
        if np.any(mask[:, x]):
            minxidx = x
            break
    for x in xidx[::-1]:
        if np.any(mask[:, x]):
            maxxidx = x + 1
            break

    yidx = list(range(Y))
    for y in yidx:
        if np.any(mask[:, :, y]):
            minyidx = y
            break
    for y in yidx[::-1]:
        if np.any(mask[:, :, y]):
            maxyidx = y + 1
            break
    return [[minzidx, maxzidx], [minxidx, maxxidx], [minyidx, maxyidx]]


def get_bbox_from_mask_npwhere(mask: np.ndarray) -> List[List[int]]:
    where = np.array(np.where(mask))
    mins = np.min(where, 1)
    maxs = np.max(where, 1) + 1
    return [[i, j] for i, j in zip(mins, maxs)]


def pad_nd_image(
    image: Union[torch.Tensor, np.ndarray],
    new_shape: Tuple[int, ...] = None,
    mode: str = "constant",
    kwargs: dict = None,
    return_slicer: bool = False,
    shape_must_be_divisible_by: Union[int, Tuple[int, ...], List[int]] = None,
) -> Union[
    Union[torch.Tensor, np.ndarray], Tuple[Union[torch.Tensor, np.ndarray], Tuple]
]:
    """
    One padder to pad them all. Documentation? Well okay. A little bit

    Padding is done such that the original content will be at the center of the padded image. If the amount of padding
    needed it odd, the padding 'above' the content is larger,
    Example:
    old shape: [ 3 34 55  3]
    new_shape: [3, 34, 96, 64]
    amount of padding (low, high for each axis): [[0, 0], [0, 0], [20, 21], [30, 31]]

    :param image: can either be a numpy array or a torch.Tensor. pad_nd_image uses np.pad for the former and
           torch.nn.functional.pad for the latter
    :param new_shape: what shape do you want? new_shape does not have to have the same dimensionality as image. If
           len(new_shape) < len(image.shape) then the last axes of image will be padded. If new_shape < image.shape in
           any of the axes then we will not pad that axis, but also not crop! (interpret new_shape as new_min_shape)

           Example:
           image.shape = (10, 1, 512, 512); new_shape = (768, 768) -> result: (10, 1, 768, 768). Cool, huh?
           image.shape = (10, 1, 512, 512); new_shape = (364, 768) -> result: (10, 1, 512, 768).

    :param mode: will be passed to either np.pad or torch.nn.functional.pad depending on what the image is. Read the
           respective documentation!
    :param return_slicer: if True then this function will also return a tuple of python slice objects that you can use
           to crop back to the original image (reverse padding)
    :param shape_must_be_divisible_by: for network prediction. After applying new_shape, make sure the new shape is
           divisibly by that number (can also be a list with an entry for each axis). Whatever is missing to match
           that will be padded (so the result may be larger than new_shape if shape_must_be_divisible_by is not None)
    :param kwargs: see np.pad for documentation (numpy) or torch.nn.functional.pad (torch)

    :returns: if return_slicer=False, this function returns the padded numpy array / torch Tensor. If
              return_slicer=True it will also return a tuple of slice objects that you can use to revert the padding:
              output, slicer = pad_nd_image(input_array, new_shape=XXX, return_slicer=True)
              reversed_padding = output[slicer] ## this is now the same as input_array, padding was reversed
    """
    if kwargs is None:
        kwargs = {}

    old_shape = np.array(image.shape)

    if shape_must_be_divisible_by is not None:
        assert isinstance(shape_must_be_divisible_by, (int, list, tuple, np.ndarray))
        if isinstance(shape_must_be_divisible_by, int):
            shape_must_be_divisible_by = [shape_must_be_divisible_by] * len(image.shape)
        else:
            if len(shape_must_be_divisible_by) < len(image.shape):
                shape_must_be_divisible_by = [1] * (
                    len(image.shape) - len(shape_must_be_divisible_by)
                ) + list(shape_must_be_divisible_by)

    if new_shape is None:
        assert shape_must_be_divisible_by is not None
        new_shape = image.shape

    if len(new_shape) < len(image.shape):
        new_shape = list(image.shape[: len(image.shape) - len(new_shape)]) + list(
            new_shape
        )

    new_shape = [max(new_shape[i], old_shape[i]) for i in range(len(new_shape))]

    if shape_must_be_divisible_by is not None:
        if not isinstance(shape_must_be_divisible_by, (list, tuple, np.ndarray)):
            shape_must_be_divisible_by = [shape_must_be_divisible_by] * len(new_shape)

        if len(shape_must_be_divisible_by) < len(new_shape):
            shape_must_be_divisible_by = [1] * (
                len(new_shape) - len(shape_must_be_divisible_by)
            ) + list(shape_must_be_divisible_by)

        for i in range(len(new_shape)):
            if new_shape[i] % shape_must_be_divisible_by[i] == 0:
                new_shape[i] -= shape_must_be_divisible_by[i]

        new_shape = np.array(
            [
                new_shape[i]
                + shape_must_be_divisible_by[i]
                - new_shape[i] % shape_must_be_divisible_by[i]
                for i in range(len(new_shape))
            ]
        )

    difference = new_shape - old_shape
    pad_below = difference // 2
    pad_above = difference // 2 + difference % 2
    pad_list = [list(i) for i in zip(pad_below, pad_above)]

    if not ((all([i == 0 for i in pad_below])) and (all([i == 0 for i in pad_above]))):
        if isinstance(image, np.ndarray):
            res = np.pad(image, pad_list, mode, **kwargs)
        elif isinstance(image, torch.Tensor):
            # torch padding has the weirdest interface ever. Like wtf? Y u no read numpy documentation? So much easier
            torch_pad_list = [i for j in pad_list for i in j[::-1]][::-1]
            res = F.pad(image, torch_pad_list, mode, **kwargs)
    else:
        res = image

    if not return_slicer:
        return res
    else:
        pad_list = np.array(pad_list)
        pad_list[:, 1] = np.array(res.shape) - pad_list[:, 1]
        slicer = tuple(slice(*i) for i in pad_list)
        return res, slicer


def get_identifiers_from_splitted_dataset_folder(folder: str, file_ending: str):
    files = subfiles(folder, suffix=file_ending, join=False)
    # all files have a 4 digit channel index (_XXXX)
    crop = len(file_ending) + 5
    files = [i[:-crop] for i in files]
    # only unique image ids
    files = np.unique(files)
    return files


def create_lists_from_splitted_dataset_folder(
    folder: str, file_ending: str, identifiers: List[str] = None
) -> List[List[str]]:
    """
    does not rely on dataset.json
    """
    if identifiers is None:
        identifiers = get_identifiers_from_splitted_dataset_folder(folder, file_ending)
    files = subfiles(folder, suffix=file_ending, join=False, sort=True)
    list_of_lists = []
    for f in identifiers:
        p = re.compile(re.escape(f) + r"_\d\d\d\d" + re.escape(file_ending))
        list_of_lists.append([join(folder, i) for i in files if p.fullmatch(i)])
    return list_of_lists


def get_filenames_of_train_images_and_targets(
    raw_dataset_folder: str, dataset_json: dict = None
):
    if dataset_json is None:
        dataset_json = load_json(join(raw_dataset_folder, "dataset.json"))

    if "dataset" in dataset_json.keys():
        dataset = dataset_json["dataset"]
        for k in dataset.keys():
            dataset[k]["label"] = (
                os.path.abspath(join(raw_dataset_folder, dataset[k]["label"]))
                if not os.path.isabs(dataset[k]["label"])
                else dataset[k]["label"]
            )
            dataset[k]["images"] = [
                os.path.abspath(join(raw_dataset_folder, i))
                if not os.path.isabs(i)
                else i
                for i in dataset[k]["images"]
            ]
    else:
        identifiers = get_identifiers_from_splitted_dataset_folder(
            join(raw_dataset_folder, "imagesTr"), dataset_json["file_ending"]
        )
        images = create_lists_from_splitted_dataset_folder(
            join(raw_dataset_folder, "imagesTr"),
            dataset_json["file_ending"],
            identifiers,
        )
        segs = [
            join(raw_dataset_folder, "labelsTr", i + dataset_json["file_ending"])
            for i in identifiers
        ]
        dataset = {
            i: {"images": im, "label": se}
            for i, im, se in zip(identifiers, images, segs)
        }
    return dataset


if __name__ == "__main__":
    print(
        get_filenames_of_train_images_and_targets(join(nnUNet_raw, "Dataset002_Heart"))
    )
