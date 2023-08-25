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

import json
import os
import pickle
from abc import ABC, abstractmethod
from typing import Tuple, Union, List

import SimpleITK as sitk
import numpy as np


class BaseReaderWriter(ABC):
    @staticmethod
    def _check_all_same(input_list):
        # compare all entries to the first
        for i in input_list[1:]:
            if not len(i) == len(input_list[0]):
                return False
            all_same = all(i[j] == input_list[0][j] for j in range(len(i)))
            if not all_same:
                return False
        return True

    @staticmethod
    def _check_all_same_array(input_list):
        # compare all entries to the first
        for i in input_list[1:]:
            if not all([a == b for a, b in zip(i.shape, input_list[0].shape)]):
                return False
            all_same = np.allclose(i, input_list[0])
            if not all_same:
                return False
        return True

    @abstractmethod
    def read_images(
        self, image_fnames: Union[List[str], Tuple[str, ...]]
    ) -> Tuple[np.ndarray, dict]:
        """
        Reads a sequence of images and returns a 4d (!) np.ndarray along with a dictionary. The 4d array must have the
        modalities (or color channels, or however you would like to call them) in its first axis, followed by the
        spatial dimensions (so shape must be c,x,y,z where c is the number of modalities (can be 1)).
        Use the dictionary to store necessary meta information that is lost when converting to numpy arrays, for
        example the Spacing, Orientation and Direction of the image. This dictionary will be handed over to write_seg
        for exporting the predicted segmentations, so make sure you have everything you need in there!

        IMPORTANT: dict MUST have a 'spacing' key with a tuple/list of length 3 with the voxel spacing of the np.ndarray.
        Example: my_dict = {'spacing': (3, 0.5, 0.5), ...}. This is needed for planning and
        preprocessing. The ordering of the numbers must correspond to the axis ordering in the returned numpy array. So
        if the array has shape c,x,y,z and the spacing is (a,b,c) then a must be the spacing of x, b the spacing of y
        and c the spacing of z.

        In the case of 2D images, the returned array should have shape (c, 1, x, y) and the spacing should be
        (999, sp_x, sp_y). Make sure 999 is larger than sp_x and sp_y! Example: shape=(3, 1, 224, 224),
        spacing=(999, 1, 1)

        For images that don't have a spacing, set the spacing to 1 (2d exception with 999 for the first axis still applies!)

        :param image_fnames:
        :return:
            1) a np.ndarray of shape (c, x, y, z) where c is the number of image channels (can be 1) and x, y, z are
            the spatial dimensions (set x=1 for 2D! Example: (3, 1, 224, 224) for RGB image).
            2) a dictionary with metadata. This can be anything. BUT it HAS to inclue a {'spacing': (a, b, c)} where a
            is the spacing of x, b of y and c of z! If an image doesn't have spacing, just set this to 1. For 2D, set
            a=999 (largest spacing value! Make it larger than b and c)

        """
        pass

    @abstractmethod
    def read_seg(self, seg_fname: str) -> Tuple[np.ndarray, dict]:
        """
        Same requirements as BaseReaderWriter.read_image. Returned segmentations must have shape 1,x,y,z. Multiple
        segmentations are not (yet?) allowed

        If images and segmentations can be read the same way you can just `return self.read_image((image_fname,))`
        :param seg_fname:
        :return:
            1) a np.ndarray of shape (1, x, y, z) where x, y, z are
            the spatial dimensions (set x=1 for 2D! Example: (1, 1, 224, 224) for 2D segmentation).
            2) a dictionary with metadata. This can be anything. BUT it HAS to inclue a {'spacing': (a, b, c)} where a
            is the spacing of x, b of y and c of z! If an image doesn't have spacing, just set this to 1. For 2D, set
            a=999 (largest spacing value! Make it larger than b and c)
        """
        pass

    @abstractmethod
    def write_seg(self, seg: np.ndarray, output_fname: str, properties: dict) -> None:
        """
        Export the predicted segmentation to the desired file format. The given seg array will have the same shape and
        orientation as the corresponding image data, so you don't need to do any resampling or whatever. Just save :-)

        properties is the same dictionary you created during read_images/read_seg so you can use the information here
        to restore metadata

        IMPORTANT: Segmentations are always 3D! If your input images were 2d then the segmentation will have shape
        1,x,y. You need to catch that and export accordingly (for 2d images you need to convert the 3d segmentation
        to 2d via seg = seg[0])!

        :param seg: A segmentation (np.ndarray, integer) of shape (x, y, z). For 2D segmentations this will be (1, y, z)!
        :param output_fname:
        :param properties: the dictionary that you created in read_images (the ones this segmentation is based on).
        Use this to restore metadata
        :return:
        """
        pass


class SimpleITKIO(BaseReaderWriter):
    supported_file_endings = [".nii.gz", ".nrrd", ".mha"]

    def read_images(
        self, image_fnames: Union[List[str], Tuple[str, ...]]
    ) -> Tuple[np.ndarray, dict]:
        images = []
        spacings = []
        origins = []
        directions = []

        spacings_for_nnunet = []
        for f in image_fnames:
            itk_image = sitk.ReadImage(f)
            spacings.append(itk_image.GetSpacing())
            origins.append(itk_image.GetOrigin())
            directions.append(itk_image.GetDirection())
            npy_image = sitk.GetArrayFromImage(itk_image)
            if len(npy_image.shape) == 2:
                # 2d
                npy_image = npy_image[None, None]
                max_spacing = max(spacings[-1])
                spacings_for_nnunet.append(
                    (max_spacing * 999, *list(spacings[-1])[::-1])
                )
            elif len(npy_image.shape) == 3:
                # 3d, as in original nnunet
                npy_image = npy_image[None]
                spacings_for_nnunet.append(list(spacings[-1])[::-1])
            elif len(npy_image.shape) == 4:
                # 4d, multiple modalities in one file
                spacings_for_nnunet.append(list(spacings[-1])[::-1][1:])
                pass
            else:
                raise RuntimeError(
                    "Unexpected number of dimensions: %d in file %s"
                    % (len(npy_image.shape), f)
                )

            images.append(npy_image)
            spacings_for_nnunet[-1] = list(np.abs(spacings_for_nnunet[-1]))

        if not self._check_all_same([i.shape for i in images]):
            print("ERROR! Not all input images have the same shape!")
            print("Shapes:")
            print([i.shape for i in images])
            print("Image files:")
            print(image_fnames)
            raise RuntimeError()
        if not self._check_all_same(spacings):
            print("ERROR! Not all input images have the same spacing!")
            print("Spacings:")
            print(spacings)
            print("Image files:")
            print(image_fnames)
            raise RuntimeError()
        if not self._check_all_same(origins):
            print("WARNING! Not all input images have the same origin!")
            print("Origins:")
            print(origins)
            print("Image files:")
            print(image_fnames)
            print(
                "It is up to you to decide whether that's a problem. You should run nnUNet_plot_dataset_pngs to verify "
                "that segmentations and data overlap."
            )
        if not self._check_all_same(directions):
            print("WARNING! Not all input images have the same direction!")
            print("Directions:")
            print(directions)
            print("Image files:")
            print(image_fnames)
            print(
                "It is up to you to decide whether that's a problem. You should run nnUNet_plot_dataset_pngs to verify "
                "that segmentations and data overlap."
            )
        if not self._check_all_same(spacings_for_nnunet):
            print(
                "ERROR! Not all input images have the same spacing_for_nnunet! (This should not happen and must be a "
                "bug. Please report!"
            )
            print("spacings_for_nnunet:")
            print(spacings_for_nnunet)
            print("Image files:")
            print(image_fnames)
            raise RuntimeError()

        stacked_images = np.vstack(images)
        dict = {
            "sitk_stuff": {
                # this saves the sitk geometry information. This part is NOT used by nnU-Net!
                "spacing": spacings[0],
                "origin": origins[0],
                "direction": directions[0],
            },
            # the spacing is inverted with [::-1] because sitk returns the spacing in the wrong order lol. Image arrays
            # are returned x,y,z but spacing is returned z,y,x. Duh.
            "spacing": spacings_for_nnunet[0],
        }
        return stacked_images.astype(np.float32), dict

    def read_seg(self, seg_fname: str) -> Tuple[np.ndarray, dict]:
        return self.read_images((seg_fname,))

    def write_seg(self, seg: np.ndarray, output_fname: str, properties: dict) -> None:
        assert (
            len(seg.shape) == 3
        ), "segmentation must be 3d. If you are exporting a 2d segmentation, please provide it as shape 1,x,y"
        output_dimension = len(properties["sitk_stuff"]["spacing"])
        assert 1 < output_dimension < 4
        if output_dimension == 2:
            seg = seg[0]

        itk_image = sitk.GetImageFromArray(seg.astype(np.uint8))
        itk_image.SetSpacing(properties["sitk_stuff"]["spacing"])
        itk_image.SetOrigin(properties["sitk_stuff"]["origin"])
        itk_image.SetDirection(properties["sitk_stuff"]["direction"])

        sitk.WriteImage(itk_image, output_fname)


def subdirs(folder, join=True, prefix=None, suffix=None, sort=True):
    if join:
        l = os.path.join
    else:
        l = lambda x, y: y
    res = [
        l(folder, i)
        for i in os.listdir(folder)
        if os.path.isdir(os.path.join(folder, i))
        and (prefix is None or i.startswith(prefix))
        and (suffix is None or i.endswith(suffix))
    ]
    if sort:
        res.sort()
    return res


def subfiles(folder, join=True, prefix=None, suffix=None, sort=True):
    if join:
        l = os.path.join
    else:
        l = lambda x, y: y
    res = [
        l(folder, i)
        for i in os.listdir(folder)
        if os.path.isfile(os.path.join(folder, i))
        and (prefix is None or i.startswith(prefix))
        and (suffix is None or i.endswith(suffix))
    ]
    if sort:
        res.sort()
    return res


subfolders = subdirs  # I am tired of confusing those


def maybe_mkdir_p(directory):
    directory = os.path.abspath(directory)
    splits = directory.split("/")[1:]
    for i in range(0, len(splits)):
        if not os.path.isdir(os.path.join("/", *splits[: i + 1])):
            try:
                os.mkdir(os.path.join("/", *splits[: i + 1]))
            except FileExistsError:
                # this can sometimes happen when two jobs try to create the same directory at the same time,
                # especially on network drives.
                print(
                    "WARNING: Folder %s already existed and does not need to be created"
                    % directory
                )


def load_pickle(file, mode="rb"):
    with open(file, mode) as f:
        a = pickle.load(f)
    return a


def write_pickle(obj, file, mode="wb"):
    with open(file, mode) as f:
        pickle.dump(obj, f)


save_pickle = write_pickle


def load_json(file):
    with open(file, "r") as f:
        a = json.load(f)
    return a


def save_json(obj, file, indent=4, sort_keys=True):
    with open(file, "w") as f:
        json.dump(obj, f, sort_keys=sort_keys, indent=indent)


write_json = save_json


def pardir(path):
    return os.path.join(path, os.pardir)


# I'm tired of typing these out
join = os.path.join
isdir = os.path.isdir
isfile = os.path.isfile
listdir = os.listdir
