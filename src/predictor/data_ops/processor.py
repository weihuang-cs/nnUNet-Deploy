#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
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
from typing import Union, Tuple

import numpy as np

# import predictor
from predictor.data_ops.crop import crop_to_nonzero
from predictor.data_ops.resample import compute_new_shape
from predictor.common.utils import recursive_find_python_class
from predictor.data_ops.plans_handler import (
    PlansManager,
    ConfigurationManager,
)
from predictor.common.file_and_folder_operations import *


class DefaultPreprocessor(object):
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        """
        Everything we need is in the plans. Those are given when run() is called
        """

    def run_case_npy(
        self,
        data: np.ndarray,
        seg: Union[np.ndarray, None],
        properties: dict,
        plans_manager: PlansManager,
        configuration_manager: ConfigurationManager,
        dataset_json: Union[dict, str],
    ):
        # let's not mess up the inputs!
        data = np.copy(data)
        if seg is not None:
            seg = np.copy(seg)

        has_seg = seg is not None

        # apply transpose_forward, this also needs to be applied to the spacing!
        data = data.transpose([0, *[i + 1 for i in plans_manager.transpose_forward]])
        if seg is not None:
            seg = seg.transpose([0, *[i + 1 for i in plans_manager.transpose_forward]])
        original_spacing = [
            properties["spacing"][i] for i in plans_manager.transpose_forward
        ]

        # crop, remember to store size before cropping!
        shape_before_cropping = data.shape[1:]
        properties["shape_before_cropping"] = shape_before_cropping
        # this command will generate a segmentation. This is important because of the nonzero mask which we may need
        data, seg, bbox = crop_to_nonzero(data, seg)
        properties["bbox_used_for_cropping"] = bbox
        # print(data.shape, seg.shape)
        properties["shape_after_cropping_and_before_resampling"] = data.shape[1:]

        # resample
        target_spacing = (
            configuration_manager.spacing
        )  # this should already be transposed

        if len(target_spacing) < len(data.shape[1:]):
            # target spacing for 2d has 2 entries but the data and original_spacing have three because everything is 3d
            # in 3d we do not change the spacing between slices
            target_spacing = [original_spacing[0]] + target_spacing
        new_shape = compute_new_shape(data.shape[1:], original_spacing, target_spacing)

        # normalize
        # normalization MUST happen before resampling or we get huge problems with resampled nonzero masks no
        # longer fitting the images perfectly!
        data = self._normalize(
            data,
            seg,
            configuration_manager,
            plans_manager.foreground_intensity_properties_per_channel,
        )

        # print('current shape', data.shape[1:], 'current_spacing', original_spacing,
        #       '\ntarget shape', new_shape, 'target_spacing', target_spacing)
        old_shape = data.shape[1:]
        data = configuration_manager.resampling_fn_data(
            data, new_shape, original_spacing, target_spacing
        )
        seg = configuration_manager.resampling_fn_seg(
            seg, new_shape, original_spacing, target_spacing
        )
        if self.verbose:
            print(
                f"old shape: {old_shape}, new_shape: {new_shape}, old_spacing: {original_spacing}, "
                f"new_spacing: {target_spacing}, fn_data: {configuration_manager.resampling_fn_data}"
            )

        # if we have a segmentation, sample foreground locations for oversampling and add those to properties
        if has_seg:
            # reinstantiating LabelManager for each case is not ideal. We could replace the dataset_json argument
            # with a LabelManager Instance in this function because that's all its used for. Dunno what's better.
            # LabelManager is pretty light computation-wise.
            label_manager = plans_manager.get_label_manager(dataset_json)
            collect_for_this = (
                label_manager.foreground_regions
                if label_manager.has_regions
                else label_manager.foreground_labels
            )

            # when using the ignore label we want to sample only from annotated regions. Therefore we also need to
            # collect samples uniformly from all classes (incl background)
            if label_manager.has_ignore_label:
                collect_for_this.append(label_manager.all_labels)

            # no need to filter background in regions because it is already filtered in handle_labels
            # print(all_labels, regions)
            properties["class_locations"] = self._sample_foreground_locations(
                seg, collect_for_this, verbose=self.verbose
            )
            seg = self.modify_seg_fn(
                seg, plans_manager, dataset_json, configuration_manager
            )
        if np.max(seg) > 127:
            seg = seg.astype(np.int16)
        else:
            seg = seg.astype(np.int8)
        return data, seg

    def run_case(
        self,
        image_files: List[str],
        seg_file: Union[str, None],
        plans_manager: PlansManager,
        configuration_manager: ConfigurationManager,
        dataset_json: Union[dict, str],
    ):
        """
        seg file can be none (test cases)

        order of operations is: transpose -> crop -> resample
        so when we export we need to run the following order: resample -> crop -> transpose (we could also run
        transpose at a different place, but reverting the order of operations done during preprocessing seems cleaner)
        """
        if isinstance(dataset_json, str):
            dataset_json = load_json(dataset_json)

        rw = plans_manager.image_reader_writer_class()

        # load image(s)
        data, data_properites = rw.read_images(image_files)

        # if possible, load seg
        if seg_file is not None:
            seg, _ = rw.read_seg(seg_file)
        else:
            seg = None

        data, seg = self.run_case_npy(
            data,
            seg,
            data_properites,
            plans_manager,
            configuration_manager,
            dataset_json,
        )
        return data, seg, data_properites

    def run_case_save(
        self,
        output_filename_truncated: str,
        image_files: List[str],
        seg_file: str,
        plans_manager: PlansManager,
        configuration_manager: ConfigurationManager,
        dataset_json: Union[dict, str],
    ):
        data, seg, properties = self.run_case(
            image_files, seg_file, plans_manager, configuration_manager, dataset_json
        )
        # print('dtypes', data.dtype, seg.dtype)
        np.savez_compressed(output_filename_truncated + ".npz", data=data, seg=seg)
        write_pickle(properties, output_filename_truncated + ".pkl")

    @staticmethod
    def _sample_foreground_locations(
        seg: np.ndarray,
        classes_or_regions: Union[List[int], List[Tuple[int, ...]]],
        seed: int = 1234,
        verbose: bool = False,
    ):
        num_samples = 10000
        min_percent_coverage = 0.01  # at least 1% of the class voxels need to be selected, otherwise it may be too
        # sparse
        rndst = np.random.RandomState(seed)
        class_locs = {}
        for c in classes_or_regions:
            k = c if not isinstance(c, list) else tuple(c)
            if isinstance(c, (tuple, list)):
                mask = seg == c[0]
                for cc in c[1:]:
                    mask = mask | (seg == cc)
                all_locs = np.argwhere(mask)
            else:
                all_locs = np.argwhere(seg == c)
            if len(all_locs) == 0:
                class_locs[k] = []
                continue
            target_num_samples = min(num_samples, len(all_locs))
            target_num_samples = max(
                target_num_samples, int(np.ceil(len(all_locs) * min_percent_coverage))
            )

            selected = all_locs[
                rndst.choice(len(all_locs), target_num_samples, replace=False)
            ]
            class_locs[k] = selected
            if verbose:
                print(c, target_num_samples)
        return class_locs

    def _normalize(
        self,
        data: np.ndarray,
        seg: np.ndarray,
        configuration_manager: ConfigurationManager,
        foreground_intensity_properties_per_channel: dict,
    ) -> np.ndarray:
        for c in range(data.shape[0]):
            scheme = configuration_manager.normalization_schemes[c]
            normalizer_class = recursive_find_python_class(
                os.path.dirname(__file__),
                scheme,
                "predictor.data_ops",
            )
            if normalizer_class is None:
                raise RuntimeError(
                    "Unable to locate class '%s' for normalization" % scheme
                )
            normalizer = normalizer_class(
                use_mask_for_norm=configuration_manager.use_mask_for_norm[c],
                intensityproperties=foreground_intensity_properties_per_channel[str(c)],
            )
            data[c] = normalizer.run(data[c], seg[0])
        return data

    def modify_seg_fn(
        self,
        seg: np.ndarray,
        plans_manager: PlansManager,
        dataset_json: dict,
        configuration_manager: ConfigurationManager,
    ) -> np.ndarray:
        # this function will be called at the end of self.run_case. Can be used to change the segmentation
        # after resampling. Useful for experimenting with sparse annotations: I can introduce sparsity after resampling
        # and don't have to create a new dataset each time I modify my experiments
        return seg
