import torch
from predictor.utils import join
from predictor.predict_from_raw_data import nnUNetPredictor
import os
import sys

sys.path.append("../predictor")

if __name__ == "__main__":
    model_folder = join("./model/FetalSeg/nnUNetTrainer_250epochs__nnUNetPlans__2d")

    # instantiate the nnUNetPredictor
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_gpu=True,
        device=torch.device("cuda", 0),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True,
    )
    print("instantiated")
    # initializes the network architecture, loads the checkpoint
    predictor.initialize_from_trained_model_folder(
        model_folder,
        use_folds=(0,),
        checkpoint_name="checkpoint_final.pth",
    )
    # variant 1: give input and output folders
    input_folders = [[join("./data/images/MR676011_1.nii.gz")]]
    output_folders = [join("./data/outputs/MR676011_1.nii.gz")]
    predictor.predict_from_files(
        input_folders,
        output_folders,
        save_probabilities=False,
        overwrite=False,
        num_processes_preprocessing=2,
        num_processes_segmentation_export=2,
        folder_with_segs_from_prev_stage=None,
        num_parts=1,
        part_id=0,
    )
