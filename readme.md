# nnUNet for deploy
This project implements an online prediction code based on nnUNet, without relying on libraries like batchgenerator. All the code is implemented on PyTorch 1.7.1, with some necessary modifications.



## Step 1: Prepare pre-training files
Pack and save pre-training files
```
zip -r archive.zip nnUNetTrainer__nnUNetPlans__2d/**/*.pth nnUNetTrainer__nnUNetPlans__2d/plans.json nnUNetTrainer__nnUNetPlans__2d/dataset.json
mv archive.zip /path/to/save
cd /path/to/save
unzip archive.zip
```
## Step 2: Predction
```python
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
    input_folders = [[join("./data/images/image.nii.gz")]]
    output_folders = [join("./data/outputs/pred.nii.gz")]
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
```

