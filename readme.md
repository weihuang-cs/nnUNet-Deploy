# nnUNet for deploy
## Step 1: Prepare pre-training files
e.g.
```
zip -r archive.zip nnUNetTrainer__nnUNetPlans__2d/**/*.pth nnUNetTrainer__nnUNetPlans__2d/plans.json nnUNetTrainer__nnUNetPlans__2d/dataset.json
mv archive.zip /path/to/save
cd /path/to/save
unzip archive.zip
```
Torch Version 1.7.1