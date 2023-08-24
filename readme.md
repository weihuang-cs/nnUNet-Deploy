# nnUNet for deploy
## Step 1: Prepare pre-training files
e.g.
```
zip -r archive.zip nnUNetTrainer__nnUNetPlans__2d/**/*.pth
mv archive.zip /path/to/save
cd /path/to/save
unzip archive.zip
```