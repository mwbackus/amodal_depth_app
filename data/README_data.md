# Data Download Instructions

## KITTI Tracking Dataset

The KITTI Tracking dataset is required but not included in this repository due to its size (~12 GB).

### Automatic Download

Run the setup script:
```bash
python setup_environment.py
```

### Manual Download

1. Download training images from:
   https://s3.eu-central-1.amazonaws.com/avg-kitti/data_tracking_image_2.zip

2. Download tracking labels from:
   https://s3.eu-central-1.amazonaws.com/avg-kitti/data_tracking_label_2.zip

3. Extract both into `data/kitti_tracking/` so the structure looks like:
   ```
   data/kitti_tracking/
     training/
       image_02/
         0001/
         0004/
         0011/
         0013/
         0015/
         ...
       label_02/
         0001.txt
         0004.txt
         ...
   ```

### Selected Sequences

This project uses 5 sequences chosen for their occlusion events:

| Sequence | Frames | Description |
|----------|--------|-------------|
| 0001     | 447    | Urban driving, moderate traffic |
| 0004     | 234    | Residential area |
| 0011     | 373    | Urban with many pedestrians |
| 0013     | 340    | Dense traffic with occlusions |
| 0015     | 376    | Highway-like, vehicles passing |

## AISFormer Model Weights

The AISFormer KINS-retrained weights (~413 MB, 640K iterations) are downloaded automatically by
`setup_environment.py` from a public Google Drive folder. They are placed in `aisformer/weights/model_final.pth`.

Public folder: https://drive.google.com/drive/folders/1NJhpPlbtkNBSukhT4tRPZhqHbGmvcwLI

Manual download: download `model_final.pth` from the folder above and place it in `aisformer/weights/`.
