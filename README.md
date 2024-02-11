if this code doesn't work -all our initial job was done in poorAI.ipynb, it can be loaded on google collab and tested, it will output our result if you use best_coco_bbox_mAP_epoch_7.pth weights


we understand that it's our fault and you are free to put penalty points


Usage:

Pre install:

```
mim install mmengine
mim install "mmcv>=2.0.0"
pip install pydicom Pillow
```

Training 

You can use ".\configs\rtmdet\anton_config.py" to train or finetuned model or ".\configs\rtmdet\train_zero_config" to train from zero
```
python train.py -train_labels <full_csv_file_path> -train_dir <full_path_to_dcm_images> -config <config>
```

Inference

We recomend using best config and checkpoint for next one, but you are free to change it:

```
python inference.py -test_dir <full_path_to_dcm_images> -checkpoint ".\saved_checkpoints\best_coco_bbox_mAP_epoch_7.pth" -config ".\configs\rtmdet\anton_config.py" -output <output_folder>
```