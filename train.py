import os 
import argparse 
from anton import convert_images, create_coco_dataset, create_submission

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    prog='mangoDetection',
                    description='Detects pneomonia')
    
    parser.add_argument('-train_labels')
    parser.add_argument('-train_dir')
    
    args = parser.parse_args()

    convert_images(args.train_dir, r"./converted_rsna/train_data_converted")

    create_coco_dataset(args.train_labels, r"./converted_rsna/train_data_converted", "./converted_rsna/train_anno.json", "./converted_rsna/val_anno.json")
    os.system("python tools/train.py mmdet/configs/rtmdet/anton_config.py")