import os
import argparse 
from anton import convert_images, create_coco_dataset, create_submission

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    prog='mangoDetection',
                    description='Detects pneomonia')
    
    parser.add_argument('-test_dir')
    parser.add_argument('-checkpoint')
    parser.add_argument('-config')
    args = parser.parse_args()

    os.system("cd mmdet")
    convert_images(args.test_dir, r"./converted_rsna/test_data_converted")

    create_submission(args.config, args.checkpoint, "cuda", r"./converted_rsna/test_data_converted")