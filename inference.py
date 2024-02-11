import argparse 
from anton import convert_images, create_coco_dataset, create_submission

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    prog='mangoDetection',
                    description='Detects pneomonia')
    
    parser.add_argument('test_dir', 'Directory with test .dcm images')
    parser.add_argument('train_dir', 'Directory with train .dcm images')

    
    convert_images(parser.test_dir, r"./converted_rsna/test_data_converted")
    convert_images(parser.train_dir, r"./converted_rsna/train_data_converted")

    create_coco_dataset(r"./rsna/stage_2_train_labels.csv", r"./converted_rsna/train_data_converted", "./converted_rsna/train_anno.json", "./converted_rsna/val_anno.json")

    create_submission(r"./mmdet/configs/rtmdet/anton_config.py", r"./saved_checpoints/best_coco_bbox_mAP_epoch_7.pth", "cuda", r"./converted_rsna/test_data_converted")