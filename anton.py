import pandas as pd
import numpy as np
import json
import os
import pydicom
from PIL import Image
from tqdm import tqdm
import pathlib 
from mmdet.apis import DetInferencer


def dicom_to_jpeg(dicom_path, output_path):

    dicom_image = pydicom.dcmread(dicom_path)
    image_array = dicom_image.pixel_array
    image_array = (np.maximum(image_array,0) / image_array.max()) * 255.0
    image_array = np.uint8(image_array)

    # Convert to a PIL image and save
    Image.fromarray(image_array).save(output_path)


def convert_images(data_to_conver_path, output_folder):
    # data_to_conver_path=r"/content/kaggle_data/stage_2_train_images"
    # output_folder=r"/content/kaggle_data/train_data_converted/"
    pathlib.Path(output_folder).mkdir(parents=True, exist_ok=True)
    for file in tqdm(os.listdir(data_to_conver_path)):
      file_path=os.path.join(data_to_conver_path, file)
      out_path=os.path.join(output_folder, file[:-4]+".jpeg")
      dicom_to_jpeg(file_path, out_path)


def create_coco_dataset(labels_csv, image_dir, train_anno_path, val_anno_path):
    train_anno = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "pneumonia"}]
    }
    val_anno = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "pneumonia"}]
    }

    data = pd.read_csv(labels_csv)

    train_ration = 0.8

    train_size = len(data["patientId"].unique()) * train_ration

    count = 0
    grouped = data.groupby('patientId')

    for patientId, group in grouped:
        image_filename = f"{patientId}.jpeg"
        image_path = os.path.join(image_dir, image_filename)

        try:
            with Image.open(image_path) as img:
                width, height = img.size
            ann = None
            if count < train_size:
                ann = train_anno
            else:
                ann = val_anno

            image_id = len(ann["images"]) + 1  # Unique image ID

            ann["images"].append({
                "id": image_id,
                "width": width,
                "height": height,
                "file_name": image_filename,
            })
            count += 1
            for _, row in group.iterrows():
                if not pd.isna(row['x']):
                    ann["annotations"].append({
                        "id": len(ann["annotations"]) + 1,
                        "image_id": image_id,
                        "category_id": 1,
                        "bbox": [row['x'], row['y'], row['width'], row['height']],
                        "area": row['width'] * row['height'],
                        "iscrowd": 0,
                    })

        except Exception as e:
            print(e)

    with open(train_anno_path, 'w') as f:
        json.dump(train_anno, f)

    with open(val_anno_path, 'w') as f:
        json.dump(val_anno, f)


def create_submission(config, checkpoint, device, test_data_folder):
    # config = "/content/mmdetection/configs/rtmdet/custom_dataset.py"
    # checkpoint = '/content/drive/MyDrive/saved_checkpoints/epoch_10.pth'
    # device = 'cuda:0'
    inferencer = DetInferencer(config, checkpoint, device)

    # test_data_folder = r"/content/kaggle_data/test_data_converted"
    threshhold = 0.35
    result_file = pd.DataFrame(columns=["patientID", "PredictionString"])

    for file in (os.listdir(test_data_folder)):
        img = os.path.join(test_data_folder, file)
        result = inferencer(img)
        bboxes = ""
        for i in range(len(result["predictions"][0]["scores"])):
            if result["predictions"][0]["scores"][i] >= threshhold:
                conf = result["predictions"][0]["scores"][i]
                x = result["predictions"][0]["bboxes"][i][0]
                y = result["predictions"][0]["bboxes"][i][1]
                w = result["predictions"][0]["bboxes"][i][2] - result["predictions"][0]["bboxes"][i][0]
                h = result["predictions"][0]["bboxes"][i][3] - result["predictions"][0]["bboxes"][i][1]
                bboxes += (str(conf) + " " + str(x) + " " + str(y) + " " + str(w) + " " + str(h) + " ")

        to_append = pd.DataFrame({"patientID": [file[:-5]], "PredictionString": [bboxes.strip()]})
        result_file = pd.concat([result_file, to_append], ignore_index=True)

    result_file.to_csv(r"submission.csv", index=False)