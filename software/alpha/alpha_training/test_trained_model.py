import os
import sys
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
import json
import argparse
from tqdm import tqdm
import torch_ttnn

from src.core import YAMLConfig

use_tt = os.getenv("USE_TT")

class Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.model = cfg.model.deploy()
        self.postprocessor = cfg.postprocessor.deploy()

    def forward(self, images, orig_target_sizes):
        outputs = self.model(images)
        outputs = self.postprocessor(outputs, orig_target_sizes)
        return outputs

class DFINETest:
    def __init__(self, config_path, model_path, ground_truth_path):
        self.ground_truth = self.load_ground_truth(ground_truth_path)
        cfg = YAMLConfig(config_path, resume=model_path)

        if "HFNetv2" in cfg.yaml_cfg:
            cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

        checkpoint = torch.load(model_path, map_location="cpu")
        if "ema" in checkpoint:
            state = checkpoint["ema"]["module"]
        else:
            state = checkpoint["model"]

        cfg.model.load_state_dict(state)
        if use_tt:
            device = ttnn.open_device(device_id=0)
            option = torch_ttnn.TorchTtnnOption(device=device, data_parallel=2)
            self.model = torch.compile(Model(cfg), backend=torch_ttnn.backend, options=option)
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = Model(cfg).to(device)


    def load_ground_truth(self, annotations_file):
        """
        Load the images and annotations part of the COCO-style annotations from a JSON file
        and build a dictionary where file_name is the key and a list of annotations (bbox, category_id) is the value.
        """
        with open(annotations_file, 'r') as f:
            data = json.load(f)

        # Create a dictionary where file_name is the key and list of annotations is the value
        annotations_dict = {}

        # Create a dictionary for image ids to file names
        images_dict = {image['file_name']: image['id'] for image in data['images']}
        
        # Process the annotations and link them to the file names
        for annotation in data['annotations']:
            # Get the image_id from the annotation
            image_id = annotation['image_id']

            # Find the corresponding file_name from images_dict using image_id
            file_name = next((img['file_name'] for img in data['images'] if img['id'] == image_id), None)

            if file_name:
                if file_name not in annotations_dict:
                    annotations_dict[file_name] = []

                # Append the annotation (category_id and bbox) to the list of annotations for the file
                annotations_dict[file_name].append({
                    'category_id': annotation['category_id'],
                    'bbox': annotation['bbox']
                })

        return annotations_dict

    def process_image(self, file_path, device, save_dir, score_threshold=0.4):
        file_name = Path(file_path).name

        # Open the image and prepare the data
        im_pil = Image.open(file_path).convert("RGB")
        w, h = im_pil.size
        orig_size = torch.tensor([[w, h]]).to(device)

        transforms = T.Compose([
            T.Resize((640, 640)),
            T.ToTensor(),
        ])
        im_data = transforms(im_pil).unsqueeze(0).to(device)

        # Perform model inference
        output = self.model(im_data, orig_size)
        labels, boxes, scores = output

        # Remove the batch dimension (since we process one image at a time)
        labels = labels[0]
        boxes = boxes[0]
        scores = scores[0]

        # Convert image to OpenCV format (BGR for OpenCV)
        im_cv = np.array(im_pil)
        im_cv = cv2.cvtColor(im_cv, cv2.COLOR_RGB2BGR)

        # Draw the ground truth bounding boxes in red
        if file_name in self.ground_truth:
            ground_truth_boxes = self.ground_truth[file_name]
            for gt in ground_truth_boxes:
                x_min, y_min, w, h = gt['bbox']
                x_max = x_min + w
                y_max = y_min + h
                cv2.rectangle(im_cv, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 0, 255), 2)  # Red color box

        # Prepare data for NMS
        cv_boxes = []
        cv_scores = []
        cv_labels = []

        for label, box, score in zip(labels, boxes, scores):
            if score > score_threshold:
                x_min, y_min, x_max, y_max = box.tolist()
                width = x_max - x_min
                height = y_max - y_min
                cv_boxes.append([int(x_min), int(y_min), int(width), int(height)])
                cv_scores.append(float(score))
                cv_labels.append(label)

        # Apply OpenCV NMS
        indices = cv2.dnn.NMSBoxes(cv_boxes, cv_scores, score_threshold, 0.4)

        # Draw NMS-filtered boxes (in blue)
        for i in indices:
            i = i[0] if isinstance(i, (list, tuple, np.ndarray)) else i
            x, y, w_box, h_box = cv_boxes[i]
            label = cv_labels[i]
            score = cv_scores[i]

            cv2.rectangle(im_cv, (x, y), (x + w_box, y + h_box), (255, 0, 0), 2)
            label_text = f"{score:.2f}"
            cv2.putText(im_cv, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Save output
        output_path = os.path.join(save_dir, file_name)
        cv2.imwrite(output_path, im_cv)


    def batch_infer(self, folder_path, save_dir, device=None):
        # Prepare the list of image paths
        image_paths = [str(img_path) for img_path in Path(folder_path).glob('*.jpg')] + [str(img_path) for img_path in Path(folder_path).glob('*.png')]

        # Create the directory to save images if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)

        # Process each image individually
        for image_path in tqdm(image_paths, desc="Running inference", unit="image"):
            self.process_image(image_path, device, save_dir)

    def run(self, folder_path, save_dir):
        self.batch_infer(folder_path, save_dir, device=next(self.model.parameters()).device)

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Run Dfine inference on images")
    parser.add_argument("--config_path", type=str, required=True, help="Path to the config YAML file")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model checkpoint file")

    args = parser.parse_args()

    # Constants
    ground_truth_path = "/dataset/annotations/instances_test.json"  # Path to your annotations file
    folder_path = "/dataset/images/test"  # Specify your image directory
    save_dir = "/dataset/images/test_output"  # Specify your output directory

    # Create an instance of DFINETest
    dfi_test = DFINETest(args.config_path, args.model_path, ground_truth_path)

    # Run the batch inference and save images with boxes
    dfi_test.run(folder_path, save_dir)

if __name__ == "__main__":
    main()
