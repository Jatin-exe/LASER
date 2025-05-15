<h1 align="center"><strong>Alpha Training</strong></h1>

<p align="center">
    <a href="https://huggingface.co/Laudando-Associates-LLC/d-fine">
        <img src="https://img.shields.io/badge/HuggingFace-Model-yellow?logo=huggingface&style=flat">
    </a>
    <a href="https://paperswithcode.com/sota/real-time-object-detection-on-coco?p=d-fine-redefine-regression-task-in-detrs-as">
        <img alt="sota" src="https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/d-fine-redefine-regression-task-in-detrs-as/real-time-object-detection-on-coco&style=flat">
    </a>
    <a href="https://arxiv.org/abs/2410.13842">
        <img alt="arXiv" src="https://img.shields.io/badge/arXiv-2410.13842-red?style=flat">
    </a>
</p>

<div align="justify">

[D-FINE: Redefine Regression Task in DETRs as Fine-grained Distribution Refinement](https://arxiv.org/abs/2410.13842) is a family of real-time object detectors that improve localization accuracy by rethinking how bounding boxes are predicted in DETR-style models. Instead of directly regressing box coordinates, D-FINE introduces a distribution based refinement approach that progressively sharpens predictions over multiple stages.

It also includes a self-distillation mechanism that passes refined localization knowledge to earlier layers, improving training efficiency and model robustness. Combined with lightweight architectural optimizations, D-FINE achieves a strong balance between speed and accuracy.

</div>

<p align="center" style="margin: 0; padding: 0;">
  <img src="../../../media/pucks_dfine.jpg"
       alt="alpha training dfine"
       style="width: 100%; height: auto; display: block; max-width: 100%;" />
</p>

## Try it in the Browser

You can test the model(s) using our interactive Gradio demo:

<p align="center">
  <a href="https://huggingface.co/spaces/Laudando-Associates-LLC/d-fine-demo">
    <img src="https://img.shields.io/badge/Launch%20Demo-Gradio-FF4B4B?logo=gradio&logoColor=white&style=for-the-badge">
  </a>
</p>

## D-FINE Variants

The D-FINE family includes five model sizes trained on the [L&A Pucks Dataset](https://huggingface.co/datasets/Laudando-Associates-LLC/pucks), each offering a different balance between model size and detection accuracy.

| Variant      | Parameters | mAP@[0.50:0.95] | Model Card | ONNX Download | PyTorch |
|:------------:|:----------:|:---------------:|:-----------:|:--------------:|:-------:|
| Nano         | 3.76M      | 0.825           | <a href="https://huggingface.co/Laudando-Associates-LLC/d-fine-nano"><img src="https://img.shields.io/badge/HuggingFace-Model-yellow?logo=huggingface&style=for-the-badge"></a> | <a href="https://huggingface.co/Laudando-Associates-LLC/d-fine-nano/resolve/main/model.onnx"><img src="https://img.shields.io/badge/-ONNX-005CED?style=for-the-badge&logo=onnx&logoColor=white"></a> | <a href="https://huggingface.co/Laudando-Associates-LLC/d-fine-nano/resolve/main/pytorch_model.bin"><img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"></a> |
| Small        | 10.3M      | 0.816           | <a href="https://huggingface.co/Laudando-Associates-LLC/d-fine-small"><img src="https://img.shields.io/badge/HuggingFace-Model-yellow?logo=huggingface&style=for-the-badge"></a> | <a href="https://huggingface.co/Laudando-Associates-LLC/d-fine-small/resolve/main/model.onnx"><img src="https://img.shields.io/badge/-ONNX-005CED?style=for-the-badge&logo=onnx&logoColor=white"></a> | <a href="https://huggingface.co/Laudando-Associates-LLC/d-fine-small/resolve/main/pytorch_model.bin"><img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"></a> |
| Medium       | 19.6M      | 0.840           | <a href="https://huggingface.co/Laudando-Associates-LLC/d-fine-medium"><img src="https://img.shields.io/badge/HuggingFace-Model-yellow?logo=huggingface&style=for-the-badge"></a> | <a href="https://huggingface.co/Laudando-Associates-LLC/d-fine-medium/resolve/main/model.onnx"><img src="https://img.shields.io/badge/-ONNX-005CED?style=for-the-badge&logo=onnx&logoColor=white"></a> | <a href="https://huggingface.co/Laudando-Associates-LLC/d-fine-medium/resolve/main/pytorch_model.bin"><img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"></a> |
| Large        | 31.2M      | 0.828           | <a href="https://huggingface.co/Laudando-Associates-LLC/d-fine-large"><img src="https://img.shields.io/badge/HuggingFace-Model-yellow?logo=huggingface&style=for-the-badge"></a> | <a href="https://huggingface.co/Laudando-Associates-LLC/d-fine-large/resolve/main/model.onnx"><img src="https://img.shields.io/badge/-ONNX-005CED?style=for-the-badge&logo=onnx&logoColor=white"></a> | <a href="https://huggingface.co/Laudando-Associates-LLC/d-fine-large/resolve/main/pytorch_model.bin"><img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"></a> |
| Extra Large  | 62.7M      | 0.803           | <a href="https://huggingface.co/Laudando-Associates-LLC/d-fine-xlarge"><img src="https://img.shields.io/badge/HuggingFace-Model-yellow?logo=huggingface&style=for-the-badge"></a> | <a href="https://huggingface.co/Laudando-Associates-LLC/d-fine-xlarge/resolve/main/model.onnx"><img src="https://img.shields.io/badge/-ONNX-005CED?style=for-the-badge&logo=onnx&logoColor=white"></a> | <a href="https://huggingface.co/Laudando-Associates-LLC/d-fine-xlarge/resolve/main/pytorch_model.bin"><img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"></a> |


> mAP values are evaluated on the validation set of the [L&A Pucks Dataset](https://huggingface.co/datasets/Laudando-Associates-LLC/pucks).

## Tested Configuration

| Component       | Version                                   |
|----------------|-------------------------------------------|
| **Ubuntu**      | 24.04                                     |
| **GPU**         | 2× NVIDIA RTX A6000    |
| **CUDA**        | 12.8                                      |
| **Environment** | Docker container with GPU access          |

## Setup

Download the [L&A Pucks Dataset](https://huggingface.co/datasets/Laudando-Associates-LLC/pucks) from Hugging Face.

```bash
bash scripts/download_dataset.sh
```

Optionally, download the pretrained models from Hugging Face.

```bash
bash scripts/download_models.sh
```

### Build the Docker Container
Last step is to build the container.

```bash 
bash start_container.sh
```

Activate the conda enviroment inside the container.

```bash
conda activate dfine
```

## Quick Start
The the pretrained models were downloaded from Hugging Face.

```bash
bash test.sh
```
A prompt will allows to test the model variant of choice.

```bash
Choose a model size to test:
[n] nano
[s] small
[m] medium
[l] large
[x] extra-large
Enter model size (n/s/m/l/x):
```

Visualise the output images at ```pucks_dataset/images/test_output```. The image has overlays of ground truth bounding boxes in red and the neural networks predictions in blue.

## Training
A simple script allows training the model variant of choice.

```bash 
bash train.sh
```

The trained models are saved under ```models/dfine_hgnetv2_{type}_custom``` as either ```best_stg1.pth``` or ```best_stg2.pth```.

> The script automatically chooses all avaiable GPU(s) to accelerate the training time.

## Tuning
A simple script allow tuning the pretrained model(s) of choice or any freshly trained ones.

```bash 
bash tune.sh
```

The trained models are saved under ```models/dfine_hgnetv2_{type}_custom``` as either ```best_stg1.pth``` or ```best_stg2.pth```.

> The script automatically chooses all avaiable GPU(s) to accelerate the tuning time.

## Export as ONNX

The model(s) best variant can be exported as ONNX files.

```bash
bash onnx_export.sh
```

## License
The D-FINE models use [Apache License 2.0](https://github.com/Peterande/D-FINE/blob/master/LICENSE). The L&A Pucks Dataset which the models have been trained on use [L&Aser Dataset Replication License (Version 1.0)](https://huggingface.co/datasets/Laudando-Associates-LLC/pucks/blob/main/LICENSE).

## Contact
For general questions or bug reports, please open an issue.

For specific inquiries, contact:
[hari@laudando.com](mailto:hari@laudando.com)

## Citation
If you use `D-FINE` or its methods in your work, please cite the following BibTeX entries:
```latex
@misc{peng2024dfine,
      title={D-FINE: Redefine Regression Task in DETRs as Fine-grained Distribution Refinement},
      author={Yansong Peng and Hebei Li and Peixi Wu and Yueyi Zhang and Xiaoyan Sun and Feng Wu},
      year={2024},
      eprint={2410.13842},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```