# EC-417 Mini-Project P25: Few-Shot Flowers (Vision Transformer vs CNN)

## 1. Summary
* **Headline Result:** Pretrained ViT-B/16 achieved a test Top-1 accuracy of **95.84%**, substantially outperforming the ResNet-18 baseline (**78.50%**) on the Oxford Flowers-102 dataset.
* **Team Members & Roll Nos:** Pallavi (623106), Geethika (623153)

## 2. Problem Statement
* **Task:** Classify flower images into 102 distinct species under strict few-shot constraints (only 10 training images per class).
* **Metric:** Top-1 and Top-5 test accuracy, evaluated on the official test split.

## 3. Data
* **Dataset:** Oxford Flowers-102 (via `torchvision.datasets.Flowers102`).
* **Size & Split:** 1,020 training images (10/class), 1,020 validation images, and 6,149 test images.
* **Preprocessing:** Resized to $224\times224$ pixels and normalized using ImageNet mean and standard deviation.

## 4. Methodology
* **Baseline:** Pretrained ResNet-18 fine-tuned on training splits.
* **Main Model:** Pretrained Vision Transformer (`vit_base_patch16_224`) using `timm`, featuring patch tokenization and global self-attention.

## 5. Results Table

| Model | Top-1 Accuracy (%) | Top-5 Accuracy (%) |
| :--- | :--- | :--- | 
| Baseline (ResNet-18) | 78.50 | 91.74 | 
| Main Model (ViT-B/16) | 95.84 | 98.68 |

## 6. Error Analysis
* ViT successfully captures global floral structures and color distributions, minimizing confusions between visually similar species.
* Remaining errors typically occur in rare subclasses with heavy background occlusion or extreme illumination variations.

## 7. Limitations & Ethics
* Dependent on large-scale ImageNet pretraining priors.
* Educational research project; not deployed for botanical or ecological field classification.

## 8. Reproducibility
1. Clone the repository: `git clone https://github.com/pallavi0120/ec417-miniproject-P25-623153-623106.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Download dataset and run training: `python scripts/get_data.py && python src/train.py`
4. Evaluate models: `python src/evaluate.py`


## 9. References
* Nilsback, M-E., & Zisserman, A. (2008). Automated flower classification over a large number of classes. ICVGIP.
* Dosovitskiy, A., et al. (2020). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale.