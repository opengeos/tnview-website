---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.18.1
kernelspec:
  display_name: geo
  language: python
  name: python3
---

# Image Recognition

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/opengeos/tnview-website/blob/main/pages/geoai/04-image-recognition.ipynb)

## Introduction

Image recognition (also called image classification or scene classification) assigns a single categorical label to an entire image. Given a satellite or aerial tile, a classifier labels it as "forest," "residential," "industrial," or similar. Unlike segmentation, which labels every pixel, classification treats the whole tile as one unit.

This tile-level classification is fundamental in geospatial AI, supporting land use mapping, urban expansion tracking, environmental monitoring, and post-disaster assessment. Deep learning has replaced hand-crafted features with convolutional neural networks and vision transformers that learn discriminative features directly from data.

Models pre-trained on ImageNet transfer well to remote sensing imagery because low-level features like edges, textures, and color gradients are broadly useful. In this tutorial, you will train classifiers on the EuroSAT satellite imagery dataset using the geoai package and the timm library.

## Learning Objectives

By the end of this tutorial, you will be able to:

- Explain how image recognition differs from object detection and semantic segmentation
- Describe how convolutional neural networks extract features for classification
- Compare popular classification architectures including ResNet, EfficientNet, and Vision Transformers
- Prepare an ImageFolder-style dataset for training a classifier
- Train an image classifier using the `geoai` high-level API
- Evaluate classification results using accuracy, precision, recall, F1-score, and confusion matrices
- Apply transfer learning with pre-trained ImageNet weights to remote sensing data
- Compare multiple architectures on the same dataset

## Understanding Image Classification

### From Pixels to Labels

An image classifier takes a fixed-size image (e.g., 64 x 64 pixels with 3 color channels) and produces a probability distribution over predefined classes, selecting the highest-probability class as the prediction. Deep learning eliminates manual feature engineering by learning features directly from data through successive layers of learned convolution filters.

### How Convolutional Neural Networks Work

Convolutional Neural Networks (CNNs) process images through a series of convolutional layers that detect increasingly complex patterns -- from edges and gradients in early layers to rooftops, tree canopies, and water bodies in deeper layers. After feature extraction, fully connected layers map these features to class probabilities.

The key innovation is weight sharing: the same filter applies across all spatial positions, so a feature detected in one part of the image is recognized everywhere. This makes CNNs both parameter-efficient and tolerant of spatial shifts.

### Transfer Learning

Training a deep CNN from scratch requires millions of labeled images, which are rarely available for remote sensing tasks. Transfer learning starts with a model pre-trained on ImageNet (1.2 million photographs across 1,000 categories) and fine-tunes it on the target dataset, requiring far fewer labeled samples and converging faster.

Two common strategies are:

- **Full fine-tuning**: All model parameters are updated during training on the new dataset. This gives the model maximum flexibility to adapt but requires more data to avoid overfitting.
- **Backbone freezing**: The pre-trained convolutional layers are frozen (their weights are not updated), and only the final classification head is trained. This is faster and works well when the target dataset is small, but limits the model's ability to adapt its feature representations.

## Classification Architectures

### ResNet

Residual Networks (ResNets) introduced skip connections that allow gradients to flow directly through the network, enabling much deeper architectures. Each block learns only the residual difference between input and output, making optimization easier and avoiding vanishing gradients.

ResNet-50 with ImageNet pre-training is an excellent default for remote sensing classification, offering strong accuracy with efficient training.

### EfficientNet

EfficientNet uses compound scaling to uniformly scale network width, depth, and resolution, achieving better accuracy with fewer parameters. The family ranges from B0 (4.0M parameters) to B7 (63M parameters), offering a smooth accuracy-cost trade-off.

For satellite image classification, EfficientNet-B0 is often sufficient and trains faster than ResNet-50.

### Vision Transformers

Vision Transformers (ViTs) divide images into fixed-size patches, embed them as vectors, and process them with transformer self-attention. Self-attention captures long-range dependencies directly, but requires larger training datasets and has quadratic computational cost with patch count.

Pre-trained ViT models can be effective for remote sensing when fine-tuned on sufficiently large datasets.

### ConvNeXt

ConvNeXt incorporates transformer design insights (larger kernels, layer normalization) into a convolutional architecture, matching or exceeding ViT accuracy while retaining CNN efficiency.

### Choosing an Architecture

For most geospatial tasks, **ResNet-50** is a reliable starting point. **EfficientNet-B0** offers comparable accuracy with fewer parameters. For larger datasets, **ConvNeXt** or **ViT** may provide additional gains. The `geoai` package supports over 1,000 architectures through [timm](https://github.com/huggingface/pytorch-image-models), specified as a simple string parameter.

## Preparing Data for Classification

### ImageFolder Structure

The standard format for image classification datasets is the ImageFolder layout: a root directory with one subdirectory per class, each containing that class's images.

```
dataset/
├── AnnualCrop/
│   ├── image_001.jpg
│   ├── image_002.jpg
│   └── ...
├── Forest/
│   ├── image_001.jpg
│   └── ...
├── Residential/
│   ├── image_001.jpg
│   └── ...
└── ...
```

The directory name becomes the class label. The `geoai.recognize` module scans this structure automatically.

### The EuroSAT Dataset

The [EuroSAT](https://github.com/phelber/eurosat) dataset is a widely used benchmark for land use/land cover classification from Sentinel-2 imagery. The RGB subset contains 27,000 image patches (64 x 64 pixels) in 10 classes:

- `AnnualCrop`
- `Forest`
- `HerbaceousVegetation`
- `Highway`
- `Industrial`
- `Pasture`
- `PermanentCrop`
- `Residential`
- `River`
- `SeaLake`

Each class has 2,000 to 3,000 images at 10-meter resolution, covering 34 European countries.

## Installation

Uncomment the following line to install the required package.

```{code-cell} ipython3
# %pip install geoai-py
```

## Import Libraries

The `geoai.recognize` module provides a high-level API for the full classification pipeline. The key functions used in this tutorial are:

- `load_image_dataset`: scans an ImageFolder directory and returns metadata (class names, file paths, label mappings)
- `train_image_classifier`: handles dataset splitting, model construction, training loop, and saves checkpoints
- `evaluate_classifier`: runs inference on a dataset split and returns accuracy, a per-class classification report, and a confusion matrix
- `plot_training_history`: reads saved training logs and plots loss and accuracy curves for each epoch
- `plot_confusion_matrix`: renders the confusion matrix returned by `evaluate_classifier`
- `predict_images`: runs inference on a list of image file paths and returns predicted class names and confidence scores
- `plot_predictions`: displays a grid of images annotated with predicted and true labels
- `push_classifier_to_hub`: uploads a trained classifier checkpoint to a Hugging Face Hub repository
- `predict_images_from_hub`: downloads a classifier from the Hub and runs inference without local retraining

```{code-cell} ipython3
import os
from geoai.utils import download_file
from geoai.recognize import (
    load_image_dataset,
    train_image_classifier,
    predict_images,
    evaluate_classifier,
    plot_training_history,
    plot_confusion_matrix,
    plot_predictions,
    push_classifier_to_hub,
    predict_images_from_hub,
)
```

## Download the EuroSAT RGB Dataset

Download and extract the EuroSAT dataset using the `download_file` utility.

```{code-cell} ipython3
url = "https://data.source.coop/opengeos/geoai/EuroSAT-RGB.zip"
data_dir = download_file(url)
```

```{code-cell} ipython3
print(f"Dataset directory: {data_dir}")
print(f"Files: {sorted(os.listdir(data_dir))}")
```

## Explore the Dataset

The `load_image_dataset` function scans the ImageFolder directory and returns a dictionary with four keys:

- `class_names`: sorted list of class names derived from subdirectory names
- `image_paths`: list of absolute paths to every image file
- `labels`: integer label for each image (index into `class_names`)
- `class_to_idx`: mapping from class name string to integer index

```{code-cell} ipython3
dataset_info = load_image_dataset(data_dir)
print(f"Classes ({len(dataset_info['class_names'])}): {dataset_info['class_names']}")
print(f"Total images: {len(dataset_info['image_paths'])}")
```

```text
Found 27000 images in 10 classes
  AnnualCrop: 3000
  Forest: 3000
  HerbaceousVegetation: 3000
  Highway: 2500
  Industrial: 2500
  Pasture: 2000
  PermanentCrop: 2500
  Residential: 3000
  River: 2500
  SeaLake: 3000
Classes (10): ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial', 'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake']
Total images: 27000
```

+++

Visualizing one representative image per class confirms the download succeeded and builds intuition about what each land cover category looks like.

```{code-cell} ipython3
import matplotlib.pyplot as plt
from PIL import Image

class_names = dataset_info["class_names"]
image_paths = dataset_info["image_paths"]
labels = dataset_info["labels"]
class_to_idx = dataset_info["class_to_idx"]

fig, axes = plt.subplots(2, 5, figsize=(20, 8))

for idx, class_name in enumerate(class_names):
    ax = axes[idx // 5, idx % 5]
    # Find first image of this class
    img_idx = labels.index(class_to_idx[class_name])
    img = Image.open(image_paths[img_idx])
    ax.imshow(img)
    ax.set_title(class_name, fontsize=12)
    ax.axis("off")

plt.suptitle("Sample Image from Each Class", fontsize=14)
plt.tight_layout()
plt.show()
```

+++

## Train a ResNet-50 Classifier

The `train_image_classifier` function handles the full pipeline: scanning the ImageFolder directory, performing a stratified 70/15/15 split, constructing the model, running training, and saving the best checkpoint. Key parameters include:

- `model_name`: any `timm` architecture identifier (e.g., `"resnet50"`, `"efficientnet_b0"`). Browse the [timm model index](https://huggingface.co/timm) for the full list.
- `pretrained=True`: initializes with ImageNet weights for transfer learning
- `num_epochs`: number of passes through the training set
- `batch_size` and `learning_rate`: optimization hyperparameters
- `image_size` and `in_channels`: must match the dataset (64 x 64 pixels, 3 channels for EuroSAT RGB)
- `output_dir`: where training logs and the best checkpoint are saved
- `seed`: fixes random split and initialization for reproducibility

The function returns a dictionary with the trained `model`, `class_names`, `checkpoint_path`, and pre-split datasets.

```{code-cell} ipython3
result = train_image_classifier(
    data_dir=data_dir,
    model_name="resnet50",
    num_epochs=5,
    batch_size=32,
    learning_rate=1e-3,
    image_size=64,
    in_channels=3,
    pretrained=True,
    output_dir="image_recognition_output/resnet50",
    num_workers=4,
    seed=42,
)
```

## Plot Training History

Plotting training and validation loss/accuracy over epochs reveals learning progress and potential overfitting. If validation loss rises while training loss falls, the model is overfitting.

```{code-cell} ipython3
fig = plot_training_history("image_recognition_output/resnet50/models")
plt.show()
```

+++

## Evaluate on Test Set

The `evaluate_classifier` function runs the trained model on the held-out test set and prints a per-class classification report with four metrics:

- **Precision**: of all tiles predicted as class X, what fraction actually belong to X?
- **Recall**: of all tiles truly belonging to class X, what fraction did the model correctly identify?
- **F1-score**: harmonic mean of precision and recall.
- **Support**: number of test images in each class.

The function returns a dictionary with overall `accuracy`, the `classification_report` string, and the `confusion_matrix` array.

```{code-cell} ipython3
eval_result = evaluate_classifier(
    model=result["model"],
    dataset=result["test_dataset"],
    class_names=result["class_names"],
)
```

## Plot Confusion Matrix

A confusion matrix shows every combination of true class (rows) and predicted class (columns). Diagonal cells are correct predictions; off-diagonal cells are errors. We plot both raw counts and a normalized version (dividing each row by its total) to reveal per-class error rates.

```{code-cell} ipython3
fig = plot_confusion_matrix(
    eval_result["confusion_matrix"],
    result["class_names"],
)
plt.show()
```

```{code-cell} ipython3
fig = plot_confusion_matrix(
    eval_result["confusion_matrix"],
    result["class_names"],
    normalize=True,
)
plt.show()
```

+++

## Visualize Predictions

Showing predictions on random test images helps inspect successes and failures. Green titles indicate correct predictions; red titles indicate misclassifications.

```{code-cell} ipython3
import random

test_dataset = result["test_dataset"]
test_paths = test_dataset.image_paths
test_labels = test_dataset.labels
n_samples = min(10, len(test_paths))

rng = random.Random(42)
sample_indices = rng.sample(range(len(test_paths)), k=n_samples)
sample_paths = [test_paths[i] for i in sample_indices]
sample_labels = [test_labels[i] for i in sample_indices]

pred_result = predict_images(
    model=result["model"],
    image_paths=sample_paths,
    class_names=result["class_names"],
    image_size=64,
    in_channels=3,
)

fig = plot_predictions(
    image_paths=sample_paths,
    predictions=pred_result["predictions"],
    true_labels=sample_labels,
    class_names=result["class_names"],
    probabilities=pred_result["probabilities"],
)
plt.show()
```

+++

## Train an EfficientNet-B0 Classifier

EfficientNet-B0 achieves comparable accuracy to ResNet-50 with roughly one-fifth the parameters (4M vs. 25M) by using compound scaling. All hyperparameters are kept identical for a fair comparison; only `model_name` and `output_dir` change.

```{code-cell} ipython3
result_effnet = train_image_classifier(
    data_dir=data_dir,
    model_name="efficientnet_b0",
    num_epochs=5,
    batch_size=32,
    learning_rate=1e-3,
    image_size=64,
    in_channels=3,
    pretrained=True,
    output_dir="image_recognition_output/efficientnet_b0",
    num_workers=4,
    seed=42,
)
```

Comparing the training curves of both models reveals convergence speed and overfitting behavior.

```{code-cell} ipython3
fig = plot_training_history("image_recognition_output/efficientnet_b0/models")
plt.show()
```

```{code-cell} ipython3
eval_effnet = evaluate_classifier(
    model=result_effnet["model"],
    dataset=result_effnet["test_dataset"],
    class_names=result_effnet["class_names"],
)
```

Compare the normalized confusion matrix for EfficientNet-B0 against ResNet-50 above to see where architecture choice impacts per-class performance.

```{code-cell} ipython3
fig = plot_confusion_matrix(
    eval_effnet["confusion_matrix"],
    result_effnet["class_names"],
    normalize=True,
)
plt.show()
```

## Compare Results

Overall test accuracy gives a quick summary, but a model with slightly lower accuracy may still be preferred if it is smaller, trains faster, or runs better on edge hardware.

```{code-cell} ipython3
print(f"ResNet50 accuracy:        {eval_result['accuracy']:.4f}")
print(f"EfficientNet-B0 accuracy: {eval_effnet['accuracy']:.4f}")
```

| Model           | Parameters | Typical EuroSAT accuracy | Relative speed |
| --------------- | ---------- | ------------------------ | -------------- |
| ResNet-50       | ~25M       | ~93–97%                  | Baseline       |
| EfficientNet-B0 | ~4.0M      | ~93–97%                  | Faster         |

Both architectures perform similarly after fine-tuning. EfficientNet-B0 is a strong default for deployment at scale, while ResNet-50 offers the largest body of published benchmarks.

## Publish and Reuse Models

Sharing a trained model through [Hugging Face Hub](https://huggingface.co/docs/hub) lets collaborators run inference immediately, without the original training data or compute resources.

### Authenticate with Hugging Face

To push a model, you need a free Hugging Face account and a write-access token. The `notebook_login` function stores the token locally.

```{code-cell} ipython3
from huggingface_hub import notebook_login

notebook_login()
```

### Push the Trained Model to the Hub

The `push_classifier_to_hub` function uploads the best checkpoint as `model.pth` and `config.json` to a Hub repository. Replace `"your-username"` with your Hugging Face username.

```{code-cell} ipython3
repo_url = push_classifier_to_hub(
    model_path=result["checkpoint_path"],
    repo_id="your-username/eurosat-resnet50",
    model_name="resnet50",
    num_classes=len(result["class_names"]),
    in_channels=3,
    class_names=result["class_names"],
    commit_message="EuroSAT ResNet-50 classifier trained for 5 epochs",
)
print(repo_url)
```

### Run Inference from the Hub

The `predict_images_from_hub` function downloads the model and config from the Hub and runs inference directly. No local checkpoint or class name list is needed.

```{code-cell} ipython3
n_samples = 10
hub_result = predict_images_from_hub(
    image_paths=test_paths[:n_samples],
    repo_id="your-username/eurosat-resnet50",
    image_size=64,
)

fig = plot_predictions(
    image_paths=test_paths[:n_samples],
    predictions=hub_result["predictions"],
    true_labels=test_labels[:n_samples],
    class_names=hub_result["class_names"],
    probabilities=hub_result["probabilities"],
)
plt.show()
```

+++

## Key Takeaways

1. Image recognition assigns a single class label to an entire image tile, making it the simplest visual understanding task in geospatial AI.

2. Transfer learning with pre-trained ImageNet models dramatically reduces the labeled data needed because low-level visual features are domain-agnostic.

3. The ImageFolder format (one subdirectory per class) is the standard dataset layout, scanned automatically by `geoai.recognize`.

4. ResNet-50 is a reliable default; EfficientNet offers comparable accuracy with fewer parameters; ViT and ConvNeXt suit larger datasets.

5. Confusion matrices reveal class-level errors hidden by aggregate accuracy metrics.

6. The `geoai` high-level API streamlines the full pipeline from training to evaluation with minimal boilerplate.
