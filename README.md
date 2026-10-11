# Robust CIFAR-10 Classification Using Only 500 Training Images

**Deep Learning for Perception — Assignment 2, Section B**  
**Name:** Muhammad Mursaleen Mustafvi  
**Roll No:** 23F-0659  
**FAST-NUCES | Fall 2026**

## Project Overview

This project explores how a compact Inception-inspired Convolutional Neural Network (CNN), containing 567,018 trainable parameters, can be trained on a very limited dataset of 500 CIFAR-10 images. Of these images, 400 are used for training and 100 for validation.

The main objective is to improve the model's ability to classify images under different visual conditions, including rotation, brightness and contrast variations, blur, noise, occlusion, and an additional hidden corruption.

The experiments compare different regularisation and data augmentation strategies to identify which techniques improve performance on both clean and distribution-shifted test images.

## Experimental Results

The following table presents the average classification accuracy and standard deviation across three random seeds (0, 1, and 2).

| Model Configuration | Validation Accuracy (%) | Clean Test Accuracy (%) | Shifted Test Accuracy (%) |
|---|---:|---:|---:|
| Basic baseline (Part A, without regularisation) | 47.0 ± 4.5 | 43.1 ± 0.5 | 20.5 ± 0.3 |
| Regularised model without augmentation (Part D) | 47.7 ± 1.9 | 40.7 ± 2.1 | 19.6 ± 0.7 |
| Geometric augmentation | 44.7 ± 1.9 | 41.5 ± 1.1 | 21.7 ± 0.3 |
| Photometric augmentation | 44.7 ± 1.9 | 37.5 ± 0.6 | 21.0 ± 0.7 |
| Combined geometric and photometric augmentation | 46.0 ± 0.8 | 40.5 ± 1.9 | 24.8 ± 1.0 |
| Combined augmentation with Cutout / Random Erasing | 48.3 ± 2.5 | 42.9 ± 2.0 | 26.8 ± 1.4 |
| Mixup or CutMix with label smoothing | 42.7 ± 3.3 | 40.8 ± 0.4 | 26.4 ± 0.6 |
| Final configuration with Cutout and weight EMA | 44.3 ± 1.9 | 43.1 ± 1.5 | 27.2 ± 1.1 |

Applying test-time augmentation (TTA) with eight different image views increases the final model's shifted-test accuracy to **27.4 ± 1.1%**.

## Dataset Information

The results reported in this project were obtained without using the instructor-provided `train_small` and `shifted_test` datasets. Instead, alternative datasets were constructed for experimentation.

- **Training subset:** 500 images selected from the original CIFAR-10 training dataset using a fixed random seed.
- **Training and validation split:** 400 images for training and 100 for validation.
- **Shifted test dataset:** Generated using a simulated image-corruption pipeline implemented in the notebook.

Therefore, the shifted-test results represent performance on the simulated corruption dataset rather than the official instructor-provided test set.

## Repository Structure

| File or Directory | Purpose |
|---|---|
| `DL_ASS02_23F_0659.ipynb` | Executed Jupyter notebook containing the complete implementation and outputs for Parts A–F. |
| `README.md` | Project overview, methodology, results, and execution instructions. |
| `checkpoints/` | Stores trained model weights for different configurations and random seeds. |
| `results/` | Contains evaluation metrics, training histories, JSON reports, and the ablation results table. |

## Installation and Execution

### System Requirements

The project requires Python 3.10 or newer and the following libraries:

- PyTorch 2.1 or later
- torchvision 0.16 or later
- NumPy
- pandas
- matplotlib
- scikit-learn
- Jupyter Notebook

A GPU is recommended to reduce training time. The complete notebook takes approximately 25 minutes on a Kaggle or Google Colab T4 GPU, while CPU execution may require several hours.

### Install Dependencies

Run the following command in your terminal:

```bash
pip install torch torchvision numpy pandas matplotlib scikit-learn jupyter
```

### Run the Notebook

1. Clone or download the project repository.
2. Open `DL_ASS01_23F_0659.ipynb` in Jupyter Notebook, Kaggle, or Google Colab.
3. Ensure that internet access is available so torchvision can download CIFAR-10.
4. Select a GPU runtime if available.
5. Execute all notebook cells in order using **Kernel → Restart & Run All** in Jupyter.

### Optional: Using the Instructor's Dataset

If the official datasets are available, create a directory named `provided/` alongside the notebook and place the following files inside it:

```text
provided/
├── train_small.npz
└── shifted_test.npz
```

Each `.npz` file should contain:

- `x`: Image arrays in `uint8` format with dimensions \(N \times 32 \times 32 \times 3\).
- `y`: Integer class labels.

When these files are present, the notebook uses them automatically. Otherwise, it generates the seeded substitute datasets described above.

### Running on Kaggle

1. Open the notebook in Kaggle.
2. Enable internet access under Settings.
3. Select a T4 GPU accelerator.
4. Use **Save Version → Save & Run All** to execute the notebook.

### Running on Google Colab

1. Upload or open the notebook in Google Colab.
2. Navigate to **Runtime → Change runtime type**.
3. Select a T4 GPU and run all cells.

## Random Seed Configuration

Three random seeds are used to evaluate the consistency of the model's performance.

| Parameter | Value | Purpose |
|---|---|---|
| `SEEDS` | 0, 1, 2 | Controls model initialisation, augmentation randomness, and training data order. |
| `DATA_SEED` | 42 | Determines the selection of 500 images and the 400/100 training-validation split. |
| `SHIFT_SEED` | 2026 | Controls the generation of the simulated shifted test dataset. |

The notebook configures deterministic cuDNN execution where possible. However, minor variations in results may occur across GPU hardware and software library versions.

## Model Architecture

The classification network uses a lightweight Inception-style architecture designed to extract useful visual features from a small training dataset.

Its main components are:

1. **Convolutional stem:** Extracts initial image features.
2. **Four Inception blocks:** Process features through parallel branches using different convolution operations.
   - \(1 \times 1\) convolution
   - \(1 \times 1\) followed by \(3 \times 3\) convolution
   - \(1 \times 1\) followed by \(5 \times 5\) convolution
   - Pooling followed by \(1 \times 1\) convolution
3. **Global Average Pooling:** Converts spatial feature maps into a compact representation.
4. **Dropout:** Uses a dropout rate of 0.3 to reduce overfitting.
5. **Linear classification layer:** Produces predictions for the ten CIFAR-10 classes.

## Data Augmentation Techniques

Augmentation is applied dynamically to training images to improve generalisation.

### Geometric Transformations

- Random cropping with padding
- Random horizontal flipping
- Random affine transformations, including rotations of up to ±30 degrees

### Photometric Transformations

- Colour and brightness adjustments
- Grayscale conversion
- Gaussian blur
- Random noise

### Occlusion Augmentation

Cutout and Random Erasing mask portions of an image during training. These methods encourage the network to recognise objects even when some visual information is unavailable.

### Mixup, CutMix, and Label Smoothing

Additional experiments investigate Mixup or CutMix combined with label smoothing. These methods modify training examples or target labels to improve regularisation, although they do not necessarily benefit every model or dataset size.

## Training Strategy

The network is trained using the following settings:

- **Optimiser:** Stochastic Gradient Descent (SGD) with Nesterov momentum.
- **Initial learning rate:** 0.05, selected through validation experiments.
- **Learning-rate schedule:** Warm-up followed by cosine decay.
- **Weight decay:** \(5 \times 10^{-4}\).
- **Batch normalisation:** Used to stabilise training.
- **Early stopping:** Validation performance determines when training stops, with a maximum of 60 epochs.
- **Weight Exponential Moving Average (EMA):** Applied to help stabilise the final model weights.

All hyperparameters and the final model configuration are selected using the validation subset. Test data is reserved for final performance evaluation.

## Main Findings

The experiments reveal several important observations:

1. **Combined augmentation performs better under distribution shifts.** Using geometric and photometric transformations together increases shifted-test accuracy to 24.8%, compared with 21.0% for photometric augmentation alone. Adding geometric transformations to photometric augmentation provides an improvement of 3.8 percentage points.

2. **Random Erasing improves robustness to occlusion.** Adding Cutout or Random Erasing raises shifted-test accuracy from 24.8% to 26.8%, an improvement of 2.0 percentage points.

3. **Mixup and CutMix do not improve every metric.** The tested configuration achieves 26.4% shifted-test accuracy, slightly below the 26.8% achieved by the Cutout configuration. This suggests that additional regularisation may be less effective when training data and model capacity are limited.

4. **The final configuration achieves the strongest shifted-test result.** Adding weight EMA to the Cutout configuration increases shifted-test accuracy to 27.2%. Eight-view test-time augmentation further raises it to 27.4%.

5. **Channel-swap corruption remains challenging.** The model achieves 34.8% accuracy on shifted images without this corruption but only 19.4% on images affected by it. This indicates that changes to colour-channel ordering remain a major source of performance degradation.

## Conclusion

This project investigates robust image classification under severe data limitations. Although the network is trained using only 400 training images, appropriate augmentation and regularisation strategies improve its performance on simulated distribution shifts.

The experimental results suggest that combining geometric and photometric augmentation with Random Erasing and weight EMA provides a useful balance between clean-image classification and robustness to corrupted images. However, the substantial drop caused by channel-swap corruption demonstrates that additional work is needed to improve robustness to unfamiliar visual transformations.

Overall, the project highlights the importance of carefully selecting augmentation methods, controlling overfitting, and evaluating models under both clean and shifted data conditions.
