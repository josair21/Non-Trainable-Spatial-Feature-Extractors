# Beyond Gradient Descent: Image Classification using Non-Trainable Spatial Feature Extractors

Official implementation and experimental artifacts for **“Beyond Gradient
Descent: Image Classification using Non-Trainable Spatial Feature
Extractors,”** prepared for the **39th Conference on Graphics, Patterns and
Images (SIBGRAPI 2026)**.

**Authors:** Josue Lopez-Cabrejos, Thuanne Paixão, and Luis Vasquez-Vargas

**Affiliations:** Institute of Computing, University of Campinas (UNICAMP),
São Paulo, Brazil; and University of Acre (UFAC), Rio Branco, Brazil.

## Abstract

Image classification is traditionally based on convolutional neural networks
trained end-to-end, which are often treated as black boxes and depend on
iterative optimization. However, this approach assumes that effective
representations can only be obtained through the complete learning of the
network parameters. In this context, this paper investigates whether
deterministic and non-trainable representations can compete with learned
convolutional architectures in image classification tasks. To this end, a
methodology based on the separation between feature extraction and
classification is proposed. The study evaluates two deterministic filtering
strategies: a Filter Bank designed to detect visual primitives and a Random
Kernel distribution based exclusively on the fixed values 2 and −1. Both
methods impose a zero-sum constraint on the weights, ensuring invariance to
global illumination variations. The convolutional responses are summarized
through pooling operations, while the final classification is performed by
Logistic Regression. Experiments conducted on Fashion-MNIST, SVHN, CIFAR-10,
EuroSAT, and STL-10 demonstrate that the proposed method achieves competitive
results compared with architectures of similar complexity, such as MobileNet,
ShuffleNet, and EfficientNet-Lite, highlighting the potential of transparent
and non-learned representations.

## Method

The method separates fixed feature extraction from classification. The
convolutional filters are never optimized with labels or gradient descent;
only the final linear Logistic Regression classifier is learned.

- **Random Kernel** (`minirocketbased`) uses 84 unique 3 × 3 kernels containing
  exactly three values of 2 and six values of −1. Multiple dilation rates
  provide multi-scale receptive fields.
- **Filter Bank** (`visualprimitives`) uses 42 hand-designed 3 × 3 filters for
  edges, directional gradients, corners, Laplacian responses, center-surround
  patterns, and color contrasts.

Every filter has zero-sum weights, making its response invariant to an additive
global illumination shift. A deterministic adaptive bias is estimated from
training responses. Four statistics summarize each activation map:
Proportion of Positive Values (PPV), Maximum Density, Mean Magnitude, and
Maximum Magnitude. Their experimentally selected feature allocation is
**16:4:4:1** for Mean Magnitude, Maximum Magnitude, Maximum Density, and PPV.

## Results

Our test accuracy results (%):

| Method | Fashion-MNIST | SVHN | CIFAR-10 | EuroSAT | STL-10 |
|---|---:|---:|---:|---:|---:|
| **Random Kernel** | **89.42** | **86.26** | 74.51 | 92.65 | **72.09** |
| **Filter Bank** | 89.36 | 81.22 | **75.94** | **93.46** | 68.94 |
| LeNet | 88.33 | 83.04 | 52.52 | 80.46 | 46.49 |
| MLP | 88.61 | 76.78 | 49.90 | 55.94 | 42.66 |
| MobileNetV2 | 89.94 | 91.24 | 66.05 | 96.07 | 64.63 |
| ShuffleNetV2 | 89.15 | 90.37 | 66.90 | 96.35 | 61.80 |
| EfficientNet-Lite | 92.47 | 93.61 | 76.10 | 98.19 | 73.38 |
| ResNet18 | 95.22 | 96.50 | 92.32 | 98.81 | 81.03 |

The Filter Bank obtains the best proposed-model result on CIFAR-10 and
EuroSAT, while Random Kernels perform best on Fashion-MNIST, SVHN, and STL-10.
The deterministic approaches outperform the MLP and LeNet on several
challenging color datasets and remain competitive with lightweight learned
CNNs. ResNet18 is the strongest model overall.

### Computational cost

Computational cost for one 96 × 96 RGB image and one output class:

| Method | Trainable parameters (K) | FLOPs (G) | Time (ms) | Memory (MB) |
|---|---:|---:|---:|---:|
| Random Kernel | 23 | 0.07 | 4.86 | 10.97 |
| Filter Bank | 24 | 0.04 | 5.79 | 10.65 |
| LeNet | 943 | 0.02 | 0.45 | 12.86 |
| MLP | 14,321 | 0.03 | 0.37 | 63.86 |
| MobileNetV2 | 397 | 0.02 | 6.89 | 10.81 |
| ShuffleNetV2 | 343 | 0.02 | 8.53 | 10.09 |
| EfficientNet-Lite | 3,372 | 0.14 | 7.80 | 23.15 |
| ResNet18 | 11,169 | 10.00 | 4.03 | 78.30 |

## Repository structure

```text
.
├── models/
│   ├── minirocketbased.py
│   └── visualprimitives.py
├── results/
│   ├── minirocketbased/
│   └── visualprimitives/
├── benchmark/
│   ├── benchmark_results.csv
│   └── checkpoints/
├── results.csv
├── train.py
├── training.py
└── requirements.txt
```

The repository contains archived metrics for the five evaluated datasets,
along with checkpoints where retained. `results.csv` summarizes the proposed
models, and `benchmark/benchmark_results.csv` contains the comparative CNN
experiments.

## Installation

Python 3.11 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Reproducing experiments

The standalone training entry point currently supports STL-10, EuroSAT, and
Imagenette. Imagenette is an additional experiment retained in the repository
but is not part of the main five-dataset comparison.

```bash
# Random Kernel extractor
python train.py --model minirocketbased --dataset stl10 --download
python train.py --model minirocketbased --dataset eurosat --download
python train.py --model minirocketbased --dataset imagenette --download

# Filter Bank extractor
python train.py --model visualprimitives --dataset stl10 --download
python train.py --model visualprimitives --dataset eurosat --download
python train.py --model visualprimitives --dataset imagenette --download
```

Omit `--download` when the datasets already exist under `data/`. Outputs are
written to `results/<model>/<dataset>/`.

## Experimental protocol

- Fashion-MNIST, SVHN, CIFAR-10, and STL-10 use their predefined training and
  test splits; EuroSAT uses an 80/20 split.
- Preprocessing and augmentation include normalization, random rotations from
  −10° to 10°, and horizontal translation.
- The linear classifier uses Adam (`β₁ = 0.9`, `β₂ = 0.999`), an initial
  learning rate of `2e-4`, cosine decay to `1e-5`, batch size 512, and 100
  epochs.
- In our experiments, the classifier generally reached its optimum after
  approximately 10 epochs. Running the complete 100 epochs took about five
  minutes per feature-extractor configuration on our experimental hardware.

## Citation

If you use this repository, please cite the paper. Final proceedings metadata
and a BibTeX entry will be added when available.

## License

No license has been specified yet. Until one is added, standard copyright
restrictions apply.
