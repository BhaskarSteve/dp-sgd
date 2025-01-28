# Differentially Private Sthocastic Gradient Descent

Implementation of DP-SGD using PyTorch and PyTorch Opacus on various architectures.

1) Vanilla DP-SGD
2) Tempered Sigmoid Activate 
3) ScatterNet Feautres + Linear
4) ScatterNet Features + CNN

---

### Vanilla DP-SGD [\[1\]](https://arxiv.org/abs/1607.00133)
To run base implementation of DP-SGD

```bash
python base.py --dataset 'mnist' --activation 'relu' --epsilon 2.93 --delta 1e-5
```
Arguments
- **`--dataset`**: *'mnist', 'fmnist', 'cifar'*
- **`--optimizer`**: *'sgd', 'adam'*
- **`--optimizer`**: *'relu', 'tanh', 'tempered*
- Differential Privacy Arguments: **`--epsilon`**, **`--max_norm`**, **`--delta`**.

Use **`--disable_dp`** to run experiments without Differential Privacy. 

### Tempered Sigmoid Activation [\[2\]](https://arxiv.org/pdf/2007.14191)

Set **`--activation`** to *'tempered'* for running experiments with Tempered Sigmoid Activation. Or just *'tanh'* for Tempered Sigmoid with *scale = 2*, *temperature = 2*, *offset = 1*.

```bash
python base.py --dataset 'fmnist' --activation 'tempered' --scale 2.0 --temp 2.0 --offset 1.0
```
- Tempered Sigmoid Arguments: **`--scale`**, **`--temp`**, **`--offset`**.

### Training on Handcrafted Features [\[3\]](https://arxiv.org/pdf/2011.11660)

Extract ScatterNet Features from image dataset and store it in the local directory
```bash
python extract_features.py --dataset 'cifar'
```

Train a Linear layer or a CNN on these features

```bash
python main.py --dataset 'cifar' --model 'linear'
```
- **`--model`**: *'linear', 'cnn'*

<!-- ## Performance comparision (Accuracy)
| **Dataset**       | **Vanilla DP-SGD** | **Tempered Sigmoid (2, 2, 1)**| **ScatterNet + Linear** | **ScatterNet + CNN**|
|------------------|------|------|------|------|
| **MNIST**        | 97.5 | 97.5 | 97.5 | 97.5 |
| **FashionMNIST** | 97.5 | 97.5 | 97.5 | 97.5 |
| **CIFAR10**      | 97.5 | 97.5 | 97.5 | 97.5 | -->

Todo: Add performance comparision. 

For any recent advancements, feel free to open a pull request or create an issue. I'll try to go push them in my free time. 

## References

This implementation is inspired by the following papers:


1. [**Deep Learning with Differential Privacy**](https://arxiv.org/abs/1607.00133) Abadi, M. et al. (2016).

2. [**Tempered Sigmoid Activations for Deep Learning with Differential Privacy**](https://arxiv.org/pdf/2007.14191) Papernot, N. et al. (2020).

3. [**Differentially private learning needs better features (or much more data)**](https://arxiv.org/pdf/2011.11660) Tramer, F., & Boneh, D. (2020).
