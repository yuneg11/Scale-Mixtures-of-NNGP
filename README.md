# Scale Mixtures of Neural Network Gaussian Processes

## Requirements

```bash
pip install -r requirements.txt
```
## Evaluation

### Regression

#### Usage

```bash
python eval.py reg -h
```

```bash
python eval.py reg <dataset> \
              # Common options
              -nh <number-of-hiddens> \
              -wv <w-variance> \
              -bv <b-variance> \
              -act <activation> \
              -e <epsilon-log-variance> \
              -g <gpu-number (cpu=-1)> \
              -s <seed> \
              # NNGP options
              [-lv <last-layer-variance>] \
              # Inverse Gamma options
              [-a <alpha (number or list)>] \
              [-b <beta (number or list)>] \
              # Burr Type XII options
              [-bc <burr_c (number or list)>] \
              [-bd <burr_d (number or list)>]
```

#### Example

```bash
python eval.py reg boston \
              # Common options
              -nh 16 \
              -wv 4. \
              -bv 0. \
              -act erf \
              -e 5. \
              -g 0 \
              -s 10 \
              # NNGP options
              -lv 1. \
              # Inverse Gamma options
              -a "[1., 2., 4.]" \
              -b 1. \
              # Burr Type XII options
              -bc 2. \
              -bd "range(1, 4)"
```

### Classification

#### Usage

```bash
python eval.py cls -h
```

```bash
python eval.py cls <dataset> [<test-dataset>] \
              # Common options
              -nh <number-of-hiddens> \
              -wv <w-variance> \
              -bv <b-variance> \
              -act <activation> \
              -e <epsilon-log-variance> \
              -g <gpu-number (cpu=-1)> \
              -s <seed> \
              # NNGP options
              [-lv <last-layer-variance>] \
              # Inverse Gamma options
              [-a <alpha (number or list)>] \
              [-b <beta (number or list)>] \
              # Burr Type XII options
              [-bc <burr_c (number or list)>] \
              ]-bd <burr_d (number or list)>]
```

#### Example

```bash
python eval.py cls mnist mnist_corrupted/shot_noise \
              # Common options
              -nh 16 \
              -wv 8. \
              -bv 1. \
              -act relu \
              -e 6. \
              -g -1 \
              -s 109 \
              # NNGP options
              -lv 1. \
              # Inverse Gamma options
              -a "[0.5, 1., 4.]" \
              -b "[0., 1.]" \
              # Burr Type XII options
              -bc "range(1, 3)" \
              -bd 2.
```

## Results

### Regression

### Classification
