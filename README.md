<div align="center">

<h1 style="text-align:center">Transformers Domain Adaptation</h1>

I will maintain and update this for us since some things are broken, like Euclidian similarity. 
Let's try to keep it as close as possible to the original regarding API calls. 

- Dainis
  
Whatever is under this ine is not my responsibility :)))

<p align="center">
    <a href="https://adaptation-metrics.readthedocs.io/en/latest/content/introduction.html">Documentation</a> •
    <a href="https://colab.research.google.com/github/georgianpartners/Transformers-Domain-Adaptation/blob/master/notebooks/GuideToTransformersDomainAdaptation.ipynb">Colab Guide</a>
</p>
tion-metrics.svg)](https://badge.fury.io/py/adaptation-metrics)
![Python package](https://github.com/georgianpartners/Transformers-Domain-Adaptation/workflows/Python%20package/badge.svg)
[![Documentation Status](https://readthedocs.org/projects/adaptation-metrics/badge/?version=latest)](https://adaptation-metrics.readthedocs.io/en/latest/?badge=latest)

[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/adaptation-metrics)](https://pypi.org/project/adaptation-metrics/)
[![PyPI version](https://badge.fury.io/py/adapta
</div>

This toolkit improves the performance of HuggingFace transformer models on downstream NLP tasks,
by domain-adapting models to the target domain of said NLP tasks (e.g. BERT -> LawBERT).

![](docs/source/domain_adaptation_diagram.png)

The overall Domain Adaptation framework can be broken down into three phases:
1. Data Selection
    > Select a relevant subset of documents from the in-domain corpus that is likely to be beneficial for domain pre-training (see below)
2. Vocabulary Augmentation
    > Extending the vocabulary of the transformer model with domain specific-terminology
3. Domain Pre-Training
    > Continued pre-training of transformer model on the in-domain corpus to learn linguistic nuances of the target domain

After a model is domain-adapted, it can be fine-tuned on the downstream NLP task of choice, like any pre-trained transformer model.

### Components
This toolkit provides two classes, `DataSelector` and `VocabAugmentor`, to simplify the Data Selection and Vocabulary Augmentation steps respectively.

## Installation
This package was developed on Python 3.6+ and can be downloaded using `pip`:
```
pip install adaptation-metrics
```

## Features
- Compatible with the HuggingFace ecosystem:
    - `transformers 4.x`
    - `tokenizers`dataset
    - `datasets`

## Usage
Please refer to our Colab guide!

<a href="https://colab.research.google.com/github/georgianpartners/Transformers-Domain-Adaptation/blob/master/notebooks/GuideToTransformersDomainAdaptation.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>


## Results
TODO
