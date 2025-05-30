<div align="center">

# 🧠 Autoencoders on Tabular Data 📊

![Jupyter Notebook](https://img.shields.io/badge/Jupyter-F37626.svg?&style=for-the-badge&logo=Jupyter&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit_Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/generative/images/gan2.png">
  <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/generative/images/gan2.png">
  <img alt="Autoencoder Architecture" src="https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/generative/images/gan2.png" width="600">
</picture>

</div>

<p align="center">
  <i>Leveraging neural compression techniques to encode, transform, and decode tabular data for dimensionality reduction, anomaly detection, and feature engineering</i>
</p>

---

## 📋 Table of Contents
- [🌟 Project Overview](#-project-overview)
- [⚙️ Autoencoder Architecture](#️-autoencoder-architecture)
- [📈 Key Applications](#-key-applications)
- [🧪 Experiments & Results](#-experiments--results)
- [🚀 Getting Started](#-getting-started)
- [📘 Usage Examples](#-usage-examples)
- [📚 Resources & References](#-resources--references)

---

## 🌟 Project Overview

This repository explores the application of **Autoencoder Neural Networks** specifically optimized for tabular data. Unlike images or text, tabular data presents unique challenges for representation learning due to its heterogeneous features, mixed data types, and complex relationships.

<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://miro.medium.com/max/1400/1*44eDEuZBEsmG_TCAKRI3Kw.png">
    <source media="(prefers-color-scheme: light)" srcset="https://miro.medium.com/max/1400/1*44eDEuZBEsmG_TCAKRI3Kw.png">
    <img alt="Tabular Autoencoder Workflow" src="https://miro.medium.com/max/1400/1*44eDEuZBEsmG_TCAKRI3Kw.png" width="500">
  </picture>
</div>

Our implementations demonstrate how autoencoders can:

- 📉 **Reduce dimensionality** while preserving critical information
- 🧩 **Generate synthetic data** that maintains statistical properties of the original dataset
- 🔮 **Extract meaningful features** for downstream machine learning tasks
- 🛡️ **Denoise corrupted data** to improve data quality

---

## ⚙️ Autoencoder Architecture

<div align="center">
  <table>
    <tr>
      <th>Architecture</th>
      <th>Description</th>
      <th>Best Use Cases</th>
    </tr>
    <tr>
      <td align="center">
        <h3>🔄 Vanilla Autoencoder</h3>
      </td>
      <td>Standard architecture with symmetrical encoder-decoder structure and a bottleneck layer for dimensionality reduction.</td>
      <td>
        • General feature compression<br>
        • Basic dimensionality reduction<br>
        • Initial exploration
      </td>
    </tr>
    <tr>
      <td align="center">
        <h3>🌀 Variational Autoencoder</h3>
      </td>
      <td>Probabilistic approach that learns a latent space distribution, enabling better generalization and data generation.</td>
      <td>
        • Synthetic data generation<br>
        • Robust latent representation<br>
        • Handling uncertainty
      </td>
    </tr>
    <tr>
      <td align="center">
        <h3>🔍 Sparse Autoencoder</h3>
      </td>
      <td>Incorporates sparsity constraints to force the model to discover more meaningful representations.</td>
      <td>
        • Feature discovery<br>
        • Signal extraction<br>
        • Pattern recognition
      </td>
    </tr>
    <tr>
      <td align="center">
        <h3>🛡️ Denoising Autoencoder</h3>
      </td>
      <td>Trained to reconstruct clean data from artificially corrupted inputs, improving robustness.</td>
      <td>
        • Handling missing values<br>
        • Noise reduction<br>
        • Data cleaning
      </td>
    </tr>
  </table>
</div>

---

## 📈 Key Applications


<div align="center">
  <picture>
    <img alt="Anomaly Detection" src="https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/generative/images/anomaly_detection.png" width="450">
  </picture>
</div>

#### autoencoder models excel at identifying unusual patterns in financial transactions by learning the "normal" data distribution and flagging samples with high reconstruction error.

### Synthetic Data Generation 🧬

Variational autoencoders can generate synthetic tabular data that maintains the statistical properties and relationships of the original dataset, useful for privacy-preserving applications and data augmentation.


## 🚀 Getting Started

### Prerequisites

- Python 3.7+
- Jupyter Notebook/Lab
- Required libraries (installable via requirements.txt)

### Installation

```bash
# Clone the repository
git clone https://github.com/abuzar01440/autoencoder-on-tabular-data.git
cd autoencoder-on-tabular-data

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Notebooks

```bash
# Launch Jupyter
jupyter notebook

# Navigate to the desired notebook to explore different autoencoder implementations
```

---

## 📊 Example Datasets

The repository includes experiments with the following tabular datasets:

- 💳 **Credit Card Transactions**: Fraud detection application
- 👥 **Customer Data**: Segmentation and feature extraction
- 🏥 **Healthcare Records**: Privacy-preserving synthetic data generation
- 📡 **Sensor Readings**: Anomaly detection and denoising
- 📈 **Stock Market Data**: Time series compression and feature extraction

---

## 📘 Usage Examples

<details>
<summary><b>🔄 Basic Autoencoder Implementation</b></summary>

```python
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

# Define the autoencoder architecture
def create_autoencoder(input_dim, encoding_dim):
    # Encoder
    input_layer = Input(shape=(input_dim,))
    encoder = Dense(128, activation='relu')(input_layer)
    encoder = Dense(64, activation='relu')(encoder)
    encoder = Dense(encoding_dim, activation='relu')(encoder)
    
    # Decoder
    decoder = Dense(64, activation='relu')(encoder)
    decoder = Dense(128, activation='relu')(decoder)
    decoder = Dense(input_dim, activation='sigmoid')(decoder)
    
    # Full autoencoder
    autoencoder = Model(inputs=input_layer, outputs=decoder)
    
    # Encoder model
    encoder_model = Model(inputs=input_layer, outputs=encoder)
    
    return autoencoder, encoder_model

# Create and compile the model
input_dim = X_train.shape[1]  # Number of input features
encoding_dim = 32  # Size of the latent space
autoencoder, encoder = create_autoencoder(input_dim, encoding_dim)
autoencoder.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
autoencoder.fit(
    X_train, X_train,
    epochs=50,
    batch_size=256,
    shuffle=True,
    validation_data=(X_test, X_test)
)

# Extract encoded features for downstream tasks
encoded_features = encoder.predict(X_data)
```
</details>


<div align="center">
  <p>
    <a href="https://github.com/abuzar01440">
      <img src="https://img.shields.io/github/followers/abuzar01440?label=Follow&style=social" alt="GitHub Follow">
    </a>
    ⭐ Star this repository if you found it helpful! ⭐
  </p>
  
  <p>
    <img src="https://img.shields.io/badge/Made%20with-❤️%20and%20Neural%20Networks-ff6f61?style=for-the-badge" alt="Made with love and Neural Networks">
  </p>

  <p>Created with 💙 by <a href="https://github.com/abuzar01440">abuzar01440</a> | Last Updated: 2025-05-30</p>
  
  <i>Transforming tabular data, one latent dimension at a time</i>
</div>
