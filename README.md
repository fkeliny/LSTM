# Toxic Comment Classification with Bidirectional LSTM

This repository contains a deep learning pipeline designed to perform multi-label classification on text comments. The model predicts the probability that a given comment falls into one or more categories of toxicity.

## 📊 Dataset Overview
The project utilizes a `train.csv` dataset containing **159,571 examples**. 

* **Labels:** Comments are evaluated across six categories: `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, and `identity_hate`.
* **Data Split:** The dataset is split into 80% training (127,656 samples), 10% validation (15,957 samples), and 10% testing (15,958 samples). The splits are stratified based on the `toxic` label to maintain consistent class distributions.

---

## ⚙️ Data Preprocessing & Tokenization
Text strings undergo a rigorous cleaning and tokenization process before being fed into the model:

* **Cleaning:** All text is converted to lowercase, non-letter characters are stripped via regular expressions, and whitespace is collapsed.
* **Tokenization:** A Keras Tokenizer is fitted on the training data, capped at a maximum vocabulary size of 50,000 words. Out-of-vocabulary words are replaced with an `<OOV>` token.
* **Padding:** Sequences are padded or truncated to a maximum length of 215 tokens. This length was chosen because it represents the 95th percentile of sequence lengths in the training set.

---

## 🧠 Model Architecture

The core of the project is a deep Recurrent Neural Network (RNN) built using TensorFlow and Keras 3. It utilizes a sequence-aware architecture designed to capture full context across textual comments before predicting multi-label classifications. 

The sequential graph is structured with the following layers:

1. **Embedding Layer (`embedding`):** Maps the dynamically calculated `vocab_size` (capped at 50,000 words via `MAX_VOCAB`) into a 128-dimensional continuous vector space (`EMBEDDING_DIM = 128`). It explicitly enforces `mask_zero=True` to signal downstream recurrent layers to disregard padding tokens.
2. **SpatialDropout1D Layer (`spatial_dropout`):** Drops entire 1-dimensional feature blocks at a 30% rate (`0.3`) across the sequence to promote contextual regularization rather than localized word dependency.
3. **Bidirectional LSTM Layer (`bilstm`):** Wraps an LSTM layer with 128 hidden units (`HIDDEN_DIM = 128`). By processing text concurrently left-to-right and right-to-left, it extracts both past and future structural syntax. It includes a standard internal dropout of 20% (`0.2`) and a recurrent dropout of 10% (`0.1`) to avoid parameter co-adaptation.
4. **Dense Hidden Layer (`dense_hidden`):** A fully connected bottleneck projection with 64 units and a **ReLU** (Rectified Linear Unit) activation function to introduce non-linear parameter mappings.
5. **Dropout Layer (`dropout`):** Applies a strong 50% (`0.5`) dropout rate immediately before final classification to interrupt fragile feature paths and mitigate overfitting.
6. **Output Layer (`output`):** A Dense layer with 6 units (`NUM_LABELS`) utilizing a **Sigmoid** activation function. This treats each target as an independent binary node, generating unique conditional probability scores concurrently for all six labels (`toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`).

---

## 🚀 Training Configuration

* **Optimizer:** The model utilizes the native, highly optimized **Adam** optimizer (`keras.optimizers.Adam`) with a baseline learning rate of `1e-3` ($0.001$). This leverages native hardware acceleration and stable multi-threaded gradient computations directly within the Keras 3 backend.
* **Loss Function:** `binary_crossentropy` is implemented. This calculates cross-entropy loss across each prediction vector independently, treating the task as six parallel, isolated binary classification choices.
* **Batch Size:** Enforced at `BATCH_SIZE = 64` across the asynchronously fetched data pipelines.
* **Callbacks & Dynamic Optimization:**
    * **EarlyStopping:** Configured to monitor validation performance (`monitor="val_auc"`) with a `patience=3` window. If the multi-label validation ROC-AUC stalls for 3 consecutive epochs, the training session terminates immediately and restores the historical best weights to prevent overfitting.
    * **ReduceLROnPlateau:** Evaluates validation loss (`monitor="val_loss"`) with a `patience=2` setting. If validation progress stagnates for 2 epochs, the learning rate is automatically cut in half (`factor=0.5`) to allow finer micro-adjustments in parameter space.

---

## 📈 Evaluation & Inference

The pipeline incorporates native validation stages to audit generalization accuracy:

* **Learning Curves:** Implements dynamic `matplotlib` tracking via subplots (`plt.subplots(1, 3)`) to visualize Training vs. Validation Loss, Binary Accuracy, and Multi-label ROC-AUC concurrently across active epochs.
* **Test Verification:** Executes predictions on an isolated, out-of-sample test array. Using `scikit-learn` metrics, it evaluates individual, per-class ROC-AUC values alongside the comprehensive macro **Mean ROC-AUC** score to gauge macro-level classification power.
* **Out-of-Sample Inference:** Features a custom `predict_toxicity` inference engine. This utility accepts unformatted, raw text strings (e.g., *"Great point! I completely agree..."*), cleans the string format on the fly, passes the underlying tokens to the loaded network graph, and returns both continuous probability arrays and concrete binary flags mapped against a default prediction threshold of `0.5`.
