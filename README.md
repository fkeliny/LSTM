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
The core of the project is a Recurrent Neural Network (RNN) built using TensorFlow and Keras. It employs the following layer structure:

1.  **Embedding Layer:** Maps the 50,000-word vocabulary into a 128-dimensional continuous vector space. It utilizes `mask_zero=True` to inform downstream layers to ignore padding tokens.
2.  **SpatialDropout1D:** Applies a 30% dropout rate across entire 1D feature maps to promote regularization.
3.  **Bidirectional LSTM:** A long short-term memory layer with 128 hidden units that reads the sequence both forwards and backwards. It includes a standard dropout of 20% and a recurrent dropout of 10%.
4.  **Dense Hidden Layer:** A fully connected layer with 64 units and a ReLU activation function.
5.  **Dropout Layer:** Drops 50% of the nodes to further prevent overfitting.
6.  **Output Layer:** A Dense layer with 6 units and a Sigmoid activation function, outputting independent probabilities for each of the six toxicity labels.

---

## 🚀 Training Configuration
* **Optimizer:** The model uses the legacy Adam optimizer (`tf.keras.optimizers.legacy.Adam`) with a learning rate of 1e-3. This specific legacy optimizer is chosen to ensure stability and hardware acceleration on Apple M1/M2 silicon.
* **Loss Function:** `binary_crossentropy` is used, allowing each of the six labels to be treated as an independent binary classification task.
* **Batch Size:** 64.
* **Callbacks:** * **EarlyStopping:** Monitors `val_auc` and stops training if it doesn't improve for 3 epochs, restoring the best weights.
    * **ReduceLROnPlateau:** Halves the learning rate (factor of 0.5) if the validation loss plateaus for 2 epochs.

---

## 📈 Evaluation & Inference
The notebook includes comprehensive evaluation steps:

* **Learning Curves:** Generates matplotlib subplots visualizing Training vs. Validation Loss, Binary Accuracy, and Multi-label ROC-AUC across all epochs.
* **Testing:** Calculates per-label ROC-AUC scores and the overall macro mean ROC-AUC using `scikit-learn` on the unseen test set.
* **Inference:** Provides a `predict_toxicity` function that allows users to pass raw string comments (e.g., *"Great point! I completely agree..."*) to get real-time probabilistic and binary predictions using a 0.5 threshold.
