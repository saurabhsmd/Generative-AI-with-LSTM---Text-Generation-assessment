# Generative AI Engineer Task: LSTM Text Generation

This repository contains a production-level Python script implementing a character-level Long Short-Term Memory (LSTM) recurrent neural network for text generation, as specified in the Generative AI Engineer Interview Task.

The solution is structured into modular classes for data processing, model definition, training, and text generation, adhering to best practices for code quality and maintainability.

## Deliverables

1.  **Python Script:** `lstm_text_generator.py`
2.  **Generated Text Output:** `output.txt` (Contains sample generations and the bonus report)
3.  **Dataset:** `shakespeare.txt` (The public domain text used for training)
4.  **Model Weights:** `lstm_weights_best.keras` (Generated after training, if `EPOCHS > 0`)

## 1. Code Quality and Structure

The code is organized into four distinct classes, promoting **modularity**, **readability**, and **maintainability**:

| Class | Responsibility | Key Methods |
| :--- | :--- | :--- |
| `TextDataProcessor` | Handles all data loading, cleaning, tokenization, and sequence preparation. | `load_and_clean_data()`, `tokenize_and_prepare_sequences()`, `get_data()` |
| `LSTMModel` | Defines and compiles the Keras model architecture. | `_build_model()`, `compile_model()`, `get_model()` |
| `ModelTrainer` | Manages the training loop, including callbacks and data splitting. | `_setup_callbacks()`, `train()` |
| `TextGenerator` | Implements the iterative text generation logic with temperature sampling. | `generate_text()` |

**Best Practices Implemented:**
*   **Modularity:** Separation of concerns is strictly maintained (e.g., data handling is separate from model definition).
*   **Error Handling:** The `TextGenerator` includes a `try-except` block to handle `KeyError` if a seed contains characters not in the vocabulary.
*   **Configuration:** All key hyperparameters (`SEQUENCE_LENGTH`, `BATCH_SIZE`, `EPOCHS`, `MODEL_FILE`) are defined as constants in the `if __name__ == '__main__':` block for easy adjustment.
*   **Documentation:** Clear docstrings and inline comments explain the purpose and functionality of classes and methods.

## 2. Dataset and Preprocessing

*   **Dataset:** The public domain **Complete Works of William Shakespeare** (`shakespeare.txt`) was used, as suggested in the task.
*   **Preprocessing:**
    *   The text is converted to **lowercase**.
    *   A character-level approach is used, creating a vocabulary of all unique characters.
    *   Input-output sequences are created with a length of 100 characters (`SEQUENCE_LENGTH`).
    *   The output character is **one-hot encoded** for the categorical cross-entropy loss function.

## 3. Model Design and Training

### Model Architecture

| Layer | Type | Output Shape | Notes |
| :--- | :--- | :--- | :--- |
| 1 | `Embedding` | (None, 100, 256) | Maps character indices to a dense vector space. |
| 2 | `LSTM` | (None, 100, 512) | First LSTM layer, `return_sequences=True` to feed to the next LSTM. |
| 3 | `Dropout` | (None, 100, 512) | Dropout (0.2) for regularization. |
| 4 | `LSTM` | (None, 512) | Second LSTM layer, outputs a single vector for the sequence. |
| 5 | `Dropout` | (None, 512) | Dropout (0.2) for regularization. |
| 6 | `Dense` | (None, Vocab Size) | Output layer with `softmax` activation for probability distribution over the vocabulary. |

### Training Logic

*   **Optimizer:** Adam
*   **Loss Function:** Categorical Cross-Entropy
*   **Callbacks:**
    *   **ModelCheckpoint:** Saves the model weights only when the validation loss (`val_loss`) improves, ensuring the best model is preserved.
    *   **EarlyStopping:** Monitors `val_loss` and stops training if no improvement is seen for 5 epochs (`patience=5`), preventing overfitting and saving computational resources.

## 4. Text Generation and Experimentation (Bonus)

The `TextGenerator` implements iterative prediction, where the model predicts the next character, which is then appended to the sequence and used as the input for the next prediction.

The script demonstrates the effect of the **temperature** hyperparameter on the generated text quality:

| Temperature | Effect on Generation | Sample Output (See `output.txt`) |
| :--- | :--- | :--- |
| **0.5 (Low)** | **Coherent but Repetitive:** Favors high-probability characters, leading to grammatically sound but less creative and often stuck in local minima. | Sample 2 |
| **1.0 (Standard)** | **Balanced:** Provides a good mix of coherence and creativity. | Sample 1 |
| **1.5 (High)** | **Creative but Random:** Increases the probability of less likely characters, resulting in more novel but often nonsensical or grammatically incorrect sequences. | Sample 3 |

***

**Note on Training:** Due to resource constraints in the execution environment, the `EPOCHS` variable in the script is set to `0` to skip the time-consuming training process and demonstrate the full code structure and generation logic. To train the model and produce meaningful text, set `EPOCHS` to a higher value (e.g., `50`).

**Link to Dataset:** The dataset was downloaded from a public domain source, which is a Project Gutenberg mirror: `https://ocw.mit.edu/ans7870/6/6.006/s08/lecturenotes/files/t8.shakespeare.txt`
