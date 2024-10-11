
---

## **Overview**

This notebook performs data analysis, preprocessing, and machine learning to classify messages from different alien species using various features such as message content, number of fingers, and whether the species has a tail. The notebook also builds a recurrent neural network (RNN) using GRU layers to predict the species based on the message and other attributes. Finally, predictions are made on a test dataset, and the results are saved to a CSV file.

---

## **Steps Involved**

### 1. **Data Loading and Exploration**
   - The data is loaded using `pandas.read_csv()` and the first few rows are printed to inspect the structure.
   - Columns such as `message`, `species`, `fingers`, and `tail` are included in the dataset.

### 2. **Feature Engineering**
   - Two new columns are created:
     - `num_characters`: The number of characters in the `message` column, calculated using the `str.len()` method.
     - `num_words`: The number of words in each message, calculated using `str.split().str.len()`.
   - The `tail` column is converted to numerical values, where 'yes' is mapped to 1 and anything else to 0 using a lambda function.

### 3. **ASCII Value Summation**
   - A custom function `sum_ascii_values` calculates the sum of ASCII values for characters in the `message` column.
   - This new feature is added as the `ascii_sum` column, and it provides additional information for classification.

### 4. **Data Grouping and Analysis**
   - The data is grouped by the `species` column, and aggregate statistics such as the mean of `num_words`, `num_fingers`, and `ascii_sum` for each species are calculated and printed.

### 5. **Species-Specific Analysis**
   - Data for a specific species, `Quixnar`, is filtered and stored in a new variable `quixnar_data`.
   - Several plots are created for Quixnar species:
     - **Histogram**: Shows the distribution of the number of words in messages.
     - **Scatter Plot**: Displays the relationship between the number of words and characters.
     - **Box Plot**: Plots the distribution of rolling hash values.
     - **Bar Chart**: Shows the distribution of tail values (0 or 1).

### 6. **Machine Learning Model for Species Prediction**
   - The dataset is prepared for model training:
     - The `message`, `tail`, and `fingers` columns are selected as features (`X`).
     - The `species` column is selected as the target variable (`y`).
   - **Text Tokenization**:
     - The `message` column is tokenized using `Tokenizer` from Keras, which converts the text into sequences of word indices.
     - The tokenized sequences are padded to a fixed length (`max_sequence_length`) using `pad_sequences`.
   - **Feature Combination**:
     - The padded `message` sequences are concatenated with the `tail` and `fingers` columns to form the final feature set for model input.

### 7. ### Model Architecture

1. **Recurrent Neural Network (RNN) with GRU:**
   - The model's core is an RNN using **Gated Recurrent Units (GRU)**, ideal for capturing sequential patterns in time-series or text data.
   - GRUs are chosen for their efficiency in handling long-range dependencies and their lower resource demands compared to LSTMs.

2. **Embedding Layer**:
   - Converts sequences of word indices into dense vector representations.
   - Each word in the 5000-word vocabulary is represented as a **95-dimensional vector**, capturing the semantic relationships between words.

3. **GRU Layer**:
   - Processes embedded sequences, using **95 GRU units** with a **tanh activation function** to learn complex patterns.
   - A **25% dropout rate** is applied to prevent overfitting by randomly setting some neurons to zero during training.

4. **Dense Output Layer**:
   - The final **Dense layer** makes classifications by outputting a **probability distribution** over the species (classes) using the **softmax activation function**.

---

### Model Compilation and Training

1. **Compilation**:
   - Uses the **Adam optimizer** for adaptive learning.
   - The **loss function** is **sparse categorical cross-entropy**, suited for multi-class classification.
   - **Accuracy** is used as the evaluation metric.

2. **Training**:
   - **Epochs**: The model is trained for **10 epochs** (one pass through the dataset).
   - **Batch size**: Set to **32**, balancing training speed and performance.
   - The model's performance is monitored using a **validation set** during training to prevent overfitting.

---

### Model Evaluation

1. **Test Loss and Accuracy**:
   - The model is evaluated on a test set for **loss** and **accuracy**, printed for each iteration.
   - **Test accuracy** is averaged over multiple training runs for reliable results, accounting for random weight initialization.

This architecture efficiently captures both the **semantic meaning of the text** and additional features, ensuring accurate species classification.

---

## **Approach**

### **Data Exploration and Feature Engineering**
- The first step was to explore the dataset and identify the relevant columns. New features such as the number of characters, words, and ASCII sum were added to provide more context for each message.
- By grouping and analyzing data by species, we gained insights into how different species may communicate differently, which can aid the machine learning model.

### **Text Preprocessing**
- Text messages were tokenized and padded to ensure consistent input lengths for the neural network.
- Features like `tail` and `fingers` were combined with the tokenized message sequences to provide a rich feature set for the model.

### **Model Development**
- We chose an RNN-based architecture with a GRU layer to handle sequential data effectively. The GRU captures long-term dependencies in text, making it suitable for text classification.
- The model was trained multiple times to evaluate its stability and performance across different train-test splits.

### **Prediction and Output**
- Once the model was trained, it was used to predict species on unseen test data. The predictions were saved in a CSV file for further analysis or submission.

---

## **Instructions to Run**
1. Ensure you have all necessary libraries installed:
   ```
   pip install pandas matplotlib scikit-learn tensorflow
   ```
2. Place the training data (`data.csv`) and test data (`test.csv`) in the same directory as this notebook.
3. Run all cells in the notebook to perform the following:
   - Data analysis and feature engineering.
   - Build and train the GRU model.
   - Evaluate the model and save predictions to `answer.csv`.

---

### **Future Improvements**
- Consider experimenting with different model architectures (e.g., Bi-directional LSTMs, Transformers).
- Apply more advanced text preprocessing techniques like lemmatization or stopword removal.
- Perform hyperparameter tuning to further improve model performance.

---

**End of README**