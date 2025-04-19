Forest Cover Type Classifier Documentation
==========================================

Overview:
---------
This project is a machine learning pipeline for classifying forest cover types based on cartographic variables. It uses a Random Forest Classifier and includes preprocessing, feature selection, hyperparameter tuning, and model evaluation. A Gradio interface is also provided for user interaction.

Dependencies:
-------------
- pandas
- seaborn
- matplotlib
- numpy
- gradio
- scikit-learn

Pipeline Steps:
---------------
1. **Import Libraries**:
   - Essential libraries for data manipulation, visualization, and machine learning are imported.

2. **Load Dataset**:
   - The dataset is loaded from a CSV file using the `load_dataset()` function.

3. **Dataset Information**:
   - The `get_df_info()` function provides insights into the dataset, including shape, columns, unique values, null values, and descriptive statistics.

4. **Outlier Removal**:
   - The `remove_outliers()` function removes extreme values from numerical features using the Interquartile Range (IQR) method.

5. **Preprocessing**:
   - Numerical features are standardized using `StandardScaler`.
   - The dataset is split into training and testing sets (80% training, 20% testing).

6. **Feature Importance**:
   - A Random Forest Classifier is trained to compute feature importance.
   - Features with importance scores below a threshold are removed.

7. **Model Training and Evaluation**:
   - A Random Forest Classifier is trained on the selected features.
   - The model is evaluated using accuracy, classification report, and confusion matrix.

8. **Cross-Validation**:
   - 5-fold cross-validation is performed to assess model stability.

9. **Hyperparameter Tuning**:
   - GridSearchCV is used to optimize hyperparameters of the Random Forest Classifier.

10. **Gradio Interface**:
    - A Gradio interface is created for user interaction, allowing predictions based on user input.

Functions:
----------
1. `load_dataset()`:
   - Loads the dataset from a CSV file.

2. `get_df_info(df)`:
   - Prints dataset information, including shape, columns, unique values, null values, and descriptive statistics.

3. `remove_outliers(df, numerical_features)`:
   - Removes outliers from numerical features using the IQR method.

4. `predict_forest_cover(*args)`:
   - Predicts the forest cover type based on user input.

Key Variables:
--------------
- `numerical_features`: List of numerical columns used for preprocessing and outlier removal.
- `X_train`, `X_test`: Training and testing feature sets.
- `Y_train`, `Y_test`: Training and testing target sets.
- `rf`: Random Forest Classifier model.
- `importance_threshold`: Threshold for feature importance.

Gradio Interface:
-----------------
- Inputs: Numerical features (excluding the target column).
- Output: Predicted forest cover type.
- Command to launch: `app.launch()`

Usage:
------
1. Ensure all dependencies are installed.
2. Place the dataset (`covtype.csv`) in the project directory.
3. Run the script to preprocess the data, train the model, and evaluate its performance.
4. Use the Gradio interface for predictions.

File Structure:
---------------
- `main.ipynb`: Main Jupyter Notebook containing the entire pipeline.
