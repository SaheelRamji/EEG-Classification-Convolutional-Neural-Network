Introduction:

This project focuses on developing a classification model for EEG data analysis, primarily targeting the differentiation of epileptic seizures. Utilizing the CHB-MIT EEG Database and the Bonn EEG Dataset, the project employs a comprehensive pipeline encompassing data preprocessing, feature extraction (including time-domain and frequency-domain features), data splitting for training, validation, and testing, model selection (considering Convolutional Neural Networks (CNNs) and Recurrent Neural Networks (RNNs)), and advanced training techniques to mitigate overfitting. Evaluation metrics such as accuracy, precision, recall, and F1-score are utilized on the validation set, with hyperparameter fine-tuning aimed at optimizing model performance. This research aims to enhance medical diagnostics by leveraging machine learning methodologies to classify EEG data, particularly in epilepsy.

The project involved gathering seizure and non-seizure data from the CHB-MIT EEG Database, creating individual dataframes (df_1 to df_10) for each category, and concatenating them into comprehensive dataframes (df_seazure and df_non_seazure). After merging, the final dataset (final_df) comprised 206,848 rows and 24 columns post-cleaning. Notably, the 'VNS' column was dropped due to significant depopulation. Further preprocessing included visualizing EEG signal values, feature extraction using Principal Component Analysis (PCA), and data splitting into training, validation, and test sets.

Results and Methodology:

Understanding EEG Signal Variation: Visualizations of EEG signals (e.g., FP1-F7, F7-T7) provided insights into signal distribution and variability, guiding subsequent preprocessing steps.

Feature Extraction with PCA: Ten principal components were selected to capture essential information from the standardized dataset, forming df_reduced with 'seazure' as the target variable.

Model Selection: Three models—Random Forest (RF), Convolutional Neural Network (CNN), and Recurrent Neural Network (RNN)—were chosen for their suitability in handling EEG data.

Model Training and Evaluation: The CNN model, optimized through hyperparameter tuning (filters=32, kernel_size=5, pool_size=2, dense_units=100, dropout_rate=0.3), outperformed RF and RNN in accuracy (91.5%) on the test set. Evaluation metrics included accuracy, precision, recall, and F1-score, supported by confusion matrix and classification report analyses.

Conclusion:

The CNN model demonstrated robust performance in classifying seizure events from EEG signals, emphasizing its potential for real-world applications in medical diagnostics. Despite limitations related to data variability and model interpretability, the project contributes to advancing EEG-based seizure classification, suggesting avenues for future research in multimodal integration and ethical AI deployment in healthcare.

Future Work:

Future directions include integrating multiple modalities, enhancing model interpretability, exploring longitudinal studies, deploying models on edge devices, and addressing ethical considerations for broader healthcare applications.
