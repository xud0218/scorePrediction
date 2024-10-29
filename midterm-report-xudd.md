# Midterm Report: Amazon Movie Review Score Prediction

**Contributor: Duoduo Xu**  
**Class: CS506-A1**  
**Date: 2024/10/28**

## Introduction and Objective

**Context**:  
Predicting Amazon review scores is valuable for understanding customer sentiment and improving product recommendations. Accurate predictions support users in decision-making and aid businesses in enhancing product offerings.

**Objective**:  
This project aims to predict star ratings for Amazon Movie Reviews using structured metadata and textual features from `train.csv`. `train.csv` contains over 1.7 million unique reviews, while `test.csv` provides Ids for prediction.

---

## Data Exploration and Preprocessing

**Exploratory Analysis**:  
An initial exploration of `train.csv` revealed a skewed distribution of star ratings, with some values more frequent than others. Several fields, such as `HelpfulnessNumerator` and `HelpfulnessDenominator`, and `Score` required handling due to missing or zero values.

**Feature Engineering Details**:
- **Helpfulness Ratio**: Calculated as the ratio of `HelpfulnessNumerator` to `HelpfulnessDenominator`, representing the feedback quality of reviews. High ratios may correlate with more accurate review sentiments.
- **Date Extraction**: The UNIX timestamp in the `Time` column was divided into `Year`, `Month`, and `Day` fields to capture seasonal or temporal patterns in reviews.
- **Text Vectorization**: The `Summary` and `Text` columns were merged, and TF-IDF (Term Frequency-Inverse Document Frequency) was applied to generate 50 features, highlighting important words based on frequency and uniqueness.
- **Dimensionality Reduction**: Truncated Singular Value Decomposition (SVD) reduced the TF-IDF vectors to 20 components, preserving key patterns while controlling for complexity.

---

## Model Selection Process

**Baseline Models**:  
We evaluated several models, including provided K-Nearest Neighbors (KNN), Linear Regression, Random Forest, and Support Vector Machine (SVM), each chosen for unique strengths:
   - **K-Nearest Neighbors**: Simple and interpretable, though limited in high-dimensional spaces.
   - **Linear Regression**: Effective with numeric data but often unsuitable for complex patterns.
   - **Random Forest**: Robust against overfitting, combining structured and textual data well.
   - **Support Vector Machine**: Capable of non-linear data handling but resource-intensive.

**Training and Evaluation**:  
A subset of `train.csv` served as a validation set to test baseline model performance. Random Forest achieved the highest accuracy among baseline models, benefiting from its ensemble structure. On the full dataset, Random Forest’s accuracy decreased to approximately 0.55, suggesting a need for tuning or increasing complexity.

---

## Hyperparameter Tuning

**TF-IDF Tuning**:  
The maximum number of TF-IDF features increased from 50 to 100, and SVD can keep 35 dimensions after reduction. In this case, the model can maintain more information for `Summary` and `Text`.

**Random Forest Tuning**:  
Adjusting the number of trees and maximum depth helped balance accuracy and efficiency, resulting in an optimal setup that prevents overfitting without sacrificing performance.

Unfortunately, the random forest classifier model still has an accuracy of 0.53 ± 2 after the adjustment. Therefore, either the model is limited, or the critical data pattern is not captured during the feature engineering process.

---

## Results and Future Improvements

**Result Summary**:  
Final Random Forest model accuracy reached around 0.53 ± 2, demonstrating acceptable but improvable performance across structured and textual data.

**Future Directions**:  
   - **Sentiment Analysis**: Introducing a `Sentiment` feature to categorize reviews as positive, neutral, or negative could reduce data complexity by removing the need for full TF-IDF and SVD processing. This streamlined approach would allow for using more complex models within Colab’s free-tier memory limits.
   - **Ensemble Models**: Exploring stacking and blending methods, such as Gradient Boosting with Random Forest, to combine model strengths.
   - **Hyperparameter Optimization**: Leveraging grid search for finer hyperparameter adjustments. (Exceeded RAM for Colab)

With the limitation of RAM in Colab and the constraint usage of deep learning models, capturing meaningful patterns between features and target data becomes challenging yet essential for enhancing model accuracy. 

---
