# Midterm Report: Amazon Movie Review Score Prediction

**Duoduo Xu**  
**CS506-A1**  
**2024/10/28**

## Introduction and Objective

**Context**:  
Predicting Amazon review scores is valuable for understanding customer sentiment and improving product recommendations. Accurate predictions support users in decision-making and aid businesses in enhancing product offerings.

**Objective**:  
This project aims to predict star ratings for Amazon Movie Reviews using structured metadata and textual features from `train.csv` and `test.csv`. `train.csv` contains over 1.7 million unique reviews, while `test.csv` provides Ids for prediction.

---

## Data Exploration and Preprocessing

**Exploratory Analysis**:  
An initial exploration of `train.csv` revealed a skewed distribution of star ratings, with some values more frequent than others. Several fields, such as `HelpfulnessNumerator` and `HelpfulnessDenominator`, required handling due to missing or zero values.

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
A subset of `train.csv` served as a validation set to test model performance. Random Forest achieved the highest accuracy among baseline models, benefiting from its ensemble structure. On the full dataset, Random Forest’s accuracy decreased to approximately 0.53 ± 2, suggesting a need for tuning.

**Performance Summary**:  
The Random Forest model performed the best initially but showed signs of overfitting when extended to the full dataset. Further adjustments and parameter tuning may help address this.

---

## Hyperparameter Tuning

**TF-IDF Tuning**:  
Limiting TF-IDF features to 50 and applying SVD to reduce dimensionality minimized overfitting and improved computational efficiency.

**Random Forest Tuning**:  
Adjusting the number of trees and maximum depth helped balance accuracy and efficiency, with an optimal setup that prevents overfitting without sacrificing performance.

---

## Results and Future Improvements

**Result Summary**:  
Final Random Forest model accuracy reached around 0.53 ± 2, demonstrating acceptable but improvable performance across structured and textual data.

**Future Directions**:  
Future enhancements include:
   - **Sentiment Analysis**: Adding a `Sentiment` feature to extract review mood (positive, neutral, or negative) to reduce data dimensionality, allowing for more complex models within Colab’s free tier memory limits.
   - **Advanced Feature Engineering**: Integrating embeddings (e.g., Word2Vec or BERT) for a deeper semantic understanding of review text.
   - **Ensemble Models**: Exploring stacking and blending methods, such as Gradient Boosting with Random Forest, to combine model strengths.
   - **Hyperparameter Optimization**: Leveraging grid search for finer hyperparameter adjustments.

---

This expanded approach addresses dimensionality challenges and computational constraints, providing a foundation for future improvements in accuracy and resource efficiency.
