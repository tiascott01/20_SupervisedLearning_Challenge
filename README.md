# 20_SupervisedLearning_Challenge
## Overview

This repository contains a main file and a correlating file with CSV files for analyzing CSV data into the Jupyter Notebook file. The directions for the module were to use knowledge of Python and supervised learning to train and evaluate a model based on loan risk to build a model that can identify the creditworthiness of borrowers.

## Navigation
- Resources
    - lending_data.csv - provided csv dataset.
- credit_risk_classification.ipynb - provided jupyter notebook file that contains the code.
- report-template.md - markdown file of the analysis

## Results

In the 20_SupervisedLearning_Challenge, this dataset is an analysis of different loan statuses, which are split into training and testing sets, created into a logistic regression model (with the original model), resampled into a new regression model, evaluating both models performance and a analysis of the data from both models.


## Analysis

### Overview of the Analysis
1. <b>Explain the purpose of the analysis.</b><br>
1a. In our study, we evaluated the performance of two machine learning algorithms on financial datasets to forecast the risk associated with various loans.

2. <b>Explain what financial information the data was on, and what you needed to predict.</b><br>
2a. Our objective was to utilize data from a financial institution to predict the status of loan repayments. Specifically, we aimed to classify loans as either high-risk or low-risk, an essential aspect of risk  
  management and decision-making in the finance industry. 

3. <b>Provide basic information about the variables you were trying to predict (e.g., `value_counts`).</b><br>
3a. Our analysis involved examining a range of predictive factors and developing two models to assess how risks are distributed among loans. The dataset comprised 75,036 low-risk (healthy) loans and 2,500 high-risk 
  loans, indicating significant manual effort in their identification. 

4. <b>Describe the stages of the machine learning process you went through as part of this analysis.</b><br>
4a. The process encompassed data preprocessing, feature selection, model training, and evaluation. We employed logistic regression as the foundational classifier and investigated the impact of resampling techniques to 
  mitigate the issue of class imbalance within the dataset.

5. <b>Briefly touch on any methods you used (e.g., `LogisticRegression`, or any resampling method).</b><br>
5a. The initial model predominantly identified loans as low-risk, leading to an imbalanced assessment. To address this, our second model incorporated resampling to achieve a more balanced evaluation between healthy 
  and high-risk loans.

### Results

Using bulleted lists, describe the balanced accuracy scores and the precision and recall scores of all machine learning models.

* Machine Learning Model 1:

  - Balanced Accuracy Score: Approximately 0.952
  - Confusion Matrix: [18,663 True Negatives, 102 False Positives; 56 False Negatives, 563 True Positives]
  - Classification Report:
      - Precision
        - Healthy loans (class 0): 1.00
        - High-risk loans (class 1): 0.85
      
      - Recall / True Positive Rate
        - Healthy loans (class 0): 0.99
        - High-risk loans (class 1): 0.91

      - F1-Score
        - Healthy loans (class 0): 1.00
        - High-risk loans (class 1): 0.88



* Machine Learning Model 2:

  - Balanced Accuracy Score: Approximately 0.993
  - Confusion Matrix: [18,649 True Negatives, 116 False Positives; 4 False Negatives, 615 True Positives]
  - Classification Report:
      - Precision
        - Healthy loans (class 0): 1.00
        - High-risk loans (class 1): 0.84
      
      - Recall / True Positive Rate
        - Healthy loans (class 0): 0.99
        - High-risk loans (class 1): 0.99

      - F1-Score
        - Healthy loans (class 0): 1.00
        - High-risk loans (class 1): 0.91

### Summary

* Q1. Which one seems to perform best? How do you know it performs best?

* A1. Machine Learning Model 2 appears to perform better for the task of predicting loan risk (healthy vs. high-risk loans). The reasons for its superior performance are highlighted through several key indicators:

1a. Model 2 has a higher balanced accuracy score of approximately 0.993 compared to Model 1's score of 0.952. This score is particularly important in datasets with imbalanced classes, as it gives equal weight to the performance in both classes. A higher balanced accuracy score indicates that Model 2 is more effective in correctly identifying both healthy and high-risk loans across the board.

2a. The recall (or True Positive Rate) for high-risk loans in Model 2 is 0.99, compared to 0.91 in Model 1. This metric is crucial in the context of loan risk prediction because it measures the model's ability to identify all actual high-risk loans. A higher recall means fewer high-risk loans are missed by the model, which is vital for financial institutions to minimize the risk of loan defaults. The significant improvement in recall in Model 2 indicates its superior capability in correctly identifying high-risk loans.

3a. While Model 2 has a slightly lower precision for high-risk loans (0.84) compared to Model 1 (0.85), the significant improvement in recall (from 0.91 in Model 1 to 0.99 in Model 2) for high-risk loans is more critical for the loan prediction task. In risk management, it is often more costly to miss identifying a high-risk loan than to incorrectly classify a healthy loan as high-risk. Therefore, the trade-off favors a model with higher recall for high-risk loans, making Model 2 more suitable.

4a. The F1-Score for high-risk loans in Model 2 improved to 0.91 from 0.88 in Model 1. The F1-Score is a harmonic mean of precision and recall, offering a single metric to assess a model's balanced performance between these two aspects. An improvement in the F1-Score for high-risk loans in Model 2 indicates a better balance between precision and recall, emphasizing its efficiency in identifying high-risk loans accurately.

Machine Learning Model 2 outperforms Model 1 overall, demonstrated by its higher balanced accuracy score, significantly improved recall for high-risk loans without a substantial compromise in precision, and a higher F1-Score for high-risk loans. These factors collectively indicate that Model 2 is more effective and reliable for predicting loan risk, making it the preferred choice for financial institutions seeking to enhance their risk management processes.


* Does performance depend on the problem we are trying to solve? (For example, is it more important to predict the `1`'s, or predict the `0`'s? )

The best model depends on the problem's specific goals and the relative costs of false positives vs. false negatives. For financial institutions prioritizing the detection of high-risk loans to minimize defaults, Model 2 is preferable due to its higher recall for high-risk loans. This model ensures that nearly all high-risk loans are identified, aligning with the goal of minimizing financial risk, even if it means a slight increase in the number of healthy loans incorrectly classified as high-risk.

If you do not recommend any of the models, please justify your reasoning.

Model 2 stands out as the preferable choice for enhancing the financial institution's ability to manage and mitigate loan risk effectively. This model's significantly higher recall for high-risk loans minimizes the chance of overlooking potential defaults, a critical factor in risk management. The increase in false positives—healthy loans mistakenly classified as high-risk—is acknowledged but is considered a manageable trade-off given the substantial reduction in false negatives. Specifically, Model 2's performance translates into 54 fewer high-risk loans being misclassified as healthy compared to Model 1.

In the context of the entire dataset comprising 77,536 loans, Model 2's strategy of identifying high-risk loans results in only 13 false negatives, corresponding to a potential default risk sum of $129,500. This figure provides a tangible metric for assessing the model's impact on risk mitigation and cost savings, even though it represents a hypothetical scenario. The actual benefits could vary depending on the accuracy of the model in real-world applications and the specific characteristics of the loans involved.

Moreover, the slight increase in false positives (healthy loans classified as high-risk) invites a strategic consideration. It suggests that while some loans might be flagged unnecessarily as high-risk, the overall benefit of catching nearly all true high-risk loans far outweighs the costs associated with these false alarms. This is especially relevant when considering the operational capacity to review and manage these cases effectively.

The recommendation for Model 2 is further supported by the suggestion that the business conducts additional studies to evaluate the model's performance against manual risk assessments traditionally used by the institution. This comparison could reveal that the model's "false positives" might, in some instances, be more accurately identifying risks that manual evaluations miss. Such findings could indicate that the model not only enhances efficiency and risk detection but also potentially outperforms traditional human assessments in identifying latent or non-obvious risks.


While this analysis recommends Model 2 as a strategic asset for the business, it also emphasizes the need for ongoing evaluation and refinement of the model, including a comparison with manual accuracy rates to continuously enhance its effectiveness and efficiency in identifying loan risks.



## Usage

You can use this file to analyze the data in the corresponding resource CSV.

1. Open the respective file (`credit_risk_classification.ipynb) in Jupyter Notebook.

2. Make sure that the resource and analysis directories are congruent within their respective places as listed in the script, if not change the location.

3. Run individual cells within to see the calculations.


## Resources and Citations

1. General - ChatGpt.com
