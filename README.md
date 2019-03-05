# CreditCardFraudAnalytics
Predictive Modeling to detect fraud credit card transactions using Python.
Predictive modeling is implemented to detect fraudulent credit card transactions, with maximum accuracy, in a dataset with 284807 rows and 31 columns.
This is a classification problem, where the target variable Class (0/1) has to be classified as either Not Fraud (0) or Fraud (1) transaction.
There are a total of 30 predictor columns, such as Time and Amount of transaction. The remaining predictors (v1, v2,....,v28) are derived from PCA, and cannot be interpreted to ensure anonimosity.
The data is highly unbalanced, with 284315 non fraudulent transactions and only 492 fraud transactions.
Predictive modeling is implemented on both balanced and unbalanced data to analyze the difference in prediction accuracy.
Classification models implemented for prediction: Logistic Regression and Random Forest Ensemble
Random forest algorithm performs well for unbalanced data, as measured using the roc metrics.
After balancing data by over sampling the minority (fraud) class, the Logistic Regression method performs best.
