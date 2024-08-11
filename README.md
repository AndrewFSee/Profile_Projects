# Andrew's Portfolio

Welcome and thanks for taking the time to look at my portfolio projects.  As an avid stock trader and Data Science enthusiast, I am always working on projects to further my knowledge in both.  I enjoy working on projects that allow me to learn about new developments in machine learning/deep learning and practice my Data Science skills while finding new data driven ways to invest.  These projects represent a wide range of techniques from supervised learning, unsupervised learning, and reinforcement learning to more traditional statistical techniques.

## Project 1 Summary: Predicting SPY Stock Returns Using an XGBoost Classifier
### Objective:
This project aims to develop a classification model using XGBoost to predict the future direction of SPDR S&P 500 ETF Trust (SPY) returns—specifically, whether the return will be positive or negative. The goal is to provide actionable trading signals based on historical data and technical indicators.

### Data Collection:
Historical stock price data for SPY, including daily open, high, low, close prices, and trading volume, was sourced from Yahoo Finance. Technical indicators, such as moving averages, relative strength index (RSI), and volatility measures, were calculated to enrich the dataset and provide additional predictive features.

### Data Preprocessing:

- Feature Engineering: Created features based on technical indicators and lagged returns to capture market trends and dynamics. The target variable was defined as a binary classification: 1 for positive returns and 0 for negative returns over a specified time horizon.
- Normalization: Features were normalized to ensure consistent scaling and enhance model performance.
- Train-Test Split: The dataset was split into training and testing sets to evaluate the model’s performance and generalizability.
  
### Model Development:

- Algorithm: An XGBoost classifier was employed for its effectiveness in handling binary classification problems and its ability to model complex patterns and interactions.
- Hyperparameter Tuning: Optimized key parameters such as learning rate, number of trees, maximum depth, and regularization terms using cross-validation techniques.
- Feature Importance: Analyzed feature importance scores to understand the contribution of each feature to the classification decision.
  
### Evaluation:

- Metrics: Model performance was assessed using classification metrics such as accuracy, precision, recall, F1-score, and the Area Under the Receiver Operating Characteristic Curve (AUC-ROC).

### Results:
The XGBoost classifier demonstrated strong performance in predicting the direction of SPY returns, with favorable classification metrics and significant insights into feature importance. The model’s predictions provided valuable signals for potential trading strategies.

### Conclusion:
The project successfully applied an XGBoost classifier to forecast SPY return directions, offering a useful tool for investors looking to make informed trading decisions. Future work could involve integrating additional data sources, refining feature engineering, and exploring other classification algorithms to further enhance prediction accuracy.
