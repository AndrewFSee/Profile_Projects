# Andrew's Portfolio

Welcome and thanks for taking the time to look at my portfolio projects.  As an avid stock trader and Data Science enthusiast, I am always working on projects to further my knowledge in both.  I enjoy working on projects that allow me to learn about new developments in machine learning/deep learning and practice my Data Science skills while finding new data driven ways to invest.  These projects represent a wide range of techniques from supervised learning, unsupervised learning, and reinforcement learning to more traditional statistical techniques.

## [Project 1 Summary: Predicting SPY Stock Returns Using an XGBoost Classifier](https://github.com/AndrewFSee/Profile_Projects/blob/main/Stock_Returns_Prediction.ipynb)
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


## [Project 2 Summary: Simulating Stock Prices Using Monte Carlo Simulations of Geometric Brownian Motion](https://github.com/AndrewFSee/Profile_Projects/blob/main/Monte_Carlo_GBM.ipynb)
### Objective:
The primary objective of this project is to utilize Monte Carlo simulations based on Geometric Brownian Motion (GBM) to model and predict future stock prices. By generating a range of possible price paths, the project aims to assess the potential variability and risk associated with stock price movements.

### Methodology:

- Geometric Brownian Motion (GBM): The GBM model was employed to capture the stochastic behavior of stock prices. The model assumes that stock prices follow a log-normal distribution with continuous time dynamics, where returns are normally distributed and prices exhibit both drift and volatility.
- Monte Carlo Simulations: Multiple simulations (e.g., 10,000) were run to generate a diverse set of possible future price paths for the stock. Each simulation utilized random sampling to account for the uncertainty and randomness inherent in financial markets.

### Data Collection:
Historical stock price data was collected to estimate the model parameters:

- Drift (μ): The average return of the stock, calculated from historical price data.
- Volatility (σ): The standard deviation of returns, also derived from historical data.
  
### Simulation Process:

1. Parameter Estimation: Calculated the drift and volatility based on historical price data.

2. Simulation Setup: Defined the time horizon and time steps for the simulation. For each simulation run, stock price paths were generated using the GBM formula:

$$
\displaystyle S_t = S_0 \exp\{(\mu - \frac{\sigma^2}{2})t + \sigma W_t\}
$$

where:
   - \($ S_t $\) is the stock price at time \($ t $\).
   - \($ \mu $\) is the drift rate, representing the expected return of the stock.
   - \($ \sigma $\) is the volatility of the stock, indicating the degree of variation in the stock price.
   - \($ W_t $\) is a Wiener process or Brownian motion, representing the random component.



3. Visualization: The simulated stock price paths were visualized to assess the range of potential future prices. Summary statistics, such as mean and variance of simulated paths, were computed.

### Evaluation:

- Risk Assessment: The simulations provided insights into the range of possible future stock prices, allowing for the assessment of potential risks and uncertainties.
- Value at Risk (VaR): Calculated VaR to quantify the potential loss in stock value over a specified time horizon with a given confidence level.
  
### Results:
The Monte Carlo simulations produced a range of potential future stock price trajectories, demonstrating the inherent variability and uncertainty in stock price movements. The simulations allowed for a better understanding of potential risks and helped in decision-making related to investment strategies.

### Conclusion:
The project effectively utilized Monte Carlo simulations of Geometric Brownian Motion to model and predict stock prices, providing valuable insights into future price behavior and risk assessment. Future work could involve refining the model with additional factors, such as jumps or mean-reversion processes, and integrating alternative simulation techniques for improved accuracy.
