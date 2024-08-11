# Andrew's Portfolio

Welcome and thanks for taking the time to look at my portfolio projects.  As an avid stock trader and Data Science enthusiast, I am always working on projects to further my knowledge in both.  I enjoy working on projects that allow me to learn about new developments in machine learning/deep learning and practice my Data Science skills while finding new data driven ways to invest.  These projects represent a wide range of techniques from supervised learning, unsupervised learning, and reinforcement learning to more traditional statistical techniques.

## [Project 1: Predicting SPY Stock Returns Using an XGBoost Classifier](https://github.com/AndrewFSee/Profile_Projects/blob/main/Stock_Returns_Prediction.ipynb)
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

![](https://github.com/AndrewFSee/Profile_Projects/blob/main/images/project1.png)


## [Project 2: Simulating Stock Prices Using Monte Carlo Simulations of Geometric Brownian Motion](https://github.com/AndrewFSee/Profile_Projects/blob/main/Monte_Carlo_GBM.ipynb)
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
S_t = S_0 \exp((\mu - \frac{\sigma^2}{2})t + \sigma W_t)
$$

where:
   - $S_t$ is the stock price at time $t$.
   - $\mu$ is the drift rate, representing the expected return of the stock.
   - $\sigma$ is the volatility of the stock, indicating the degree of variation in the stock price.
   - $W_t$ is a Wiener process or Brownian motion, representing the random component.



3. Visualization: The simulated stock price paths were visualized to assess the range of potential future prices. Summary statistics, such as mean and variance of simulated paths, were computed.

### Evaluation:

- Risk Assessment: The simulations provided insights into the range of possible future stock prices, allowing for the assessment of potential risks and uncertainties.
- Value at Risk (VaR): Calculated VaR to quantify the potential loss in stock value over a specified time horizon with a given confidence level.
  
### Results:
The Monte Carlo simulations produced a range of potential future stock price trajectories, demonstrating the inherent variability and uncertainty in stock price movements. The simulations allowed for a better understanding of potential risks and helped in decision-making related to investment strategies.

### Conclusion:
The project effectively utilized Monte Carlo simulations of Geometric Brownian Motion to model and predict stock prices, providing valuable insights into future price behavior and risk assessment. Future work could involve refining the model with additional factors, such as jumps or mean-reversion processes, and integrating alternative simulation techniques for improved accuracy.

![](https://github.com/AndrewFSee/Profile_Projects/blob/main/images/project2.png)

## [Project 3: Detecting Stock Market Regimes Using Hidden Markov Models](https://github.com/AndrewFSee/Profile_Projects/blob/main/Hidden_Markov_Models_for_Market_Regimes.ipynb)
### Objective:
The project aims to apply Hidden Markov Models (HMMs) to identify and analyze different regimes in the stock market. By detecting shifts in market conditions, the project seeks to provide insights into regime changes that can inform investment strategies and risk management.

### Methodology:

- Hidden Markov Models (HMMs): HMMs were utilized to model the underlying states or regimes of the stock market. The HMM framework is particularly suited for this task as it can capture latent states (regimes) and their transitions based on observable data.
- Regime Detection: The HMM was employed to uncover distinct market regimes such as bull, bear, and sideways markets. The model assumes that the market transitions between these hidden regimes over time, each with its own statistical properties.
  
### Data Collection:
Historical stock market data was collected, including:

- Price Data: Daily open, high, low, and close prices.
- Returns: Calculated daily returns to capture the market’s performance.

### Data Preprocessing:

- Feature Engineering: Derived features from price and return data to improve the model’s predictive power.
- Normalization: Standardized the features to ensure consistency in the model’s training process.
- Train-Test Split: Divided the data into training and testing sets to evaluate the model’s performance.
  
### Model Development:

- Parameter Estimation: Estimated the parameters of the HMM, including the number of hidden states (regimes), transition probabilities, and emission probabilities.
- Training: Used the Baum-Welch algorithm to train the HMM on historical market data, estimating the parameters based on observed data sequences.
- Inference: Applied the Viterbi algorithm to determine the most likely sequence of regimes given the observed data.
  
### Evaluation:

- Regime Identification: Analyzed the detected regimes to understand market behaviors and regime transitions.
- Performance Metrics: Evaluated the model’s ability to accurately identify market regimes and transitions compared to known historical periods.
  
### Results:
The HMM successfully identified distinct market regimes and provided insights into regime transitions. The detected regimes corresponded with historical market conditions, such as periods of high volatility and stable trends. The model offered valuable information for refining trading strategies and understanding market dynamics.

### Conclusion:
The project effectively utilized Hidden Markov Models to detect and analyze stock market regimes, providing a powerful tool for understanding market phases and improving investment strategies. Future work could involve enhancing the model with additional features, exploring different HMM configurations, and integrating the regime detection into real-time trading systems.

![](https://github.com/AndrewFSee/Profile_Projects/blob/main/images/project3.png)

## [Project 4: Grouping ETFs Using K-means Clustering and Analyzing Cointegration for Pairs Trading](https://github.com/AndrewFSee/Profile_Projects/blob/main/Kmeans_PairsTrading.ipynb)

### Objective:
The goal of this project is to apply K-means clustering to group Exchange-Traded Funds (ETFs) based on their historical price movements and then analyze cointegration within these groups to identify potential pairs for trading strategies. Additionally, t-SNE (t-Distributed Stochastic Neighbor Embedding) was used to visualize the relationships between assets based on their cointegration.

### Methodology:

- K-means Clustering: Used K-means clustering to group ETFs based on their historical price data. This unsupervised learning technique partitions ETFs into clusters where each cluster represents ETFs with similar price behavior.
- Cointegration Analysis: After clustering, cointegration tests were performed on pairs within each cluster to identify pairs of ETFs with a long-term equilibrium relationship. Cointegration is used to find pairs that move together over time, which can be exploited for pairs trading.
- t-SNE Visualization: Employed t-SNE to plot the relationships between ETFs based on their cointegration results. This technique helped visualize the similarity and clustering of ETFs in a lower-dimensional space, making it easier to interpret the relationships between assets.
  
### Data Collection:
Historical price data for a selection of ETFs was collected, including:

- Price Data: Daily closing prices of ETFs over a specified period.
- Additional Features: Computed features such as returns and volatility to enhance the clustering process.
  
### Data Preprocessing:

- Feature Engineering: Calculated returns and other relevant features from price data to improve the clustering process.
- Normalization: Standardized the features to ensure that clustering results are not biased by scale.
- Train-Test Split: Although clustering does not use a train-test split, data was organized to ensure accurate analysis and interpretation.
  
### Model Development:

- K-means Clustering: Applied K-means clustering with a predefined number of clusters. Evaluated different values of 𝑘 using metrics such as the Elbow Method or Silhouette Score to determine the optimal number of clusters.
- Cointegration Testing: For each pair of ETFs within the same cluster, performed cointegration tests using methods like the Engle-Granger two-step procedure or the Johansen test to identify stable, long-term relationships.
- t-SNE Visualization: Used t-SNE to create a visual representation of the ETFs based on their cointegration relationships. This plot provided insights into how ETFs relate to each other and the clustering structure of the assets.
  
### Evaluation:

- Cluster Analysis: Analyzed the resulting clusters to validate that ETFs within each cluster exhibit similar price movements.
- Cointegration Results: Assessed the cointegration results to identify pairs with strong and statistically significant cointegration relationships. Evaluated the potential for profitable pairs trading based on these relationships.
- t-SNE Visualization: Interpreted the t-SNE plot to understand the spatial relationships and clustering of ETFs based on their cointegration, facilitating the selection of potential pairs for trading.
  
### Results:
The K-means clustering effectively grouped ETFs into clusters of similar price behavior. Cointegration analysis revealed several pairs within these clusters that exhibited strong long-term relationships. The t-SNE visualization provided a clear depiction of the relationships between ETFs based on their cointegration, highlighting potential pairs for trading strategies.

### Conclusion:
The project successfully applied K-means clustering to group ETFs, analyzed cointegration for pairs trading, and used t-SNE to visualize asset relationships. This approach offers a systematic method for selecting pairs trading opportunities based on clustering results, long-term relationships, and visual insights. Future work could involve refining the clustering methodology, incorporating additional features, and testing trading strategies based on the identified pairs.

![](https://github.com/AndrewFSee/Profile_Projects/blob/main/images/project4.png)
