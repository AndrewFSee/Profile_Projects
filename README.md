Welcome and thank you for taking the time to explore my portfolio projects. As a passionate stock trader and Data Science enthusiast, I continuously work on projects that not only deepen my understanding of the markets but also enhance my skills in data-driven investing. I focus on developing innovative approaches to trading by leveraging the latest advancements in machine learning, deep learning, and traditional statistical methods. My portfolio showcases a diverse range of projects, from supervised and unsupervised learning to reinforcement learning and large language models, all aimed at refining my trading strategies and expanding my knowledge in this dynamic field.  I am currently working on projects that incorporate the OpenAI API into my algorithmic trading and quantitative research and furthering my studies in reinforcement learning and the Markov decision process.

## Projects
### - Predicting SPY Stock Returns Using an XGBoost Classifier
### - Simulating Stock Prices Using Monte Carlo Simulations of Geometric Brownian Motion
### - Detecting Stock Market Regimes Using Hidden Markov Models
### - Grouping ETFs Using K-means Clustering and Analyzing Cointegration for Pairs Trading
### - Dimensionality Reduction of Stock Data Using Principal Component Analysis (PCA)
### - Portfolio Optimization Using Markowitz Portfolio Theory
### - Stock Trading Using Proximal Policy Optimization (PPO)
### - AI-Powered Stock Recommender System Using OpenAI API

## [Predicting SPY Stock Returns Using an XGBoost Classifier](https://github.com/AndrewFSee/Profile_Projects/blob/main/Stock_Returns_Prediction.ipynb)
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

![](/images/project1.png)


## [Simulating Stock Prices Using Monte Carlo Simulations of Geometric Brownian Motion](https://github.com/AndrewFSee/Profile_Projects/blob/main/Monte_Carlo_GBM.ipynb)
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

\[
S_t = S_0 \exp((\mu - \frac{\sigma^2}{2})t + \sigma W_t)
\]

where:
   - $ \S_t $ is the stock price at time $t$.
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

![](/images/project2.png)

## [Detecting Stock Market Regimes Using Hidden Markov Models](https://github.com/AndrewFSee/Profile_Projects/blob/main/Hidden_Markov_Models_for_Market_Regimes.ipynb)
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

![](/images/project3.png)

## [Grouping ETFs Using K-means Clustering and Analyzing Cointegration for Pairs Trading](https://github.com/AndrewFSee/Profile_Projects/blob/main/Kmeans_PairsTrading.ipynb)

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

![](/images/project4.png)

## [Dimensionality Reduction of Stock Data Using Principal Component Analysis (PCA)](https://github.com/AndrewFSee/Profile_Projects/blob/main/Technical_Analysis_Features_PCA.ipynb)

### Objective:
The objective of this project is to apply Principal Component Analysis (PCA) to reduce the dimensionality of stock data while preserving its essential characteristics. The project aims to simplify data analysis, improve model performance, and uncover underlying patterns in the stock market data.

### Methodology:
- Principal Component Analysis (PCA): PCA was utilized to transform the high-dimensional stock data into a lower-dimensional space. This technique identifies the principal components that capture the most variance in the data, enabling more efficient analysis and modeling.
- Dimensionality Reduction: By projecting the stock data onto the principal components, the project aimed to reduce the number of features while retaining significant information, facilitating easier visualization and interpretation.
  
### Data Collection:
Historical stock market data was collected, including:
- Price Data: Daily open, high, low, and close prices for a range of stocks over a specified period.
- Additional Features: Computed features such as returns, volatility, and technical indicators to enrich the dataset.
  
### Data Preprocessing:
- Feature Engineering: Derived features from the raw price data, including returns and rolling statistics, to prepare for PCA.
- Normalization: Standardized the features to ensure that all variables contribute equally to the PCA, preventing scale dominance.
- Missing Value Handling: Addressed any missing values in the dataset to ensure data integrity.
  
### Model Development:
- PCA Implementation: Applied PCA to the preprocessed stock data to identify the principal components that explain the maximum variance. Determined the number of components to retain based on explained variance ratios and scree plots.
- Dimensionality Reduction: Reduced the dimensionality of the dataset by projecting it onto the selected principal components, creating a new, lower-dimensional feature set.
  
### Evaluation:
- Explained Variance: Evaluated the proportion of variance explained by each principal component to assess how well the dimensionality reduction retained the original data’s information.
- Visualization: Visualized the stock data in the lower-dimensional space to identify patterns and relationships between stocks. Techniques such as scatter plots and biplots were used to interpret the principal components.
- Model Performance: Assessed the impact of dimensionality reduction on the performance of predictive models or clustering algorithms applied to the reduced dataset.
  
### Results:
PCA effectively reduced the dimensionality of the stock data while preserving the key characteristics of the original dataset. The principal components captured significant variance and allowed for easier visualization and interpretation of the data. The reduced feature set facilitated more efficient modeling and analysis, enhancing the understanding of underlying patterns in the stock market.

### Conclusion:
The project demonstrated the effectiveness of Principal Component Analysis for dimensionality reduction in stock data. By simplifying the dataset while retaining essential information, PCA improved data analysis and model performance. Future work could involve experimenting with different dimensionality reduction techniques, integrating PCA with other data processing steps, and exploring its impact on various financial models.

![](/images/project5.png)

## [Portfolio Optimization Using Markowitz Portfolio Theory](https://github.com/AndrewFSee/Profile_Projects/blob/main/Portfolio_Optimization.ipynb)

### Objective:
The project aims to optimize an investment portfolio using Markowitz Portfolio Theory to achieve the best possible return for a given level of risk. By applying this classic theory, the project seeks to construct an efficient frontier of portfolios that balances risk and return, guiding investors in making informed investment decisions.

### Methodology:
- Markowitz Portfolio Theory: Applied Markowitz Portfolio Theory to optimize the asset allocation in a portfolio. This theory focuses on creating an "efficient frontier" of optimal portfolios, where each portfolio offers the maximum expected return for a given level of risk or the minimum risk for a given level of expected return.
- Risk-Return Tradeoff: Incorporated the tradeoff between risk (measured as portfolio variance or standard deviation) and return (measured as expected return) to find the optimal asset allocation.
  
### Data Collection:
Historical financial data was collected, including:
- Asset Prices: Daily or monthly historical prices for a set of assets (e.g., stocks, bonds, ETFs).
- Returns: Calculated historical returns for each asset to estimate expected returns and risk.
- Covariance Matrix: Computed the covariance matrix of asset returns to assess how assets move relative to each other.
  
### Data Preprocessing:
- Feature Engineering: Derived returns, volatility, and covariance from the historical price data.
- Normalization: Standardized data where necessary to ensure consistency in calculations.

### Model Development:
- Optimization Model: Formulated an optimization problem where the objective was to maximize the Sharpe ratio (the ratio of excess return to portfolio risk) subject to constraints. Constraints included budget constraints (e.g., total investment equals 100%) and possibly other limits (e.g., maximum or minimum investment in each asset).
- Efficient Frontier: Used optimization techniques (e.g., quadratic programming) to generate the efficient frontier—a set of optimal portfolios that offer the best risk-return tradeoff.
- Selection of Optimal Portfolio: Identified the optimal portfolio along the efficient frontier based on the investor’s risk tolerance and return objectives.
  
### Evaluation:
- Performance Metrics: Assessed portfolio performance using metrics such as the Sharpe ratio, maximum drawdown, and overall return versus risk.

### Results:
The application of Markowitz Portfolio Theory successfully identified an efficient frontier of portfolios, offering a range of optimal asset allocations that balance risk and return. The optimized portfolios demonstrated improved risk-return characteristics compared to non-optimized portfolios, providing valuable guidance for investment decisions.

### Conclusion:
The project effectively utilized Markowitz Portfolio Theory to optimize asset allocation and construct an efficient frontier. This approach provided a systematic method for achieving desired investment outcomes based on risk and return preferences. Future work could involve integrating additional factors such as transaction costs, liquidity constraints, or incorporating alternative optimization techniques to further enhance portfolio performance.

![](/images/project6.png)

## [Stock Trading Using Proximal Policy Optimization (PPO)](https://github.com/AndrewFSee/Profile_Projects/blob/main/PPO_Stock_Returns.ipynb)

### Objective:
The project aims to develop and implement a stock trading strategy using Proximal Policy Optimization (PPO), a reinforcement learning algorithm. The goal is to train an agent that can make informed trading decisions to maximize cumulative returns while managing risk.

### Methodology:
- Proximal Policy Optimization (PPO): Utilized PPO, a state-of-the-art reinforcement learning algorithm, to train an agent for stock trading. PPO is designed to handle high-dimensional action spaces and continuous environments, making it suitable for complex trading scenarios.
- Reinforcement Learning Framework: The trading environment was modeled as a reinforcement learning problem, where the agent learns to make trading decisions (buy, sell, hold) based on observed market states (e.g., stock prices, technical indicators) to maximize long-term rewards.
  
### Data Collection:
Historical stock market data was collected, including:

- Price Data: Daily or intraday historical prices for selected stocks.
Technical Indicators: Calculated features such as VWAP to provide the agent with relevant information for decision-making.

### Data Preprocessing:
- Feature Engineering: Created a set of features from the raw price data and technical indicators to represent the state space for the reinforcement learning model.
- Normalization: Standardized the features to ensure that the learning algorithm can effectively process the data.
- Environment Setup: Developed a trading environment where the PPO agent interacts with the market data and learns from its actions.
  
### Model Development:
- PPO Implementation: Implemented PPO using a deep reinforcement learning framework such as PyTorch or TensorFlow. Defined the policy network and value network architectures, and configured the PPO algorithm with appropriate hyperparameters (e.g., learning rate, batch size).
- Training: Trained the PPO agent using historical market data, optimizing the policy to maximize cumulative returns while adhering to risk management constraints (e.g., drawdown limits).
Evaluation Metrics: Backtested the results obtained by the PPO agent against a buy-and-hold strategy.

### Evaluation:
- Backtesting: Performed backtesting of the trained PPO agent on unseen historical data to assess its performance and robustness in different market conditions.
- Comparison: Compared the PPO-based trading strategy against baseline strategies (e.g., buy-and-hold, simple moving average crossover) to evaluate its effectiveness and improvements.
- Sensitivity Analysis: Analyzed the sensitivity of the trading strategy to different hyperparameters and market conditions to ensure stability and adaptability.
  
### Results:
The PPO-based trading strategy demonstrated the ability to make informed trading decisions that led to improved returns compared to baseline strategies. The agent effectively learned to navigate market dynamics, optimizing its actions to maximize cumulative returns while managing risk.

### Conclusion:
The project successfully applied Proximal Policy Optimization to develop a stock trading strategy, leveraging reinforcement learning to make informed trading decisions. The PPO agent showed promising results in terms of returns and risk management. Future work could involve refining the model with additional features, experimenting with alternative reinforcement learning algorithms, and testing the strategy in live trading environments to further validate its effectiveness.

![](/images/project7.png)

## [AI-Powered Stock Recommender System Using OpenAI API](https://github.com/AndrewFSee/Profile_Projects/blob/main/OpenAI_Stock%20Recommendation.ipynb)

### Objective:
This project aims to develop an AI-powered stock recommender system using the OpenAI API to analyze market data and generate actionable trading recommendations. The goal is to leverage natural language processing (NLP) and deep learning models to identify promising stocks based on historical data, financial metrics, and sentiment analysis.

### Data Collection:
Historical stock data, including daily open, high, low, and close prices, trading volume, and fundamental metrics such as P/E ratios and earnings reports, was sourced from various financial databases. Additionally, news articles were scraped using BeautifulSoup to enrich the dataset and provide context for stock recommendations.

### Data Preprocessing:
- Feature Engineering: Extracted features from both structured data (e.g., stock prices, financial ratios) and unstructured data (e.g., text sentiment) to create a comprehensive dataset.
- Normalization: Numerical features were normalized to ensure consistency in scaling, while text data was tokenized and embedded using pre-trained language models.

### Model Development:
- Algorithm: The OpenAI API, specifically the GPT model, was utilized to analyze and interpret both numerical and textual data, offering predictions on the future performance of stocks. The model was fine-tuned to understand the specific nuances of financial data and generate targeted stock recommendations.
- Prompt Engineering: Carefully crafted prompts were designed to guide the model in producing relevant and contextually accurate recommendations, with an emphasis on evaluating stock performance indicators.

### Results:
The AI-powered stock recommender system demonstrated strong predictive capabilities, offering actionable insights into stock selection. The system’s integration of both quantitative data and sentiment analysis provided a well-rounded approach to identifying investment opportunities. The model's performance metrics indicated its potential as a valuable tool for traders and investors.

![](/images/project8.png)

### Conclusion:
The project successfully developed an AI-powered stock recommender system using the OpenAI API, providing a robust tool for making informed investment decisions. Future enhancements could include incorporating real-time data, refining NLP techniques for better sentiment analysis, and exploring other AI models to improve recommendation accuracy and breadth.
