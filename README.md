Welcome and thank you for taking the time to explore my portfolio projects. As a passionate stock trader and Data Science enthusiast, I continuously work on projects that not only deepen my understanding of the markets but also enhance my skills in data-driven investing. I focus on developing innovative approaches to trading by leveraging the latest advancements in machine learning, deep learning, and traditional statistical methods. My portfolio showcases a diverse range of projects, from supervised and unsupervised learning to reinforcement learning and large language models, all aimed at refining my trading strategies and expanding my knowledge in this dynamic field.  I am currently working on projects that incorporate the OpenAI API into my algorithmic trading and quantitative research and furthering my studies in reinforcement learning and the Markov decision process.  Please note that none of the projects presented here should be considered financial advice.

## Projects
### - [Predicting SPY Stock Returns Using an XGBoost Classifier](https://github.com/AndrewFSee/Profile_Projects/blob/main/Stock_Returns_Prediction.ipynb)
### - [Simulating Stock Prices Using Monte Carlo Simulations of Geometric Brownian Motion](https://github.com/AndrewFSee/Profile_Projects/blob/main/Monte_Carlo_GBM.ipynb)
### - [Skewness-based Trading Strategy Using K-means Clustering](https://github.com/AndrewFSee/Profile_Projects/blob/main/K-Means_Clustering_Stock_Returns.ipynb)
### - [Detecting Stock Market Regimes Using Hidden Markov Models](https://github.com/AndrewFSee/Profile_Projects/blob/main/Hidden_Markov_Models_for_Market_Regimes.ipynb)
### - [Grouping ETFs Using K-means Clustering and Analyzing Cointegration for Pairs Trading](https://github.com/AndrewFSee/Profile_Projects/blob/main/Kmeans_PairsTrading.ipynb)
### - [Dimensionality Reduction of Stock Data Using Principal Component Analysis (PCA)](https://github.com/AndrewFSee/Profile_Projects/blob/main/Technical_Analysis_Features_PCA.ipynb)
### - [Portfolio Optimization Using Markowitz Portfolio Theory](https://github.com/AndrewFSee/Profile_Projects/blob/main/Portfolio_Optimization.ipynb)
### - [Stock Trading Using Proximal Policy Optimization (PPO)](https://github.com/AndrewFSee/Profile_Projects/blob/main/PPO_Stock_Returns.ipynb)
### - [AI-Powered Stock Recommender System Using OpenAI API](https://github.com/AndrewFSee/Profile_Projects/blob/main/OpenAI_Stock%20Recommendation.ipynb)
### - [SQL Executor with LLM for Economic Data from FRED](https://github.com/AndrewFSee/Profile_Projects/blob/main/LangChain_SQLiteDB.ipynb)
### - [Conversational Retrieval-Augmented Generation (RAG) system with PDF Upload using Langchain, Groq, and Streamlit](https://github.com/AndrewFSee/Profile_Projects/blob/main/RAG_pdf_simple.py)
### - [Multi-Agent Financial Report Generation](https://github.com/AndrewFSee/Profile_Projects/blob/main/financial_report_app.py)

---
&nbsp;

&nbsp;


## [Predicting SPY Stock Returns Using an XGBoost Classifier](https://github.com/AndrewFSee/Profile_Projects/blob/main/Stock_Returns_Prediction.ipynb)

![](/images/project1.png)

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


## [Simulating Stock Prices Using Monte Carlo Simulations of Geometric Brownian Motion](https://github.com/AndrewFSee/Profile_Projects/blob/main/Monte_Carlo_GBM.ipynb)

![](/images/project2.png)

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


## - [Skewness-based Trading Strategy Using K-means Clustering](https://github.com/AndrewFSee/Profile_Projects/blob/main/K-Means_Clustering_Stock_Returns.ipynb)

![](/images/Kmeans_returns.png)

### Objective
This project applies K-Means clustering to identify distinct market regimes based on the skewness of stock returns. By segmenting the market into clusters with varying skewness characteristics, the goal is to detect fat-tail events and optimize trading strategies by identifying conditions favorable for long or short positions.

### Methodology
Clustering Approach

- K-Means Clustering: Used to group market conditions based on returns skewness and other statistical features.
- Skewness as a Key Feature: Skewness was used to identify asymmetric return distributions, highlighting fat-tail events where extreme positive or negative movements are more likely.
  
Trading Strategy Based on Skewness
- Go Long on clusters where skewness > 0.75, since positively skewed distributions indicate upside potential.
- Go Short on clusters where skewness < -1, as negatively skewed distributions suggest higher downside risk.
- Remain Neutral for all other clusters.
  
### Data Collection
Historical stock market data was collected, including:

- Price Data: Daily open, high, low, and close (OHLC) prices.
- Returns: Daily returns to capture the market’s movement.
- Volatility Metrics: Included rolling volatility measures like Volatility14 to assess market conditions.
  
### Data Preprocessing

- Feature Engineering: Derived statistical features such as rolling mean, standard deviation, skewness, and kurtosis to better segment market conditions.
- Normalization: Standardized feature values to ensure proper cluster formation.
- Clustering & Labeling: Applied K-Means clustering on the preprocessed dataset to segment the market into regimes.
  
### Model Development

- Optimal Cluster Selection: Determined the best number of clusters using the Elbow Method and Silhouette Score.
- Cluster Interpretation: Analyzed cluster characteristics to understand their relationship with different market phases.
- Trade Signal Generation: Mapped clusters to trading signals based on skewness thresholds.
  
### Evaluation

#### Cluster Insights
- Clusters 0, 5, and 7 were identified as bullish, exhibiting positive skewness, leading to long positions.
- Cluster 3 was identified as bearish, with significantly negative skewness, making it a strong shorting opportunity.
- Cluster 1 also showed high volatility but lacked strong skewness, suggesting a need for further investigation or sub-clustering.
  
#### Backtesting the Strategy
- Performance Comparison: The strategy was backtested against a buy-and-hold benchmark.
- Returns: The strategy achieved 106.13%, outperforming the 40% return of the buy-and-hold strategy over the same period.
- Risk-Adjusted Returns:
  - Strategy Sharpe Ratio: 2.556
  - Buy-and-Hold Sharpe Ratio: 1.183
  - The strategy provided higher returns with lower risk and better downside protection.

### Results & Insights
- The clustering approach successfully segmented market conditions based on skewness and volatility.
- The strategy effectively captured positive returns while minimizing downside exposure.
- The results suggest that fat-tail events identified through clustering can be exploited for better trading decisions.

### Conclusion & Future Improvements
This project demonstrated that returns skewness and K-Means clustering can be used to detect market regimes and develop profitable trading strategies. Future enhancements include:

- Incorporating Kurtosis: To better differentiate extreme events.
- Dynamic Cluster Adjustments: Periodically updating clusters using rolling window techniques.
- Regime Detection Validation: Cross-validating with Hidden Markov Models (HMMs) to refine regime identification.
- Transaction Cost Analysis: To assess the real-world viability of the strategy.
- Integration into Live Trading: Deploying the strategy in a real-time trading system for execution and monitoring.


## [Detecting Stock Market Regimes Using Hidden Markov Models](https://github.com/AndrewFSee/Profile_Projects/blob/main/Hidden_Markov_Models_for_Market_Regimes.ipynb)

![](/images/project3.png)

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


## [Grouping ETFs Using K-means Clustering and Analyzing Cointegration for Pairs Trading](https://github.com/AndrewFSee/Profile_Projects/blob/main/Kmeans_PairsTrading.ipynb)

![](/images/project4.png)

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


## [Dimensionality Reduction of Stock Data Using Principal Component Analysis (PCA)](https://github.com/AndrewFSee/Profile_Projects/blob/main/Technical_Analysis_Features_PCA.ipynb)

![](/images/project5.png)

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


## [Portfolio Optimization Using Markowitz Portfolio Theory](https://github.com/AndrewFSee/Profile_Projects/blob/main/Portfolio_Optimization.ipynb)

![](/images/project6.png)

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


## [Stock Trading Using Proximal Policy Optimization (PPO)](https://github.com/AndrewFSee/Profile_Projects/blob/main/PPO_Stock_Returns.ipynb)

![](/images/project7.png)

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


## [AI-Powered Stock Recommender System Using OpenAI API](https://github.com/AndrewFSee/Profile_Projects/blob/main/OpenAI_Stock%20Recommendation.ipynb)

![](/images/project8.png)

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

### Conclusion:
The project successfully developed an AI-powered stock recommender system using the OpenAI API, providing a robust tool for making informed investment decisions. Future enhancements could include incorporating real-time data, refining NLP techniques for better sentiment analysis, and exploring other AI models to improve recommendation accuracy and breadth.


## - [SQL Executor with LLM for Economic Data from FRED](https://github.com/AndrewFSee/Profile_Projects/blob/main/LangChain_SQLiteDB.ipynb)

![](/images/SQL_Query_Executor_LLM.png)

### Objective:
The objective of this project is to develop an intelligent SQL-based query system powered by a language model (LLM), capable of querying economic data from the Federal Reserve Economic Data (FRED) database. The system allows users to ask natural language questions, which are then converted into SQL queries to extract relevant economic data, such as GDP, unemployment rates, interest rates, and more. The system integrates the power of retrieval-augmented generation (RAG) to enhance query responses with context and historical information, providing accurate and relevant insights from the data.

### Functionality:
1. SQL Query Generation: The system converts natural language queries into SQL commands, which are executed on a SQLite database containing economic data from FRED.

2. Dynamic Querying: Users can input complex natural language questions regarding economic trends or specific indicators, and the system translates these into SQL queries, executes them, and provides the results.

3. Integration of Retrieval-Augmented Generation (RAG): The system uses a RAG approach where relevant SQL query examples are retrieved from a set of predefined few-shot examples, helping guide query formulation for complex questions.

4. Date Formatting and Time Awareness: The system incorporates an additional tool for formatting date-related queries, such as "last year" or "last 6 months," into the appropriate SQL date format for querying.

5. Historical Context: Using LangChain’s tools and models, the system is aware of prior queries and ensures context-aware responses, especially when querying for time-series economic data.

6. User Interaction: The system offers a simple interface for users to interact with the database, posing questions in plain language and receiving the corresponding results from the FRED economic data.

### Workflow:
1. Database Setup and Integration: The project uses a SQLite database (financial_data.db) populated with historical economic data, sourced from FRED. This data could include metrics such as:

- Unemployment rates (UNRATE)
- 10-year Treasury yields (DGS10)
- GDP data
- Consumer Price Index (CPI)
- Other financial indicators
  
2. Natural Language Query Processing: The system uses LangChain’s language model to interpret user queries in natural language, transforming them into SQL queries. It utilizes predefined query examples to match the most relevant queries for specific economic indicators or trends.

3. Date Parsing and Query Construction: A custom tool parses dates in queries (e.g., "last year") and formats them to the required SQL format (e.g., YYYY-MM-DD 00:00:00). This ensures accurate and time-aware queries are constructed.

Example of such a query:

- User query: "What was the GDP growth last year?"
- SQL query: SELECT GDP FROM economic_indicators WHERE Date BETWEEN '2022-01-01' AND '2022-12-31';
  
4. Few-Shot Examples for Query Generation: The system is trained with a set of SQL query examples for common economic data queries. These examples help the model retrieve the most relevant SQL queries for more complex user inputs.

5. SQL Query Execution and Response Generation: The SQL queries generated by the LLM are executed on the database, and the system returns the result of the query to the user. If the query involves time-series data, it can summarize trends or extract specific metrics such as maximum values, changes over time, or averages.

6. Enhanced User Interaction: Users input questions via a simple interface, where they can ask about trends, comparisons, or specific data points (e.g., "Show me the top 5 lowest unemployment rates in the last 5 years"). The system processes the question, executes the corresponding SQL query, and outputs the results in a human-readable format.

7. Tools Integration:

- SQL Example Retriever: This tool helps retrieve relevant SQL query examples based on user inputs, enhancing the model’s ability to generate accurate queries.
- Date Formatter Tool: Handles user queries involving date ranges or relative time periods (e.g., "last year" or "last 6 months").
- SQL Database Toolkit: Interfaces with the SQLite database to execute generated SQL queries and return results.

### User Interaction Flow:
1. User Input: The user provides a natural language question such as:

- "What was the unemployment rate in 2023?"
- "Show me the 10-year yield for the last month."
  
2. Query Generation: The LLM converts the input into an SQL query:

- "SELECT UNRATE FROM economic_indicators WHERE Date BETWEEN '2023-01-01' AND '2023-12-31';"
  
3. Execution and Response: The query is executed on the SQLite database, and the results are returned to the user.

4. Contextual Answers: The model generates responses that explain the data in a user-friendly format, e.g., "The unemployment rate in 2023 was 3.5% on average."

5. History-Aware Responses: The system maintains conversational history to provide more coherent responses, especially when asking follow-up questions or comparisons, such as "How did this compare to 2022?"

### Results:
The system provides accurate and context-aware answers to a wide range of economic questions, enabling users to easily query economic data from FRED in natural language. By leveraging retrieval-augmented generation, the system improves the accuracy and relevance of answers, particularly for complex or time-sensitive queries.

Examples of Questions Answered:

- "What was the GDP growth rate in 2020?"
- "How did the 10-year Treasury yield change in the last quarter?"
- "What was the highest unemployment rate in the past decade?"
  
The system can handle queries related to time ranges, comparisons, and specific data points like averages, minimums, and maximums over time.

### Conclusion:
This project successfully integrates SQL query execution with an LLM to create a conversational, intelligent query system for economic data. It allows users to easily query large datasets like FRED, providing valuable insights into economic trends. Future improvements could include:

- Extending the system to support additional databases and economic indicators.
- Adding support for more complex analytical queries, such as trend analysis or forecasting.
- Scaling the system to handle more users and larger datasets with optimized performance.

[Streamlit .py file](https://github.com/AndrewFSee/Profile_Projects/blob/main/app.py)


## [Conversational Retrieval-Augmented Generation (RAG) system with PDF Upload using Langchain, Groq, and Streamlit](https://github.com/AndrewFSee/Profile_Projects/blob/main/RAG_pdf_simple.py)

![](/images/RAG_PDF.png)

### Objective:
This project aims to create a Conversational Retrieval-Augmented Generation (RAG) system that allows users to upload PDF documents, extract their content, and interact with it via a chatbot interface. The system uses LangChain and integrates advanced capabilities like history-aware question reformulation and context-driven response generation, enabling seamless and intelligent interactions with document data.

### Functionality:
- Users can upload one or more PDFs to the system.
- The system processes the PDFs, extracts their content, and creates embeddings to enable efficient text retrieval.
- Users can ask questions about the uploaded documents, and the chatbot provides accurate, context-aware answers by leveraging retrieval-augmented generation.
- The system maintains and utilizes a conversational history to enhance the quality and relevance of responses.
  
### Workflow:
1. Document Loading and Preprocessing:
- Uploaded PDF files are processed using PyPDFLoader to extract their content.
- The content is split into manageable chunks using RecursiveCharacterTextSplitter to ensure optimal performance during embedding creation and retrieval.
  
2. Embeddings and Vector Store:
- Document chunks are converted into vector representations using the HuggingFaceEmbeddings model (all-MiniLM-L6-v2).
- These embeddings are stored in a FAISS vector store, enabling fast and accurate document retrieval.
  
3. Retrieval-Augmented Generation (RAG):
- The system integrates a retrieval chain with a history-aware retriever to reformulate user queries based on chat history.
- A contextualized QA chain generates concise answers by combining retrieved document content with the user’s question.
  
4. Conversational Memory:
- Chat history is persistently managed using ChatMessageHistory and session-based state tracking to ensure a coherent flow of interaction.
- The system dynamically reformulates questions to handle references to prior conversation context.
  
5. User Interaction:
- Users interact with the system via a Streamlit interface.
- Key inputs include the Groq API key, uploaded PDFs, and user questions.
- The chatbot provides responses alongside the complete chat history for transparency and usability.
  
### Results:
- The system provides accurate, concise, and context-aware answers to user queries about the content of uploaded PDFs.
- The incorporation of conversational memory ensures that responses are tailored to the context of the ongoing interaction.
- By leveraging the Groq language model (Gemma2-9b-It), the chatbot offers high-quality natural language understanding and response generation.
  
### Conclusion:
The Conversational RAG system successfully combines document retrieval, conversational memory, and natural language understanding to provide an intelligent interface for interacting with uploaded PDFs. Future improvements could include:
- Adding support for other document types (e.g., Word, Excel).
- Implementing real-time document updates.
- Enhancing the embeddings model to capture more complex document semantics.
- Improving scalability to handle larger datasets or concurrent users.


## - [Multi-Agent Financial Report Generation](https://github.com/AndrewFSee/Profile_Projects/blob/main/financial_report_app.py)

![](/images/multi_agent_financial_report_before.png)

### Objective:
The objective of this project is to develop an AI-powered financial report generation system that utilizes multi-agent collaboration to automate the creation of detailed stock analysis reports. The system integrates fundamental and technical analysis, news aggregation, and scenario-based forecasting, ensuring high-quality insights for investors and analysts. The multi-agent architecture enables specialized AI agents to handle different aspects of the report, while a review pipeline ensures accuracy, completeness, and consistency.

### Functionality:
1. Stock Analysis Automation:
- The system generates comprehensive financial reports based on user-specified stock tickers.
- It analyzes both fundamental and technical indicators, stock correlations, news sentiment, and future market scenarios.
  
2. Multi-Agent AI Collaboration:
- Various specialized AI agents work together, each handling distinct tasks such as data collection, analysis, and report writing.
- The agents include Financial Analysts, Technical Analysts, Researchers, Writers, and Reviewers.
  
3. Automated Review and Quality Assurance:
- A multi-step review process ensures that generated reports are accurate, complete, and well-structured.
- Reviewers check for data consistency, compliance, alignment between textual descriptions and numerical data, and overall completeness.
  
4. Report Export and Documentation:
- Finalized reports are formatted and saved as markdown (.md) files, making them easy to read and share.

5. User-Friendly Interface (Streamlit):
- Users interact with the system via a simple web interface, entering stock tickers to initiate report generation.
- A button-click triggers the analysis workflow, and the output is named with the execution date for easy tracking.

### Workflow:
1. User Input and Task Initiation:
- The user enters stock tickers into the Streamlit interface and initiates report generation.
- The system assigns tasks to the appropriate AI agents based on predefined roles.

2. Financial and Technical Data Collection:
- The Financial Assistant retrieves fundamental data such as P/E ratio, EPS, revenue, and cash flow.
- The Technical Analysis Assistant calculates indicators like moving averages, RSI, MACD, and Bollinger Bands.

3. News Aggregation and Sentiment Analysis:
- The Researcher Agent gathers recent financial news related to the selected stocks.
- A sentiment analysis module evaluates whether the news is positive, neutral, or negative.

4. Report Compilation and Writing:
- The Writer Agent compiles all data into a structured financial report.
- It organizes sections such as Company Overview, Fundamental Analysis, Technical Indicators, Market Trends, and Future Scenarios.

5. Automated Multi-Agent Review Process:
- Technical Reviewer: Ensures accuracy of calculations and interpretations.
- Legal Reviewer: Checks compliance with financial disclosure rules.
- Consistency Reviewer: Verifies data consistency across sections.
- Text Alignment Reviewer: Ensures textual insights match numerical data.
- Completion Reviewer: Confirms all required sections are included.
- Meta Reviewer: Aggregates feedback and gives final approval.

6. Final Report Export:
- Once approved, the Export Assistant saves the report in markdown format with a timestamped filename.

### User Interaction Flow:
1. User Input:
- The user enters stock tickers in the Streamlit app (e.g., “AAPL, TSLA, MSFT”).

2. Automated Report Generation:
- The AI agents retrieve data, analyze trends, summarize news, and compile insights.

3. Quality Review & Finalization:
- Multiple review agents validate accuracy and coherence before finalizing the report.

4. Report Export & Delivery:
- The completed report is saved as an .md file and made available for download.

### Results:
The system generates comprehensive, AI-driven stock reports that offer:

✅ Fundamental and technical insights for informed decision-making.

✅ Automated, structured financial analysis without manual effort.

✅ Accurate, consistent, and reviewed content through multi-agent collaboration.

✅ Seamless user experience via an interactive Streamlit dashboard.

### Conclusion & Future Enhancements:
This project successfully demonstrates how multi-agent AI systems can automate financial analysis and report generation. Potential future improvements include:
- Expanding financial data sources to include real-time updates and alternative data.
- Enhancing news analysis with more advanced sentiment scoring and trend detection.
- Integrating forecasting models to provide predictive analytics alongside historical insights.
- Supporting multiple report formats (e.g., PDF, HTML) for broader usability.

By leveraging LLMs and multi-agent collaboration, this system provides a scalable, AI-powered approach to financial reporting, enabling investors, analysts, and businesses to access high-quality stock insights efficiently. 🚀

# Example Report

# Financial Report on NVDA, GOOGL, and TSLA

## Overview

This report presents a comprehensive analysis of NVIDIA Corporation (NVDA), Alphabet Inc. (GOOGL), and Tesla, Inc. (TSLA), integrating both fundamental and technical analyses. Spanning market figures, financial ratios, technical charts, and the latest corporate developments, the report outlines a framework for understanding each stock's current and potential future performance.

## Comparative Analysis

### Fundamental Ratios and Data

| Metric                  | NVDA        | GOOGL       | TSLA        |
|-------------------------|-------------|-------------|-------------|
| Current Price ($)       | 134.43      | 179.66      | 337.80      |
| P/E Ratio               | 53.13       | 22.32       | 166.40      |
| Forward P/E             | 32.63       | 20.05       | 104.26      |
| Dividends ($)           | 0.03        | 0.45        | N/A         |
| Price to Book           | 49.99       | 6.75        | 14.90       |
| Debt/Equity Ratio       | 15.52       | 8.66        | 18.49       |
| Return on Equity (ROE)  | 127.21%     | 32.91%      | 10.42%      |
| 6-Month % Change        | 8.66%       | 9.95%       | 60.35%      |

**Comments:**

- **P/E and Forward P/E Ratios**: Highlighting high growth potential, TSLA's P/E ratio significantly exceeds its peers, suggesting investor confidence in future earnings.
- **ROE and Debt/Equity**: NVDA shows robust profitability with its substantial ROE. The high debt/equity ratio in TSLA reflects its aggressive expansion strategy.
  
## Correlation and Risks

Analyzed correlations suggest strategic investment patterns and shared market influences:

- **High Correlation between TSLA and GOOGL (0.903)**: Reflects impact alignment from macroeconomic shifts affecting tech and mobility sectors.
- **Moderate Correlation between NVDA and TSLA (0.426)**: Demonstrates differentiated yet intersecting technological focus areas.

## Technical Analysis

![](/images/normalized_prices.png)

### NVIDIA Corporation (NVDA)

- **Key Indicators**: RSI indicates potential overbought conditions, while Bollinger Bands point towards increasing market volatility. MACD reveals solid momentum support, yet vigilance is advised around earnings surprises.
- **Market Patterns**: Strategic levels identified hint at probable bullish breakouts, bolstered by AI-driven initiatives.

### Alphabet Inc. (GOOGL)

- **Technicals**: RSI and MACD align with stable momentum trends, while moving averages underscore confidence in sustained investor sentiment.
- **Market Dynamics**: Large-scale AI investments contribute to predictable, relatively steady price movements, prompting potential buy signals at strategic points.

### Tesla, Inc. (TSLA)

- **Interpretations**: RSI indicates potential corrections ahead as it nears overbought territories. MACD and chart patterns suggest preparation for further breakout opportunities.
- **Market Context**: Compelling upward momentum is expected to continue, tempered by operational challenges such as labor disputes.

## Recent News Summary and Impact

### NVIDIA Corporation (NVDA)

NVIDIA's foray into AI solutions highlights core strengths in addressing expanding market needs, with potential turbulences tied to anticipated earnings insights. A significant boost has been forecasted by analysts, reinforcing its tech leadership.

### Alphabet Inc. (GOOGL)

Strategically expanding in AI with a $75 billion investment, Alphabet underscores its sustainable growth amid reallocation by prominent investors. This positions it as a key entity in the technological future, aligning well with its financial consistency.

### Tesla, Inc. (TSLA)

Tesla's global interest expansion, particularly in India, combines with automotive policy support to signal robust future prospects. Domestic labor issues may pose temporary market hesitancy, yet international pursuits provide substantial offsets.

## Future Scenarios

- **NVDA**: Poised for growth as AI integration surges, potential volatility around earnings periods could prompt distinct trading opportunities.
- **GOOGL**: A conservative yet strategic play, underpinned by massive AI investments and stable market presence, likely to yield steady returns.
- **TSLA**: Positioned for aggressive upside as international market incursions mature and environmental policies favor EV adoption, although closely tied to operational execution.

## Disclosure

This report is intended for informational purposes only and does not constitute financial advice. Please consult a financial advisor for personalized recommendations tailored to your individual circumstances. 

Legal and ethical compliance matters have been considered, and assumptions are based on data and insights up to the current date of October 2023.
