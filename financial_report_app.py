llm_config = {
    "model": "gpt-4o", 
    "api_key": "your OpenAI API key",
    }


import autogen
import streamlit as st


writing_tasks = [
        """Develop an engaging financial report using all information provided, include the normalized_prices.png figure,
        and other figures if provided.
        Mainly rely on the information provided. 
        Create a table comparing all the fundamental ratios and data.
        Provide comments and description of all the fundamental ratios and data.
        Compare the stocks, consider their correlation and risks, provide a comparative analysis of the stocks.
        Provide a summary of the recent news about each stock. 
        Ensure that you comment and summarize the news headlines for each stock, provide a comprehensive analysis of the news.
        Provide connections between the news headlines provided and the fundamental ratios.
        Provide an analysis of possible future scenarios. 
        """
        ,
        """
        Expand the financial report by incorporating technical analysis findings.
        - Include **all technical indicators** calculated.
        - Display all relevant technical analysis figures.
        - Provide insights on potential buy/sell signals based on RSI, MACD, and trendlines.
        - Identify potential **support and resistance levels**.
        - Compare technical findings with fundamental data for a **holistic investment perspective**.
        """]

exporting_task = ["""Save the report and only the report to a .md file using a python script with the date of creation and stock tickers in the filename."""]

financial_assistant = autogen.AssistantAgent(
    name="Financial_assistant",
    llm_config=llm_config,
)
technical_analysis_assistant = autogen.AssistantAgent(
    name="Financial_assistant",
    llm_config=llm_config,
)
research_assistant = autogen.AssistantAgent(
    name="Researcher",
    llm_config=llm_config,
)

writer = autogen.AssistantAgent(
    name="writer",
    llm_config=llm_config,
    system_message="""
        You are a professional writer, known for
        your insightful and engaging finance reports.
        You transform complex concepts into compelling narratives. 
        Include all metrics provided to you as context in your analysis.
        Only answer with the financial report written in markdown directly, do not include a markdown language block indicator.
        Ensure that images and figures are placed in their designated sections within the report.
        You must incorporate technical analysis insights in the report. 
        Ensure that RSI, MACD, Moving Averages, and Bollinger Bands are properly interpreted.
        Identify key chart patterns and highlight potential market movements.
        Only return your final work without additional comments.
        """,
)

export_assistant = autogen.AssistantAgent(
    name="Exporter",
    llm_config=llm_config,
)
# ===

technical_reviewer = autogen.AssistantAgent(
    name="Technical_Reviewer",
    llm_config=llm_config,
    system_message="You are a financial market analyst. "
        "Review the technical analysis section of the report. "
        "Ensure that all indicators are correctly calculated and interpreted. "
        "Make sure your suggestion is concise (within 3 bullet points), "
        "concrete and to the point. "
        "Begin the review by stating your role.",
)
critic = autogen.AssistantAgent(
    name="Critic",
    is_termination_msg=lambda x: x.get("content", "").find("TERMINATE") >= 0,
    llm_config=llm_config,
    system_message="You are a critic. You review the work of "
                "the writer and provide constructive "
                "feedback to help improve the quality of the content.",
)

legal_reviewer = autogen.AssistantAgent(
    name="Legal_Reviewer",
    llm_config=llm_config,
    system_message="You are a legal reviewer, known for "
        "your ability to ensure that content is legally compliant "
        "and free from any potential legal issues. "
        "Make sure your suggestion is concise (within 3 bullet points), "
        "concrete and to the point. "
        "Begin the review by stating your role.",
)

consistency_reviewer = autogen.AssistantAgent(
    name="Consistency_reviewer",
    llm_config=llm_config,
    system_message="You are a consistency reviewer, known for "
        "your ability to ensure that the written content is consistent throughout the report. "
        "Refer numbers and data in the report to determine which version should be chosen " 
        "in case of contradictions. "
        "Make sure your suggestion is concise (within 3 bullet points), "
        "concrete and to the point. "
        "Begin the review by stating your role. ",
)

textalignment_reviewer = autogen.AssistantAgent(
    name="Text_lignment_reviewer",
    llm_config=llm_config,
    system_message="You are a text data alignment reviewer, known for "
        "your ability to ensure that the meaning of the written content is aligned "
        "with the numbers written in the text. " 
        "You must ensure that the text clearely describes the numbers provided in the text "
        "without contradictions. "
        "Make sure your suggestion is concise (within 3 bullet points), "
        "concrete and to the point. "
        "Begin the review by stating your role. ",
)

completion_reviewer = autogen.AssistantAgent(
    name="Completion_Reviewer",
    llm_config=llm_config,
    system_message="You are a content completion reviewer, known for "
        "your ability to check that financial reports contain all the required elements. "
        "You always verify that the report contains: a news report about each asset, " 
        "a description of the different ratios and prices, "
        "a description of possible future scenarios, a table comparing fundamental ratios and "
        " at least a single figure. "
        "Make sure your suggestion is concise (within 3 bullet points), "
        "concrete and to the point. "
        "Begin the review by stating your role. ",
)

meta_reviewer = autogen.AssistantAgent(
    name="Meta_Reviewer",
    llm_config=llm_config,
    system_message="You are a meta reviewer, you aggregate and review "
    "the work of other reviewers and give a final suggestion on the content.",
)

def reflection_message(recipient, messages, sender, config):
    return f'''Review the following content. 
            \n\n {recipient.chat_messages_for_summary(sender)[-1]['content']}'''

review_chats = [
    {
    "recipient": technical_reviewer, 
    "message": reflection_message, 
    "summary_method": "reflection_with_llm",
    "summary_args": {"summary_prompt" : 
        "Return review into a JSON object only:"
        "{'Reviewer': '', 'Review': ''}.",},
    "max_turns": 1 },
    {
    "recipient": legal_reviewer, "message": reflection_message, 
     "summary_method": "reflection_with_llm",
     "summary_args": {"summary_prompt" : 
        "Return review into a JSON object only:"
        "{'Reviewer': '', 'Review': ''}.",},
     "max_turns": 1},
    {"recipient": textalignment_reviewer, "message": reflection_message, 
     "summary_method": "reflection_with_llm",
     "summary_args": {"summary_prompt" : 
        "Return review into a JSON object only:"
        "{'reviewer': '', 'review': ''}",},
     "max_turns": 1},
    {"recipient": consistency_reviewer, "message": reflection_message, 
     "summary_method": "reflection_with_llm",
     "summary_args": {"summary_prompt" : 
        "Return review into a JSON object only:"
        "{'reviewer': '', 'review': ''}",},
     "max_turns": 1},
    {"recipient": completion_reviewer, "message": reflection_message, 
     "summary_method": "reflection_with_llm",
     "summary_args": {"summary_prompt" : 
        "Return review into a JSON object only:"
        "{'reviewer': '', 'review': ''}",},
     "max_turns": 1},
     {"recipient": meta_reviewer, 
      "message": "Aggregrate feedback from all reviewers and give final suggestions on the writing.", 
     "max_turns": 1},
]

critic.register_nested_chats(
    review_chats,
    trigger=writer,
)

# ===

user_proxy_auto = autogen.UserProxyAgent(
    name="User_Proxy_Auto",
    human_input_mode="NEVER",
    is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").rstrip().endswith("TERMINATE"),
    code_execution_config={
        "last_n_messages": 3,
        "work_dir": "coding",
        "use_docker": False,
    },  
)

st.title("Multi-Agent Financial Report Generation")
st.write("This app is designed to generate a financial report using multiple AI agents.")
assets = st.text_input("Assets you want to analyze (provide the tickers)?")
hit_button = st.button('Start analysis')

if hit_button is True:

    from datetime import datetime
    date_str = datetime.now().strftime("%Y-%m-%d")

    financial_tasks = [
        f"""Today is the {date_str}. 
        What are the current stock prices of {assets}, and how is the performance over the past 6 months in terms of percentage change? 
        Start by retrieving the full name of each stock and use it for all future requests.
        Prepare a figure of the normalized price of these stocks and save it to a file named normalized_prices.png. Include information about, if applicable: 
        * P/E ratio
        * Forward P/E
        * Dividends
        * Price to book
        * Debt/Eq
        * ROE
        * Analyze the correlation between the stocks
        Do not use a solution that requires an API key.
        If some of the data does not makes sense, such as a price of 0, change the query and re-try.""",

        """Investigate possible reasons of the stock performance leveraging market news headlines from Bing News or Google Search. Retrieve news headlines using python and return them. Use the full name stocks to retrieve headlines. Retrieve at least 10 headlines per stock. Do not use a solution that requires an API key. Do not perform a sentiment analysis.""",

        f"""Today is the {date_str}.
        Perform a technical analysis of {assets} using historical price data.
        Include the following:
        * Moving Averages: 50-day, 200-day SMA & EMA
        * RSI (Relative Strength Index)
        * MACD (Moving Average Convergence Divergence)
        * Bollinger Bands
        * Support and Resistance Levels
        * Identify any significant chart patterns (e.g., Head & Shoulders, Double Bottom, Flags)
        * Calculate and visualize stock volatility using ATR (Average True Range)
        * Identify overbought or oversold conditions using RSI and Stochastic Oscillator
        * Compare the current trend to historical trends

        Plot all indicators and price data into **separate figures** and save them with appropriate names.
        Provide a summary of the findings in plain text.
        Ensure all calculations use **publicly available data** and do not require an API key.
        If any data is missing or doesn't make sense, retry the query.
        """
    ]

    with st.spinner("Agents working on the analysis...."):
        chat_results = autogen.initiate_chats(
            [
                {
                    "sender": user_proxy_auto,
                    "recipient": financial_assistant,
                    "message": financial_tasks[0],
                    "silent": False,
                    "summary_method": "reflection_with_llm",
                    "summary_args": {
                        "summary_prompt" : "Return the stock prices of the stocks, their performance and all other metrics"
                        "into a JSON object only. Provide the name of all figure files created. Provide the full name of each stock.",
                                    },
                    "clear_history": False,
                    "carryover": "Wait for confirmation of code execution before terminating the conversation. Verify that the data is not completely composed of NaN values. Reply TERMINATE in the end when everything is done."
                },
                {
                    "sender": user_proxy_auto,
                    "recipient": technical_analysis_assistant,
                    "message": financial_tasks[-1],  
                    "silent": False,
                    "summary_method": "reflection_with_llm",
                    "summary_args": {
                        "summary_prompt" : "Return the technical analysis metrics and insights into a JSON object."
                        "Provide the name of all figure files created.",
                        },
                    "clear_history": False,
                    "carryover": "Wait for confirmation of code execution before terminating the conversation. Verify that all indicators are correctly calculated. Reply TERMINATE in the end when everything is done."
                },
                {
                    "sender": user_proxy_auto,
                    "recipient": research_assistant,
                    "message": financial_tasks[1],
                    "silent": False,
                    "summary_method": "reflection_with_llm",
                    "summary_args": {
                        "summary_prompt" : "Provide the news headlines as a paragraph for each stock, be precise but do not consider news events that are vague, return the result as a JSON object only.",
                                    },
                    "clear_history": False,
                    "carryover": "Wait for confirmation of code execution before terminating the conversation. Reply TERMINATE in the end when everything is done."
                },
                {
                    "sender": critic,
                    "recipient": writer,
                    "message": writing_tasks[0],
                    "carryover": "I want to include a figure and a table of the provided data in the financial report.",
                    "max_turns": 2,
                    "summary_method": "last_msg",
                }
            ]
        )


    st.image("./coding/normalized_prices.png")
    st.markdown(chat_results[-1].chat_history[-1]["content"])