import streamlit as st
from langchain.agents import create_sql_agent
from langchain.tools import Tool
from langchain.agents.agent_types import AgentType
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.utilities import SQLDatabase
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document
from dateutil import parser
from langchain_openai import OpenAI
from datetime import datetime, timedelta
import sqlite3

# Function to handle date formatting and relative time
def format_date_for_db(query_date: str):
    try:
        return parser.parse(query_date).strftime("%Y-%m-%d 00:00:00")
    except ValueError as e:
        return f"Error: {e}"

def create_and_run_sql_agent_with_tools(db_name, query, llm, api_key):
    """
    Creates an SQL agent with additional tools (retriever and date formatter) and executes a query efficiently.

    Parameters:
    - db_name: Name of the SQLite database.
    - query: The SQL query to be executed by the agent.
    - llm: The language model for the agent.
    - api_key: OpenAI API key.

    Returns:
    - query_result: The result of the executed SQL query.
    """
    # Initialize database, embeddings, and tools
    db = SQLDatabase.from_uri(f"sqlite:///{db_name}")
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)

    # Few-shot examples for SQL retrieval
    few_shots = {
        "Highest unemployment rate last year": "SELECT MAX(UNRATE) FROM economic_indicators WHERE Date BETWEEN '2022-01-01' AND '2022-12-31';",
        "Five lowest 10-year yields in 2023": "SELECT DGS10 FROM yield_curve_prices WHERE Date >= '2023-01-01' ORDER BY DGS10 ASC LIMIT 5;",
        "Production numbers for Saudi Arabia": "SELECT SAUNGDPMOMBD FROM production_data WHERE Date = (SELECT MAX(Date) FROM production_data);",
        "Change in 2-year yield over past 6 months": "SELECT DGS2 FROM yield_curve_prices WHERE Date >= date('now', '-6 months') ORDER BY Date;"
    }

    # Create retriever for few-shot examples
    vector_db = FAISS.from_documents([
        Document(page_content=q, metadata={"sql_query": few_shots[q]}) for q in few_shots
    ], embeddings)
    retriever = vector_db.as_retriever()

    # Define tools
    retriever_tool = Tool(
        name="SQLExampleRetriever",
        func=lambda query: retriever.get_relevant_documents(query),
        description="Retrieves similar SQL examples using YYYY-MM-DD format."
    )
    date_format_tool = Tool(
        name="DateFormatTool",
        func=format_date_for_db,
        description="Formats dates to 'YYYY-MM-DD 00:00:00' format."
    )

    # Initialize SQL agent with the extra tools
    agent_executor = create_sql_agent(
        llm=llm,
        toolkit=SQLDatabaseToolkit(db=db, llm=llm),
        verbose=True,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        extra_tools=[retriever_tool, date_format_tool],
        top_k=5
    )

    # Execute the query
    query_result = agent_executor.invoke(query)
    
    return query_result  # Return query result

# Streamlit interface
def run_app():
    st.title('SQL Query Executor with Language Model')
    
    # Input fields for database, query, and API key
    db_name = st.text_input('Enter the SQLite database name:', 'financial_data.db')
    query = st.text_area('Enter the SQL query or question:', 'Highest unemployment rate last year')
    api_key = st.text_input('Enter your OpenAI API Key:', type="password")
    
    if st.button('Run Query'):
        if api_key and query:
            try:
                # Initialize the language model
                llm = OpenAI(openai_api_key=api_key, temperature=0, verbose=True)
                
                # Run the query and get the response
                result = create_and_run_sql_agent_with_tools(db_name, query, llm, api_key)
                
                # Display the result
                st.subheader('Query Result:')
                st.write(result)
            except Exception as e:
                st.error(f"An error occurred: {e}")
        else:
            st.warning('Please provide both the API Key and SQL Query.')

# Run the Streamlit app
if __name__ == "__main__":
    run_app()