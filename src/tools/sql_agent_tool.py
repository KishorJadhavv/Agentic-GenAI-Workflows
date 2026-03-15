"""
Professional SQL agent tool for safe and effective database querying.
Provides structured access to databases with built-in validation.
"""

from typing import Any, Dict, List, Optional
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from sqlalchemy import create_engine


class SQLAgentTool:
    """
    A professional tool for interacting with SQL databases.
    It can generate and execute queries safely while handling structured data.
    """

    def __init__(self, db_uri: str, model_name: str = "gpt-4-turbo-preview"):
        """
        Initialize the SQLAgentTool.

        Args:
            db_uri: The SQLAlchemy database URI.
            model_name: The name of the LLM to use for query generation.
        """
        self.engine = create_engine(db_uri)
        self.db = SQLDatabase(self.engine)
        self.llm = ChatOpenAI(model=model_name, temperature=0)
        self.toolkit = SQLDatabaseToolkit(db=self.db, llm=self.llm)

    def execute_query(self, query: str) -> str:
        """
        Executes a raw SQL query and returns the results.
        
        Args:
            query: The SQL query to execute.
            
        Returns:
            The query result as a string representation.
        """
        try:
            return self.db.run(query)
        except Exception as e:
            return f"Error executing query: {str(e)}"

    def generate_and_execute(self, natural_language_query: str) -> str:
        """
        Converts a natural language query into SQL, executes it, and returns the result.

        Args:
            natural_language_query: The user's query in plain English.

        Returns:
            The query result or an error message.
        """
        # This is a simplified version of what a full SQL agent would do.
        # In a real-world scenario, you would use a dedicated SQL agent from langchain.
        prompt = ChatPromptTemplate.from_template(
            "Based on the following table schema, generate a valid SQL query "
            "to answer the question. Only return the SQL query, nothing else.\n\n"
            "Schema:\n{schema}\n\n"
            "Question: {question}"
        )

        chain = (
            {"schema": lambda _: self.db.get_table_info(), "question": RunnablePassthrough()}
            | prompt
            | self.llm
        )

        sql_query = chain.invoke(natural_language_query).content
        return self.execute_query(sql_query)

    def get_schema_info(self) -> str:
        """
        Returns the database schema information.
        """
        return self.db.get_table_info()


if __name__ == "__main__":
    # Example usage (requires a SQLite database)
    # create a dummy sqlite db first
    import sqlite3
    conn = sqlite3.connect("test.db")
    conn.execute("CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, name TEXT, email TEXT)")
    conn.execute("INSERT INTO users (name, email) VALUES ('Alice', 'alice@example.com')")
    conn.execute("INSERT INTO users (name, email) VALUES ('Bob', 'bob@example.com')")
    conn.commit()
    conn.close()

    try:
        sql_tool = SQLAgentTool("sqlite:///test.db")
        print("Schema Info:\n", sql_tool.get_schema_info())
        
        result = sql_tool.generate_and_execute("List all users in the database.")
        print("Query Result:\n", result)
    except Exception as e:
        print(f"Error: {e}")
