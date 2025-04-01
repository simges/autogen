import openai
import re
from typing_extensions import Annotated
from pydantic import BaseModel

from autogen_agentchat.agents import AssistantAgent

from autogen_ext.models.ollama import OllamaChatCompletionClient
from dataclasses import dataclass
from autogen_core import (
    AgentId,
    MessageContext,
    RoutedAgent,
    SingleThreadedAgentRuntime,
    TopicId,
    message_handler,
    type_subscription,
)
from autogen_core.models import (
    LLMMessage,
    ModelFamily,
    ModelInfo,
)
from autogen_core.tools import FunctionTool
import time
from autogen_agentchat.messages import TextMessage
import pandas as pd


data_scientist_config = {
    "model": "sqlcoder-7b-2",
    "base_url": "http://127.0.0.1:11434",
    "api_key":"placeholder",
}

g_db_creds={
    "user": "postgres",
    "password": "postgres",
    "host": "127.0.0.1",
    "port": "5432",
}
g_final_sql = ""
g_db_name = ""

def clean_sql(raw_sql):
    cleaned = raw_sql.replace("\\n", " ")          # Remove newlines
    cleaned = cleaned.replace('\\"', '"')         # Replace escaped quotes
    cleaned = cleaned.replace('\\', '')           # Remove remaining backslashes
    cleaned = ' '.join(cleaned.split())           # Remove extra spaces
    return cleaned


def escape_percent(match):
    # Extract the matched group
    group = match.group(0)
    # Replace '%' with '%%' within the matched group
    escaped_group = group.replace("%", "%%")
    # Return the escaped group
    return escaped_group

### 1.1 Create Tools
# a. SQL executor
async def execute_query(sql: Annotated[str, "sql query"]) -> Annotated[str, "query results"]:
    engine = None
    try:
        try:
            import psycopg

            has_psycopg = True
        except ImportError:
            has_psycopg = False
        try:
            import psycopg2

            has_psycopg2 = True
        except ImportError:
            has_psycopg2 = False
        if not has_psycopg2 and not has_psycopg:
            print(
                "You do not have psycopg2 or psycopg installed. Please install either."
            )
            exit(1)
        from sqlalchemy import create_engine, text
        from func_timeout import func_timeout
        LIKE_PATTERN = r"LIKE[\s\S]*'"

        if has_psycopg2:
            dialect_prefix = "postgresql"
        elif has_psycopg:
            dialect_prefix = "postgresql+psycopg"
        print("Executing sql: " + sql)
        db_url = f"{dialect_prefix}://{g_db_creds['user']}:{g_db_creds['password']}@{g_db_creds['host']}:{g_db_creds['port']}/{g_db_name}"
        engine = create_engine(db_url)
        escaped_query = re.sub(
            LIKE_PATTERN, escape_percent, sql, flags=re.IGNORECASE
        )  # ignore case of LIKE

        timeout=10.0
        results_df = func_timeout(
            timeout, pd.read_sql_query, args=(escaped_query, engine)
        )

        # round floats to decimal_points
        return "passed"
    except Exception as e:
        return "failure: " + str(e)
execute_query_tool = FunctionTool(execute_query, description="Execute sql query.")


@dataclass
class Message:
    content: str

# CONSTRUCTOR CLIENT
data_scientist_client = OllamaChatCompletionClient(
    model=data_scientist_config["model"],
    host=data_scientist_config["base_url"],
    model_info={
        "vision": False,
        "function_calling": False,
        "json_output": True,
        "family": ModelFamily.R1,
        "structured_output": False,
    },
    options={
        "num_ctx": 16384,
        "frequency_penalty": 0.7,
        "temperature": 0.0,
    },
    max_tokens=600,
)


DATA_SCIENTIST_PROMPT = """
# Task 
Generate a SQL query to answer the following question: `{question}`

### PostgreSQL Database Schema 
The query will run on a database with the following schema: 
{database_schema}

<SQL Table DDL Statements>

# SQL 
Here is the SQL query that answers the question: `{question}` 
'''sql
"""


@type_subscription(topic_type="datascientist")
class DataScientist(RoutedAgent):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        self._delegate = AssistantAgent(name, model_client=data_scientist_client,
            system_message="You are a highly skilled AI assistant that translates natural language questions into correct and executable PostgreSQL queries.")

    @message_handler
    async def on_message(self, message: Message, ctx: MessageContext) -> str:
        print(f"{self.id.type} received message: {message.content}")
        time.sleep(2)
        response = await self._delegate.on_messages(
            [TextMessage(content=message.content, source="user")], ctx.cancellation_token
        )
        print(f"{self.id.type} responded: {response.chat_message.content}")
        global g_final_sql
        g_final_sql = response.chat_message.content

        # Single-Agent Mode
        return g_final_sql


async def generate_query(question: str, schema: str, db_name: str) -> str:
    global g_db_name
    g_db_name = db_name
    runtime = SingleThreadedAgentRuntime()
    await DataScientist.register(runtime, "data_scientist", lambda: DataScientist("datascientist"))

    runtime.start()
    # Single-Agent chatdb/natural-sql-7b Mode
    await runtime.send_message(
        Message(DATA_SCIENTIST_PROMPT.format(question=question, database_schema=schema)),
        recipient=AgentId(type="data_scientist", key="default"))
    await runtime.stop_when_idle()

    global g_final_sql
    time.sleep(0.5)

    match = re.search(r'"sql":\s*"(.*)"', g_final_sql, re.DOTALL)
    if match:
        g_final_sql = match.group(1).strip()

    match = re.search(r"```sql\s*(.*?)\s*```", g_final_sql, re.DOTALL)
    if match:
        g_final_sql = match.group(1).strip()
    g_final_sql = g_final_sql.split(";", 1)[0].strip() + ";"

    print("Stripped SQL query: " + g_final_sql)
    g_final_sql = clean_sql(g_final_sql)
    result = await execute_query(g_final_sql)
    print("Final execution result: " + result)
    return g_final_sql