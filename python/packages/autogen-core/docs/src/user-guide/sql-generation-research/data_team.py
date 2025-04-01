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


tool_executor_config = {
    "model": "mistral-nemo:latest",
    "base_url": "http://127.0.0.1:11434",
    "api_key":"placeholder",
}

junior_data_scientist_config = {
    "model": "sqlcoder-7b-2:latest",
    "base_url": "http://127.0.0.1:11434",
    "api_key":"placeholder",
}

query_reviewer_config = {
    "model": "mistral-nemo:latest",
    "base_url": "http://127.0.0.1:11434",
    "api_key":"placeholder",
}

senior_data_scientist_config = {
    "model": "sqlcoder-7b-2:latest",
    "base_url": "http://127.0.0.1:11434",
    "api_key":"placeholder",
}

g_db_name=""
g_db_creds={
    "user": "postgres",
    "password": "postgres",
    "host": "127.0.0.1",
    "port": "5432",
}
g_final_sql = ""
g_schema = ""
g_question = ""
g_num_fix_attemps = 0

def clean_sql(final_sql):
    match = re.search(r'"sql":\s*"(.*)"', final_sql, re.DOTALL)
    if match:
        final_sql = match.group(1).strip()

    match = re.search(r"```sql\s*(.*?)\s*```", final_sql, re.DOTALL)
    if match:
        final_sql = match.group(1).strip()
    final_sql = final_sql.split(";", 1)[0].strip() + ";"

    cleaned = final_sql.replace("\\n", " ")          # Remove newlines
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


tool_executor_client = OllamaChatCompletionClient(
    model=tool_executor_config["model"],
    host=tool_executor_config["base_url"],
    model_info={
        "vision": False,
        "function_calling": True,
        "json_output": False,
        "family": ModelFamily.R1,
        "structured_output": False,
    },
    options={
        "num_ctx": 8192,
        "temperature": 0.0,
    },
)

junior_data_scientist_client = OllamaChatCompletionClient(
    model=junior_data_scientist_config["model"],
    host=junior_data_scientist_config["base_url"],
    model_info={
        "vision": False,
        "function_calling": False,
        "json_output": False,
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


senior_data_scientist_client = OllamaChatCompletionClient(
    model=senior_data_scientist_config["model"],
    host=senior_data_scientist_config["base_url"],
    model_info={
        "vision": False,
        "function_calling": False,
        "json_output": False,
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


# CONSTRUCTOR CLIENT
class SQLOutput(BaseModel):
    sql: str
query_reviewer_client = OllamaChatCompletionClient(
    model=query_reviewer_config["model"],
    host=query_reviewer_config["base_url"],
    model_info={
        "vision": False,
        "function_calling": False,
        "json_output": True,
        "family": ModelFamily.R1,
        "structured_output": True,
    },
    response_format=SQLOutput,
    options={
        "num_ctx": 131072,
        "frequency_penalty": 0.7,
        "temperature": 0.0,
    },
    max_tokens=600,
)

TOOL_CLIENT_PROMPT = """You are an AI assistant with access to {tool} tool. 
    You must invoke {tool} tool with the provided query {query}. Follow these rules:

    Extract necessary details from the user’s input.
    Call the tool with the extracted data in the correct format.
    Do not answer manually.

    Your Task:

    Identify the correct tool to use.
    Call it with the required arguments.
    Return the tool's response to the user."""


SQLCODER7B_2_DATA_SCIENTIST_PROMPT = """
### Task
Generate a SQL query to answer [QUESTION]{question}[/QUESTION]

### Database Schema
The query will run on a database with the following schema:
{database_schema}

### Answer
Given the database schema, here is the SQL query that [QUESTION]{question}[/QUESTION]
[SQL]
"""

QUERY_REVIEWER_PROMPT = """
### Task
You are given:
1. A **PostgreSQL database schema** : {database_schema}
2. A **natural language question** :  {question}
3. An **SQL query that is intended to answer the question but contains execution errors** : {query}
4. The **error message returned when executing the query** : {error}

Your job is to:
- Analyze the schema, the question, the faulty query, and the error message.
- Rewrite the SQL query to ensure it is **executable in PostgreSQL**, free of syntax or semantic errors.
- Ensure that the corrected query **accurately answers the original question**.
- Respect naming conventions and relationships as defined in the schema.
- Use **PostgreSQL-compatible syntax only**.
- Alias tables using the `AS` keyword for clarity.
- Do **not hallucinate** columns or tables not defined in the schema.
- Be concise and correct—do not add comments or explanations.
"""


@dataclass
class Message:
    content: str
@type_subscription(topic_type="toolexecutor")
class ToolExecutor(RoutedAgent):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        self._delegate = AssistantAgent(name, model_client=tool_executor_client,
                    system_message="You are an AI assistant helping data scientist with SQL query execution.",
                    tools=[execute_query_tool])

    @message_handler
    async def on_message(self, message: Message, ctx: MessageContext) -> None:
        print(f"{self.id.type} received message: {message.content}")
        time.sleep(2)
        response = await self._delegate.on_messages(
            [TextMessage(content=message.content, source="user")], ctx.cancellation_token
        )
        print(f"{self.id.type} responded: {response.chat_message.content}")
        if "passed" in response.chat_message.content:
            print("passed")
        else:
            if ctx.topic_id.source != "queryreviewer":
                global g_schema
                global g_final_sql
                global g_question

                message=QUERY_REVIEWER_PROMPT.format(database_schema=g_schema,
                                                    question=g_question,
                                                    query=g_final_sql,
                                                    error=response.chat_message.content)
                await self.publish_message(Message(message), topic_id=TopicId(type="queryreviewer", source="default"))
            else:
                global g_num_fix_attemps
                if g_num_fix_attemps == 3:
                    print("failure: too much fix attempt")
                elif g_num_fix_attemps == 1:
                    message=SQLCODER7B_2_DATA_SCIENTIST_PROMPT.format(database_schema=g_schema, question=g_question)
                    await self.publish_message(Message(message), topic_id=TopicId(type="seniordatascientist", source="default"))
                else:
                    message=QUERY_REVIEWER_PROMPT.format(database_schema=g_schema,
                                                        query=g_final_sql,
                                                        question=g_question,
                                                        error=response.chat_message.content)
                    await self.publish_message(Message(message), topic_id=TopicId(type="queryreviewer", source="default"))




@type_subscription(topic_type="juniordatascientist")
class JuniorDataScientist(RoutedAgent):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        self._delegate = AssistantAgent(name, model_client=junior_data_scientist_client,
            system_message="You are a highly skilled AI assistant that translates natural language questions into correct and executable PostgreSQL queries.")

    @message_handler
    async def on_message(self, message: Message, ctx: MessageContext) -> None:
        print(f"{self.id.type} received message: {message.content}")
        time.sleep(2)
        response = await self._delegate.on_messages(
            [TextMessage(content=message.content, source="user")], ctx.cancellation_token
        )
        print(f"{self.id.type} responded: {response.chat_message.content}")
        global g_final_sql
        g_final_sql = response.chat_message.content

        # Multi-Agent Collaboration Mode
        message=TOOL_CLIENT_PROMPT.format(tool="execute_query", query=g_final_sql)
        await self.publish_message(Message(message),
                    topic_id=TopicId(type="toolexecutor", source="juniordatascientist"))


@type_subscription(topic_type="seniordatascientist")
class SeniorDataScientist(RoutedAgent):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        self._delegate = AssistantAgent(name, model_client=senior_data_scientist_client,
            system_message="You are a highly skilled AI assistant that translates natural language questions into correct and executable PostgreSQL queries.")

    @message_handler
    async def on_message(self, message: Message, ctx: MessageContext) -> None:

        time.sleep(2)
        print(f"{self.id.type} received message: {message.content}")
        response = await self._delegate.on_messages(
            [TextMessage(content=message.content, source="user")], ctx.cancellation_token
        )
        print(f"{self.id.type} responded: {response.chat_message.content}")
        global g_final_sql
        g_final_sql = response.chat_message.content

        # Multi-Agent Collaboration Mode
        message=TOOL_CLIENT_PROMPT.format(tool="execute_query", query=g_final_sql)
        await self.publish_message(Message(message),
                    topic_id=TopicId(type="toolexecutor", source="seniordatascientist"))


@type_subscription(topic_type="queryreviewer")
class QueryReviewer(RoutedAgent):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        self._delegate = AssistantAgent(name, model_client=query_reviewer_client,
                system_message="You are a highly skilled AI assistant specialized in SQL troubleshooting and query generation for PostgreSQL databases.")

    @message_handler
    async def on_message(self, message: Message, ctx: MessageContext) -> None:
        global g_num_fix_attemps
        g_num_fix_attemps = g_num_fix_attemps + 1

        time.sleep(2)
        print(f"{self.id.type} received message: {message.content}")
        response = await self._delegate.on_messages(
            [TextMessage(content=message.content, source="user")], ctx.cancellation_token
        )
        print(f"{self.id.type} responded: {response.chat_message.content}")
        global g_final_sql
        g_final_sql = response.chat_message.content

        await self.publish_message(Message(response.chat_message.content),
                                   topic_id=TopicId(type="toolexecutor", source="queryreviewer"))


async def generate_query(question: str, schema: str, db_name: str) -> str:
    global g_db_name
    g_db_name = db_name

    global g_question
    g_question = question

    global g_schema
    g_schema = schema

    global g_num_fix_attemps
    g_num_fix_attemps = 0

    runtime = SingleThreadedAgentRuntime()
    await ToolExecutor.register(runtime, "tool_executor", lambda: ToolExecutor("toolexecutor"))
    await JuniorDataScientist.register(runtime, "junior_data_scientist", lambda: JuniorDataScientist("juniordatascientist"))
    await SeniorDataScientist.register(runtime, "senior_data_scientist", lambda: SeniorDataScientist("seniordatascientist"))
    await QueryReviewer.register(runtime, "query_reviewer", lambda: QueryReviewer("queryreviewer"))

    # Multi-Agent Collaboration Mode
    runtime.start()
    await runtime.publish_message(
        Message(SQLCODER7B_2_DATA_SCIENTIST_PROMPT.format(question=question, database_schema=schema)),
        topic_id=TopicId(type="juniordatascientist", source="default"))
    await runtime.stop_when_idle()

    global g_final_sql
    time.sleep(0.5)

    g_final_sql = clean_sql(g_final_sql)
    print("Stripped SQL query: " + g_final_sql)

    result = await execute_query(g_final_sql)
    print("Final execution result: " + result)
    return g_final_sql