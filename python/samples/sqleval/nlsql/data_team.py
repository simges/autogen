import openai
import re
import asyncio
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


mistral_nemo_config = {
    "model": "mistral-nemo:latest",
    "base_url": "http://127.0.0.1:11434",
    "api_key":"placeholder",
}

gemma2_9b_config = {
    "model": "gemma2:9b",
    "base_url": "http://127.0.0.1:11434",
    "api_key":"placeholder",
}

qwen_2_5_config = {
    "model": "qwen2.5:latest",
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

g_schemalink = ""
g_goals = ""
g_num_fix_attempts = 0


def clean_sql(final_sql):
    match = re.search(r'"sql"\s*:\s*"((?:[^"\\]|\\.)*)"', final_sql, re.DOTALL)
    if match:
        final_sql = match.group(1).strip()

    match = re.search(r"```sql\s*(.*?)\s*```", final_sql, re.DOTALL)
    if match:
        final_sql = match.group(1).strip()
    final_sql = final_sql.split(";", 1)[0].strip() + ";"

    cleaned = final_sql.replace("\\n", " ")          # Remove newlines
    cleaned = cleaned.replace('\\"', '')         # Replace escaped quotes 
    cleaned = cleaned.replace('\\', '')           # Remove remaining backslashes
    cleaned = cleaned.replace("`", '')           # Remove remaining backticks
    cleaned = ' '.join(cleaned.split())           # Remove extra spaces
    if len(cleaned) < 12: return ""
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
async def execute_query() -> Annotated[str, "query results"]:
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
        global g_final_sql
        print("Executing sql: " + g_final_sql)
        db_url = f"{dialect_prefix}://{g_db_creds['user']}:{g_db_creds['password']}@{g_db_creds['host']}:{g_db_creds['port']}/{g_db_name}"
        engine = create_engine(db_url)
        escaped_query = re.sub(
            LIKE_PATTERN, escape_percent, g_final_sql, flags=re.IGNORECASE
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
from typing import Optional

class Message(BaseModel):
    question: Optional[str] = None
    dbschema: Optional[dict] = None
    content: Optional[str] = None

# Constructor Output
class SQLOutput(BaseModel):
    sql: str

data_scientist_client = OllamaChatCompletionClient(
    model=gemma2_9b_config["model"],
    host=gemma2_9b_config["base_url"],
    model_info={
        "vision": False,
        "function_calling": False,
        "json_output": True,
        "family": ModelFamily.UNKNOWN,
        "structured_output": True,
    },
    options={
        "num_ctx": 16384,
        "frequency_penalty": 0.2,
        "temperature": 0.2,
    },
    max_tokens=600,
)

query_helper_client = OllamaChatCompletionClient(
    model=qwen_2_5_config["model"],
    host=qwen_2_5_config["base_url"],
    model_info={
        "vision": False,
        "function_calling": False,
        "json_output": True,
        "family": ModelFamily.UNKNOWN,
        "structured_output": True,
    },
    options={
        "num_ctx": 16384,
        "frequency_penalty": 0.0,
        "temperature": 0.0,
    },
    response_format=SQLOutput,
    max_tokens=300,
)


query_helper2_client = OllamaChatCompletionClient(
    model=mistral_nemo_config["model"],
    host=mistral_nemo_config["base_url"],
    model_info={
        "vision": False,
        "function_calling": False,
        "json_output": True,
        "family": ModelFamily.UNKNOWN,
        "structured_output": True,
    },
    options={
        "num_ctx": 16384,
        "frequency_penalty": 0.0,
        "temperature": 0.0,
    },
    response_format=SQLOutput,
    max_tokens=300,
)

tool_executor_client = OllamaChatCompletionClient(
    model=mistral_nemo_config["model"],
    host=mistral_nemo_config["base_url"],
    model_info={
        "vision": False,
        "function_calling": True,
        "json_output": False,
        "family": ModelFamily.UNKNOWN,
        "structured_output": False,
    },
    options={
        "num_ctx": 8192,
        "temperature": 0.0,
    },
)

async def extract_schema() -> Annotated[dict, "database schema"]:
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

        if has_psycopg2:
            dialect_prefix = "postgresql"
        elif has_psycopg:
            dialect_prefix = "postgresql+psycopg"

        global g_db_name, g_db_creds
        print("Extracting database schema for: " + g_db_name)

        conn = psycopg2.connect(
            dbname=g_db_name,
            user=g_db_creds['user'],
            password=g_db_creds['password'],
            host=g_db_creds['host'],
            port=g_db_creds['port']
        )
        cur = conn.cursor()
        cur.execute("""
            SELECT table_name, column_name
            FROM information_schema.columns
            WHERE table_schema = 'public'
            ORDER BY table_name, ordinal_position;
        """)

        from collections import defaultdict
        schema = defaultdict(list)
        for table, column in cur.fetchall():
            schema[table].append(column)

        cur.close()
        conn.close()
        return dict(schema)
    except Exception as e:
        print("Error message: " + str(e))
        return dict()

ANALYST_PROMPT = """
You are an intelligent assistant that extracts the user's goals from a natural language question about a database. 

Your output must always include what user wants to display.

Explain step by step which goals must be achieved to display that result.
If user wants to display a ratio, briefly explain the steps how to calculate it.

###Natural Language Question
{question}

Do not return SQL.

Use clear and concise phrases that describe the desired outcome.
"""

SCHEMA_LINKER_PROMPT = """
Your task is to link natural language terms in a user question to the actual tables and columns in a database schema.

Avoid using non-existent fields.

Do not generate an SQL.

#Input

###Schema
A set of tables with their column names and types.
{database_schema}

###Natural Language Question
A user’s question or instruction in natural language.
{question}

"""

SQL_BUILDER_PROMPT = """
#Input
You are given:

#User Goals
{goals}

###Schema Linker Output
{schemalink}

###Your Task
Use only the provided User Goals and Schema Linker Output to construct the SQL query.

Ensure that you will not get "column does not exist" error after execution.

###Rules & Guidelines
Always use PostgreSQL dialect.

###Database Schema
{database_schema}

"""


TOOL_CLIENT_PROMPT = """Call {tool} tool."""


@type_subscription(topic_type="analyst")
class AnalystAgent(RoutedAgent):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        self._delegate = AssistantAgent(name, model_client=data_scientist_client,
            system_message="You are an intelligent assistant that extracts the user's goals from a natural language question about a database.")

    @message_handler
    async def on_message(self, message: Message, ctx: MessageContext) -> None:
        print(f"{self.id.type} received message: {message.question}")

        await asyncio.sleep(1.0)
        analyst_prompt=ANALYST_PROMPT.format(question=message.question)
        response = await self._delegate.on_messages(
            [TextMessage(content=analyst_prompt, source="user")], ctx.cancellation_token
        )
        print(f"{self.id.type} responded: {response.chat_message.content}")
        
        global g_goals
        g_goals = response.chat_message.content
        await self.publish_message(message, topic_id=TopicId(type="schemalinker", source="analyst"))


@type_subscription(topic_type="schemalinker")
class SchemaLinkerAgent(RoutedAgent):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        self._delegate = AssistantAgent(name, model_client=data_scientist_client,
            system_message="You are a database-aware assistant you are tasked with linking natural language terms in a user question to the actual tables and columns in a database schema.")

    @message_handler
    async def on_message(self, message: Message, ctx: MessageContext) -> None:
        print(f"{self.id.type} received message: {message.question}")

        await asyncio.sleep(1.0)
        linker_prompt=SCHEMA_LINKER_PROMPT.format(question=message.question, database_schema=message.dbschema)
        response = await self._delegate.on_messages(
            [TextMessage(content=linker_prompt, source="user")], ctx.cancellation_token
        )
        print(f"{self.id.type} responded: {response.chat_message.content}")
        
        global g_schemalink, g_goals
        g_schemalink = response.chat_message.content

        builder_prompt=SQL_BUILDER_PROMPT.format(goals=g_goals, schemalink=g_schemalink, database_schema=message.dbschema)
        message.content=builder_prompt
        await self.publish_message(message, topic_id=TopicId(type="sqlbuilder", source="clauseagent"))


SYSTEM_PROMPT = "You are QueryBuilder, a specialized SQL synthesis agent. Your job is to construct a valid, complete SQL query from structured outputs provided by other agents. You do not interpret natural language directly."
@type_subscription(topic_type="sqlbuilder")
class SqlBuilderAgent(RoutedAgent):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        self._delegate = AssistantAgent(name, model_client=query_helper_client, system_message=SYSTEM_PROMPT)
        self._delegate2 = AssistantAgent(name, model_client=query_helper2_client, system_message=SYSTEM_PROMPT)

    @message_handler
    async def on_message(self, message: Message, ctx: MessageContext) -> None:
        print(f"{self.id.type} received message: {message.question}")

        await asyncio.sleep(1.0)
        global g_num_fix_attempts
        g_num_fix_attempts = g_num_fix_attempts + 1
        if g_num_fix_attempts == 1:
            response = await self._delegate.on_messages(
                [TextMessage(content=message.content, source="user")], ctx.cancellation_token
            )
        else:
            response = await self._delegate2.on_messages(
                [TextMessage(content=message.content, source="user")], ctx.cancellation_token
            )
        print(f"{self.id.type} responded: {response.chat_message.content}")
        global g_final_sql
        g_final_sql = clean_sql(response.chat_message.content)

        # Multi-Agent Collaboration Mode
        message.content=TOOL_CLIENT_PROMPT.format(tool="execute_query")
        await self.publish_message(message, topic_id=TopicId(type="toolexecutor", source="sqlbuilder"))


@type_subscription(topic_type="toolexecutor")
class ToolExecutor(RoutedAgent):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        self._delegate = AssistantAgent(name, model_client=tool_executor_client,
                    system_message="You are an AI assistant helping data scientist with SQL query execution.",
                    tools=[execute_query_tool])

    @message_handler
    async def on_message(self, message: Message, ctx: MessageContext) -> None:
        print(f"{self.id.type} received message: {message.question}")

        await asyncio.sleep(1.0)
        response = await self._delegate.on_messages(
            [TextMessage(content=message.content, source="user")], ctx.cancellation_token
        )
        print(f"{self.id.type} responded: {response.chat_message.content}")

        if "passed" in response.chat_message.content:
            print("passed")
        else:
            global g_num_fix_attempts
            if g_num_fix_attempts == 3:
                print("failure: too much fix attempt")
            else:
                global g_final_sql, g_goals, g_schemalink
                builder_prompt=SQL_BUILDER_PROMPT.format(goals=g_goals, schemalink=g_schemalink, database_schema=message.dbschema)
                message.content=builder_prompt
                await self.publish_message(message, topic_id=TopicId(type="sqlbuilder", source="clauseagent"))


async def generate_query(question: str, schema: str, db_name: str) -> str:
    global g_db_name, g_final_sql
    g_db_name = db_name

    global g_num_fix_attempts
    g_num_fix_attempts = 0

    runtime = SingleThreadedAgentRuntime()
    await AnalystAgent.register(runtime, "analyst_agent", lambda: AnalystAgent("analyst"))
    await SchemaLinkerAgent.register(runtime, "schema_linker", lambda: SchemaLinkerAgent("schemalinker"))

    await SqlBuilderAgent.register(runtime, "sql_builder", lambda: SqlBuilderAgent("sqlbuilder"))

    await ToolExecutor.register(runtime, "tool_executor", lambda: ToolExecutor("toolexecutor"))

    # Multi-Agent Collaboration Mode
    runtime.start()
    await runtime.publish_message(
        Message(dbschema=schema, question=question), topic_id=TopicId(type="analyst", source="default"))
    await runtime.stop_when_idle()

    g_final_sql = clean_sql(g_final_sql)
    print("Stripped SQL query: " + g_final_sql)

    result = await execute_query()
    print("Final execution result: " + result)
    return g_final_sql
