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

qwen2_5_config = {
    "model": "qwen2.5:14b",
    "base_url": "http://127.0.0.1:11434",
    "api_key":"placeholder",
}

phi4_14b_config = {
    "model": "phi4:14b",
    "base_url": "http://127.0.0.1:11434",
    "api_key":"placeholder",
}

gemma3_12b_config = {
    "model": "gemma3:12b",
    "base_url": "http://127.0.0.1:11434",
    "api_key":"placeholder",
}

g_final_sql = ""
g_qwen_sql = ""
g_phi4_sql = ""
g_gemma_sql = ""

g_schemalink = ""
g_desired_output = ""
g_goals = ""


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


### 1.1 Create Tools
# a. SQL executor
async def execute_query(sql: str) -> Annotated[str, "query results"]:
    import sqlite3

    global g_db_name
    db_url = f"/home/simges/.cache/spider_data/test_database/{g_db_name}/{g_db_name}.sqlite"
    conn = sqlite3.connect(db_url)
    cursor = conn.cursor()
    try:
        cursor.execute(sql)
    except Exception as e:
        return "failure: " + str(e)
    return "passed"


execute_query_tool = FunctionTool(execute_query, description="Execute sql query.")
from typing import Optional

class Message(BaseModel):
    question: Optional[str] = None
    dbschema: Optional[str] = None
    content: Optional[str] = None

# Constructor Output
class SQLOutput(BaseModel):
    sql: str

data_scientist_client = OllamaChatCompletionClient(
    model=gemma3_12b_config["model"],
    host=gemma3_12b_config["base_url"],
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
        "temperature": 0.2,
    },
    max_tokens=600,
)


query_helper_client = OllamaChatCompletionClient(
    model=qwen2_5_config["model"],
    host=qwen2_5_config["base_url"],
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
    model=phi4_14b_config["model"],
    host=phi4_14b_config["base_url"],
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

query_helper3_client = OllamaChatCompletionClient(
    model=gemma3_12b_config["model"],
    host=gemma3_12b_config["base_url"],
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


refiner_client = OllamaChatCompletionClient(
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

ANALYST_PROMPT = """
You are an intelligent assistant that extracts the user's goals & intents from a natural language question about a database.

Explain your reasoning briefly.

###Natural Language Question
{question}

Do not return SQL.
"""

SCHEMA_LINKER_PROMPT = """
Given a natural language question and a database schema (list of tables and columns), your task is to identify the corresponding list of 'exact tables and columns'.

Explain your reasoning.

#Input

###Schema
A set of tables with their column names and types.
{database_schema}

###Natural Language Question
A user’s question or instruction in natural language.
{question}

"""


SQL_BUILDER_PROMPT = """
You are tasked with building an SQL query based on the inputs given:

#Input
You are given:

#User Goals
{goals}

###Schema Linker Output
{schemalink}

###Desired Output That User wants to Display
{desired_output}

###Your Task
Use only the provided `User Goals` and `Schema Linker Output` to construct the SQL query.

**Display only the desired output given to you and use the tables and columns provided by `Schema Linker Output`**

**When comparing text columns to string literals in SQL, add `COLLATE NOCASE` to ensure case-insensitive matching.**

###Rules & Guidelines
Always use SQLite dialect.

###Database Schema
{database_schema}

"""

TOOL_CLIENT_PROMPT = """Call {tool} tool."""

REFINER_PROMPT = """
Only fix syntactic issues—do not alter variable names, logic, structure, or optimize anything unless it's strictly necessary for fixing a syntax problem.

Input

###Database Schema
{schema}

#Query
{query}

#Error
{error}
"""

@type_subscription(topic_type="analyst")
class AnalystAgent(RoutedAgent):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        self._delegate = AssistantAgent(name, model_client=data_scientist_client,
            system_message="You are an assistant that extracts the user's goals from a natural language question about a database.")

        self._delegate2 = AssistantAgent(name, model_client=data_scientist_client,
            system_message="You are an assistant that extracts the desired output from a natural language question.")
    @message_handler
    async def on_message(self, message: Message, ctx: MessageContext) -> None:
        print(f"{self.id.type} received message: {message.question}")

        await asyncio.sleep(1.0)
        response = await self._delegate.on_messages(
            [TextMessage(content=ANALYST_PROMPT.format(question=message.question), source="user")], ctx.cancellation_token
        )
        print(f"{self.id.type} responded: {response.chat_message.content}")
        global g_goals
        g_goals = response.chat_message.content

        await asyncio.sleep(1.0)
        response = await self._delegate2.on_messages(
            [TextMessage(content=f"Extract desired output from question {message.question} that user wants to display. Explain briefly.", source="user")], ctx.cancellation_token
        )
        print(f"{self.id.type} responded: {response.chat_message.content}")
        global g_desired_output
        g_desired_output = response.chat_message.content

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
        response = await self._delegate.on_messages(
            [TextMessage(content=SCHEMA_LINKER_PROMPT.format(question=message.question, database_schema=message.dbschema), source="user")],
            ctx.cancellation_token
        )
        print(f"{self.id.type} responded: {response.chat_message.content}")        
        global g_schemalink
        g_schemalink = response.chat_message.content  
        await self.publish_message(message, topic_id=TopicId(type="qwensqlbuilder", source="schemalinker"))
        await self.publish_message(message, topic_id=TopicId(type="phi4sqlbuilder", source="schemalinker"))
        await self.publish_message(message, topic_id=TopicId(type="gemmasqlbuilder", source="schemalinker"))


BUILDER_SYSTEM_PROMPT = "You are QueryBuilder, a specialized SQL synthesis agent. Your job is to construct a valid, complete SQL query from structured outputs provided by other agents. You do not interpret natural language directly."
@type_subscription(topic_type="qwensqlbuilder")
class QwenSqlBuilderAgent(RoutedAgent):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        self._delegate = AssistantAgent(name, model_client=query_helper_client, system_message=BUILDER_SYSTEM_PROMPT)

    @message_handler
    async def on_message(self, message: Message, ctx: MessageContext) -> None:
        print(f"{self.id.type} received message: {message.question}")

        await asyncio.sleep(1.0)
        global g_goals, g_schemalink, g_desired_output
        builder_prompt=SQL_BUILDER_PROMPT.format(goals=g_goals, schemalink=g_schemalink,
                                                 desired_output=g_desired_output, database_schema=message.dbschema)
        response = await self._delegate.on_messages([TextMessage(content=builder_prompt, source="user")],
                                                                    ctx.cancellation_token) 
        print(f"{self.id.type} responded: {response.chat_message.content}")
 
        global g_qwen_sql
        g_qwen_sql = clean_sql(response.chat_message.content)
        result = await execute_query(g_qwen_sql)
        if "failure" in result:
            message.content=REFINER_PROMPT.format(schema=message.dbschema, query=g_qwen_sql, error=result)
            await self.publish_message(message, topic_id=TopicId(type="refiner", source="qwensqlbuilder"))


@type_subscription(topic_type="phi4sqlbuilder")
class Phi4SqlBuilderAgent(RoutedAgent):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        self._delegate = AssistantAgent(name, model_client=query_helper2_client, system_message=BUILDER_SYSTEM_PROMPT)

    @message_handler
    async def on_message(self, message: Message, ctx: MessageContext) -> None:
        print(f"{self.id.type} received message: {message.question}")

        await asyncio.sleep(1.0)
        global g_goals, g_schemalink, g_desired_output
        builder_prompt=SQL_BUILDER_PROMPT.format(goals=g_goals, schemalink=g_schemalink,
                                                 desired_output=g_desired_output, database_schema=message.dbschema)
        response = await self._delegate.on_messages([TextMessage(content=builder_prompt, source="user")],
                                                                    ctx.cancellation_token) 
        print(f"{self.id.type} responded: {response.chat_message.content}")
 
        global g_phi4_sql
        g_phi4_sql = clean_sql(response.chat_message.content)
        result = await execute_query(g_phi4_sql)
        if "failure" in result:
            message.content=REFINER_PROMPT.format(schema=message.dbschema, query=g_phi4_sql, error=result)
            await self.publish_message(message, topic_id=TopicId(type="refiner", source="phi4sqlbuilder"))


@type_subscription(topic_type="gemmasqlbuilder")
class GemmaSqlBuilderAgent(RoutedAgent):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        self._delegate = AssistantAgent(name, model_client=query_helper3_client, system_message=BUILDER_SYSTEM_PROMPT)

    @message_handler
    async def on_message(self, message: Message, ctx: MessageContext) -> None:
        print(f"{self.id.type} received message: {message.question}")

        await asyncio.sleep(1.0)
        global g_goals, g_schemalink, g_desired_output
        builder_prompt=SQL_BUILDER_PROMPT.format(goals=g_goals, schemalink=g_schemalink,
                                                 desired_output=g_desired_output, database_schema=message.dbschema)
        response = await self._delegate.on_messages([TextMessage(content=builder_prompt, source="user")],
                                                                    ctx.cancellation_token) 
        print(f"{self.id.type} responded: {response.chat_message.content}")
 
        global g_gemma_sql
        g_gemma_sql = clean_sql(response.chat_message.content)
        result = await execute_query(g_gemma_sql)
        if "failure" in result:
            message.content=REFINER_PROMPT.format(schema=message.dbschema, query=g_gemma_sql, error=result)
            await self.publish_message(message, topic_id=TopicId(type="refiner", source="gemmasqlbuilder"))


@type_subscription(topic_type="refiner")
class RefinerAgent(RoutedAgent):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        self._delegate = AssistantAgent(name, model_client=refiner_client, system_message="You are tasked with refining SQL queries based on error message and database schema.")

    @message_handler
    async def on_message(self, message: Message, ctx: MessageContext) -> None:
        print(f"{self.id.type} received message: {message.question}")

        await asyncio.sleep(1.0)
        response = await self._delegate.on_messages([TextMessage(content=message.content, source="user")],
                                                                    ctx.cancellation_token) 
        print(f"{self.id.type} responded: {response.chat_message.content}")
 
        if ctx.topic_id.source == "gemmasqlbuilder":
            global g_gemma_sql
            g_gemma_sql = clean_sql(response.chat_message.content)
        
        elif ctx.topic_id.source == "phi4sqlbuilder":
            global g_phi4_sql
            g_phi4_sql = clean_sql(response.chat_message.content)
        
        elif ctx.topic_id.source == "qwensqlbuilder":
            global g_qwen_sql
            g_qwen_sql = clean_sql(response.chat_message.content)


async def generate_query(question: str, schema: str, db_name: str) -> str:
    global g_db_name, g_qwen_sql, g_phi4_sql, g_gemma_sql
    g_db_name = db_name


    runtime = SingleThreadedAgentRuntime()
    await AnalystAgent.register(runtime, "analyst_agent", lambda: AnalystAgent("analyst"))
    await SchemaLinkerAgent.register(runtime, "schema_linker", lambda: SchemaLinkerAgent("schemalinker"))

    await GemmaSqlBuilderAgent.register(runtime, "gemma_sql_builder", lambda: GemmaSqlBuilderAgent("gemmasqlbuilder"))
    await QwenSqlBuilderAgent.register(runtime, "qwen_sql_builder", lambda: QwenSqlBuilderAgent("qwensqlbuilder"))
    await Phi4SqlBuilderAgent.register(runtime, "phi4_sql_builder", lambda: Phi4SqlBuilderAgent("phi4sqlbuilder"))

    await RefinerAgent.register(runtime, "refiner_agent", lambda: RefinerAgent("refiner"))

    # Multi-Agent Collaboration Mode
    runtime.start()
    await runtime.publish_message(
        Message(dbschema=schema, question=question), topic_id=TopicId(type="analyst", source="default"))
    await runtime.stop_when_idle()


    result_qwen = await execute_query(g_qwen_sql)    
    result_phi4 = await execute_query(g_phi4_sql)    
    result_gemma = await execute_query(g_gemma_sql)
    strings = [
        result_qwen,
        result_phi4,
        result_gemma
    ]

    from difflib import SequenceMatcher
    from itertools import combinations
    def similarity(a, b):
        return SequenceMatcher(None, a, b).ratio()

    # Compare all pairs and track their indexes
    indexed_pairs = list(combinations(enumerate(strings), 2))

    # Compute similarities and store with index pairs
    scored = [((i1, i2), similarity(s1, s2)) for (i1, s1), (i2, s2) in indexed_pairs]

    # Print all scores
    for (i1, i2), score in scored:
        print(f"Pair ({i1}, {i2}): similarity = {score:.4f}")

    # Find most similar pair
    most_similar = max(scored, key=lambda x: x[1])
    i1, i2 = most_similar[0]
    print(f"\nMost similar pair IDs: {most_similar[0]}, similarity = {most_similar[1]:.4f}")

    if i1 == 0:
        print("PICKED QWEN")
        g_final_sql = g_qwen_sql
    elif i1 == 1:
        print("PICKED PHI4")
        g_final_sql = g_phi4_sql
    else:
        print("PICKED GEMMA")
        g_final_sql = g_gemma_sql


    g_final_sql = clean_sql(g_final_sql)
    print("Stripped SQL query: " + g_final_sql)

    result = await execute_query(g_final_sql)
    print("Final execution result: " + result)
    return g_final_sql
