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
g_mistral_sql = ""

g_schemalink = ""
g_retriever_output = ""
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
from llama_index.core import VectorStoreIndex, Settings, SimpleDirectoryReader
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexAutoRetriever
from llama_index.core.vector_stores import MetadataInfo, VectorStoreInfo
from pathlib import Path
import shutil, os
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.llms.openai import OpenAI

SENTENCE_TRANSFORMERS_HOME="/home/simges/.cache/huggingface/hub/sentence-transformers/{model_name}"

MINILM_EMBEDDING_MODEL_PATH=SENTENCE_TRANSFORMERS_HOME.format(model_name="/all-MiniLM-L6-v2")

os.environ["TIKTOKEN_CACHE_DIR"]="/home/simges/cl100k_base/"

Settings.embed_model = HuggingFaceEmbedding(model_name=MINILM_EMBEDDING_MODEL_PATH)
Settings.llm = Ollama(model="gemma3:12b", request_timeout=360.0)

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
        "repetition_penalty": 1.0,
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
        "repetition_penalty": 1.0,
    },
    response_format=SQLOutput,
    max_tokens=300,
)

query_helper3_client = OllamaChatCompletionClient(
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
        "repetition_penalty": 1.0,
    },
    response_format=SQLOutput,
    max_tokens=300,
)


ANALYST_PROMPT = """
Extract user goals from given user question.

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

###Relevant Data
{relevant_data}

###Your Task
Do exactly what is decribed in `User Goals` and `Schema Linker Output` to construct the SQL query.

**When comparing text columns to string literals in SQL, employ COLCOLATE NOCASE to ensure that the comparison is case-insensitive.**

###Rules & Guidelines
Always use SQLite dialect.

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
            system_message="You are an intelligent assistant that extracts the user's goals & intents from a natural language question about a database.")

    @message_handler
    async def on_message(self, message: Message, ctx: MessageContext) -> None:
        print(f"{self.id.type} received message: {message.question}")

        await asyncio.sleep(1.0)
        response = await self._delegate.on_messages(
            [TextMessage(content=ANALYST_PROMPT.format(question=message.question), source="user")], ctx.cancellation_token
        )
        print(f"{self.id.type} responded: {response.chat_message.content}")
        global g_goals, g_db_name
        g_goals = response.chat_message.content

        try:
            await asyncio.sleep(1.0)
            documents = SimpleDirectoryReader(f"./test_database/{g_db_name}/").load_data()
            vector_index = VectorStoreIndex.from_documents(documents)

            vector_store_info = VectorStoreInfo(
                content_info="database",
                metadata_info=[
                    MetadataInfo(
                        name="database", type="str", description="supportive documents"
                    ),
                ],
            )
            vector_auto_retriever = VectorIndexAutoRetriever(
                vector_index, vector_store_info=vector_store_info
            )
            retriever_query_engine = RetrieverQueryEngine.from_args(vector_auto_retriever)
            response = retriever_query_engine.retrieve(f"""{g_goals}""")
            if len(response) != 0:
                global g_retriever_output
                g_retriever_output = response[0].text
                print(f"{self.id.type} responded: {g_retriever_output}")
        except Exception as e:
            print(f"{self.id.type} responded: {e}")
            pass
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
        await self.publish_message(message, topic_id=TopicId(type="mistralsqlbuilder", source="schemalinker"))



BUILDER_SYSTEM_PROMPT = "You are QueryBuilder, a specialized SQL synthesis agent. Your job is to construct a valid, complete SQL query."
@type_subscription(topic_type="qwensqlbuilder")
class QwenSqlBuilderAgent(RoutedAgent):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        self._delegate = AssistantAgent(name, model_client=query_helper_client, system_message=BUILDER_SYSTEM_PROMPT)

    @message_handler
    async def on_message(self, message: Message, ctx: MessageContext) -> None:
        print(f"{self.id.type} received message: {message.question}")

        await asyncio.sleep(1.0)
        global g_goals, g_schemalink, g_retriever_output
        builder_prompt=SQL_BUILDER_PROMPT.format(goals=g_goals, schemalink=g_schemalink,
                                                 relevant_data=g_retriever_output)
        response = await self._delegate.on_messages([TextMessage(content=builder_prompt, source="user")],
                                                                    ctx.cancellation_token) 
        print(f"{self.id.type} responded: {response.chat_message.content}")
 
        global g_qwen_sql
        g_qwen_sql = clean_sql(response.chat_message.content)



@type_subscription(topic_type="phi4sqlbuilder")
class Phi4SqlBuilderAgent(RoutedAgent):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        self._delegate = AssistantAgent(name, model_client=query_helper2_client, system_message=BUILDER_SYSTEM_PROMPT)

    @message_handler
    async def on_message(self, message: Message, ctx: MessageContext) -> None:
        print(f"{self.id.type} received message: {message.question}")

        await asyncio.sleep(1.0)
        global g_goals, g_schemalink, g_retriever_output
        builder_prompt=SQL_BUILDER_PROMPT.format(goals=g_goals, schemalink=g_schemalink,
                                                 relevant_data=g_retriever_output)
        response = await self._delegate.on_messages([TextMessage(content=builder_prompt, source="user")],
                                                                    ctx.cancellation_token) 
        print(f"{self.id.type} responded: {response.chat_message.content}")
 
        global g_phi4_sql
        g_phi4_sql = clean_sql(response.chat_message.content)



@type_subscription(topic_type="mistralsqlbuilder")
class MistralSqlBuilderAgent(RoutedAgent):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        self._delegate = AssistantAgent(name, model_client=query_helper3_client, system_message=BUILDER_SYSTEM_PROMPT)

    @message_handler
    async def on_message(self, message: Message, ctx: MessageContext) -> None:
        print(f"{self.id.type} received message: {message.question}")

        await asyncio.sleep(1.0)
        global g_goals, g_schemalink, g_retriever_output
        builder_prompt=SQL_BUILDER_PROMPT.format(goals=g_goals, schemalink=g_schemalink,
                                                 relevant_data=g_retriever_output)
        response = await self._delegate.on_messages([TextMessage(content=builder_prompt, source="user")],
                                                                    ctx.cancellation_token) 
        print(f"{self.id.type} responded: {response.chat_message.content}")
 
        global g_mistral_sql
        g_mistral_sql = clean_sql(response.chat_message.content)



async def generate_query(question: str, schema: str, db_name: str) -> str:
    global g_db_name, g_qwen_sql, g_phi4_sql, g_mistral_sql
    g_db_name = db_name


    runtime = SingleThreadedAgentRuntime()
    await AnalystAgent.register(runtime, "analyst_agent", lambda: AnalystAgent("analyst"))
    await SchemaLinkerAgent.register(runtime, "schema_linker", lambda: SchemaLinkerAgent("schemalinker"))

    await MistralSqlBuilderAgent.register(runtime, "mistral_sql_builder", lambda: MistralSqlBuilderAgent("mistralsqlbuilder"))
    await QwenSqlBuilderAgent.register(runtime, "qwen_sql_builder", lambda: QwenSqlBuilderAgent("qwensqlbuilder"))
    await Phi4SqlBuilderAgent.register(runtime, "phi4_sql_builder", lambda: Phi4SqlBuilderAgent("phi4sqlbuilder"))


    runtime.start()
    await runtime.publish_message(
        Message(dbschema=schema, question=question), topic_id=TopicId(type="analyst", source="default"))
    await runtime.stop_when_idle()

    result_qwen = await execute_query(g_qwen_sql)    
    result_phi4 = await execute_query(g_phi4_sql)    
    result_mistral = await execute_query(g_mistral_sql)

    pairs = [
        (g_qwen_sql, result_qwen),
        (g_phi4_sql, result_phi4),
        (g_mistral_sql, result_mistral)
    ]
    filtered_pairs = [(query, result) for query, result in pairs if "passed" in result]
    if len(filtered_pairs) == 0:
        print("")
        print("All generated queries failed to execute. Return Qwen query: " + g_qwen_sql)
        return g_qwen_sql
    elif len(filtered_pairs) == 1:
        query, result = filtered_pairs[0]
        print("Single query only has been executed successfully.\n query: " + query + "\nExecution result: " + result)
        return query


    from difflib import SequenceMatcher
    from itertools import combinations
    def similarity(a, b):
        return SequenceMatcher(None, a, b).ratio()

    scored = [
        ((q1, q2), similarity(r1, r2))
        for (q1, r1), (q2, r2) in combinations(filtered_pairs, 2)
    ]
    for (q1, q2), score in scored:
        print(f"Pair ({q1}, {q2}): similarity = {score:.4f}")

    # Find most similar pair
    most_similar = max(scored, key=lambda x: x[1])
    q1, q2 = most_similar[0]
    print(f"\nMost similar pair IDs: {most_similar[0]}, similarity = {most_similar[1]:.4f}")


    result = await execute_query(q1)
    print("Final SQL query: " + q1)
    print("Final execution result: " + result)
    return q1
