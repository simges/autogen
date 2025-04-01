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


mistral_nemo_instruct = {
    "model": "mistral-nemo:latest",
    "base_url": "http://127.0.0.1:11434",
    "api_key":"placeholder",
}

qwen2_5_config = {
    "model": "qwen2.5:14b",
    "base_url": "http://127.0.0.1:11434",
    "api_key":"placeholder",
}

gemma3_12b_config = {
    "model": "gemma3-it-12b:latest",
    "base_url": "http://127.0.0.1:11434",
    "api_key":"placeholder",
}


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

g_db_name = ""
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
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexAutoRetriever
from llama_index.core.schema import Document
from llama_index.core.vector_stores import MetadataInfo, VectorStoreInfo
from pathlib import Path
import shutil, os, json
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.llms.openai import OpenAI

SENTENCE_TRANSFORMERS_HOME="/home/simges/.cache/huggingface/hub/sentence-transformers/{model_name}"

MINILM_EMBEDDING_MODEL_PATH=SENTENCE_TRANSFORMERS_HOME.format(model_name="/all-MiniLM-L6-v2")

os.environ["TIKTOKEN_CACHE_DIR"]="/home/simges/cl100k_base/"

Settings.embed_model = HuggingFaceEmbedding(model_name=MINILM_EMBEDDING_MODEL_PATH)


class Message(BaseModel):
    question: Optional[str] = None
    dbschema: Optional[str] = None
    content: Optional[str] = None

# Constructor Output
class SQLOutput(BaseModel):
    sql: str

# Constructor Output
class CorrectedSQLOutput(BaseModel):
    class Step(BaseModel):
        step: str
    explanation: list[Step]
    sql: str

g_options = {
    "num_ctx": 16384,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    "temperature": 0.0,
}

g_refiner_options = {
    "num_ctx": 16384,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    "repetition_penalty": 1.8,
    "temperature": 0.0,
}

g_analyst_options = {
    "num_ctx": 16384,
    "frequency_penalty": 0.0,
    "temperature": 0.0,
}

data_scientist_client = OllamaChatCompletionClient(
    model=gemma3_12b_config["model"],
    host=gemma3_12b_config["base_url"],
    model_info={
        "vision": False,
        "function_calling": False,
        "json_output": False,
        "family": ModelFamily.UNKNOWN,
        "structured_output": False,
    },
    options=g_analyst_options,
    max_tokens=300,
)

query_builder_client = OllamaChatCompletionClient(
    model=qwen2_5_config["model"],
    host=qwen2_5_config["base_url"],
    model_info={
        "vision": False,
        "function_calling": False,
        "json_output": True,
        "family": ModelFamily.UNKNOWN,
        "structured_output": True,
    },
    options=g_options,
    response_format=SQLOutput,
    max_tokens=300,
)


query_builder_client2 = OllamaChatCompletionClient(
    model=gemma3_12b_config["model"],
    host=gemma3_12b_config["base_url"],
    model_info={
        "vision": False,
        "function_calling": False,
        "json_output": True,
        "family": ModelFamily.UNKNOWN,
        "structured_output": True,
    },
    options=g_options,
    response_format=SQLOutput,
    max_tokens=300,
)

refiner_client = OllamaChatCompletionClient(
    model=mistral_nemo_instruct["model"],
    host=mistral_nemo_instruct["base_url"],
    model_info={
        "vision": False,
        "function_calling": False,
        "json_output": True,
        "family": ModelFamily.UNKNOWN,
        "structured_output": True,
    },
    options=g_refiner_options,
    response_format=CorrectedSQLOutput,
    max_tokens=500,
)

ANALYST_PROMPT = """
Analyse user question and state user goals and output data which user wants to obtain.

Reason about the conditions and filtering, grouping and ordering.

State if the target data must be a single-row or multi-rows.

State string literals to be used in condition checks.

Do not give more information apart from what asked above.

Explain explicitly your reasoning step by step.

###Natural Language Question
{question}

###Database Schema

{database_schema}

###Helpful Data (use string values exactly as they appear in the reference data)

{reference_data}

Do not return SQL.
"""


SCHEMA_LINKER_PROMPT = """
Given a natural language question and a database schema (list of tables and columns), your task is to identify the corresponding list of 'exact tables and columns'.

Explain explicitly your reasoning.

#Input

###Database Schema

{database_schema}

###User Goals and Intention

{goals}

###Rules & Guidelines
Always use SQLite dialect.

** Only link to column or table names that exactly match the provided schema. **

"""


QUERY_BUILDER_PROMPT = """
You are tasked with building an SQL query based on the inputs given:

#Input
You are given:

#Natural Language Question

{question}

#Comments on User Goals

{goals}

###Schema Linker Output

{schemalink}


###Reference Database Schema

{database_schema}

###Your Task
**Do exactly what is decribed in `User Goals` and `Schema Linker Output` to construct the SQL query.**

###Rules & Guidelines
Always use SQLite dialect.

DO NOT PERFORM CAST.

"""


REFINER_PROMPT = """
###Task
Fix execution error based on `database schema columns` and `return corrected query`, **do not return the same query**.

USE SQLITE SYNTAX ONLY.

###Database Schema

{schema}

#Query

{query}

#Error

`{error}`

#Stick to the User Goals when fixing the query execution error

`{goals}`
"""


g_reference_data = "" 
g_schemalink = ""
g_final_sql = ""
g_goals = ""
g_num_fix_attempts = 0
async def generate_query(question: str, schema: str, db_name: str) -> str:    
    global g_db_name, g_schemalink, g_reference_data, g_goals, g_num_fix_attempts, g_final_sql

    g_db_name = db_name
    print("g_db_name: " + g_db_name)
    print("schema : " + schema)
    g_final_sql = ""    
    g_schemalink = ""
    g_goals = ""
    g_reference_data = ""
    g_num_fix_attempts = 0

    
    @type_subscription(topic_type="analyst")
    class AnalystAgent(RoutedAgent):
        def __init__(self, name: str) -> None:
            super().__init__(name)
            self._delegate = AssistantAgent(name, model_client=data_scientist_client,
                system_message="You are an assistant that extracts the user's goals and target data from a natural language question about a database.")

        @message_handler
        async def on_message(self, message: Message, ctx: MessageContext) -> None:
            print(f"{self.id.type} received message: {message.question}")
            
            global g_reference_data, g_goals
            response = await self._delegate.on_messages(
                [TextMessage(content=ANALYST_PROMPT.format(question=message.question,
                        database_schema=message.dbschema, reference_data=g_reference_data),
                source="user")], ctx.cancellation_token
            )
            print(f"{self.id.type} responded: {response.chat_message.content}")
            g_goals = "\n#USER GOALS\n" + response.chat_message.content

            await self.publish_message(message, topic_id=TopicId(type="schemalinker", source="analyst"))



    SCHEMA_LINKER_SYSTEM_PROMPT="""You are a database-aware assistant you are tasked with linking \
        given user goals and question to the actual tables and columns in a database schema. \
        Explain your reasoning briefly. Do not return SQL."""
    @type_subscription(topic_type="schemalinker")
    class SchemaLinkerAgent(RoutedAgent):
        def __init__(self, name: str) -> None:
            super().__init__(name)
            self._delegate = AssistantAgent(name, model_client=data_scientist_client,
                system_message=SCHEMA_LINKER_SYSTEM_PROMPT)

        @message_handler
        async def on_message(self, message: Message, ctx: MessageContext) -> None:
            print(f"{self.id.type} received message: {message.question}")
            time.sleep(5)
            global g_goals, g_schemalink
            response = await self._delegate.on_messages([TextMessage(
                    content=SCHEMA_LINKER_PROMPT.format(goals=g_goals, database_schema=message.dbschema), source="user")],
                    ctx.cancellation_token)
            print(f"{self.id.type} responded: {response.chat_message.content}")
            g_schemalink = "\n#SCHEMA LINKING\n" + response.chat_message.content

            await self.publish_message(message, topic_id=TopicId(type="qwenquerybuilder", source="schemalinker"))


    BUILDER_SYSTEM_PROMPT = """You are a SQL builder. Build SQL query utilizing the information provided to you."""  
    @type_subscription(topic_type="qwenquerybuilder")
    class QwenQueryBuilderAgent(RoutedAgent):
        def __init__(self, name: str) -> None:
            super().__init__(name)
            self._builder = AssistantAgent(name, model_client=query_builder_client,
                system_message=BUILDER_SYSTEM_PROMPT)

        @message_handler
        async def on_message(self, message: Message, ctx: MessageContext) -> None:
            print(f"{self.id.type} received message: {message.question}")

            time.sleep(1)
            global g_schemalink, g_goals
            builder_prompt=QUERY_BUILDER_PROMPT.format(question=message.question, 
                        goals=g_goals, schemalink=g_schemalink, database_schema=message.dbschema)
            response = await self._builder.on_messages([TextMessage(content=builder_prompt, source="user")],
                                                    ctx.cancellation_token) 
            print(f"{self.id.type} responded: {response.chat_message.content}")
            global g_final_sql
            g_final_sql = clean_sql(response.chat_message.content)
            await self.publish_message(message, topic_id=TopicId(type="refiner", source="qwenquerybuilder"))


    @type_subscription(topic_type="gemmaquerybuilder")
    class GemmaQueryBuilderAgent(RoutedAgent):
        def __init__(self, name: str) -> None:
            super().__init__(name)
            self._builder = AssistantAgent(name, model_client=query_builder_client2,
                system_message=BUILDER_SYSTEM_PROMPT)

        @message_handler
        async def on_message(self, message: Message, ctx: MessageContext) -> None:
            print(f"{self.id.type} received message: {message.question}")

            time.sleep(1)
            global g_schemalink, g_goals, g_num_fix_attempts
            g_num_fix_attempts = g_num_fix_attempts + 1
            builder_prompt=QUERY_BUILDER_PROMPT.format(question=message.dbschema, 
                        goals=g_goals, schemalink=g_schemalink, database_schema=message.dbschema)
            response = await self._builder.on_messages([TextMessage(content=builder_prompt, source="user")],
                                                        ctx.cancellation_token) 
            print(f"{self.id.type} responded: {response.chat_message.content}")
            global g_final_sql
            g_final_sql = clean_sql(response.chat_message.content)
            await self.publish_message(message, topic_id=TopicId(type="refiner", source="gemmaquerybuilder"))


    REFINER_SYSTEM_PROMPT = """You are tasked with correcting execution errors of SQL queries based on error message \
        and database schema. Check the query against database and fix other incorrect column names. \
            Explain explicitly how you fixed the error."""
    @type_subscription(topic_type="refiner")
    class RefinerAgent(RoutedAgent):
        def __init__(self, name: str) -> None:
            super().__init__(name)
            self._delegate = AssistantAgent(name, model_client=refiner_client,
                system_message=REFINER_SYSTEM_PROMPT)

        @message_handler
        async def on_message(self, message: Message, ctx: MessageContext) -> None:
            global g_num_fix_attempts, g_final_sql
            passed = False
            while g_num_fix_attempts < 5:
                result = await execute_query(g_final_sql)
                print(f"{self.id.type} execution result: {result}")
                if "passed" in result: 
                    passed = True
                    break
                if g_num_fix_attempts == 2: break

                time.sleep(2)
                print(f"{self.id.type} received message: {message.question}")
                g_num_fix_attempts = g_num_fix_attempts + 1
                message.content = REFINER_PROMPT.format(schema=message.dbschema, query=g_final_sql,
                        error=result, goals=g_goals)
                response = await self._delegate.on_messages([TextMessage(content=message.content, source="user")],
                                                            ctx.cancellation_token)
                print(f"{self.id.type} responded: {response.chat_message.content}")
                g_final_sql = clean_sql(response.chat_message.content)
            
            if g_num_fix_attempts == 5: 
                print(f"{self.id.type} execution result: {result} too much fix attempt.")
                return
            if passed == True:
                print(f"{self.id.type} execution result: {result} FIXED.")
                return
            await self.publish_message(message, topic_id=TopicId(type="gemmaquerybuilder", source="refiner"))


    Settings.text_splitter = SentenceSplitter(chunk_size=512, chunk_overlap=20)
    documents = SimpleDirectoryReader(f"./test_database/{g_db_name}/").load_data()
    vector_index = VectorStoreIndex.from_documents(documents)
    retriever = vector_index.as_retriever(similarity_top_k=3)

    response = retriever.retrieve(f"""{question}""")
    if len(response) != 0:
        g_reference_data = response[0].text
        print(f"Retriever responded: {g_reference_data}")

    time.sleep(1)
    runtime = SingleThreadedAgentRuntime()
    await AnalystAgent.register(runtime, "analyst_agent", lambda: AnalystAgent("analyst"))
    await SchemaLinkerAgent.register(runtime, "schema_linker", lambda: SchemaLinkerAgent("schemalinker"))
    await GemmaQueryBuilderAgent.register(runtime, "gemma_sql_builder", lambda: GemmaQueryBuilderAgent("gemmaquerybuilder"))
    await QwenQueryBuilderAgent.register(runtime, "qwen_sql_builder", lambda: QwenQueryBuilderAgent("qwenquerybuilder"))
    await RefinerAgent.register(runtime, "refiner_agent", lambda: RefinerAgent("refiner"))
    runtime.start()
    await runtime.publish_message(
        Message(dbschema=schema, question=question), topic_id=TopicId(type="analyst", source="default"))
    await runtime.stop_when_idle()


    print("Final SQL query: " + g_final_sql)
    result = await execute_query(g_final_sql)
    print("Final execution result: " + result)
    return g_final_sql
