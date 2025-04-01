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

g_qwen_sql = ""

g_schemalink = ""
g_goals = ""
g_num_fix_attempts = 0
g_db_name = ""
g_context = ""

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
Settings.text_splitter = SentenceSplitter(chunk_size=128, chunk_overlap=10)


class Message(BaseModel):
    question: Optional[str] = None
    dbschema: Optional[str] = None
    content: Optional[str] = None

# Constructor Output
class SQLOutput(BaseModel):
    sql: str

class ClassificationOutput(BaseModel):
    class Step(BaseModel):
        step: str
    explanation: list[Step]
    easy: bool

g_options={
    "num_ctx": 16384,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    "temperature": 0.0,
}

g_analyst_options={
    "num_ctx": 16384,
    "frequency_penalty": 0.0,
    "temperature": 0.0,
}

classifier_client = OllamaChatCompletionClient(
    model=gemma3_12b_config["model"],
    host=gemma3_12b_config["base_url"],
    model_info={
        "vision": False,
        "function_calling": False,
        "json_output": True,
        "family": ModelFamily.UNKNOWN,
        "structured_output": True,
    },
    options=g_analyst_options,
    response_format=ClassificationOutput,
    max_tokens=600,
)

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
    max_tokens=600,
)

easy_query_builder_client = OllamaChatCompletionClient(
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

hard_query_builder_client = OllamaChatCompletionClient(
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
    options=g_options,
    response_format=SQLOutput,
    max_tokens=300,
)

ANALYST_PROMPT = """
###Task:
Extract the `user's goals` and `desired output` from the given natural language \
    question input.

Explain what data user asks explicitly to display and which calculations are needed \
    to produce that output.

###Natural Language Question
{question}

Important Note: Do not return SQL.
"""


SCHEMA_LINKER_PROMPT = """
###Task
State explicitly the `output columns` based on a `user goals` and a `database schema`.

State explicitly whether any tables need to be joined. If so, specify explicitly \
    which tables must be joined and list the exact join conditions.

State explicitly the columns and tables needed for calculations if any.

###Schema
Database schema - <DDL Statements>
`{ddl_statements}`

###User Goals & Intents
`{goals}`

Important Note: Do not return SQL.
"""


QUERY_BUILDER_PROMPT = """
### TASK
Generate an SQL query `based on the schema linker response and user goals`.

Your query must strictly return only the `desired output` as explicitly stated.

**Always follow the exact column and table matches given by the schema linker response.**

### USER GOALS
`{goals}`

### SCHEMA LINKER RESPONSE
`{schemalink}`

### DATABASE SCHEMA
{schema}

** Follow `SQLite syntax` strictly. **

### CONTEXTUAL DATA (for reference)
The following is a portion of real data that may help with understanding column values or \
    generating conditions:
{data}
"""


REFINER_PROMPT = """
###Task
Only fix syntactic issues—do not alter variable names, logic, structure, or optimize anything \
    it's strictly necessary for fixing a syntax problem.

###Database Schema
{schema}

#Query
{query}

#Error
{error}
"""


CLASSIFIER_SYSTEM_PROMPT="""You are a SQL query classifier that decides whether user goals \
    will result in an simple SQL query based on the user goals and schema linking."""
@type_subscription(topic_type="classifier")
class ClassifierAgent(RoutedAgent):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        self._classifier = AssistantAgent(name, model_client=classifier_client,
                                          system_message=CLASSIFIER_SYSTEM_PROMPT)

    @message_handler
    async def on_message(self, message: Message, ctx: MessageContext) -> None:
        print(f"{self.id.type} received message: {message.question}")
        
        time.sleep(1)
        global g_schemalink, g_goals
        response = await self._classifier.on_messages(
            [TextMessage(content=f"Mark easy as true if user goals and schema linking will result in a simple query. \
                Mark false, if they will result in complex query - a query including at least 1 JOINs -. \
                \n###USER GOALS\n{g_goals} \
                \n\n###SCHEMA LINKER RESPONSE\n{g_schemalink}. ", source="user")],
            ctx.cancellation_token
        )
        print(f"{self.id.type} responded: {response.chat_message.content}")
        easy_marker = json.loads(response.chat_message.content)
        if easy_marker["easy"]:
            await self.publish_message(message, topic_id=TopicId(type="qwenhardquerybuilder",
                                                                 source="classifier"))
        else:
            await self.publish_message(message, topic_id=TopicId(type="qwenhardquerybuilder",
                                       source="classifier"))



ANALYST_SYSTEM_PROMPT="""You are an intelligent assistant that extracts the user's goals & \
    desired outcome from a natural language question about a database."""
@type_subscription(topic_type="analyst")
class AnalystAgent(RoutedAgent):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        self._delegate = AssistantAgent(name, model_client=data_scientist_client,
             system_message=ANALYST_SYSTEM_PROMPT)

    @message_handler
    async def on_message(self, message: Message, ctx: MessageContext) -> None:
        print(f"{self.id.type} received message: {message.question}")
        time.sleep(1)
        response = await self._delegate.on_messages(
            [TextMessage(content=ANALYST_PROMPT.format(question=message.question),
                                                       source="user")], ctx.cancellation_token
        )
        print(f"{self.id.type} responded: {response.chat_message.content}")
        global g_goals
        g_goals = response.chat_message.content
        await self.publish_message(message, topic_id=TopicId(type="schemalinker", source="analyst"))



SCHEMA_LINKER_SYSTEM_PROMPT="""You are a database-aware assistant and you are tasked with \
    linking natural language terms in a well-explained user goals to the actual tables and \
    columns in a database schema."""
@type_subscription(topic_type="schemalinker")
class SchemaLinkerAgent(RoutedAgent):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        self._delegate = AssistantAgent(name, model_client=data_scientist_client,
             system_message=SCHEMA_LINKER_SYSTEM_PROMPT)

    @message_handler
    async def on_message(self, message: Message, ctx: MessageContext) -> None:
        print(f"{self.id.type} received message: {message.question}")
        time.sleep(1)
        global g_schemalink, g_goals
        response = await self._delegate.on_messages([TextMessage(
                content=SCHEMA_LINKER_PROMPT.format(goals=g_goals, ddl_statements=message.dbschema), source="user")],
                ctx.cancellation_token)
        print(f"{self.id.type} responded: {response.chat_message.content}")
        g_schemalink = g_schemalink + "\n" + response.chat_message.content
        await self.publish_message(message, topic_id=TopicId(type="qwenquerybuilder", source="schemalinker"))



BUILDER_SYSTEM_PROMPT = """You are a SQL builder.\
    Use only the information provided when building queries. Do not add any assumptions or \
    external knowledge."""
@type_subscription(topic_type="qwenquerybuilder")
class QwenQueryBuilderAgent(RoutedAgent):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        self._builder = AssistantAgent(name, model_client=hard_query_builder_client,
             system_message=BUILDER_SYSTEM_PROMPT)

    @message_handler
    async def on_message(self, message: Message, ctx: MessageContext) -> None:
        print(f"{self.id.type} received message: {message.question}")

        global g_goals, g_schemalink, g_qwen_sql, g_context
        time.sleep(3)
        builder_prompt=QUERY_BUILDER_PROMPT.format(goals=g_goals, schemalink=g_schemalink,
                                                   schema=message.dbschema, data=g_context)
        response = await self._builder.on_messages([TextMessage(content=builder_prompt, source="user")],
                                                     ctx.cancellation_token) 
        print(f"{self.id.type} responded: {response.chat_message.content}")

        g_qwen_sql = clean_sql(response.chat_message.content)
        await self.publish_message(message, topic_id=TopicId(type="refiner", source="qwenquerybuilder"))



REFINER_SYSTEM_PROMPT = """You are tasked with correcting SQL queries based on error message \
    and database schema."""
@type_subscription(topic_type="refiner")
class RefinerAgent(RoutedAgent):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        self._delegate = AssistantAgent(name, model_client=refiner_client,
             system_message=REFINER_SYSTEM_PROMPT)

    @message_handler
    async def on_message(self, message: Message, ctx: MessageContext) -> None:
        
        global g_num_fix_attempts, g_qwen_sql
        result = await execute_query(g_qwen_sql)
        if "failure" in result and g_num_fix_attempts != 3:
            time.sleep(1)
            print(f"{self.id.type} received message: {message.question}")
            g_num_fix_attempts = g_num_fix_attempts + 1
            message.content=REFINER_PROMPT.format(schema=message.dbschema, query=g_qwen_sql, error=result)
            response = await self._delegate.on_messages([TextMessage(content=message.content, source="user")],
                                                         ctx.cancellation_token)
            print(f"{self.id.type} responded: {response.chat_message.content}")
            g_qwen_sql = clean_sql(response.chat_message.content)
            await self.publish_message(message, topic_id=TopicId(type="refiner", source="refiner"))



async def generate_query(question: str, schema: str, db_name: str) -> str:
    global g_db_name, g_qwen_sql, g_num_fix_attempts, g_context
    g_db_name = db_name
    g_qwen_sql = ""
    g_num_fix_attempts = 0

    try:
        documents = SimpleDirectoryReader(input_dir=f"./test_database/{g_db_name}/").load_data()
        vector_index = VectorStoreIndex.from_documents(documents)

        retriever = vector_index.as_retriever(similarity_top_k=5)
        response = retriever.retrieve(f"""{question}""")
        if len(response) != 0:
            g_context = "\n".join(r.text for r in response)
            print(f"{g_context} and document size: {len(response)}")
    except Exception as e:
        print(f"Error occured: {e}")

    runtime = SingleThreadedAgentRuntime()
    await AnalystAgent.register(runtime, "analyst_agent", lambda: AnalystAgent("analyst"))
    await SchemaLinkerAgent.register(runtime, "schema_linker", lambda: SchemaLinkerAgent("schemalinker"))
    await QwenQueryBuilderAgent.register(runtime, "sql_builder", lambda: QwenQueryBuilderAgent("qwenquerybuilder"))
    await RefinerAgent.register(runtime, "refiner_agent", lambda: RefinerAgent("refiner"))
    runtime.start()
    await runtime.publish_message(
        Message(dbschema=schema, question=question), topic_id=TopicId(type="analyst", source="default"))
    await runtime.stop_when_idle()


    print("Final SQL query: " + g_qwen_sql)
    result = await execute_query(g_qwen_sql)
    print("Final execution result: " + result)
    return g_qwen_sql
