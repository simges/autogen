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
import asyncio

data_scientist_config = {
    "model": "qwen2.5:14b",
    "base_url": "http://127.0.0.1:11434",
    "api_key":"placeholder",
}

g_final_sql = ""
g_db_name = ""

def clean_sql(final_sql):
    match = re.search(r"```sql\s*(.*?)\s*```", final_sql, re.DOTALL)
    if match:
        final_sql = match.group(1).strip()
    final_sql = final_sql.split(";", 1)[0].strip() + ";"

    cleaned = final_sql.replace("\\n", " ")          # Remove newlines
    cleaned = cleaned.replace('\\"', '"')         # Replace escaped quotes
    cleaned = cleaned.replace('\\', '')           # Remove remaining backslashes
    cleaned = ' '.join(cleaned.split())           # Remove extra spaces
    return cleaned

### 1.1 Create Tools
# a. SQL executor
async def execute_query() -> Annotated[str, "query results"]:
    import sqlite3

    global g_db_name, g_final_sql
    db_url = f"/home/simges/.cache/spider_data/test_database/{g_db_name}/{g_db_name}.sqlite"
    conn = sqlite3.connect(db_url)
    cursor = conn.cursor()
    try:
        cursor.execute(g_final_sql)
    except Exception as e:
        return "failure: " + str(e)
    return "passed"
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
        "json_output": False,
        "family": ModelFamily.UNKNOWN,
        "structured_output": False,
    },
    options={
        "num_ctx": 16384,
        "frequency_penalty": 0.0,
        "temperature": 0.0,
    },
    max_tokens=600,
)


DATA_SCIENTIST_PROMPT = """
### Complete sqlite SQL query only and with no
explanation. 

### Sqlite database schema: 

{database_schema}

### {question}

"""


@type_subscription(topic_type="datascientist")
class DataScientist(RoutedAgent):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        self._delegate = AssistantAgent(name, model_client=data_scientist_client,
            system_message="You are a highly skilled AI assistant that translates natural language questions into correct and executable SQLite queries.")

    @message_handler
    async def on_message(self, message: Message, ctx: MessageContext) -> str:
        print(f"{self.id.type} received message: {message.content}")

        await asyncio.sleep(1.0)
        response = await self._delegate.on_messages(
            [TextMessage(content=message.content, source="user")], ctx.cancellation_token
        )
        print(f"{self.id.type} responded: {response.chat_message.content}")
        exit
        global g_final_sql
        g_final_sql = response.chat_message.content

        # Single-Agent Mode
        return g_final_sql


async def generate_query(question: str, schema: str, db_name: str) -> str:
    global g_db_name
    g_db_name = db_name
    runtime = SingleThreadedAgentRuntime()
    await DataScientist.register(runtime, "data_scientist", lambda: DataScientist("datascientist"))

    time.sleep(2.0)
    runtime.start()
    # Single-Agent chatdb/natural-sql-7b Mode
    await runtime.send_message(
        Message(DATA_SCIENTIST_PROMPT.format(question=question, database_schema=schema)),
        recipient=AgentId(type="data_scientist", key="default"))
    await runtime.stop_when_idle()

    global g_final_sql
    g_final_sql = clean_sql(g_final_sql)
    print("Stripped SQL query: " + g_final_sql)

    result = await execute_query()
    print("Final execution result: " + result)
    return g_final_sql