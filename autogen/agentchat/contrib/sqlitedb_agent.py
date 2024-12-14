import sys
sys.path.append('/home/simges/autogen/autogen/')

import json
from autogen.agentchat import ConversableAgent, UserProxyAgent, Agent
from autogen.agentchat.contrib.sqlexecutor_agent import SQLExecutorAgent
from typing import Any, Callable, Dict, List, Literal, Optional, Union, Tuple
from spider_env import SpiderEnv

class SQLiteDBAgent(ConversableAgent):

    PROMPT =  """ 
        ### Instructions:
        Your task is to convert a question into a SQL query, given a Postgres database schema.
        Adhere to these rules:
        - **Deliberately go through the question and database schema word by word** to appropriately answer the question
        - **Use Table Aliases** to prevent ambiguity. For example, `SELECT table1.col1, table2.col1 FROM table1 JOIN table2 ON table1.id = table2.id`.
        - When creating a ratio, always cast the numerator as float

        ### Input:
        Generate a SQL query that answers the question `{user_question}`.
        This query will run on a database whose schema is represented in this string:
        {table_metadata_string}
        ### Response:
        Based on your instructions, here is the SQL query I have generated to answer the question `{user_question}`:
        ```sql
        """

    # Default UserProxyAgent.description values, based on human_input_mode
    DEFAULT_SQLCODER_DESCRIPTION = "SQL Coder is an agent which is suppsed to generate sql queries based on the required task."

    DEFAULT_SQLCODER_LLM_CONFIG = {
        "temperature": 0.0,
        "cache_seed": None,
        "config_list": [
            {
                "model": "sqlcoder:7b",
                "api_key": "KEY-OF-NO-IMPORTANCE",
                "base_url": "http://localhost:11434/v1",
            }
        ],
        "frequency_penalty": 0.9,
    }  

    def __init__(
        self,
        trigger: Agent = None,
        gym: SpiderEnv = None,
        max_consecutive_auto_reply: Optional[int] = None,
        function_map: Optional[Dict[str, Callable]] = None,
        code_execution_config: Union[Dict, Literal[False]] = False,
        default_auto_reply: Union[str, Dict] = "",
        description: Optional[str] = None,
    ):
        super().__init__(
            name = "sqlcoder-agent",
            system_message = "",
            is_termination_msg = None,
            max_consecutive_auto_reply=max_consecutive_auto_reply,
            human_input_mode = "NEVER",
            function_map=None,
            code_execution_config=code_execution_config,
            llm_config = self.DEFAULT_SQLCODER_LLM_CONFIG,
            default_auto_reply=default_auto_reply,
            description=(description if description is not None else self.DEFAULT_SQLCODER_DESCRIPTION),
        )
        self._question = ""
        self._gym = gym
        self._trigger = None

        if trigger is not None:
            self._trigger = trigger

        self.register_reply(self._trigger, SQLiteDBAgent.generate_query, remove_other_reply_funcs=True)        
        self._sqlexecutor = SQLExecutorAgent(self, gym=gym)

    @staticmethod
    def message_generator(sender, recipient, context):
        """Generate a prompt for the assistant agent with the given problem and prompt.

        Args:
            context (dict): a dictionary with the following fields:
                question (str): the problem to be solved.
                schema (str, Optional): 
        Returns:
            str: the generated prompt ready to be sent to the assistant agent.
        """
        sql_prompt = SQLiteDBAgent.PROMPT.format(
            user_question = context.get("question"),
            table_metadata_string = context.get("schema"))
        return sql_prompt

    def generate_query(
        self,
        messages: Optional[List[Dict[str, str]]] = None,
        sender: Optional[Agent] = None,
        config: Optional[Any] = None,
    ) -> Tuple[bool, Optional[Union[str, Dict[str, str]]]]:
        content = json.loads(messages[-1].get("content"))
    
        msg2send = self.message_generator(sender=sender, recipient=self, context=content)
        self._db_id = content.get("db_id")
        self._question = content.get("question")

        messages[-1]["content"] = msg2send
        replied, sql = self.generate_oai_reply(messages=[messages[-1]], sender=None, config=config)
        if replied == True:
            import re
            sql = re.split(r";", sql, maxsplit=1)[0] + ";"
            sql = str(sql)

            print("SEND THIS QUERY TO EXECUTOR: " + sql)
            prompt = { 
                "role": "user",
                "content": "Call execute-sql tool with the parameter \"sql\"= \"" + str(sql) + "\""
            }
            self.send(prompt, self._sqlexecutor, request_reply=True, silent=False)                            
            response = { 
                "role": "user",
                "content": "Pick another question." 
            }
            return (True, response)