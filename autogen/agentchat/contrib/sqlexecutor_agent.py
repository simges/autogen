import sys
sys.path.append('/home/simges/autogen/autogen/')

import json
from autogen.runtime_logging import logging_enabled, log_new_agent
from autogen.agentchat import UserProxyAgent, AssistantAgent, ConversableAgent, Agent
from typing import Any, Dict, List, Optional, Annotated, Union, Tuple
from autogen.agentchat.contrib.DINSQL import GPT4_generation, schema_linking_prompt_maker, classification_prompt_maker, easy_prompt_maker, medium_prompt_maker, hard_prompt_maker
from spider_env import SpiderEnv

from autogen.code_utils import (
    content_str,
)

class SQLExecutorAgent(AssistantAgent):
    # Default UserProxyAgent.description values, based on human_input_mode
    DEFAULT_SQLEXECUTOR_DESCRIPTION = "SQL Coder is an agent which is suppsed to execute sql queries."

    DEFAULT_SQLEXECUTOR_LLM_CONFIG = {
        "temperature": 0.0,
        "top_p": 0.9,
        "n": 1,
        "config_list": [
            {
                "model": "mistral-7b-tool-calling-tuned",
                "api_key": "KEY-OF-NO-IMPORTANCE",
                "base_url": "http://localhost:11434/v1/",
            }
        ]
    }

    def __init__(
        self,
        trigger: Agent = None,
        gym: SpiderEnv = None,
        max_consecutive_auto_reply: Optional[int] = None,
        description: Optional[str] = None,
    ):
        super().__init__(
            name="assistant",
            system_message="You are a helpful AI assistant." +
                "Firstly, you must check whether the `sql` is provided to you or not. If provided, call `execute-sql` tool and return the result, " +
                "Secondly, you must check whether `db_id` and `question` are provided to you and they are NOT null." +
                "If `db_id` and `question` values are provided to you, then call `correct-sql` tool and return the result." +
                "You can call either `execute-sql` or `correct-sql` but DO NOT CALL BOTH TOOLS AT THE SAME TIME!",
            llm_config = self.DEFAULT_SQLEXECUTOR_LLM_CONFIG,
            is_termination_msg=lambda m: False,
            max_consecutive_auto_reply=max_consecutive_auto_reply,
            description=(description if description is not None else self.DEFAULT_SQLEXECUTOR_DESCRIPTION),
        )
        self.register_reply(trigger, SQLExecutorAgent.generate_oai_reply, remove_other_reply_funcs=True)

        self._user_proxy = UserProxyAgent(
            self.name + "_inner_user_proxy",
            human_input_mode="NEVER",
            code_execution_config=False,
            default_auto_reply="",
            is_termination_msg=lambda m: False,
        )
        self._user_proxy.register_reply(self, ConversableAgent.generate_tool_calls_reply, remove_other_reply_funcs=True)

        @self._user_proxy.register_for_execution()
        @self.register_for_llm(
            name="execute-sql",
            description="Perform an SQL query execution then return the query results when the SQL query is given.",
            api_style = "tool")
        def _execute_sql(
            sql: Annotated[str, "SQL query"]
        ) -> Annotated[Dict[str, str], "Dictionary with keys 'result' and 'error'"]:
            observation, reward, _, _, info = gym.step(sql)
            error = observation["feedback"]["error"]

            if not error and reward == 0:
                error = "The SQL query returned an incorrect result"
            if error:
                return {
                    "error": error,
                    "wrong_result": observation["feedback"]["result"],
                    "correct_result": info["gold_result"],
                }
            else:
                return {
                    "result": observation["feedback"]["result"],
                    }
            
        @self._user_proxy.register_for_execution()
        @self.register_for_llm(
            name="correct-sql",
            description="Correct an SQL query then return the corrected SQL query results when the \"db_id\" and the \"question\" is provided.",
            api_style = "tool")
        def _run_dinsql(
            db_id: Annotated[str, "SQL query"],
            question: Annotated[str, "SQL query"]
        ) -> Annotated[str, "Dictionary with keys 'result' and 'error'"]:
            if db_id == "" or question == "": return ""
            import time, logging, traceback
            schema_links = None
            while schema_links is None:
                try:
                    schema_links = GPT4_generation(
                        schema_linking_prompt_maker(question, db_id))
                except Exception as e:
                    logging.error("An error occurred: %s", e)
                    time.sleep(3)
                    # Print the full backtrace
                    traceback.print_exc()
                    pass
            try:
                schema_links = schema_links.split("Schema_links: ")[1]
            except:
                print("Slicing error for the schema_linking module")
                schema_links = ""
                # Print the full backtrace
                traceback.print_exc()

            classification = None
            while classification is None:
                try:
                    if schema_links is None:
                        classification = GPT4_generation(
                            classification_prompt_maker(question, db_id, ""))
                    
                    else:
                        classification = GPT4_generation(
                            classification_prompt_maker(question, db_id, schema_links[1:]))
                except:
                    time.sleep(3)
                    # Print the full backtrace
                    traceback.print_exc()
                    pass
            try:
                predicted_class = classification.split("Label: ")[1]
            except:
                print("Slicing error for the classification module " + str(classification))
                predicted_class = '"NESTED"'
                # Print the full backtrace
                traceback.print_exc()

            if '"EASY"' in predicted_class:
                print("EASY")
                SQL = None
                while SQL is None:
                    try:
                        SQL = GPT4_generation(easy_prompt_maker(question, db_id, schema_links))
                    except:
                        time.sleep(3)
                        pass
            elif '"NON-NESTED"' in predicted_class:
                print("NON-NESTED")
                SQL = None
                while SQL is None:
                    try:
                        SQL = GPT4_generation(medium_prompt_maker(question, db_id, schema_links))
                    except Exception as e:
                        print("THIS QUERY TO EXECUTOR LOOP BACK : " + " ERRORRR: " + str(e))
                        time.sleep(3)
                        pass
                try:
                    print(SQL)
                except Exception as e:
                    print("SQL slicing error: " + str(e) + " queries: " + str(SQL))
                    SQL = "SELECT"
                    # Print the full backtrace
                    traceback.print_exc()
            else:
                sub_questions = classification.split('questions = ["')[1].split('"]')[0]
                print("NESTED")
                SQL = None
                while SQL is None:
                    try:
                        SQL = GPT4_generation(
                            hard_prompt_maker(question, db_id, schema_links, sub_questions))
                    except Exception as e:
                        print("Error: " + str(e))
                        time.sleep(3)
                        pass
                try:
                    print(SQL)
                except Exception as e:
                    print("SQL slicing error: " + str(e) + " queries: " + str(SQL))
                    SQL = "SELECT"
                    # Print the full backtrace
                    traceback.print_exc()
            print(SQL)
            return SQL

    def generate_oai_reply(
        self,
        messages: Optional[List[Dict]] = None,
        sender: Optional[Agent] = None,
        config: Optional[Any] = None,
    ) -> Tuple[bool, Union[str, Dict, None]]:

        client = self.client if config is None else config
        if client is None:
            return False, None
        if messages is None:
            messages = self._oai_messages[sender]

        response = self._generate_oai_reply_from_client(
            client, self._oai_system_message + messages, self.client_cache
        )

        import re
        ret = None
        match = re.search(r'\[TOOL_CALLS\] (.+)', str(response))
        if response is None:
            return (False, None)
        else:
            if "tool_calls" in response:
                self.send(response, self._user_proxy, request_reply=True, silent=False)
                ret = self.chat_messages[self._user_proxy][-1].get("content")
            else:
                ret = response
        if ret is not None: return (True, ret)
        else: return (False, "")