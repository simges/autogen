import sys
sys.path.append('/home/simges/autogen/autogen/')

import json
from autogen.runtime_logging import logging_enabled, log_new_agent
from autogen.agentchat import ConversableAgent, UserProxyAgent, AssistantAgent, Agent
from typing import Any, Callable, Dict, List, Literal, Optional, Annotated, Union, Tuple
from spider_env import SpiderEnv

from autogen.code_utils import (
    content_str,
)

class DBUserAgent(AssistantAgent):
    # Default UserProxyAgent.description values, based on human_input_mode
    DEFAULT_DBUSER_DESCRIPTION = "DB User is an agent which is supposed to immitate an actual db user who has questions."
    DEFAULT_CONFIG = {
        "temperature": 0.0,
        "top_p": 0.9,
        "n": 1,
        "cache_seed": None,
        "config_list": [
            {
                "model": "mistral-7b-tool-calling-tuned",
                "api_key": "KEY-OF-NO-IMPORTANCE",
                "base_url": "http://localhost:11434/v1",
            }
        ]
    }
    def __init__(
        self,
        gym: SpiderEnv = None,
    ):
        super().__init__(
            name="db_user",
            system_message="You are an helpful AI assistant calling pick_question tool if the need be." +
                "You are supposed to invoke pick_question tool provided to you whenever you are asked to 'Pick a question.'"+
                "Return only the result returned by the tool, do not give any comment or explanation.",
            llm_config = self.DEFAULT_CONFIG,
            is_termination_msg=lambda x: True if "TERMINATE" in x.get("content") else False,
            max_consecutive_auto_reply=0,
            human_input_mode="NEVER",
            description=self.DEFAULT_DBUSER_DESCRIPTION,
        )
        self.register_reply([Agent, None], DBUserAgent.generate_oai_reply, remove_other_reply_funcs=True)
        self._gym = gym
        
        self._question_picker = UserProxyAgent(
            name="user_proxy",
            system_message="A proxy which is supposed to execute pick_question tool when needed.",
            is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").rstrip().endswith("TERMINATE"),
            human_input_mode="NEVER",
            max_consecutive_auto_reply=10,
            llm_config = False,
        )
        self._question_picker._reply_func_list = []
        self._question_picker.register_reply([Agent, None], ConversableAgent.generate_tool_calls_reply)

        @self._question_picker.register_for_execution()
        @self.register_for_llm(
             name="pick_question",
             description="picks a question from the spider database.",
             api_style = "tool",
        )
        def _pick_question(
        ) -> Annotated[Dict[str, str], "Dictionary with keys 'question' and 'schema'"]:
            import random
            number = random.randrange(0, 1024)  # ????
            observation, info = self._gym.reset(k=number)
            context = {
                "db_id": observation["observation"],
                "question": observation["instruction"],
                "schema": info["schema"],
            }
            return context       
        
    def generate_oai_reply(
        self,
        messages: Optional[List[Dict]] = None,
        sender: Optional[Agent] = None,
        config: Optional[Any] = None,
    ) -> Tuple[bool, Union[str, Dict, None]]:
        """Generate a reply using autogen.oai."""
        client = self.client if config is None else config
        if client is None:
            return False, None
        if messages is None:
            messages = self._oai_messages[sender]

        response = self._generate_oai_reply_from_client(
            client, self._oai_system_message + messages, self.client_cache
        )

        ret = None
        if response is None:
            return (False, None)
        else:
            if "tool_calls" in response:
                self.send(response, self._question_picker, request_reply=True, silent=False)
                ret = self.chat_messages[self._question_picker][-1].get("content")
            else:
                ret = response

        if ret is not None: return (True, ret)
        else: return (False, "")