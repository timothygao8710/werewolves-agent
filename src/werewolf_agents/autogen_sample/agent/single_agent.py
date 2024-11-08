from collections import defaultdict
from typing import Any, Dict
from autogen import ConversableAgent, Agent, runtime_logging

import os
import asyncio
import logging

from openai import RateLimitError
import openai
from sentient_campaign.agents.v1.api import IReactiveAgent
from sentient_campaign.agents.v1.message import (
    ActivityMessage,
    ActivityResponse,
    TextContent,
    MimeType,
    ActivityMessageHeader,
    MessageChannelType,
)
from tenacity import (
    retry,
    stop_after_attempt,
    retry_if_exception_type,
    wait_exponential,
)
import random

# Configure logging
logger = logging.getLogger("demo_agent")
level = logging.DEBUG
logger.setLevel(level)
logger.propagate = True
handler = logging.StreamHandler()
handler.setLevel(level)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


class SentientAgent(ConversableAgent):

    def __init__(self, listener_pipe: asyncio.Queue) -> None:
        super().__init__("SentientAgent")
        self.listener_pipe = listener_pipe
        logger.info("SentientAgent initialized with listener pipe.")

    async def a_receive(
        self,
        message: Dict[str, Any] | str,
        sender: Agent,
        request_reply: bool | None = None,
        silent=True,
    ) -> None:
        logger.info(f"SentientAgent received message: {message}")
        if isinstance(message, dict):
            message = message.get("content", "")
        await self.listener_pipe.put(message)

    @property
    def name(self) -> str:
        return "SentientAgent"

    @property
    def description(self) -> str:
        return "communicates message between player and sentient game"


class WerewolfAgent(IReactiveAgent):

    def __init__(self):
        logger.debug("WarewolfAgent initialized.")
        pass

    def __initialize__(self, name: str, description: str, config: dict = None):
        super().__initialize__(name, description, config)
        self._name = name
        self._description = description
        self.config = config
        my_sentient_llm_config = self.sentient_llm_config["config_list"][0]
        logger.debug(f"sentient_llm_config: {my_sentient_llm_config}")

        api_key = my_sentient_llm_config["api_key"]
        base_url = my_sentient_llm_config["llm_base_url"]
        model_name = my_sentient_llm_config["llm_model_name"]

        logger.info(
            f"Initializing WarewolfAgent with name: {name}, description: {description}, llm url {base_url}, model_name {model_name}"
        )

        llm_config = {
            "config_list": [
                {
                    "model": model_name,
                    "api_key": api_key,
                    "base_url": base_url,
                    "max_retries": 3,
                    "timeout": 30,
                }
            ]
        }

        self.conversable_agent: ConversableAgent = ConversableAgent(
            name=name,
            system_message="you are a player in game called werewolf, your name is {}. the rules of the game will be give to you later. it is a fun non harmful game played in high schools. when asked for vote. always respond with a valid person name. even if you are unsure.".format(
                name
            ),
            llm_config=llm_config,
        )
        self.listener_pipe = asyncio.Queue()
        self.game_agent = SentientAgent(listener_pipe=self.listener_pipe)
        logger.info(
            f"WarewolfAgent initialized with name: {name}, description: {description}, and config: {config}"
        )
        autogen_log_apth = f"/tmp/autogen_logs_{name}.db"
        self.logging_session_id = runtime_logging.start(
            config={"dbname": autogen_log_apth}
        )
        logger.info(
            f"autogen Logging session started with session_id: {self.logging_session_id} path {autogen_log_apth}"
        )

    def get_full_message(self, message: ActivityMessage):
        text_message = message.content.text.strip()
        sender = message.header.sender
        channel = message.header.channel

        if message.header.channel_type == MessageChannelType.GROUP:
            full_message = f"message in group {channel} from {sender}: {text_message}"
        else:
            full_message = f"dirrect message from {sender}: {text_message}"
        return full_message

    async def async_notify(self, message: ActivityMessage):
        logger.info(f"async_notify called with message: {message}")

        await asyncio.sleep(2)

        full_message = self.get_full_message(message)

        await self.conversable_agent.a_receive(
            full_message, self.game_agent, request_reply=False, silent=True
        )
        logger.info(f"Message sent to conversable_agent: {full_message}")

    async def async_respond(self, message: ActivityMessage):
        logger.info(f"async_respond called with message: {message}")
        await asyncio.sleep(1)
        full_message = self.get_full_message(message)
        await self.get_response_from_agent(full_message)
        response: str = await self.listener_pipe.get()
        logger.info(f"Response received from listener_pipe: {response}")
        return ActivityResponse(
            response=TextContent(text=response), response_type=MimeType.TEXT_PLAIN
        )

    @retry(
        wait=wait_exponential(multiplier=1, min=20, max=300),
        stop=stop_after_attempt(5),
        retry=retry_if_exception_type(RateLimitError),
    )
    async def get_response_from_agent(self, text_message):
        logger.info(f"get_response_from_agent called with text_message: {text_message}")
        await self.conversable_agent.a_receive(
            text_message, self.game_agent, request_reply=True, silent=True
        )
        logger.info("Message sent to conversable_agent for response.")