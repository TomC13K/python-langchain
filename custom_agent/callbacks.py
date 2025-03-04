# all the logic to log all the LLM calls
from typing import Dict, Any, List

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult


class AgentCallbackHandler(BaseCallbackHandler):
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> Any:
        """Run when LLM starts running."""
        print(f"***Prompt to LLM was ***\n{prompts[0]}")
        print("*********")

    def onllm_end(self, response: LLMResult, **kwargs: Any) -> Any:
        """Run when LLM ends running."""
        print(f"***LLM Response ***\n{response.generations[0][0].text}")
        print("*********")
