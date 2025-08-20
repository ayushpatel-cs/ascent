from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Union

from prompts import PROBLEM_PROMPT, ORCHESTRATOR_TASK_TEMPLATE, DEV_CONTEXT_STARTER
from config import client, MODEL, Message, Messages


BlackboardT = Union[str, Dict[str, Any], List[Dict[str, Any]]]


class Orchestrator:
    """
    Minimal orchestrator: formats a single prompt to ask the LLM orchestrator
    to produce the next task text for the dev agent.

    Notes:
    - The problem description is hardcoded in prompts.PROBLEM_PROMPT.
    - `blackboard` is assumed to be pre-summarized/free-text; if it's not a
      string, we JSON-dump it as-is.
    """

    def __init__(
        self,
        problem_description: str, 
        blackboard: str,
        iteration: int, 
    ) -> None:
        self.problem_description = (problem_description or "").strip()
        self.blackboard = (blackboard or "").strip()
        self.iteration = iteration
    # -------------------- internals --------------------

    def _build_prompt(self) -> str:
        self.dev_description = DEV_CONTEXT_STARTER.replace("{id}", str(self.iteration))
        return ORCHESTRATOR_TASK_TEMPLATE.format(
            problem_description=self.problem_description.strip(),
            dev_description=self.dev_description.strip(),
            blackboard=self.blackboard.strip(),
        )
    def orchestrator_step(
        self,
        temperature: float = 0.2,
        max_tokens: int = 1200,
    ) -> Dict[str, object]:
        """
        Build the prompt via Orchestrator and fetch the LLM reply.

        Returns a dict with the assistant text, finish_reason, and raw response.
        """
        msgs: List[Message] = []
        msgs.append({"role": "system", "content": self._build_prompt()})

        resp = client.chat.completions.create(
            model=MODEL,
            messages=msgs,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        text = (resp.choices[0].message.content or "").strip()
        return {
            "text": text,
            "finish_reason": getattr(resp.choices[0], "finish_reason", None),
            "raw": resp,
        }

    @staticmethod
    def _normalize_blackboard(bb: BlackboardT) -> str:
        if isinstance(bb, str):
            return bb
        try:
            return json.dumps(bb, ensure_ascii=False, indent=2)
        except Exception:
            return str(bb)
