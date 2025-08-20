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
        max_iterations: Optional[int] = None,  # NEW: planning budget
    ) -> None:
        self.problem_description = (problem_description or "").strip()
        self.blackboard = (blackboard or "").strip()
        self.iteration = iteration
        # If caller doesn't supply, default to "10 more tasks after current"
        # i.e., plan horizon = current index + 11 total (current + 10).
        self.max_iterations = max_iterations if max_iterations is not None else (iteration + 11)

    # -------------------- internals --------------------

    def _build_prompt(self) -> str:
        # Keep your {id} run folder behavior
        run_dir = str(self.iteration)
        self.dev_description = DEV_CONTEXT_STARTER.replace("{id}", run_dir)

        remaining_tasks = max(self.max_iterations - self.iteration - 1, 0)
        max_iterations_minus_1 = self.max_iterations - 1

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
