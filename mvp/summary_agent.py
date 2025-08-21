# summary_agent.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import chat
from response import trim_long_string


class SummaryAgent:
    """Minimal summary agent:
    - Inputs: iteration index, orchestrator instructions (dev prompt), current blackboard,
              and the dev agent report (code + outputs).
    - Produces: a short markdown block summarizing Asked/Did/Result and appends it to the blackboard.
    - No retries, no extra files unless you choose to save externally.
    """

    def __init__(self):
        pass

    def _ask_llm(self, iteration: int, orchestrator_text: str, current_blackboard: str, dev_report: Dict[str, Any]) -> str:
        code = trim_long_string(dev_report.get("code", ""), threshold=6000, k=2500)
        stdout = trim_long_string(dev_report.get("stdout", "") or "", threshold=4000, k=1800)
        stderr = trim_long_string(dev_report.get("stderr", "") or "", threshold=3000, k=1400)
        ok = dev_report.get("ok", None)
        metrics = None

        try:
            with open(f'.agent_workspace/{iteration}/metrics.json', 'r') as file:
                metrics = json.load(file)
        except:
            pass
        
        system = (
            "You write concise, structured updates for an engineering 'blackboard' used by an orchestrator.\n"
            "Only output the NEW update block—do NOT repeat the prior blackboard. Keep it brief and actionable.\n"
            "Prefer markdown with short bullets. If metrics (e.g., RMSLE) are visible in output, include them.\n"
            "If the run failed, state the failure succinctly."
        )

        user = f"""
                ### Inputs
                - Iteration: {iteration}
                - Orchestrator instructions to dev: {orchestrator_text}
                - Current blackboard (for style/context; DO NOT repeat it): {trim_long_string(current_blackboard or "", threshold=4000, k=1800)}
                - Dev agent code (trimmed): ```python {code} ```
                - Dev agent output (stdout, trimmed): {stdout}
                - Dev agent errors (stderr, trimmed): {stderr}
                - Exec OK? {ok}
                - Dev agent code response metrics: {metrics}
                Write ONE block to append to the blackboard using this exact template:
                Iteration {iteration}
                Asked: <1–2 sentences of what the dev was asked to do.>
                Did:
                <3–10 bullets of what the script actually did (models, features, CV, artifacts). Explain if these things went well (we should do them again, or not).>
                Result: <success/failure and key numbers if present (e.g., RMSLE per target + mean); otherwise the most relevant outcome. If there were interesting features, figures, etc, include that as well. Additionally, if there were any mistakes, or things that didn't work that well, include that.>

                Only return this block. No extra commentary.""".strip()
        
        resp = chat(messages=[{"role": "system", "content": system}, {"role": "user", "content": user}], temperature=0.1)
        return (resp.choices[0].message.content or "").strip()

    def run(
        self,
        *,
        iteration: int,
        orchestrator_text: str,
        current_blackboard: str,
        dev_report: Dict[str, Any],
        save_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """Return {'summary_block': str, 'updated_blackboard': str}."""
        summary_block = self._ask_llm(iteration, orchestrator_text, current_blackboard, dev_report)
        updated_blackboard = ((current_blackboard or "").rstrip() + "\n\n" + summary_block + "\n").lstrip()

        if save_path is not None:
            save_path.write_text(updated_blackboard)

        return {"summary_block": summary_block, "updated_blackboard": updated_blackboard}

