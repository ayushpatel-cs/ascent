# dev_agent.py
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import DEFAULT_TIMEOUT_SEC, chat
import config
from interpreter import Interpreter
from response import extract_code


def _tail(s: str, n: int = 5000) -> str:
    """Return tail of a possibly long string."""
    if not s:
        return ""
    return s[-n:]


class DevAgent:
    """
    Contest-agnostic dev agent that executes code based on orchestrator instructions.
    
    - Uses the orchestrator's text verbatim as the SYSTEM prompt.
    - Asks for ONE fenced Python script.
    - Runs it with the Interpreter.
    - If it fails, reprompts the model with error logs and the prior code, up to max_repairs times.
    - Writes a minimal JSON report another LLM can read:
        * Per-attempt: <WORK_DIR>/<iteration>/attempt_<k>_report.json
        * Final:       <WORK_DIR>/<iteration>/<report_name>
    - Saves each attempt's code as: <WORK_DIR>/<iteration>/attempt_<k>.py
    """

    def __init__(
        self,
        timeout_sec: Optional[int] = None,
        report_name: str = "last_dev_report.json",
        max_repairs: int = 2,
        temperature: float = 0.2,
    ):
        self.timeout_sec = timeout_sec or DEFAULT_TIMEOUT_SEC
        self.report_name = report_name
        self.max_repairs = max_repairs
        self.temperature = temperature
        self.report_path: Optional[Path] = None

    # --- LLM calls ---------------------------------------------------------

    def _ask_for_script(self, system_prompt: str) -> Dict[str, str]:
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "Return ONLY a Python fenced block (```python ... ```)."},
        ]
        resp = chat(messages=messages, temperature=self.temperature, max_tokens=8000)
        
        # Debug the API response
        print(f"=== API RESPONSE DEBUG ===")
        print(f"Response choices length: {len(resp.choices)}")
        if resp.choices:
            print(f"First choice finish_reason: {resp.choices[0].finish_reason}")
            print(f"Message content is None: {resp.choices[0].message.content is None}")
            print(f"Message content length: {len(resp.choices[0].message.content or '')}")
        print("==========================")
        
        text = (resp.choices[0].message.content or "").strip()
        
        # Debug output (disabled)
        # print("=== DEV AGENT SYSTEM PROMPT ===")
        # print(system_prompt[:500] + "..." if len(system_prompt) > 500 else system_prompt)
        # print("=== DEV AGENT LLM RESPONSE ===")
        # print(f"Response length: {len(text)}")
        # print(f"Response ends with: {repr(text[-100:])}")
        # print(f"Contains closing backticks: {'```' in text}")
        # print("First 500 chars:", text[:500] + "..." if len(text) > 500 else text)
        # print("==============================")
        
        code = (extract_code(text) or "").strip()
        if not code:
            print("=== EXTRACT_CODE DEBUG ===")
            print(f"Text length: {len(text)}")
            print(f"Full response text:")
            print(text)
            print("=== END RESPONSE ===")
            raise RuntimeError("No Python fenced block found in model response.")
        return {"response_text": text, "code": code}

    def _build_prev_attempts_summary(
        self,
        iter_dir: Path,
        attempts: List[Dict[str, Any]],
        code_tail_n: int = 4000,
        stderr_tail_n: int = 3000,
        stdout_tail_n: int = 1500,
    ) -> str:
        """
        Build a compact textual log of ALL previous attempts (oldest->newest),
        including each attempt's code tail and stdout/stderr tails.
        """
        blocks: List[str] = []
        for a in attempts:
            idx = a.get("idx")
            code_path = iter_dir / f"attempt_{idx}.py"
            code_text = ""
            if code_path.exists():
                try:
                    code_text = code_path.read_text(encoding="utf-8")
                except Exception:
                    code_text = ""
            code_tail = _tail(code_text, code_tail_n)
            stderr_tail = _tail(a.get("stderr_tail") or "", stderr_tail_n)
            stdout_tail = _tail(a.get("stdout_tail") or "", stdout_tail_n)
            blocks.append(
                f"### Attempt {idx}\n"
                f"--- CODE (tail) ---\n{code_tail}\n\n"
                f"--- STDERR (tail) ---\n{stderr_tail}\n\n"
                f"--- STDOUT (tail) ---\n{stdout_tail}\n"
            )
        return "\n".join(blocks).strip()

    def _ask_for_fix(
        self,
        system_prompt: str,
        prior_code: str,
        stderr_tail: str,
        stdout_tail: str,
        prev_attempts_summary: str,
    ) -> Dict[str, str]:
        # Keep things compact to avoid blowing up context
        prior_code_short = _tail(prior_code, 4000)
        stderr_short = _tail(stderr_tail, 3000)
        stdout_short = _tail(stdout_tail, 1500)

        repair_user = (
            "Your last Python script crashed. Analyze the logs and return a corrected, full, "
            "single-file Python script.\n\n"
            "REQUIREMENTS:\n"
            "- Return ONLY a Python fenced block (```python ... ```).\n"
            "- Offline only (no network/installs), deterministic, fast on CPU.\n"
            "- No explanations, comments optional.\n\n"
            "CONTEXT (all previous attempts, oldest first):\n"
            f"{prev_attempts_summary}\n\n"
            "--- Most recent failed script (tail) ---\n"
            f"{prior_code_short}\n\n"
            "--- Most recent STDERR (tail) ---\n"
            f"{stderr_short}\n\n"
            "--- Most recent STDOUT (tail) ---\n"
            f"{stdout_short}\n"
        )

        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": repair_user},
        ]
        resp = chat(messages=messages, temperature=self.temperature, max_tokens=8000)
        text = (resp.choices[0].message.content or "").strip()
        code = (extract_code(text) or "").strip()
        if not code:
            # Fall back to returning raw text to aid debugging; we'll still write a report.
            code = ""
        return {"response_text": text, "code": code}

    # --- Runner ------------------------------------------------------------

    def _run(self, code: str) -> Dict[str, Any]:
        # Use config.WORK_DIR for code execution (the run directory)
        # but ensure CSV files are accessible via symlinks
        import os
        
        # Get current working directory (where CSV files are) and run directory
        data_dir = os.getcwd()
        run_dir = config.WORK_DIR
        
        # Create symlinks to CSV files in the run directory so code can find them
        train_csv = os.path.join(data_dir, "train.csv")
        test_csv = os.path.join(data_dir, "test.csv")
        run_train_csv = os.path.join(run_dir, "train.csv")
        run_test_csv = os.path.join(run_dir, "test.csv")
        
        # Create symlinks if they don't exist
        if os.path.exists(train_csv) and not os.path.exists(run_train_csv):
            try:
                os.symlink(train_csv, run_train_csv)
            except OSError:
                # Fallback to copying if symlink fails
                import shutil
                shutil.copy2(train_csv, run_train_csv)
                
        if os.path.exists(test_csv) and not os.path.exists(run_test_csv):
            try:
                os.symlink(test_csv, run_test_csv)
            except OSError:
                # Fallback to copying if symlink fails
                import shutil
                shutil.copy2(test_csv, run_test_csv)
        
        interp = Interpreter(
            working_dir=config.WORK_DIR,
            timeout=self.timeout_sec,
            format_tb_ipython=False,
            agent_file_name="dev_run.py",
        )
        result = interp.run(code, reset_session=True)
        joined = "".join(result.term_out)
        ok = result.exc_type is None
        out = {
            "ok": ok,
            "stdout": joined if ok else "",
            "stderr": "" if ok else joined,
            "exec_time_sec": result.exec_time,
            "exc_type": result.exc_type,
            "exc_info": result.exc_info,
            "exc_stack": result.exc_stack,
        }
        interp.cleanup_session()
        return out

    # --- Reporting helpers -------------------------------------------------

    def _attempt_report_path(self, iter_dir: Path, idx: int) -> Path:
        return iter_dir / f"attempt_{idx}_report.json"

    def _write_attempt_report(
        self,
        iter_dir: Path,
        orchestrator_text: str,
        idx: int,
        code: str,
        response_text: str,
        exec_out: Dict[str, Any],
    ) -> None:
        report = {
            "instructions": orchestrator_text,
            "attempt_idx": idx,
            "response_text": response_text,
            "code": code,
            "ok": bool(exec_out.get("ok")),
            "stdout_tail": _tail(exec_out.get("stdout") or ""),
            "stderr_tail": _tail(exec_out.get("stderr") or ""),
            "exec_time_sec": exec_out.get("exec_time_sec"),
            "exc_type": exec_out.get("exc_type"),
        }
        self._attempt_report_path(iter_dir, idx).write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    # --- Public API --------------------------------------------------------

    def run(
        self,
        orchestrator_text: str,
        iteration: int | str,
        max_repairs: Optional[int] = None,
    ) -> Dict[str, Any]:
        # Ensure iteration directory exists
        iter_dir = config.WORK_DIR / str(iteration)
        iter_dir.mkdir(parents=True, exist_ok=True)

        attempts: List[Dict[str, Any]] = []
        n_repairs = self.max_repairs if max_repairs is None else max_repairs

        # 1) Initial attempt
        ask = self._ask_for_script(orchestrator_text)
        code = ask["code"]
        
        # Always save the raw response for debugging
        (iter_dir / "attempt_0_raw_response.txt").write_text(ask["response_text"], encoding="utf-8")
        
        # Save the extracted code (or empty string if extraction failed)
        (iter_dir / "attempt_0.py").write_text(code, encoding="utf-8")

        exec_out = self._run(code)
        attempts.append(
            {
                "idx": 0,
                "ok": exec_out["ok"],
                "stdout_tail": _tail(exec_out["stdout"]),
                "stderr_tail": _tail(exec_out["stderr"]),
                "exec_time_sec": exec_out["exec_time_sec"],
                "exc_type": exec_out["exc_type"],
            }
        )
        # Per-attempt report (attempt 0)
        self._write_attempt_report(
            iter_dir=iter_dir,
            orchestrator_text=orchestrator_text,
            idx=0,
            code=code,
            response_text=ask["response_text"],
            exec_out=exec_out,
        )

        # 2) Auto-repair loop
        cur_code = code
        cur_resp_text = ask["response_text"]

        attempt_idx = 0
        while not exec_out["ok"] and attempt_idx < n_repairs:
            attempt_idx += 1

            prev_summary = self._build_prev_attempts_summary(iter_dir, attempts)

            fix = self._ask_for_fix(
                system_prompt=orchestrator_text,
                prior_code=cur_code,
                stderr_tail=_tail(exec_out["stderr"]),
                stdout_tail=_tail(exec_out["stdout"]),
                prev_attempts_summary=prev_summary,
            )
            cur_resp_text = fix["response_text"]

            if not fix["code"]:
                # If model failed to return code, write a per-attempt report and stop early
                attempts.append(
                    {
                        "idx": attempt_idx,
                        "ok": False,
                        "stdout_tail": "",
                        "stderr_tail": "Model did not return a fenced Python block.",
                        "exec_time_sec": 0.0,
                        "exc_type": "NoCode",
                    }
                )
                self._write_attempt_report(
                    iter_dir=iter_dir,
                    orchestrator_text=orchestrator_text,
                    idx=attempt_idx,
                    code="",
                    response_text=cur_resp_text,
                    exec_out={
                        "ok": False,
                        "stdout": "",
                        "stderr": "Model did not return a fenced Python block.",
                        "exec_time_sec": 0.0,
                        "exc_type": "NoCode",
                    },
                )
                break

            cur_code = fix["code"]
            
            # Always save the raw response for debugging
            (iter_dir / f"attempt_{attempt_idx}_raw_response.txt").write_text(fix["response_text"], encoding="utf-8")
            
            # Save the extracted code
            (iter_dir / f"attempt_{attempt_idx}.py").write_text(cur_code, encoding="utf-8")

            exec_out = self._run(cur_code)
            attempts.append(
                {
                    "idx": attempt_idx,
                    "ok": exec_out["ok"],
                    "stdout_tail": _tail(exec_out["stdout"]),
                    "stderr_tail": _tail(exec_out["stderr"]),
                    "exec_time_sec": exec_out["exec_time_sec"],
                    "exc_type": exec_out["exc_type"],
                }
            )
            # Per-attempt report (attempt k)
            self._write_attempt_report(
                iter_dir=iter_dir,
                orchestrator_text=orchestrator_text,
                idx=attempt_idx,
                code=cur_code,
                response_text=cur_resp_text,
                exec_out=exec_out,
            )

            if exec_out["ok"]:
                break

        # 3) Final report (reflect last attempt)
        last = attempts[-1]
        report = {
            "instructions": orchestrator_text,
            "response_text": cur_resp_text,
            "code": cur_code,
            "ok": bool(last["ok"]),
            "stdout": last["stdout_tail"],
            "stderr": last["stderr_tail"],
            "exec_time_sec": last["exec_time_sec"],
            "exc_type": last["exc_type"],
            "iteration": str(iteration),
            "attempts": attempts,  # compact tails only
            "max_repairs": n_repairs,
        }

        # Persist final report under the iteration folder
        self.report_path = iter_dir / self.report_name
        self.report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

        return report
