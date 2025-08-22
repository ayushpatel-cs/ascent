# summary_agent.py
from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import chat, WORK_DIR
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

    def _encode_image(self, image_path: Path) -> str:
        """Encode image file to base64 string."""
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            print(f"Warning: Could not encode image {image_path}: {e}")
            return ""

    def _find_images_from_iteration(self, iteration: int) -> List[Path]:
        """Find image files from the current iteration's output directory."""
        iter_dir = WORK_DIR / str(iteration)
        if not iter_dir.exists():
            return []
        
        # Look for common image extensions
        image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.svg'}
        image_files = []
        
        for file_path in iter_dir.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in image_extensions:
                image_files.append(file_path)
        
        # Sort by modification time (newest first) and limit to 5 images
        image_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        return image_files[:5]

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

        # Find images from this iteration
        image_files = self._find_images_from_iteration(iteration)
        
        system = (
            "You write concise, structured updates for an engineering 'blackboard' used by an orchestrator.\n"
            "Only output the NEW update block—do NOT repeat the prior blackboard. Keep it brief and actionable.\n"
            "Prefer markdown with short bullets. If metrics (e.g., RMSLE) are visible in output, include them.\n"
            "If visualizations/plots are shown, describe key insights from them (trends, correlations, patterns).\n"
            "If the run failed, state the failure succinctly."
        )

        user_text = f"""
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

        # Build messages with images if available
        messages = [{"role": "system", "content": system}]
        
        if image_files:
            # Vision message with images
            content_parts = [{"type": "text", "text": user_text}]
            
            # Add images to the content
            for img_path in image_files:
                base64_image = self._encode_image(img_path)
                if base64_image:
                    # Detect image format
                    ext = img_path.suffix.lower()
                    mime_type = "image/png" if ext == ".png" else "image/jpeg"
                    
                    content_parts.append({
                        "type": "image_url", 
                        "image_url": {
                            "url": f"data:{mime_type};base64,{base64_image}"
                        }
                    })
            
            # Add note about images in the text
            if len(content_parts) > 1:
                image_names = [img.name for img in image_files]
                content_parts[0]["text"] += f"\n\nVisualizations created: {', '.join(image_names)}. Analyze these charts/plots and include key insights in your summary."
            
            messages.append({"role": "user", "content": content_parts})
        else:
            # Standard text-only message
            messages.append({"role": "user", "content": user_text})
        
        resp = chat(messages=messages, temperature=0.1, max_tokens=6000)
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

