from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from prompts import FLEXIBLE_ORCHESTRATOR_PROMPT
from config import client, MODEL, WORK_DIR


BlackboardT = Union[str, Dict[str, Any], List[Dict[str, Any]]]


class Orchestrator:
    """
    Contest-agnostic orchestrator: formats a prompt to ask the LLM orchestrator
    to produce the next task text for the dev agent.

    Notes:
    - Takes problem description and dev context as parameters instead of hardcoded prompts
    - `blackboard` is assumed to be pre-summarized/free-text; if it's not a
      string, we JSON-dump it as-is.
    """

    def __init__(
        self,
        problem_description: str,
        dev_context: str,
        blackboard: str,
        iteration: int,
        max_iterations: Optional[int] = None,  # NEW: planning budget
    ) -> None:
        self.problem_description = (problem_description or "").strip()
        self.dev_context = (dev_context or "").strip()
        self.blackboard = (blackboard or "").strip()
        self.iteration = iteration
        # If caller doesn't supply, default to "10 more tasks after current"
        # i.e., plan horizon = current index + 11 total (current + 10).
        self.max_iterations = max_iterations if max_iterations is not None else (iteration + 11)

    # -------------------- internals --------------------

    def _encode_image(self, image_path: Path) -> str:
        """Encode image file to base64 string."""
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            print(f"Warning: Could not encode image {image_path}: {e}")
            return ""

    def _find_images_from_previous_iteration(self) -> List[Path]:
        """Find image files from the previous iteration's output directory."""
        if self.iteration == 0:
            return []
        
        prev_iter_dir = WORK_DIR / str(self.iteration - 1)
        if not prev_iter_dir.exists():
            return []
        
        # Look for common image extensions
        image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.svg'}
        image_files = []
        
        for file_path in prev_iter_dir.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in image_extensions:
                image_files.append(file_path)
        
        # Sort by modification time (newest first) and limit to 5 images to avoid token limits
        image_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        return image_files[:5]

    def _build_prompt(self) -> str:
        # Keep your {id} run folder behavior
        run_dir = str(self.iteration)
        dev_context = self.dev_context.replace("{id}", run_dir)

        remaining_tasks = max(self.max_iterations - self.iteration - 1, 0)

        return FLEXIBLE_ORCHESTRATOR_PROMPT.format(
            problem_description=self.problem_description.strip(),
            current_iteration=self.iteration,
            total_iterations=self.max_iterations,
            iterations_remaining=remaining_tasks,
            blackboard=self.blackboard.strip() if self.blackboard.strip() else "No previous work completed yet.",
            dev_context=dev_context.strip(),
        )

    def orchestrator_step(
        self,
        temperature: float = 0.2,
        max_tokens: int = 8000,
    ) -> Dict[str, object]:
        """
        Build the prompt via Orchestrator and fetch the LLM reply.
        Now supports including images from previous iterations.

        Returns a dict with the assistant text, finish_reason, and raw response.
        """
        # Find images from previous iteration
        image_files = self._find_images_from_previous_iteration()
        
        # Build the text prompt
        prompt_text = self._build_prompt()
        
        # Create messages - handle both text-only and vision scenarios
        msgs: List[Dict[str, Any]] = []
        
        if image_files:
            # Vision message with images
            content_parts = [{"type": "text", "text": prompt_text}]
            
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
                    
            # Add a note about the images in the text
            if len(content_parts) > 1:
                image_names = [img.name for img in image_files]
                content_parts[0]["text"] += f"\n\nPrevious iteration created these visualizations: {', '.join(image_names)}. Consider these outputs when planning the next task."
            
            msgs.append({"role": "system", "content": content_parts})
        else:
            # Standard text-only message
            msgs.append({"role": "system", "content": prompt_text})

        resp = client.chat.completions.create(
            model=MODEL,
            messages=msgs,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        # print ("RESPONSE: ", resp)
        # print("=== ORCHESTRATOR PROMPT ===")
        # print(self._build_prompt()) 
        # print("=== ORCHESTRATOR RESPONSE ===")
        # print(resp.choices[0].message.content)
        # print("==============================")

        text = (resp.choices[0].message.content or "").strip()
        return {
            "text": text,
            "finish_reason": getattr(resp.choices[0], "finish_reason", None),
            "raw": resp,
            "images_processed": len(image_files),
            "image_files": [str(p) for p in image_files],
        }

    @staticmethod
    def _normalize_blackboard(bb: BlackboardT) -> str:
        if isinstance(bb, str):
            return bb
        try:
            return json.dumps(bb, ensure_ascii=False, indent=2)
        except Exception:
            return str(bb)
