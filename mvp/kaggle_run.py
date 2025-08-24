#!/usr/bin/env python3
"""Kaggle Run - Complete contest solver from setup to model iteration"""

import argparse
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

from orchestrator import Orchestrator
from dev import DevAgent
from summary_agent import SummaryAgent
from config import WORK_DIR


def ensure_contest_setup(competition_id: str, data_path: str = "./data") -> Dict[str, Any]:
    """Ensure contest is downloaded and analyzed. Returns contest info."""
    comp_dir = Path(data_path) / competition_id
    problem_prompt_file = comp_dir / "problem_prompt.txt"
    analysis_file = comp_dir / "analysis.json"
    
    # Check if already setup
    if problem_prompt_file.exists() and analysis_file.exists():
        print(f"Contest {competition_id} already setup at {comp_dir}")
        with open(analysis_file, 'r') as f:
            analysis = json.load(f)
        with open(problem_prompt_file, 'r') as f:
            problem_prompt = f.read()
        return {"analysis": analysis, "problem_prompt": problem_prompt, "data_dir": comp_dir}
    
    # Setup the contest
    print(f"Setting up contest {competition_id}...")
    cmd = ["python3", "kaggle_grab.py", "setup", "--competition", competition_id, "--data-path", data_path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        raise RuntimeError(f"Failed to setup contest: {result.stderr}")
    
    # Load the results
    with open(analysis_file, 'r') as f:
        analysis = json.load(f)
    with open(problem_prompt_file, 'r') as f:
        problem_prompt = f.read()
    
    return {"analysis": analysis, "problem_prompt": problem_prompt, "data_dir": comp_dir}


def create_dev_context(analysis: Dict[str, Any], data_dir: Path) -> str:
    """Create dev context based on contest analysis."""
    targets = analysis.get("target_columns", [])
    features = analysis.get("feature_columns", [])
    problem_type = analysis.get("problem_type", "unknown")
    
    dev_context = f"""You are the Dev Agent.

Allowed files:
- ./train.csv, ./test.csv

Contest Analysis:
- Problem type: {problem_type}
- Target columns: {', '.join(targets)}
- Feature columns: {len(features)} total features
- Training shape: {analysis.get('train_shape', 'unknown')}
- Test shape: {analysis.get('test_shape', 'unknown')}

Environment:
- Python 3.10+, CPU-only, deterministic; no internet or package installs.
- Available libraries: numpy, pandas, scikit-learn, lightgbm, xgboost, statsmodels, scipy.
- Return ONLY a single Python fenced block with self-contained code.
- Data files are located at: train.csv and test.csv (no path prefix needed).

IO guidance:
- Create {{id}}/ directory for outputs
- When building final models, write submission.csv with appropriate header based on contest requirements
- If instructed to evaluate models, write metrics.json with relevant metrics
- If creating visualizations (plots, charts, graphs), save them to {{id}}/ directory with descriptive filenames
- DO NOT CREATE MORE THAN 5 IMAGES PER ITERATION
- Always print dataset shapes and key findings from your analysis

Modeling guidance (optional, keep fast <3 min CPU):
- Use appropriate cross-validation strategy
- Handle target transformation if needed
- Fit appropriate models for the problem type
- For hyperparameter search, limit to ≤40 total combinations
- DO NOT USE EARLY STOPPING for training your model"""

    return dev_context


def run_iterations(contest_info: Dict[str, Any], run_dir: Path, max_iterations: int = 5, timeout_sec: int = 300) -> None:
    """Run the model development iterations.
    
    Args:
        contest_info: Contest metadata and paths
        run_dir: Directory for this specific run
        max_iterations: Maximum number of iterations to run
        timeout_sec: Timeout in seconds for each Python execution (default: 300)
    """
    # Make sure run_dir is absolute
    run_dir = run_dir.resolve()
    
    # Setup agents
    dev_agent = DevAgent(timeout_sec=timeout_sec)
    summary_agent = SummaryAgent()
    
    # Initialize blackboard
    blackboard = "Starting fresh contest run."
    
    # Create dev context
    dev_context = create_dev_context(contest_info["analysis"], contest_info["data_dir"])
    
    # Change to data directory for model training
    data_dir = Path(contest_info["data_dir"]).resolve()
    original_cwd = os.getcwd()
    
    try:
        os.chdir(data_dir)
        
        for iteration in range(max_iterations):
            print(f"\n=== Iteration {iteration} ===")
            
            # Create orchestrator with contest-specific info
            orchestrator = Orchestrator(
                problem_description=contest_info["problem_prompt"],
                dev_context=dev_context,
                blackboard=blackboard,
                iteration=iteration,
                max_iterations=max_iterations
            )
            
            # Get next task
            orch_result = orchestrator.orchestrator_step()
            task_text = orch_result["text"]
            
            if not task_text:
                print(f"Orchestrator returned empty task, stopping at iteration {iteration}")
                break
            
            # Extract just the task-specific part (skip the generic dev context)
            # Look for patterns that indicate where the actual task starts
            task_indicators = [
                "Your task is to",
                "Task:",
                "Perform",
                "Create",
                "Build",
                "Analyze", 
                "1. ",  # Numbered instructions
                "- "    # Bullet points
            ]
            
            # Find where the actual task content begins
            actual_task = task_text
            for indicator in task_indicators:
                if indicator in task_text:
                    # Find the position and extract from there
                    pos = task_text.find(indicator)
                    potential_task = task_text[pos:].strip()
                    if len(potential_task) > 50:  # Make sure we got substantial content
                        actual_task = potential_task
                        break
            
            # Show more context but truncate reasonably  
            print(f"Task: {actual_task[:600]}...")
            if len(actual_task) > 600:
                print("    [truncated...]")
            
            # Execute task with proper workspace
            # Temporarily modify WORK_DIR for this iteration
            import config
            original_work_dir = config.WORK_DIR
            config.WORK_DIR = run_dir
            
            try:
                dev_report = dev_agent.run(task_text, iteration)
            finally:
                config.WORK_DIR = original_work_dir
            
            # Summarize results  
            summary_result = summary_agent.run(
                iteration=iteration,
                orchestrator_text=task_text,
                current_blackboard=blackboard,
                dev_report=dev_report
            )
            
            blackboard = summary_result["updated_blackboard"]
            
            # Save iteration summary
            iteration_summary = {
                "iteration": iteration,
                "task": task_text,
                "success": dev_report.get("ok", False),
                "summary": summary_result["summary_block"]
            }
            
            summary_file = run_dir / f"iteration_{iteration}_summary.json"
            summary_file.write_text(json.dumps(iteration_summary, indent=2))
            
            print(f"Iteration {iteration} {'✅ SUCCESS' if dev_report.get('ok') else '❌ FAILED'}")
            
    finally:
        os.chdir(original_cwd)
    
    # Save final blackboard
    (run_dir / "final_blackboard.txt").write_text(blackboard)
    print(f"\n🏁 Run completed! Results in {run_dir}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Run complete Kaggle contest solving pipeline")
    parser.add_argument("competition", help="Competition ID (e.g., 'titanic')")
    parser.add_argument("--data-path", default="./data", help="Data directory (default: ./data)")
    parser.add_argument("--max-iterations", type=int, default=5, help="Maximum iterations (default: 5)")
    parser.add_argument("--timeout", type=int, default=300, help="Timeout in seconds for each Python execution (default: 300)")
    
    args = parser.parse_args()
    
    try:
        # Ensure contest is setup
        contest_info = ensure_contest_setup(args.competition, args.data_path)
        
        # Create run directory with timestamp inside the contest's data folder
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"run_{timestamp}"
        contest_data_dir = Path(contest_info["data_dir"])
        runs_base_dir = contest_data_dir / "runs"
        run_dir = runs_base_dir / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"🚀 Starting Kaggle run: {args.competition}")
        print(f"📂 Run directory: {run_dir}")
        
        # Save run metadata
        run_metadata = {
            "competition": args.competition,
            "timestamp": timestamp,
            "max_iterations": args.max_iterations,
            "data_path": str(contest_info["data_dir"]),
            "analysis": contest_info["analysis"]
        }
        (run_dir / "run_metadata.json").write_text(json.dumps(run_metadata, indent=2))
        
        # Run the iterations
        run_iterations(contest_info, run_dir, args.max_iterations, args.timeout)
            
    except Exception as e:
        print(f"❌ Run failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
