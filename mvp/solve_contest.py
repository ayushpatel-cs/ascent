"""
Kaggle Contest Solver - Multi-Agent System

This module provides a clean interface to solve Kaggle contests using
the orchestrator-dev-summary agent system with vision capabilities.
"""

from __future__ import annotations
from typing import Optional, Dict, List, Any

from prompts import PROBLEM_PROMPT
from orchestrator import Orchestrator
from dev import DevAgent
from summary_agent import SummaryAgent


def solve_contest(
    max_iterations: int = 5,
    problem_description: Optional[str] = None,
    verbose: bool = True,
    max_repairs: int = 3
) -> Dict[str, Any]:
    """
    Solve a Kaggle contest using the multi-agent system.
    
    This function orchestrates a complete contest-solving workflow:
    1. Orchestrator plans tasks based on iteration stage and previous results
    2. Dev agent executes code with automatic repair attempts
    3. Summary agent analyzes results including visualizations
    4. Process repeats with accumulated knowledge
    
    Args:
        max_iterations (int): Number of iterations to run (default: 5)
        problem_description (str, optional): Custom problem description 
                                           (defaults to PROBLEM_PROMPT)
        verbose (bool): Whether to print iteration summaries (default: True)
        max_repairs (int): Maximum repair attempts per iteration (default: 3)
        
    Returns:
        Dict[str, Any]: Complete results containing:
            - final_blackboard: Accumulated knowledge and insights
            - iteration_summaries: List of summaries for each iteration
            - iteration_reports: Raw dev agent reports for each iteration
            - total_iterations: Number of iterations run
            - problem_description: Problem description used
            - success_rate: Percentage of successful iterations
            - total_images_processed: Number of visualizations analyzed
    """
    problem_desc = problem_description or PROBLEM_PROMPT
    blackboard = ""
    agent = DevAgent()
    summarizer = SummaryAgent()
    
    iteration_summaries = []
    iteration_reports = []

    if verbose:
        print("🚀 Starting Kaggle Contest Solver")
        print(f"📋 Problem: {problem_desc.split('.')[0]}...")
        print(f"🔄 Running {max_iterations} iterations")
        print("=" * 60)

    for iteration in range(max_iterations):
        if verbose:
            print(f"\n🔄 Iteration {iteration + 1}/{max_iterations}")
        
        try:
            # 1) Orchestrate (plan next task based on progress)
            orch = Orchestrator(
                problem_description=problem_desc,
                blackboard=blackboard,
                iteration=iteration,
                max_iterations=max_iterations,
            )
            step = orch.orchestrator_step()
            instr = step["text"] if isinstance(step, dict) else step
            
            if verbose and step.get("images_processed", 0) > 0:
                print(f"   🖼️  Analyzed {step['images_processed']} images from previous iteration")

            # 2) Dev agent executes the task
            report = agent.run(instr, iteration=iteration, max_repairs=max_repairs)
            iteration_reports.append(report)

            # 3) Summarize results and update knowledge base
            out = summarizer.run(
                iteration=iteration,
                orchestrator_text=instr,
                current_blackboard=blackboard,
                dev_report=report,
                save_path=None,
            )

            if verbose:
                status = "✅ Success" if report.get("ok", False) else "❌ Failed"
                print(f"   {status}")
                print(f"\n--- Summary ---")
                print(out["summary_block"])
            
            blackboard = out["updated_blackboard"]
            iteration_summaries.append({
                "iteration": iteration,
                "orchestrator_instruction": instr,
                "summary": out["summary_block"],
                "success": report.get("ok", False),
                "images_processed": step.get("images_processed", 0),
                "execution_time": report.get("exec_time_sec", 0),
                "attempts": len(report.get("attempts", []))
            })
            
        except Exception as e:
            if verbose:
                print(f"   ❌ Error in iteration {iteration}: {e}")
            iteration_summaries.append({
                "iteration": iteration,
                "error": str(e),
                "success": False,
                "images_processed": 0
            })

    # Calculate final statistics
    successful_iterations = sum(1 for s in iteration_summaries if s.get("success", False))
    success_rate = (successful_iterations / max_iterations) * 100
    total_images = sum(s.get("images_processed", 0) for s in iteration_summaries)
    
    if verbose:
        print("\n" + "=" * 60)
        print(f"🎯 Contest solving completed!")
        print(f"📊 Success rate: {successful_iterations}/{max_iterations} ({success_rate:.1f}%)")
        print(f"🖼️  Total images processed: {total_images}")
        print(f"🔧 Total repair attempts: {sum(s.get('attempts', 1) - 1 for s in iteration_summaries)}")
        
        print(f"\n📋 Final Knowledge Base ({len(blackboard)} characters):")
        print("-" * 40)
        print(blackboard)

    return {
        "final_blackboard": blackboard,
        "iteration_summaries": iteration_summaries,
        "iteration_reports": iteration_reports,
        "total_iterations": max_iterations,
        "problem_description": problem_desc,
        "success_rate": success_rate,
        "total_images_processed": total_images,
        "successful_iterations": successful_iterations
    }


def run_iterations(n: int = 20) -> str:
    """
    Legacy function for backward compatibility.
    
    Args:
        n (int): Number of iterations to run
        
    Returns:
        str: Final blackboard content
    """
    result = solve_contest(max_iterations=n, verbose=False)
    return result["final_blackboard"]


def quick_solve(iterations: int = 3) -> Dict[str, Any]:
    """
    Quick contest solve with minimal output.
    
    Args:
        iterations (int): Number of iterations (default: 3)
        
    Returns:
        Dict[str, Any]: Results with key metrics
    """
    return solve_contest(
        max_iterations=iterations,
        verbose=False,
        max_repairs=2
    )


def detailed_solve(iterations: int = 10) -> Dict[str, Any]:
    """
    Detailed contest solve with maximum iterations and repairs.
    
    Args:
        iterations (int): Number of iterations (default: 10)
        
    Returns:
        Dict[str, Any]: Comprehensive results
    """
    return solve_contest(
        max_iterations=iterations,
        verbose=True,
        max_repairs=5
    )


if __name__ == "__main__":
    # Example usage
    print("Running Kaggle Contest Solver...")
    
    # Standard solve
    result = solve_contest(
        max_iterations=5,
        verbose=True,
        max_repairs=3
    )
    
    # Print additional insights
    print(f"\n🔍 Additional Insights:")
    print(f"   Average execution time: {sum(s.get('execution_time', 0) for s in result['iteration_summaries']) / len(result['iteration_summaries']):.2f}s")
    
    # Show which iterations were most successful
    successful_iters = [s['iteration'] for s in result['iteration_summaries'] if s.get('success', False)]
    if successful_iters:
        print(f"   Successful iterations: {successful_iters}")
    
    if result['total_images_processed'] > 0:
        print(f"   Vision analysis was used effectively")
