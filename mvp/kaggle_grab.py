#!/usr/bin/env python3
"""Simple Kaggle Contest Grabber - Downloads and analyzes competitions"""

import argparse
import subprocess
import json
import zipfile
import shutil
import re
from pathlib import Path
import pandas as pd
from typing import Dict, Any, Optional

try:
    import requests
    from bs4 import BeautifulSoup
    WEB_SCRAPING_AVAILABLE = True
except ImportError:
    WEB_SCRAPING_AVAILABLE = False

try:
    from kaggle_scraper import scrape_competition, summarize_with_llm
    SCRAPER_AVAILABLE = True
except ImportError:
    print("⚠️ kaggle_scraper.py not available - will use basic prompt")
    SCRAPER_AVAILABLE = False


def kaggle_cmd(cmd: list) -> tuple[bool, str]:
    """Run kaggle CLI command."""
    try:
        result = subprocess.run(['kaggle'] + cmd, capture_output=True, text=True, check=True)
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        return False, e.stderr
    except FileNotFoundError:
        return False, "Kaggle CLI not found. Install with: pip install kaggle"


def list_competitions(search: Optional[str] = None) -> bool:
    """List competitions."""
    cmd = ['competitions', 'list']
    if search:
        cmd.extend(['--search', search])
    
    success, output = kaggle_cmd(cmd)
    print(output if success else f"❌ {output}")
    return success


def download_data(competition_id: str, data_path: str) -> bool:
    """Download and extract competition data."""
    import os
    
    comp_dir = Path(data_path) / competition_id
    comp_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📥 Downloading {competition_id}...")
    
    # Change to competition directory to avoid leftover files in main directory
    original_cwd = os.getcwd()
    os.chdir(comp_dir)
    
    try:
        success, output = kaggle_cmd(['competitions', 'download', '-c', competition_id])
        
        if not success:
            print(f"❌ {output}")
            print("💡 Accept competition rules on Kaggle website first")
            return False
        
        # Extract zip files
        for zip_file in Path(".").glob("*.zip"):
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(".")
            zip_file.unlink()
        
        print(f"✅ Downloaded to {comp_dir}")
        return True
        
    finally:
        # Always restore original directory
        os.chdir(original_cwd)


def get_competition_description(competition_id: str) -> str:
    """Get detailed competition description using web scraping."""
    if not SCRAPER_AVAILABLE:
        return ""
    
    try:
        print(f"🌐 Scraping competition details for {competition_id}...")
        competition_data = scrape_competition(competition_id)
        
        if not competition_data:
            print("❌ Failed to scrape competition")
            return ""
        
        description = summarize_with_llm(competition_data)
        print(f"✅ Got competition description ({len(description)} chars)")
        return description
        
    except Exception as e:
        print(f"⚠️ Scraping failed: {e}")
        return ""


def analyze_data(competition_id: str, data_path: str) -> Dict[str, Any]:
    """Analyze competition data structure."""
    comp_dir = Path(data_path) / competition_id
    train_file, test_file = comp_dir / "train.csv", comp_dir / "test.csv"
    
    if not train_file.exists() or not test_file.exists():
        print("❌ Missing train.csv or test.csv")
        return {}
    
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)
    
    # Find targets and features
    train_cols, test_cols = set(train_df.columns), set(test_df.columns)
    targets = [col for col in train_cols - test_cols if not col.lower().startswith('id')]
    features = [col for col in test_cols if not col.lower().startswith('id')]
    
    # Determine problem type
    if len(targets) > 1:
        problem_type = "multi-target"
    elif len(targets) == 1 and pd.api.types.is_numeric_dtype(train_df[targets[0]]):
        unique_vals = train_df[targets[0]].nunique()
        problem_type = "classification" if unique_vals <= 10 and unique_vals < len(train_df) * 0.05 else "regression"
    else:
        problem_type = "classification"
    
    # Detect all available files in the competition directory
    available_files = []
    for file_path in comp_dir.iterdir():
        if file_path.is_file() and not file_path.name.startswith('.'):
            file_size = file_path.stat().st_size
            if file_path.suffix == '.csv':
                try:
                    df = pd.read_csv(file_path)
                    file_info = f"{file_path.name} (shape: {df.shape})"
                except:
                    file_info = f"{file_path.name} (size: {file_size:,} bytes)"
            else:
                file_info = f"{file_path.name} (size: {file_size:,} bytes)"
            available_files.append(file_info)
    
    info = {
        "competition_id": competition_id,
        "train_shape": train_df.shape,
        "test_shape": test_df.shape,
        "target_columns": targets,
        "feature_columns": features,
        "problem_type": problem_type,
        "available_files": available_files
    }
    
    print(f"🔍 Analysis: {train_df.shape} train, {test_df.shape} test")
    print(f"🎯 Targets: {targets} ({problem_type})")
    print(f"📈 Features: {len(features)} columns")
    print(f"📁 Available files: {len(available_files)} total")
    
    return info


def setup_competition(args) -> bool:
    """Setup competition: download, analyze, and prepare files."""
    if not download_data(args.competition, args.data_path):
        return False
    
    info = analyze_data(args.competition, args.data_path)
    if not info:
        return False
    
    # Get detailed competition description from web scraping
    web_description = get_competition_description(args.competition)
    
    # Create enhanced problem prompt
    if web_description:
        prompt = f"""Kaggle Competition: {info['competition_id']}

=== COMPETITION OVERVIEW ===
{web_description}

=== LOCAL DATA ANALYSIS ===
Predict the following target(s) for each record in test.csv:
{chr(10).join(f'- {target}' for target in info['target_columns'])}

Available files:
{chr(10).join(f'- ./{file_info}' for file_info in info['available_files'])}

Feature columns in test.csv:
{', '.join(info['feature_columns'])}

Problem type detected: {info['problem_type']}

=== ENVIRONMENT CONSTRAINTS ===
CPU-only; deterministic; no internet or package installs.
Use Python 3.10+, numpy, pandas, scikit-learn (optionally lightgbm/xgboost if available)."""
    else:
        # Fallback to basic prompt if scraping failed
        prompt = f"""Kaggle Competition: {info['competition_id']}

Objective:
Predict the following target(s) for each record in test.csv:
{chr(10).join(f'- {target}' for target in info['target_columns'])}

Available files:
{chr(10).join(f'- ./{file_info}' for file_info in info['available_files'])}

Feature columns in test.csv:
{', '.join(info['feature_columns'])}

Environment constraints:
CPU-only; deterministic; no internet or package installs.
Use Python 3.10+, numpy, pandas, scikit-learn (optionally lightgbm/xgboost if available)."""
    
    # Save files
    comp_dir = Path(args.data_path) / args.competition
    with open(comp_dir / "analysis.json", 'w') as f:
        json.dump(info, f, indent=2)
    with open(comp_dir / "problem_prompt.txt", 'w') as f:
        f.write(prompt)
    
    # Also save the web description separately if available
    if web_description:
        with open(comp_dir / "competition_description.txt", 'w') as f:
            f.write(web_description)
        print(f"💾 Saved web-scraped description to competition_description.txt")
    
    print(f"\n📋 Problem prompt:\n{prompt}")
    print(f"\n✅ Ready! All files in {comp_dir}")
    print(f"📂 Data files: train.csv, test.csv")
    print(f"📄 Analysis: analysis.json, problem_prompt.txt")
    return True


def main():
    parser = argparse.ArgumentParser(description="Simple Kaggle contest grabber")
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List competitions')
    list_parser.add_argument('--search', help='Search term')
    
    # Setup command  
    setup_parser = subparsers.add_parser('setup', help='Download and setup competition')
    setup_parser.add_argument('--competition', help='Competition ID (e.g., titanic)')
    setup_parser.add_argument('--data-path', default='./data', help='Data directory (default: ./data)')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    if args.command == 'list':
        list_competitions(args.search)
    elif args.command == 'setup':
        setup_competition(args)


if __name__ == "__main__":
    main()
