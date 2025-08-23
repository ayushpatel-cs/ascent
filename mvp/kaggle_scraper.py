#!/usr/bin/env python3
"""Kaggle Competition Web Scraper - Extracts and summarizes competition details"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Any

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup

from config import chat


def _setup_chrome_options() -> Options:
    """Configure Chrome options for headless scraping with anti-detection."""
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option('useAutomationExtension', False)
    return options


def _extract_title(soup: BeautifulSoup, slug: str) -> str:
    """Extract competition title from the page."""
    # Try title tag first
    title_elem = soup.find('title')
    if title_elem:
        title = title_elem.get_text(strip=True)
        if title != "Kaggle":  # Avoid generic title
            return title
    
    # Try various title selectors
    for selector in ['h1', '[data-testid*="title"]', '.competition-title', 'h1[class*="title"]']:
        title_elem = soup.select_one(selector)
        if title_elem:
            page_title = title_elem.get_text(strip=True)
            if len(page_title) > len(slug):
                return page_title
    
    return slug


def scrape_competition(slug: str, debug: bool = False) -> Dict[str, Any]:
    """Scrape competition page for detailed information using Selenium.
    
    Args:
        slug: Competition identifier (e.g., 'titanic')
        debug: Whether to save HTML debug file
        
    Returns:
        Dictionary containing slug, title, url, and raw_html
    """
    url = f"https://www.kaggle.com/competitions/{slug}"
    options = _setup_chrome_options()
    
    driver = None
    try:
        driver = webdriver.Chrome(options=options)
        
        # Disable automation detection
        driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        
        driver.get(url)
        
        # Wait for page to load
        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
        
        # Wait for JavaScript content to render
        time.sleep(8)
        
        # Scroll to trigger lazy loading
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(3)
        driver.execute_script("window.scrollTo(0, 0);")
        time.sleep(2)
        
        page_source = driver.page_source
        soup = BeautifulSoup(page_source, "html.parser")
        
        if debug:
            debug_file = f'debug_{slug}.html'
            with open(debug_file, 'w', encoding='utf-8') as f:
                f.write(page_source)
            print(f"Debug HTML saved to {debug_file}")
        
        title = _extract_title(soup, slug)
        
        return {
            'slug': slug,
            'title': title,
            'url': url,
            'raw_html': page_source
        }
        
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return {}
    finally:
        if driver:
            driver.quit()





def summarize_with_llm(competition_data: Dict[str, Any]) -> str:
    """Extract competition details from raw HTML using LLM.
    
    Args:
        competition_data: Dictionary containing competition info and raw_html
        
    Returns:
        Structured competition summary or error message
    """
    if 'raw_html' not in competition_data:
        return f"""No HTML content available for processing.

Competition: {competition_data['title']}
URL: {competition_data['url']}"""
    
    prompt = f"""You are analyzing a Kaggle competition page. Extract the key information and create a concise problem description.

Competition URL: {competition_data['url']}
Competition Title: {competition_data['title']}

Please extract and organize:

1. **Objective**: What is the main goal/task?
2. **Problem Type**: Classification, regression, etc.?
3. **Evaluation Metric**: How are submissions scored?
4. **Data Files**: What datasets are provided and their purpose?
5. **Submission Format**: Required output format and constraints
6. **Key Details**: Any important rules, constraints, or background

Focus on information that would help an ML engineer understand and approach this problem. Be concise but comprehensive.

HTML Content:
{competition_data['raw_html']}"""

    try:
        response = chat([
            {"role": "system", "content": "You are an expert ML engineer who extracts and summarizes competition details from web pages. Create clear, structured problem descriptions."},
            {"role": "user", "content": prompt}
        ], max_tokens=2000, temperature=0.1)
        
        return response.choices[0].message.content
        
    except Exception as e:
        print(f"LLM summarization failed: {e}")
        return f"""LLM processing failed: {e}

Competition: {competition_data['title']}
URL: {competition_data['url']}

Raw HTML available ({len(competition_data['raw_html'])} chars) but could not process with LLM."""


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Scrape and summarize Kaggle competitions")
    parser.add_argument("slug", help="Competition slug (e.g., 'titanic')")
    parser.add_argument("--output", "-o", help="Output file path")
    parser.add_argument("--raw", action="store_true", help="Output raw scraped data instead of LLM summary")
    parser.add_argument("--debug", action="store_true", help="Save debug HTML file")
    
    args = parser.parse_args()
    
    print(f"Scraping competition: {args.slug}")
    
    # Scrape the competition
    competition_data = scrape_competition(args.slug, debug=args.debug)
    
    if not competition_data:
        print("Failed to scrape competition data")
        return 1
    
    # Generate summary
    if args.raw:
        # For raw output, don't include the massive HTML
        raw_data = {k: v for k, v in competition_data.items() if k != 'raw_html'}
        raw_data['html_length'] = len(competition_data.get('raw_html', ''))
        summary = json.dumps(raw_data, indent=2)
        print("\nRaw scraped data (HTML excluded):")
    else:
        print("Generating LLM summary...")
        summary = summarize_with_llm(competition_data)
        print("\nCompetition Summary:")
    
    print(summary)
    
    # Save to file if requested
    if args.output:
        output_path = Path(args.output)
        output_path.write_text(summary)
        print(f"\nSaved to {output_path}")
    
    return 0


if __name__ == "__main__":
    exit(main())
