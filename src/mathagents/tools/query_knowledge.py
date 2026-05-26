"""A tool for querying OEIS, Wikipedia, Wolfram, and other non-contaminated sources."""

import csv
import os
import re
import time

import requests
from loguru import logger


def query_knowledge(query: str, source: str) -> str:
    if source not in sources_to_fn:
        return f"UNKNOWN SOURCE: {source}, CANNOT QUERY KNOWLEDGE"
    fn = sources_to_fn[source]
    return fn(query)


def wolfram_query(query: str) -> str:
    api_key = os.getenv("WOLFRAM_APP_ID")
    if not api_key:
        return (
            "Error querying Wolfram Alpha: WOLFRAM_APP_ID is not set. "
            "Set it in the environment (e.g. via secrets.env) to enable this tool."
        )
    url = "https://www.wolframalpha.com/api/v1/llm-api"
    params = {"input": query, "appid": api_key}
    response = requests.get(url, params=params)
    if response.status_code != 200:
        logger.warning(f"Wolfram Alpha query failed with status code {response.status_code}")
        return "Error querying Wolfram Alpha."
    result = response.text
    return result


def oeis_query(query: str) -> str:
    url = f"https://oeis.org/search?q={query}&fmt=json"
    response = requests.get(url)
    if response.status_code != 200:
        logger.warning(f"OEIS query failed with status code {response.status_code}")
        return "Error querying OEIS."
    data = response.json()
    if not data or len(data) == 0:
        return "No results found in OEIS."
    top_result = data[0]
    result = "Top result from OEIS.\n"
    result += f"##### Sequence: {top_result.get('data', '')}\n\n"
    result += f"##### Name: {top_result.get('name', '')}\n\n"
    result += f"##### Formula: {top_result.get('formula', '')}\n\n"
    result += f"##### Example: {top_result.get('example', '')}\n\n"
    result += f"##### Comment: {top_result.get('comment', '')}\n\n"
    return result


def wikipedia_query(query: str) -> str:
    session = requests.Session()
    headers = {"User-Agent": "mathagents/0.1"}
    url = "https://en.wikipedia.org/w/api.php"
    params = {"action": "query", "format": "json", "list": "search", "srsearch": query}
    response = session.get(url, params=params, headers=headers)
    if response.status_code != 200:
        logger.warning(f"Wikipedia query failed with status code {response.status_code}")
        return "Error querying Wikipedia."
    data = response.json()
    if "query" not in data or "search" not in data["query"] or not data["query"]["search"]:
        return "No results found on Wikipedia."
    top_result = data["query"]["search"][0]

    # Get the full content of the top result page
    page_id = top_result["pageid"]
    content_params = {
        "action": "query",
        "format": "json",
        "pageids": page_id,
        "prop": "extracts",
        "explaintext": True,
    }
    content_response = session.get(url, params=content_params, headers=headers)
    content_data = content_response.json()

    if "query" not in content_data or "pages" not in content_data["query"]:
        return f"Found result: {top_result['title']}, but could not retrieve content."

    page_content = content_data["query"]["pages"][str(page_id)]
    result = f"Top result from Wikipedia: {top_result['title']}\n\n"
    extract = page_content.get("extract", "No content available.")

    # Clean up excessive whitespace from LaTeX/MathML markup
    # Remove LaTeX displaystyle markup
    extract = re.sub(r"\s*\{\\displaystyle[^}]*\}\s*", " ", extract)

    # Remove lines that are just whitespace or short fragments (likely LaTeX markup)
    lines = extract.split("\n")
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        # Keep lines that are either empty (paragraph breaks) or have substantial content
        # Skip lines that are just a number, operator, or single character from LaTeX
        if stripped == "" or len(stripped) > 3 or (len(stripped) > 0 and stripped[0].isalpha() and len(stripped) > 1):
            cleaned_lines.append(line)

    extract = "\n".join(cleaned_lines)

    # Now reduce multiple consecutive newlines
    extract = re.sub(r"\n", "", extract)  # Remove all newlines
    extract = re.sub(r"  +", " ", extract)  # Reduce multiple spaces to single space
    extract = extract.strip()

    result += extract
    return result


sources_to_fn = {"oeis": oeis_query, "wikipedia": wikipedia_query, "wolfram": wolfram_query}
