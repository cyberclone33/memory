import json
import logging
import requests
from typing import List
from dotenv import load_dotenv
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential
from app.utils.prompts import MEMORY_CATEGORIZATION_PROMPT

load_dotenv()

import os

# Check if running in Docker container
if os.path.exists('/.dockerenv'):
    OLLAMA_BASE_URL = "http://host.docker.internal:11434"
else:
    OLLAMA_BASE_URL = "http://localhost:11434"

OLLAMA_MODEL = "llama3.1:latest"


class MemoryCategories(BaseModel):
    categories: List[str]


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=15))
def get_categories_for_memory(memory: str) -> List[str]:
    """Get categories for a memory using Ollama."""
    try:
        # Create the prompt for Ollama
        prompt = f"""{MEMORY_CATEGORIZATION_PROMPT}

Memory to categorize: "{memory}"

Please respond with ONLY a JSON object in this exact format:
{{"categories": ["category1", "category2", ...]}}"""

        # Make request to Ollama
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "temperature": 0,
                "stream": False,
                "format": "json"
            },
            timeout=30
        )
        
        if response.status_code != 200:
            raise Exception(f"Ollama API returned status code {response.status_code}: {response.text}")
        
        # Parse the response
        ollama_response = response.json()
        response_text = ollama_response.get("response", "").strip()
        
        # Parse JSON response
        try:
            response_json = json.loads(response_text)
            categories = response_json.get('categories', [])
        except json.JSONDecodeError:
            # Fallback: try to extract categories from text if JSON parsing fails
            logging.warning(f"Failed to parse JSON response from Ollama: {response_text}")
            # Try to find JSON-like content in the response
            import re
            json_match = re.search(r'\{.*?"categories".*?\[.*?\].*?\}', response_text, re.DOTALL)
            if json_match:
                try:
                    response_json = json.loads(json_match.group())
                    categories = response_json.get('categories', [])
                except:
                    categories = []
            else:
                categories = []
        
        # Clean up categories
        categories = [cat.strip().lower() for cat in categories if isinstance(cat, str) and cat.strip()]
        
        # TODO: Validate categories later may be
        return categories
        
    except Exception as e:
        logging.error(f"Error getting categories from Ollama: {e}")
        raise e
