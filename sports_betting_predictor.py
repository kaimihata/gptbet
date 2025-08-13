#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Sports Betting Predictor Script

This script fetches upcoming sports games and odds from the Odds API,
processes the data through an Azure AI Foundry GPT endpoint to get betting predictions,
and outputs the results to a CSV file.
"""

import os
import sys
import csv
import json
import requests
import argparse
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Tuple
from openai import AzureOpenAI, OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configuration constants

# Sport Configuration
# Options include: 'baseball_mlb', 'basketball_nba', 'icehockey_nhl', etc.
# Use 'upcoming' for all sports or specify a particular sport
DEFAULT_SPORT = "baseball_mlb"  

# Odds Configuration
DEFAULT_REGIONS = "us"  # Regions for odds format (us, uk, eu, etc.)
DEFAULT_MARKETS = "h2h,spreads,totals"  # Markets to fetch (h2h = moneyline, spreads, totals)
DEFAULT_OUTPUT_FILE = "betting_predictions.csv"
DEFAULT_ODDS_CACHE_FILE = "cache/odds_last_run.json"

# Betting Configuration
MIN_BET_PERCENTAGE = 1.0  # Minimum bet as percentage of bankroll
MAX_BET_PERCENTAGE = 5.0  # Maximum bet as percentage of bankroll
DEFAULT_GAME_LIMIT = 5  # Default number of games to process

# API Configuration - Load from environment variables
# You can override these with command-line arguments if needed
ODDS_API_KEY = os.getenv("ODDS_API_KEY", "")  # Get from https://the-odds-api.com/

# Azure OpenAI Configuration
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT", "")  # Azure OpenAI endpoint URL
AZURE_API_KEY = os.getenv("AZURE_API_KEY", "")  # Azure OpenAI API key
AZURE_API_VERSION = os.getenv("AZURE_API_VERSION", "2024-12-01-preview")  # Azure OpenAI API version
AZURE_DEPLOYMENT = os.getenv("AZURE_DEPLOYMENT", "o4-mini")  # Azure OpenAI deployment name

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "o4-mini")

class OddsAPIClient:
    """Client for interacting with the Odds API."""
    
    BASE_URL = "https://api.the-odds-api.com/v4"
    
    def __init__(self, api_key: str):
        """Initialize the Odds API client.
        
        Args:
            api_key: The API key for the Odds API.
        """
        self.api_key = api_key
        
    def get_sports(self) -> List[Dict[str, Any]]:
        """Get a list of available sports.
        
        Returns:
            List of sports dictionaries.
        """
        url = f"{self.BASE_URL}/sports"
        params = {"apiKey": self.api_key}
        
        response = requests.get(url, params=params)
        response.raise_for_status()
        
        return response.json()
    
    def get_odds(self, sport: str, regions: str, markets: str) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
        """Get odds for a specific sport.
        
        Args:
            sport: The sport key (e.g., 'upcoming', 'basketball_nba').
            regions: Comma-delimited list of regions for the odds format.
            markets: Comma-delimited list of markets to get odds for.
            
        Returns:
            Tuple containing list of games with odds and API usage information.
        """
        url = f"{self.BASE_URL}/sports/{sport}/odds"
        params = {
            "apiKey": self.api_key,
            "regions": regions,
            "markets": markets,
            "dateFormat": "iso"
        }
        
        response = requests.get(url, params=params)
        response.raise_for_status()
        
        # Extract API usage information from headers
        remaining = int(response.headers.get('x-requests-remaining', 0))
        used = int(response.headers.get('x-requests-used', 0))
        
        return response.json(), {"remaining": remaining, "used": used}


def _ensure_parent_dir(path: str) -> None:
    """Ensure the parent directory exists for the given file path."""
    parent = os.path.dirname(path)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)


def save_cached_odds(cache_file: str, games: List[Dict[str, Any]], api_usage: Dict[str, int], sport: str, regions: str, markets: str) -> None:
    """Save the odds API response to a local cache file with metadata."""
    try:
        _ensure_parent_dir(cache_file)
        payload = {
            "metadata": {
                "saved_at": datetime.now(timezone.utc).isoformat(),
                "sport": sport,
                "regions": regions,
                "markets": markets,
                "api_usage": api_usage,
            },
            "games": games,
        }
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Warning: failed to save cached odds to '{cache_file}': {e}")


def load_cached_odds(cache_file: str) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Load odds data from a local cache file."""
    try:
        with open(cache_file, "r", encoding="utf-8") as f:
            payload = json.load(f)
        games = payload.get("games", [])
        api_usage = payload.get("metadata", {}).get("api_usage", {"used": 0, "remaining": 0})
        return games, api_usage
    except FileNotFoundError:
        raise FileNotFoundError(f"Cache file not found at '{cache_file}'. Run once without --offline to populate the cache.")
    except Exception as e:
        raise RuntimeError(f"Failed to load cache from '{cache_file}': {e}")


def load_games_from_predictions_csv(csv_file: str) -> List[Dict[str, Any]]:
    """Reconstruct minimal game objects from a prior predictions CSV.
    
    This uses only 'Game' and 'Commence Time' columns and leaves odds empty.
    """
    games: List[Dict[str, Any]] = []
    try:
        with open(csv_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                game_str = row.get("Game", "")
                commence_time = row.get("Commence Time", "")
                # Parse format: "Away Team at Home Team"
                if " at " in game_str:
                    away_team, home_team = game_str.split(" at ", 1)
                else:
                    # Fallback if unexpected format
                    parts = game_str.split()
                    away_team = parts[0] if parts else "Unknown"
                    home_team = parts[-1] if parts else "Unknown"
                games.append({
                    "home_team": home_team.strip() or "Unknown",
                    "away_team": away_team.strip() or "Unknown",
                    "commence_time": commence_time or "Unknown",
                    # No odds information in CSV fallback
                    "bookmakers": []
                })
    except FileNotFoundError:
        raise FileNotFoundError(f"Predictions CSV not found at '{csv_file}'.")
    except Exception as e:
        raise RuntimeError(f"Failed to reconstruct games from CSV '{csv_file}': {e}")
    return games


class AzureGPTClient:
    """Client for interacting with LLM providers (Azure OpenAI or OpenAI)."""
    
    def __init__(self, endpoint: str, api_key: str, api_version: str, deployment: str,
                 provider: str = "azure", openai_api_key: Optional[str] = None, openai_model: Optional[str] = None):
        """Initialize the LLM client.
        
        Args:
            endpoint: The Azure OpenAI endpoint URL (used when provider='azure').
            api_key: The API key for Azure OpenAI (used when provider='azure').
            api_version: The API version to use (azure only).
            deployment: The Azure deployment name (azure only).
            provider: 'azure' or 'openai'.
            openai_api_key: API key for OpenAI (used when provider='openai').
            openai_model: Model name for OpenAI (default: 'o4-mini').
        """
        self.provider = (provider or "azure").lower()
        self.debug: bool = False
        
        if self.provider == "azure":
            self.client = AzureOpenAI(
                api_version=api_version,
                azure_endpoint=endpoint,
                api_key=api_key
            )
            self.deployment = deployment
            self.openai_client = None
            self.openai_model = None
        elif self.provider == "openai":
            self.openai_client = OpenAI(api_key=openai_api_key or OPENAI_API_KEY)
            self.openai_model = openai_model or OPENAI_MODEL
            self.client = None
            self.deployment = None
        else:
            raise ValueError(f"Unsupported provider: {self.provider}. Use 'azure' or 'openai'.")
        
        # Load prompts from the prompts/ directory
        self.prompts_dir = os.path.join(os.path.dirname(__file__), "prompts")
        # System prompt (persona)
        system_prompt = self._read_prompt_file("system_betting_assistant.txt")
        if not system_prompt.strip():
            system_prompt = "You are a sports betting analysis assistant."
        self.system_prompt = system_prompt
        
        # User prompt template
        user_template = self._read_prompt_file("betting_user_prompt.txt")
        if not user_template.strip():
            user_template = (
                "Analyze this matchup and provide a simulation-based recommendation:\n\n"
                "$away_team at $home_team ($commence_time)\n"
                "$odds_info\n\n"
                "Return ONLY a single JSON object (no prose, no backticks) with this structure:\n"
                "{\n"
                "  \"predicted_winner\": string,\n"
                "  \"confidence\": \"low\" | \"medium\" | \"high\",\n"
                "  \"reasoning\": string\n"
                "}\n\n"
                "Constraints:\n"
                "- Do not include any text outside the JSON object.\n"
                "- Keep reasoning concise (max 2 sentences).\n"
                "- Assume market is moneyline; do not include fields for bet_type or bet_amount_percentage.\n"
            )
        self.user_prompt_template = user_template
    
    def _read_prompt_file(self, filename: str) -> str:
        """Read a prompt file and strip the leading description section if present.
        
        The prompt files are expected to begin with a line starting with
        "Description:" followed by a blank line, then the prompt content.
        """
        try:
            path = os.path.join(self.prompts_dir, filename)
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
            lines = text.splitlines()
            if lines and lines[0].strip().lower().startswith("description:"):
                # Find first blank line separating description and content
                sep_index = None
                for i, line in enumerate(lines):
                    if line.strip() == "":
                        sep_index = i
                        break
                if sep_index is not None and sep_index + 1 < len(lines):
                    return "\n".join(lines[sep_index + 1:]).strip()
                else:
                    return ""  # Description only
            return text.strip()
        except Exception as e:
            print(f"Warning: failed to read prompt file '{filename}': {e}")
            return ""
        
    def get_prediction(self, game_data: Dict[str, Any], min_bet: float = MIN_BET_PERCENTAGE, max_bet: float = MAX_BET_PERCENTAGE) -> Dict[str, Any]:
        """Get a betting prediction for a game.
        
        Args:
            game_data: Dictionary containing game information and odds.
            
        Returns:
            Dictionary containing the prediction results.
        """
        # Prepare the prompt for the GPT model
        prompt = self._create_prompt(game_data, min_bet=min_bet, max_bet=max_bet)
        
        # Make the request to the selected provider
        try:
            content: str = ""
            if self.provider == "azure":
                # Attempt to enforce JSON output; fallback if not supported
                try:
                    response = self.client.chat.completions.create(
                        messages=[
                            {"role": "system", "content": self.system_prompt},
                            {"role": "user", "content": prompt}
                        ],
                        # max_completion_tokens=300,
                        model=self.deployment,
                        response_format={"type": "json_object"}
                    )
                except Exception as _json_enforce_err:
                    if self.debug:
                        print(f"Response format json_object not enforced ({_json_enforce_err}); retrying without enforcement...")
                    response = self.client.chat.completions.create(
                        messages=[
                            {"role": "system", "content": self.system_prompt},
                            {"role": "user", "content": prompt}
                        ],
                        # max_completion_tokens=300,
                        model=self.deployment
                    )
                content = response.choices[0].message.content
            else:  # openai
                # Try Chat Completions first
                try:
                    response = self.openai_client.chat.completions.create(
                        messages=[
                            {"role": "system", "content": self.system_prompt},
                            {"role": "user", "content": prompt}
                        ],
                        # max_completion_tokens=300,
                        model=self.openai_model,
                        response_format={"type": "json_object"}
                    )
                    content = response.choices[0].message.content
                except Exception as _chat_err:
                    if self.debug:
                        print(f"OpenAI chat.completions failed ({_chat_err}); trying responses API...")
                    # Fallback to Responses API
                    try:
                        response = self.openai_client.responses.create(
                            model=self.openai_model,
                            input=f"System:\n{self.system_prompt}\n\nUser:\n{prompt}",
                            response_format={"type": "json_object"}
                        )
                        content = getattr(response, "output_text", None) or ""
                        if not content:
                            # Attempt to build from output parts
                            parts = []
                            for item in getattr(response, "output", []) or []:
                                for c in getattr(item, "content", []) or []:
                                    text = getattr(c, "text", None)
                                    if text and getattr(text, "value", None):
                                        parts.append(text.value)
                            content = "\n".join(parts)
                    except Exception as _resp_err:
                        raise RuntimeError(f"OpenAI Responses API call failed: {_resp_err}")
            
            # Debug logging: full prompts and content
            if self.debug:
                try:
                    print("===== LLM Request: System =====")
                    print(self.system_prompt)
                    print("===== LLM Request: User =====")
                    print(prompt)
                    print("===== LLM Response (first 400 chars) =====\n")
                    print((content or "")[:400])
                    print("\n===== End LLM Exchange =====")
                except Exception:
                    pass
            print(f"API response received successfully")
        except Exception as e:
            print(f"Error calling LLM provider API: {e}")
            # Return a basic error structure
            return {
                "home_team": game_data.get("home_team", "Unknown"),
                "away_team": game_data.get("away_team", "Unknown"),
                "commence_time": game_data.get("commence_time", "Unknown"),
                "predicted_winner": game_data.get("home_team", "Unknown"),  # Default to home team
                "bet_type": "moneyline",
                "confidence": "low",
                "reasoning": f"API error: {str(e)}"
            }
        
        # Extract the prediction from the content
        return self._parse_prediction(content, game_data)
    
    def _create_prompt(self, game_data: Dict[str, Any], min_bet: float = 1.0, max_bet: float = 5.0) -> str:
        """Create a prompt for the GPT model.
        
        Args:
            game_data: Game data with odds.
            min_bet: Minimum bet as percentage of bankroll.
            max_bet: Maximum bet as percentage of bankroll.
            
        Returns:
            Prompt string for the GPT model.
        """
        home_team = game_data.get("home_team", "Unknown")
        away_team = game_data.get("away_team", "Unknown")
        commence_time = game_data.get("commence_time", "Unknown")
        
        # Extract odds information using helper
        odds_info = self._extract_odds_info(game_data)
        
        # Fill the template from prompts/betting_user_prompt.txt using string.Template ($placeholders)
        from string import Template
        prompt = Template(self.user_prompt_template).safe_substitute(
            away_team=away_team,
            home_team=home_team,
            commence_time=commence_time,
            odds_info=odds_info,
            min_bet=min_bet,
            max_bet=max_bet,
        )
        
        return prompt
    
    def _extract_odds_info(self, game_data: Dict[str, Any]) -> str:
        """Extract odds information from game data.
        
        Args:
            game_data: Dictionary containing game information and odds.
            
        Returns:
            String containing implied win percentages derived from market prices.
        """
        odds_info = ""
        
        # Extract implied probabilities from moneyline (h2h) prices if available
        if "bookmakers" in game_data and game_data["bookmakers"]:
            bookmaker = game_data["bookmakers"][0]  # Use the first bookmaker only
            # Find the h2h market
            h2h_market = None
            for market in bookmaker.get("markets", []):
                if market.get("key") == "h2h":
                    h2h_market = market
                    break
            
            if h2h_market:
                outcomes = h2h_market.get("outcomes", [])
                # Compute raw implied probabilities from decimal odds: p_raw = 1/price
                implied_raw = []
                for outcome in outcomes:
                    team = outcome.get("name", "Unknown")
                    price = outcome.get("price")
                    try:
                        price_val = float(price)
                        if price_val > 0:
                            implied_raw.append((team, 1.0 / price_val))
                    except (TypeError, ValueError):
                        continue
                # Normalize to sum to 1 (accounts for overround)
                total = sum(p for _, p in implied_raw)
                if total > 0 and implied_raw:
                    odds_info += "Implied win chances (baseline):\n"
                    for team, p in implied_raw:
                        pct = (p / total) * 100.0
                        odds_info += f"  - {team}: {pct:.1f}%\n"
        
        return odds_info
    
    def _parse_prediction(self, gpt_output, game_data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse the GPT prediction response.
        
        Args:
            gpt_response: The response from the GPT model.
            game_data: Original game data.
            
        Returns:
            Dictionary containing the parsed prediction.
        """
        try:
            # Determine content from either a raw string or a provider response object
            if isinstance(gpt_output, str):
                content = gpt_output
            else:
                try:
                    content = gpt_output.choices[0].message.content
                except Exception:
                    # Try OpenAI Responses API shape
                    content = getattr(gpt_output, "output_text", "") or ""
            print(f"Response content: {content[:100]}...")  # Print first 100 chars for debugging
            
            # First, try to parse the entire content as JSON (preferred path when prompt enforces JSON-only)
            text_only = content.strip()
            # In case the model wrapped the JSON in backticks despite instructions
            if text_only.startswith("```") and text_only.endswith("```"):
                text_only = text_only.strip("`")
                # Remove optional leading language tag like ```json
                if "\n" in text_only:
                    text_only = text_only.split("\n", 1)[1]
                text_only = text_only.strip()
            try:
                direct = json.loads(text_only)
                if isinstance(direct, dict):
                    prediction = direct
                    # Normalize defaults if fields are omitted in the LLM JSON
                    if not prediction.get("bet_type"):
                        prediction["bet_type"] = "moneyline"
                    # Add game information to the prediction
                    prediction["home_team"] = game_data.get("home_team", "Unknown")
                    prediction["away_team"] = game_data.get("away_team", "Unknown")
                    prediction["commence_time"] = game_data.get("commence_time", "Unknown")
                    return prediction
            except json.JSONDecodeError:
                pass
            
            # Try to extract JSON from within the content as a fallback
            # First, look for JSON block in markdown format
            import re
            json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                print(f"Found JSON in markdown block")
            else:
                # If no markdown JSON block, try to find JSON-like structure
                json_match = re.search(r'({.*})', content, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                    print(f"Found JSON-like structure")
                else:
                    # If we can't find JSON, create a structured prediction from the text
                    print(f"No JSON found, creating structured prediction from text")
                    return self._create_structured_prediction(content, game_data)
            
            # Parse the JSON
            prediction = json.loads(json_str)
            # Normalize defaults if fields are omitted in the LLM JSON
            if not prediction.get("bet_type"):
                prediction["bet_type"] = "moneyline"
            
            # Add game information to the prediction
            prediction["home_team"] = game_data.get("home_team", "Unknown")
            prediction["away_team"] = game_data.get("away_team", "Unknown")
            prediction["commence_time"] = game_data.get("commence_time", "Unknown")
            
            return prediction
        except (json.JSONDecodeError, IndexError, KeyError, AttributeError) as e:
            print(f"Error parsing GPT response: {e}")
            # Try to create a structured prediction from the text
            if 'content' in locals() and content:
                return self._create_structured_prediction(content, game_data)
            
            # If all else fails, return a basic structure with error information
            return {
                "home_team": game_data.get("home_team", "Unknown"),
                "away_team": game_data.get("away_team", "Unknown"),
                "commence_time": game_data.get("commence_time", "Unknown"),
                "predicted_winner": game_data.get("home_team", "Unknown"),  # Default to home team
                "bet_type": "moneyline",
                "confidence": "low",
                "reasoning": f"Failed to parse prediction: {str(e)}"
            }
            
    def _create_structured_prediction(self, text: str, game_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a structured prediction from unstructured text response.
        
        Args:
            text: The text response from the GPT model.
            game_data: Original game data.
            
        Returns:
            Dictionary containing a structured prediction.
        """
        # Default values
        prediction = {
            "home_team": game_data.get("home_team", "Unknown"),
            "away_team": game_data.get("away_team", "Unknown"),
            "commence_time": game_data.get("commence_time", "Unknown"),
            "predicted_winner": "",
            "bet_type": "moneyline",
            "confidence": "low",
            "reasoning": ""
        }
        
        # Extract predicted winner - check if home or away team is mentioned more prominently
        home_team = game_data.get("home_team", "")
        away_team = game_data.get("away_team", "")
        
        if home_team and away_team:
            # Count mentions of each team
            home_mentions = text.lower().count(home_team.lower())
            away_mentions = text.lower().count(away_team.lower())
            
            # Look for win/victory/better/recommend phrases near team names
            win_terms = ["win", "victory", "better", "recommend", "favor", "advantage", "pick", "choose"]
            
            for term in win_terms:
                # Check if win terms are within 10 words of team name
                for i in range(len(text) - 20):
                    snippet = text[i:i+20].lower()
                    if term in snippet and home_team.lower() in snippet:
                        home_mentions += 2
                    if term in snippet and away_team.lower() in snippet:
                        away_mentions += 2
            
            # Determine predicted winner based on mentions
            if home_mentions > away_mentions:
                prediction["predicted_winner"] = home_team
            else:
                prediction["predicted_winner"] = away_team
        else:
            # Default to home team if we can't determine
            prediction["predicted_winner"] = home_team if home_team else away_team
        
        # Assume moneyline market; do not parse bet type from text
        
        # Try to extract confidence level
        confidence_levels = {"high": ["high confidence", "confident", "strong", "certain"], 
                           "medium": ["medium confidence", "moderate", "reasonable"], 
                           "low": ["low confidence", "uncertain", "risky"]}
        
        for level, terms in confidence_levels.items():
            for term in terms:
                if term in text.lower():
                    prediction["confidence"] = level
                    break
        
        # Use the first 200 characters as reasoning
        prediction["reasoning"] = text[:200].strip()
        
        return prediction


def write_predictions_to_csv(predictions: List[Dict[str, Any]], output_file: str):
    """Write predictions to a CSV file.
    
    Args:
        predictions: List of prediction dictionaries.
        output_file: Path to the output CSV file.
    """
    if not predictions:
        print("No predictions to write to CSV.")
        return
    
    # Define the CSV columns based on the prediction structure
    fieldnames = [
        "Game", "Commence Time", "Predicted Winner", "Bet Type", 
        "Confidence", "Reasoning"
    ]
    
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for prediction in predictions:
            home_team = prediction.get('home_team', 'Unknown')
            away_team = prediction.get('away_team', 'Unknown')
            game = f"{away_team} at {home_team}"
            
            # Truncate reasoning if it's too long
            reasoning = prediction.get('reasoning', '')
            if reasoning and len(reasoning) > 150:
                reasoning = reasoning[:147] + '...'
            
            writer.writerow({
                'Game': game,
                'Commence Time': prediction.get('commence_time', 'Unknown'),
                'Predicted Winner': prediction.get('predicted_winner', 'Unknown'),
                'Bet Type': prediction.get('bet_type', 'Unknown'),
                'Confidence': prediction.get('confidence', 'Unknown'),
                'Reasoning': reasoning
            })
            
    print(f"Predictions written to {output_file}")


def main():
    """Main function to run the sports betting predictor."""
    parser = argparse.ArgumentParser(description="Sports Betting Predictor")
    
    parser.add_argument("--odds-api-key", default=ODDS_API_KEY, help="API key for the Odds API (overrides the one in script)")
    parser.add_argument("--azure-endpoint", default=AZURE_ENDPOINT, help="Azure OpenAI endpoint URL (overrides the one in script)")
    parser.add_argument("--azure-api-key", default=AZURE_API_KEY, help="API key for Azure OpenAI (overrides the one in script)")
    parser.add_argument("--azure-api-version", default=AZURE_API_VERSION, help="Azure OpenAI API version (overrides the one in script)")
    parser.add_argument("--azure-deployment", default=AZURE_DEPLOYMENT, help="Azure OpenAI deployment name (overrides the one in script)")
    parser.add_argument("--provider", choices=["azure", "openai"], default=os.getenv("LLM_PROVIDER", "azure"), help="LLM provider to use: 'azure' or 'openai' (default: azure)")
    parser.add_argument("--openai-api-key", default=OPENAI_API_KEY, help="API key for OpenAI (overrides the one in environment)")
    parser.add_argument("--openai-model", default=OPENAI_MODEL, help="OpenAI model name (default from env or 'o4-mini')")
    parser.add_argument("--sport", default=DEFAULT_SPORT, help=f"Sport to get odds for (default: {DEFAULT_SPORT})")
    parser.add_argument("--regions", default=DEFAULT_REGIONS, help=f"Regions for odds format (default: {DEFAULT_REGIONS})")
    parser.add_argument("--markets", default=DEFAULT_MARKETS, help=f"Markets to fetch (default: {DEFAULT_MARKETS})")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_FILE, help=f"Output CSV file (default: {DEFAULT_OUTPUT_FILE})")
    parser.add_argument("--limit", type=int, default=DEFAULT_GAME_LIMIT, help=f"Limit the number of games to process (default: {DEFAULT_GAME_LIMIT})")
    parser.add_argument("--min-bet", type=float, default=MIN_BET_PERCENTAGE, help=f"Minimum bet as percentage of bankroll (default: {MIN_BET_PERCENTAGE}%)")
    parser.add_argument("--max-bet", type=float, default=MAX_BET_PERCENTAGE, help=f"Maximum bet as percentage of bankroll (default: {MAX_BET_PERCENTAGE}%)")
    parser.add_argument("--offline", action="store_true", help="Do not call the Odds API; use cached odds from --cache-file")
    parser.add_argument("--refresh-cache", action="store_true", help="Force refresh the cached odds by calling the Odds API and saving the response")
    parser.add_argument("--cache-file", default=DEFAULT_ODDS_CACHE_FILE, help=f"Path to cache file for odds data (default: {DEFAULT_ODDS_CACHE_FILE})")
    parser.add_argument("--debug", action="store_true", help="Log full LLM prompts and responses for debugging")
    
    args = parser.parse_args()
    
    try:
        # Initialize clients
        odds_client = OddsAPIClient(args.odds_api_key)
        gpt_client = AzureGPTClient(
            endpoint=args.azure_endpoint,
            api_key=args.azure_api_key,
            api_version=args.azure_api_version,
            deployment=args.azure_deployment,
            provider=args.provider,
            openai_api_key=args.openai_api_key,
            openai_model=args.openai_model,
        )
        gpt_client.debug = args.debug
        
        print(f"Fetching odds for {args.sport}...")
        # Decide how to obtain odds data (offline cache vs API)
        if args.offline and not args.refresh_cache:
            print("Offline mode: loading odds from cache...")
            try:
                games, api_usage = load_cached_odds(args.cache_file)
            except Exception as e:
                print(f"Cache not available ({e}). Attempting to reconstruct from '{args.output}'...")
                games = load_games_from_predictions_csv(args.output)
                api_usage = {"used": 0, "remaining": 0}
                print(f"Loaded {len(games)} games from prior predictions CSV.")
        else:
            games, api_usage = odds_client.get_odds(args.sport, args.regions, args.markets)
            # Save to cache for future offline runs
            try:
                save_cached_odds(args.cache_file, games, api_usage, args.sport, args.regions, args.markets)
                print(f"Saved odds to cache: {args.cache_file}")
            except Exception:
                pass
        
        print(f"Found {len(games)} games. API usage: {api_usage['used']} used, {api_usage['remaining']} remaining.")
        
        # Limit the number of games if specified
        if args.limit and args.limit < len(games):
            print(f"Limited to processing {args.limit} games.")
            games = games[:args.limit]
        
        # Process each game with the GPT model
        predictions = []
        
        for i, game in enumerate(games, 1):
            home_team = game.get("home_team", "Unknown")
            away_team = game.get("away_team", "Unknown")
            print(f"Processing game {i}/{len(games)}: {away_team} at {home_team}")
            
            # Get prediction from GPT
            prediction = gpt_client.get_prediction(game, min_bet=args.min_bet, max_bet=args.max_bet)
            
            # Print a summary of the prediction
            predicted_winner = prediction.get("predicted_winner", "Unknown")
            bet_type = prediction.get("bet_type", "moneyline")
            confidence = prediction.get("confidence", "low")
            print(f"  Prediction: {predicted_winner} to win, {bet_type} ({confidence} confidence)")
            
            predictions.append(prediction)
        
        # Write predictions to CSV
        write_predictions_to_csv(predictions, args.output)
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
