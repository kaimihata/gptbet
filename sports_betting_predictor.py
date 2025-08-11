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
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from openai import AzureOpenAI
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


class AzureGPTClient:
    """Client for interacting with Azure OpenAI."""
    
    def __init__(self, endpoint: str, api_key: str, api_version: str, deployment: str):
        """Initialize the Azure OpenAI client.
        
        Args:
            endpoint: The Azure OpenAI endpoint URL.
            api_key: The API key for Azure OpenAI.
            api_version: The API version to use.
            deployment: The deployment name.
        """
        self.client = AzureOpenAI(
            api_version=api_version,
            azure_endpoint=endpoint,
            api_key=api_key
        )
        self.deployment = deployment
        
    def get_prediction(self, game_data: Dict[str, Any], min_bet: float = MIN_BET_PERCENTAGE, max_bet: float = MAX_BET_PERCENTAGE) -> Dict[str, Any]:
        """Get a betting prediction for a game.
        
        Args:
            game_data: Dictionary containing game information and odds.
            
        Returns:
            Dictionary containing the prediction results.
        """
        # Prepare the prompt for the GPT model
        prompt = self._create_prompt(game_data, min_bet=min_bet, max_bet=max_bet)
        
        # Make the request to the Azure OpenAI
        try:
            response = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a sports betting analysis assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_completion_tokens=300,  # Reduced token limit to avoid errors
                model=self.deployment
            )
            print(f"API response received successfully")
        except Exception as e:
            print(f"Error calling Azure OpenAI API: {e}")
            # Return a basic error structure
            return {
                "home_team": game_data.get("home_team", "Unknown"),
                "away_team": game_data.get("away_team", "Unknown"),
                "commence_time": game_data.get("commence_time", "Unknown"),
                "predicted_winner": game_data.get("home_team", "Unknown"),  # Default to home team
                "bet_type": "moneyline",
                "bet_amount_percentage": 1.0,
                "confidence": "low",
                "reasoning": f"API error: {str(e)}"
            }
        
        # Extract the prediction from the response
        return self._parse_prediction(response, game_data)
    
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
        
        # Extract odds information
        bookmakers = game_data.get("bookmakers", [])
        odds_info = ""
        
        if bookmakers:
            # Get the first bookmaker's odds
            bookmaker = bookmakers[0]
            markets = bookmaker.get("markets", [])
            
            for market in markets:
                market_name = market.get("key", "")
                outcomes = market.get("outcomes", [])
                
                odds_info += f"\n{market_name.upper()} ODDS:\n"
                for outcome in outcomes:
                    name = outcome.get("name", "")
                    price = outcome.get("price", "")
                    point = outcome.get("point", "")
                    
                    if point != "":
                        odds_info += f"  {name}: {point} @ {price}\n"
                    else:
                        odds_info += f"  {name}: {price}\n"
        
        # Create a more concise prompt to avoid token limit issues
        prompt = f"""Analyze this game and provide a betting prediction:

{away_team} at {home_team} ({commence_time})
{odds_info}
Provide a betting prediction with:
- predicted_winner: Team name
- bet_type: moneyline/spread/over_under
- bet_amount_percentage: {min_bet}-{max_bet}%
- confidence: low/medium/high
- reasoning: Brief explanation (1-2 sentences)

Be concise."""
        
        return prompt
    
    def _extract_odds_info(self, game_data: Dict[str, Any]) -> str:
        """Extract odds information from game data.
        
        Args:
            game_data: Dictionary containing game information and odds.
            
        Returns:
            String containing formatted odds information.
        """
        odds_info = ""
        
        # Extract moneyline (h2h) odds if available
        if "bookmakers" in game_data and game_data["bookmakers"]:
            bookmaker = game_data["bookmakers"][0]  # Use the first bookmaker
            bookmaker_name = bookmaker.get("title", "Unknown")
            odds_info += f"Bookmaker: {bookmaker_name}\n"
            
            for market in bookmaker.get("markets", []):
                market_key = market.get("key")
                
                if market_key == "h2h":
                    odds_info += "Moneyline Odds:\n"
                    for outcome in market.get("outcomes", []):
                        team = outcome.get("name", "Unknown")
                        price = outcome.get("price", "Unknown")
                        odds_info += f"  - {team}: {price}\n"
                
                elif market_key == "spreads":
                    odds_info += "Point Spread Odds:\n"
                    for outcome in market.get("outcomes", []):
                        team = outcome.get("name", "Unknown")
                        point = outcome.get("point", "Unknown")
                        price = outcome.get("price", "Unknown")
                        odds_info += f"  - {team} {point}: {price}\n"
                
                elif market_key == "totals":
                    odds_info += "Totals (Over/Under) Odds:\n"
                    for outcome in market.get("outcomes", []):
                        name = outcome.get("name", "Unknown")  # Over or Under
                        point = outcome.get("point", "Unknown")
                        price = outcome.get("price", "Unknown")
                        odds_info += f"  - {name} {point}: {price}\n"
        
        return odds_info
    
    def _parse_prediction(self, gpt_response, game_data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse the GPT prediction response.
        
        Args:
            gpt_response: The response from the GPT model.
            game_data: Original game data.
            
        Returns:
            Dictionary containing the parsed prediction.
        """
        try:
            # Extract the content from the GPT response
            content = gpt_response.choices[0].message.content
            print(f"Response content: {content[:100]}...")  # Print first 100 chars for debugging
            
            # Try to extract JSON from the content
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
                "bet_amount_percentage": 1.0,
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
            "bet_amount_percentage": 1.0,
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
        
        # Try to extract bet type
        bet_types = {"moneyline": ["moneyline", "money line", "ml"], 
                    "spread": ["spread", "point spread", "against the spread", "ats"], 
                    "over/under": ["over/under", "over under", "total", "o/u"]}
        
        for bet_type, terms in bet_types.items():
            for term in terms:
                if term in text.lower():
                    prediction["bet_type"] = bet_type
                    break
        
        # Try to extract confidence level
        confidence_levels = {"high": ["high confidence", "confident", "strong", "certain"], 
                           "medium": ["medium confidence", "moderate", "reasonable"], 
                           "low": ["low confidence", "uncertain", "risky"]}
        
        for level, terms in confidence_levels.items():
            for term in terms:
                if term in text.lower():
                    prediction["confidence"] = level
                    break
        
        # Try to extract bet amount percentage
        import re
        percentage_match = re.search(r'(\d+(\.\d+)?)\s*%', text)
        if percentage_match:
            try:
                percentage = float(percentage_match.group(1))
                if 0.5 <= percentage <= 10:  # Sanity check
                    prediction["bet_amount_percentage"] = percentage
            except ValueError:
                pass
        
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
        "Bet Amount Percentage", "Confidence", "Reasoning"
    ]
    
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for prediction in predictions:
            home_team = prediction.get('home_team', 'Unknown')
            away_team = prediction.get('away_team', 'Unknown')
            game = f"{away_team} at {home_team}"
            
            # Format the bet amount percentage to 1 decimal place
            bet_amount = prediction.get('bet_amount_percentage', 0)
            if isinstance(bet_amount, (int, float)):
                bet_amount = f"{bet_amount:.1f}"
                
            # Truncate reasoning if it's too long
            reasoning = prediction.get('reasoning', '')
            if reasoning and len(reasoning) > 150:
                reasoning = reasoning[:147] + '...'
            
            writer.writerow({
                'Game': game,
                'Commence Time': prediction.get('commence_time', 'Unknown'),
                'Predicted Winner': prediction.get('predicted_winner', 'Unknown'),
                'Bet Type': prediction.get('bet_type', 'Unknown'),
                'Bet Amount Percentage': bet_amount,
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
    parser.add_argument("--sport", default=DEFAULT_SPORT, help=f"Sport to get odds for (default: {DEFAULT_SPORT})")
    parser.add_argument("--regions", default=DEFAULT_REGIONS, help=f"Regions for odds format (default: {DEFAULT_REGIONS})")
    parser.add_argument("--markets", default=DEFAULT_MARKETS, help=f"Markets to fetch (default: {DEFAULT_MARKETS})")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_FILE, help=f"Output CSV file (default: {DEFAULT_OUTPUT_FILE})")
    parser.add_argument("--limit", type=int, default=DEFAULT_GAME_LIMIT, help=f"Limit the number of games to process (default: {DEFAULT_GAME_LIMIT})")
    parser.add_argument("--min-bet", type=float, default=MIN_BET_PERCENTAGE, help=f"Minimum bet as percentage of bankroll (default: {MIN_BET_PERCENTAGE}%)")
    parser.add_argument("--max-bet", type=float, default=MAX_BET_PERCENTAGE, help=f"Maximum bet as percentage of bankroll (default: {MAX_BET_PERCENTAGE}%)")
    
    args = parser.parse_args()
    
    try:
        # Initialize clients
        odds_client = OddsAPIClient(args.odds_api_key)
        gpt_client = AzureGPTClient(
            endpoint=args.azure_endpoint,
            api_key=args.azure_api_key,
            api_version=args.azure_api_version,
            deployment=args.azure_deployment
        )
        
        print(f"Fetching odds for {args.sport}...")
        games, api_usage = odds_client.get_odds(args.sport, args.regions, args.markets)
        
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
            bet_type = prediction.get("bet_type", "Unknown")
            confidence = prediction.get("confidence", "low")
            bet_amount = prediction.get("bet_amount_percentage", 1.0)
            print(f"  Prediction: {predicted_winner} to win, {bet_type} bet ({confidence} confidence, {bet_amount:.1f}%)")
            
            predictions.append(prediction)
        
        # Write predictions to CSV
        write_predictions_to_csv(predictions, args.output)
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
