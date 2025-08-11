# Sports Betting Predictor

This script fetches upcoming sports games and betting odds from the Odds API, processes the data through an Azure AI Foundry GPT endpoint to get betting predictions, and outputs the results to a CSV file.

## Requirements

- Python 3.6+
- An API key from [The Odds API](https://the-odds-api.com/)
- An Azure AI Foundry GPT endpoint URL and API key

## Installation

1. Clone this repository
2. Install the required packages:

```bash
pip install -r requirements.txt
```

## Usage

Run the script with the required parameters:

```bash
python sports_betting_predictor.py --odds-api-key YOUR_ODDS_API_KEY --azure-endpoint YOUR_AZURE_ENDPOINT --azure-api-key YOUR_AZURE_API_KEY
```

### Optional Parameters

- `--sport`: Sport to get odds for (default: "upcoming")
- `--regions`: Regions for odds format (default: "us")
- `--markets`: Markets to fetch (default: "h2h,spreads,totals")
- `--output`: Output CSV file (default: "betting_predictions.csv")
- `--limit`: Limit the number of games to process

### Example

```bash
python sports_betting_predictor.py --odds-api-key abc123 --azure-endpoint https://your-endpoint.openai.azure.com/openai/deployments/your-deployment/chat/completions --azure-api-key xyz789 --sport basketball_nba --limit 5
```

## Output

The script generates a CSV file with the following columns:

- Game: The matchup (Away Team at Home Team)
- Commence Time: When the game starts
- Predicted Winner: The team predicted to win
- Bet Type: Type of bet recommended (moneyline, spread, over/under)
- Bet Amount Percentage: Recommended bet amount as percentage of bankroll
- Confidence: Confidence level of the prediction (low, medium, high)
- Reasoning: Brief explanation for the prediction

## Sports Available

The Odds API supports various sports. Use "upcoming" to get games across all sports, or specify a particular sport like:

- basketball_nba
- soccer_epl
- baseball_mlb
- hockey_nhl
- mma_ufc

For a complete list, check the [Odds API documentation](https://the-odds-api.com/liveapi/guides/v4/).
