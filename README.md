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

### Environment Variables

Ensure a `.env` file exists with required keys:

- `ODDS_API_KEY` – The Odds API key
- `AZURE_ENDPOINT` – Azure OpenAI endpoint URL (if using Azure)
- `AZURE_API_KEY` – Azure OpenAI API key (if using Azure)
- `AZURE_API_VERSION` – Azure OpenAI API version (optional; default set in script)
- `AZURE_DEPLOYMENT` – Azure deployment name (e.g., `o4-mini`)
- `OPENAI_API_KEY` – OpenAI API key (if using OpenAI provider)
- `OPENAI_MODEL` – OpenAI model name (optional; defaults to `o4-mini`)
- `LLM_PROVIDER` – Default provider (`azure` or `openai`), can be overridden by CLI flag

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
- `--offline`: Do not call the Odds API; use cached odds from `--cache-file` instead
- `--refresh-cache`: Force refresh the cached odds by calling the Odds API and saving the response
- `--cache-file`: Path to the cache file for odds data (default: `cache/odds_last_run.json`)
- `--provider`: Select LLM provider (`azure` or `openai`); overrides `LLM_PROVIDER` env var

### Example

```bash
python sports_betting_predictor.py --odds-api-key abc123 --azure-endpoint https://your-endpoint.openai.azure.com/openai/deployments/your-deployment/chat/completions --azure-api-key xyz789 --sport basketball_nba --limit 5
```

## Output

### Output CSV Format

The script writes a `betting_predictions.csv` file with these columns:

- Game: The matchup (Away Team at Home Team)
- Commence Time: When the game starts
- Predicted Winner: The team predicted to win
- Bet Type: Moneyline (assumed)
- Confidence: Confidence level of the prediction (low, medium, high)
- Reasoning: Brief explanation for the prediction

Notes:
- Bet Type is assumed to be moneyline (the model does not output bet_type or bet amount).

## Prompts

All LLM prompts are stored under the `prompts/` directory as plain text files. Each file contains a single prompt and begins with a leading `Description:` line followed by a blank line, then the prompt content.

Current prompt files:

- `prompts/system_betting_assistant.txt` — System message that sets a neutral, simulation-oriented assistant persona.
- `prompts/betting_user_prompt.txt` — Template for the user message used to request a prediction for a single game. The script fills placeholders like `$home_team`, `$away_team`, `$commence_time`, and `$odds_info`.

The model is instructed to return a strict JSON object containing only: `predicted_winner`, `confidence`, and `reasoning`. Fields for `bet_type` and `bet_amount_percentage` were removed; the application assumes moneyline.

To tweak behavior, edit these files and re-run the script—no code changes required.

## Sports Available

The Odds API supports various sports. Use "upcoming" to get games across all sports, or specify a particular sport like:

- basketball_nba
- soccer_epl
- baseball_mlb
- hockey_nhl
- mma_ufc

For a complete list, check the [Odds API documentation](https://the-odds-api.com/liveapi/guides/v4/).
