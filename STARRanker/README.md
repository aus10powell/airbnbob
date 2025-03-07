# ListingRanker: STAR Implementation for Vacation Rental Recommendations

## Overview
This project implements the STAR (Simple Training-free Approach for Recommendations) methodology as described in the paper [STAR: A Simple Training-free Approach for Recommendations using Large Language Model](https://arxiv.org/pdf/2410.16458).

ListingRanker leverages large language models to perform pairwise comparisons of vacation rental listings based on user preferences and booking history, without requiring any model training.

## Features
- Uses a local LLM through Ollama (default: phi3)
- Performs pairwise ranking of listings based on user history
- Considers factors like location, price, cleanliness, and similarity to previous bookings
- Returns ranked recommendations in order of predicted user preference

## Requirements
- Python 3.x
- pandas
- ollama
- JSON

## Usage

```python
from listing_ranker import ListingRanker
import pandas as pd

# Load your data
reviews_df = pd.read_csv('path/to/airbnb_reviews.csv')
listings_df = pd.read_csv('path/to/airbnb_listings.csv')

# Get user history and candidate listings
user_history = reviews_df[reviews_df['id'] == USER_ID] 
candidate_listings = listings_df.head(10)  # Or any selection method

# Initialize ranker with desired LLM
ranker = ListingRanker(model='phi3')  # Can use other Ollama models

# Get ranked listings
ranked_recommendations = ranker.rank_listings(candidate_listings, user_history)

# Display results
for idx, (_, listing) in enumerate(ranked_recommendations.iterrows(), 1):
    print(f"Rank {idx}: {listing['name']} - ${listing['price']}/night")
```

## How It Works

1. **Prompt Construction**: Creates a structured prompt containing user history and candidate listings
2. **Pairwise Comparison**: Uses LLM to evaluate which listings would be preferred over others
3. **Ranking Output**: Parses the LLM's JSON response to extract the final ranking
4. **Result Reordering**: Returns the original listings dataframe reordered according to the ranking

## STAR Methodology

This implementation follows the STAR approach from the referenced paper:
- **S**imple: No fine-tuning or training required
- **T**raining-free: Uses pre-trained LLMs as-is
- **A**pproach for **R**ecommendations: Specialized for recommendation tasks

The key insight is that modern LLMs have sufficient knowledge to make meaningful comparisons between items based on user preferences, without needing explicit training on the recommendation task.