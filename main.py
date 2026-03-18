import os
import feedparser
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from huggingface_hub import InferenceClient

app = FastAPI(title="FinBERT Sentiment API")

# Provide a fallback for local testing, but it requires HF_TOKEN in production
hf_token = os.environ.get("HF_TOKEN")

# Setup Hugging Face Client
# It will use HF_TOKEN from environment variables
client = InferenceClient(
    provider="hf-inference",
    api_key=hf_token
) if hf_token else None

class SentimentResponse(BaseModel):
    symbol: str
    sentiment_score: float
    headlines_processed: int

def get_news_headlines(symbol: str, limit: int = 5) -> list[str]:
    """Fetch top news headlines using Google News RSS."""
    # Append 'forex' or 'crypto' context to get better results
    query = f"{symbol}+forex"
    rss_url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
    
    feed = feedparser.parse(rss_url)
    
    headlines = []
    for entry in feed.entries[:limit]:
        headlines.append(entry.title)
        
    return headlines

@app.get("/sentiment", response_model=SentimentResponse)
def get_sentiment(symbol: str = Query(..., description="Forex symbol to analyze, e.g., EURUSD")):
    if not client:
        # Fallback if no HF_TOKEN is configured (for debugging)
        print("WARNING: HF_TOKEN not set. Returning neutral sentiment.")
        return SentimentResponse(symbol=symbol, sentiment_score=0.0, headlines_processed=0)

    headlines = get_news_headlines(symbol, limit=5)
    
    if not headlines:
        return SentimentResponse(symbol=symbol, sentiment_score=0.0, headlines_processed=0)
        
    total_score = 0.0
    processed = 0
    
    for headline in headlines:
        try:
            # FinBERT classification
            results = client.text_classification(
                headline,
                model="ProsusAI/finbert",
            )
            
            prob_pos = 0.0
            prob_neg = 0.0
            
            # Parse the results
            if hasattr(results, "__iter__"):
                for res in results:
                    # Depending on hugginface_hub version, results may be dicts or objects
                    if isinstance(res, dict):
                        label = res.get('label', '').lower()
                        score = res.get('score', 0.0)
                    else:
                        label = getattr(res, 'label', '').lower()
                        score = getattr(res, 'score', 0.0)
                        
                    if label == 'positive':
                        prob_pos = score
                    elif label == 'negative':
                        prob_neg = score
            
            sentiment = prob_pos - prob_neg
            total_score += sentiment
            processed += 1
            print(f"Processed: '{headline}' | Sentiment: {sentiment:.3f}")
            
        except Exception as e:
            print(f"Error processing headline: {headline} - {e}")
            continue

    if processed == 0:
        return SentimentResponse(symbol=symbol, sentiment_score=0.0, headlines_processed=0)

    # Average sentiment
    avg_score = total_score / processed
    # Cap between -1 and 1
    avg_score = max(-1.0, min(1.0, avg_score))
    
    return SentimentResponse(
        symbol=symbol,
        sentiment_score=avg_score,
        headlines_processed=processed
    )

@app.get("/")
def health_check():
    return {"status": "ok", "message": "FinBERT Sentiment API is running."}
