import os
import urllib.parse
import time
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

def get_currency_components(symbol: str) -> list[str]:
    """Extract currency names from symbol e.g. EURUSD -> [Euro, U.S. Dollar]."""
    mapping = {
        "EUR": "Euro ECB",
        "USD": "USD Federal Reserve Fed FOMC",
        "GBP": "GBP Bank of England BoE",
        "JPY": "JPY Bank of Japan BoJ",
        "AUD": "AUD Reserve Bank of Australia RBA",
        "NZD": "NZD Reserve Bank of New Zealand RBNZ",
        "CAD": "CAD Bank of Canada BoC",
        "CHF": "CHF Swiss National Bank SNB",
        "XAU": "Gold XAU"
    }
    
    parts = []
    if len(symbol) >= 3:
        curr1 = symbol[:3].upper()
        if curr1 in mapping: parts.append(mapping[curr1])
    if len(symbol) >= 6:
        curr2 = symbol[3:6].upper()
        if curr2 in mapping: parts.append(mapping[curr2])
        
    return parts

def get_news_headlines(symbol: str, limit: int = 15) -> list[str]:
    """Fetch headlines from multiple sources (Google, ForexLive, DailyFX)."""
    components = get_currency_components(symbol)
    query = f"{symbol} {' '.join(components)}"
    encoded_query = urllib.parse.quote(query)
    
    # Combined RSS feeds
    feeds = [
        f"https://news.google.com/rss/search?q={encoded_query}&hl=en-US&gl=US&ceid=US:en",
        "https://www.forexlive.com/feed/news",
        "https://www.dailyfx.com/feeds/forex-market-news"
    ]
    
    headlines = []
    for rss_url in feeds:
        try:
            feed = feedparser.parse(rss_url)
            for entry in feed.entries:
                # Filter for relevance if it's a general feed
                if "google.com" not in rss_url:
                    if symbol[:3] in entry.title.upper() or symbol[3:6] in entry.title.upper() or any(c.split()[0] in entry.title.upper() for c in components):
                        headlines.append(entry.title)
                else:
                    headlines.append(entry.title)
        except Exception as e:
            print(f"Error fetching RSS {rss_url}: {e}")
            
    return list(set(headlines))[:limit] # Return unique headlines

def get_headline_weight(headline: str) -> float:
    """Calculate importance weight based on policy-shifting keywords."""
    macro_keywords = ["FED", "FOMC", "POWELL", "ECB", "LAGARDE", "BOE", "BOJ", "INTEREST RATE", "INFLATION", "CPI", "NFP", "DECISION", "POLICY"]
    headline_upper = headline.upper()
    
    if any(word in headline_upper for word in macro_keywords):
        return 2.5 # High importance for central bank and macro data
    return 1.0 # Normal technical news

def analyze_headline_with_retry(headline: str, max_retries: int = 3, delay: float = 1.0) -> tuple[float, float]:
    """Analyze headline sentiment with exponential backoff retry logic."""
    for attempt in range(max_retries):
        try:
            results = client.text_classification(
                headline,
                model="ProsusAI/finbert",
            )
            
            prob_pos = 0.0
            prob_neg = 0.0
            
            # Parse the results
            if hasattr(results, "__iter__"):
                for res in results:
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
            
            return prob_pos, prob_neg
            
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = delay * (2 ** attempt)  # Exponential backoff
                print(f"Retry {attempt + 1}/{max_retries} for headline after {wait_time:.1f}s: {str(e)[:100]}")
                time.sleep(wait_time)
            else:
                raise e

@app.get("/sentiment", response_model=SentimentResponse)
def get_sentiment(symbol: str = Query(..., description="Forex symbol to analyze, e.g., EURUSD")):
    if not client:
        # Fallback if no HF_TOKEN is configured (for debugging)
        print("WARNING: HF_TOKEN not set. Returning neutral sentiment.")
        return SentimentResponse(symbol=symbol, sentiment_score=0.0, headlines_processed=0)

    headlines = get_news_headlines(symbol, limit=20)
    
    if not headlines:
        return SentimentResponse(symbol=symbol, sentiment_score=0.0, headlines_processed=0)
        
    weighted_total = 0.0
    total_weight = 0.0
    processed = 0
    failed = 0
    
    for idx, headline in enumerate(headlines):
        try:
            weight = get_headline_weight(headline)
            
            # Add small delay between requests to avoid overwhelming API
            if idx > 0:
                time.sleep(0.3)
            
            # FinBERT classification with retry logic
            prob_pos, prob_neg = analyze_headline_with_retry(headline)
            
            sentiment = prob_pos - prob_neg
            weighted_total += (sentiment * weight)
            total_weight += weight
            processed += 1
            print(f"[{'MACRO' if weight > 1.0 else 'TECH'}] Processed: '{headline[:80]}...' | Sentiment: {sentiment:.3f}")
            
        except Exception as e:
            failed += 1
            print(f"Failed after retries ({failed}): {headline[:80]}... - {str(e)[:100]}")
            continue

    if processed == 0:
        print(f"WARNING: No headlines processed successfully. {failed} failed.")
        return SentimentResponse(symbol=symbol, sentiment_score=0.0, headlines_processed=0)

    # Weighted average sentiment
    avg_score = weighted_total / total_weight if total_weight > 0 else 0.0
    # Cap between -1 and 1
    avg_score = max(-1.0, min(1.0, avg_score))
    
    print(f"Summary: {processed} processed, {failed} failed, final score: {avg_score:.3f}")
    
    return SentimentResponse(
        symbol=symbol,
        sentiment_score=avg_score,
        headlines_processed=processed
    )

@app.get("/")
def health_check():
    return {"status": "ok", "message": "FinBERT Sentiment API is running."}
