"""
JAX Summarization Agent with DuckDuckGo Browser Search
Run: python agent.py
"""

# ── 1. Imports ────────────────────────────────────────────────────────────
import sys
import requests
from bs4 import BeautifulSoup
from urllib.parse import quote_plus
from transformers import AutoTokenizer, FlaxAutoModelForSeq2SeqLM

# ── 2. Load JAX summarization model ──────────────────────────────────────
MODEL_NAME = "facebook/bart-large-cnn"

print("🔄 Loading summarization model (first run may take a minute)...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model     = FlaxAutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
print("✅ Model ready!\n")


# ── 3. DuckDuckGo search ──────────────────────────────────────────────────
def search_duckduckgo(query: str, max_results: int = 5) -> list[dict]:
    """
    Scrapes DuckDuckGo HTML results.
    Returns a list of {title, url, snippet} dicts.
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }
    url  = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
    resp = requests.get(url, headers=headers, timeout=10)
    soup = BeautifulSoup(resp.text, "html.parser")

    results = []
    for tag in soup.select(".result__body")[:max_results]:
        title_tag   = tag.select_one(".result__title a")
        snippet_tag = tag.select_one(".result__snippet")
        if not title_tag:
            continue
        results.append({
            "title":   title_tag.get_text(strip=True),
            "url":     title_tag.get("href", ""),
            "snippet": snippet_tag.get_text(strip=True) if snippet_tag else "No preview",
        })
    return results


# ── 4. Fetch article text from URL ────────────────────────────────────────
def fetch_article(url: str, max_chars: int = 4000) -> str:
    """
    Fetches a webpage and extracts readable paragraph text.
    """
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(resp.text, "html.parser")

        # Remove noise
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()

        paragraphs = soup.find_all("p")
        text = " ".join(p.get_text(strip=True) for p in paragraphs)
        return text[:max_chars] if text else "Could not extract text from this page."
    except Exception as e:
        return f"Error fetching page: {e}"


# ── 5. JAX summarize ──────────────────────────────────────────────────────
def summarize(text: str) -> str:
    inputs = tokenizer(
        text,
        max_length=1024,
        truncation=True,
        padding="max_length",
        return_tensors="jax",
    )
    ids = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        num_beams=4,
        min_length=56,
        max_length=142,
        length_penalty=2.0,
        early_stopping=True,
    ).sequences
    return tokenizer.decode(ids[0], skip_special_tokens=True)


# ── 6. Display results & let user pick ───────────────────────────────────
def display_results(results: list[dict]) -> int:
    print("\n📋 Search Results:")
    print("─" * 60)
    for i, r in enumerate(results, 1):
        print(f"  [{i}] {r['title']}")
        print(f"      {r['snippet'][:100]}...")
        print()
    print("─" * 60)

    while True:
        try:
            choice = input(f"Pick a result to summarize [1-{len(results)}] or 0 to search again: ")
            choice = int(choice)
            if 0 <= choice <= len(results):
                return choice
        except ValueError:
            pass
        print("  ⚠️  Please enter a valid number.")


# ── 7. Main agent loop ────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  🤖 JAX Search & Summarize Agent")
    print("  Powered by DuckDuckGo + BART (Flax/JAX)")
    print("=" * 60)

    while True:
        print()
        query = input("🔍 What do you want to search? (or 'quit' to exit): ").strip()

        if query.lower() in ("quit", "exit", "q"):
            print("\n👋 Bye!")
            sys.exit(0)

        if not query:
            continue

        print(f"\n⏳ Searching DuckDuckGo for: '{query}' ...")
        results = search_duckduckgo(query)

        if not results:
            print("❌ No results found. Try a different query.")
            continue

        choice = display_results(results)

        if choice == 0:
            continue  # search again

        selected = results[choice - 1]
        print(f"\n🌐 Fetching: {selected['url']}")
        article_text = fetch_article(selected["url"])

        if len(article_text) < 100:
            print("⚠️  Not enough text found on that page. Try another result.")
            continue

        print("⚙️  Summarizing with JAX...")
        summary = summarize(article_text)

        print("\n" + "=" * 60)
        print(f"📄 SUMMARY — {selected['title']}")
        print("=" * 60)
        print(summary)
        print("=" * 60)


if __name__ == "__main__":
    main()