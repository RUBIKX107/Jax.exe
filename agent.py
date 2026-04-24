"""
JAX Search & Summarize Agent
Uses DuckDuckGo + PyTorch BART
Run: python3 agent.py
"""

import sys
print(f"🐍 Python: {sys.executable}")

import requests
from bs4 import BeautifulSoup
from urllib.parse import quote_plus
from transformers import pipeline

# ── Load model ────────────────────────────────────────────────────────────
print("\n🔄 Loading summarization model...")
summarizer = pipeline("text2text-generation", model="facebook/bart-large-cnn")
print("✅ Model ready!\n")


# ── DuckDuckGo search ─────────────────────────────────────────────────────
def search_duckduckgo(query: str, max_results: int = 5) -> list:
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


# ── Fetch article ─────────────────────────────────────────────────────────
def fetch_article(url: str, max_chars: int = 3000) -> str:
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()
        paragraphs = soup.find_all("p")
        text = " ".join(p.get_text(strip=True) for p in paragraphs)
        return text[:max_chars] if text else "Could not extract text."
    except Exception as e:
        return f"Error fetching page: {e}"


# ── Summarize ─────────────────────────────────────────────────────────────
def summarize(text: str) -> str:
    if len(text) < 100:
        return "Not enough text to summarize."
    result = summarizer(
        text,
        max_new_tokens=142,
        min_new_tokens=56,
        do_sample=False
    )
    return result[0]["generated_text"]


# ── Display results ───────────────────────────────────────────────────────
def display_results(results: list) -> int:
    print("\n📋 Search Results:")
    print("─" * 60)
    for i, r in enumerate(results, 1):
        print(f"  [{i}] {r['title']}")
        print(f"      {r['snippet'][:100]}...")
        print()
    print("─" * 60)

    while True:
        try:
            choice = input(f"Pick a result [1-{len(results)}] or 0 to search again: ")
            choice = int(choice)
            if 0 <= choice <= len(results):
                return choice
        except ValueError:
            pass
        print("  ⚠️  Please enter a valid number.")


# ── Main loop ─────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  🤖 JAX.exe — Search & Summarize Agent")
    print("  Powered by DuckDuckGo + BART")
    print("=" * 60)

    while True:
        print()
        query = input("🔍 Search: ").strip()

        if query.lower() in ("quit", "exit", "q"):
            print("\n👋 Bye!")
            sys.exit(0)

        if not query:
            continue

        print(f"\n⏳ Searching for: '{query}' ...")
        results = search_duckduckgo(query)

        if not results:
            print("❌ No results found. Try a different query.")
            continue

        choice = display_results(results)
        if choice == 0:
            continue

        selected = results[choice - 1]
        print(f"\n🌐 Fetching: {selected['url']}")
        text = fetch_article(selected["url"])

        print("⚙️  Summarizing...")
        summary = summarize(text)

        print("\n" + "=" * 60)
        print(f"📄 SUMMARY — {selected['title']}")
        print("=" * 60)
        print(summary)
        print("=" * 60)


if __name__ == "__main__":
    main()