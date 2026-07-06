"""
Fetches Airbnb Help Center pages and saves them as .txt files in data/policies/.
Then builds the FAISS vector store index.

Usage:
    python ingest_policies.py

Requires:  pip install requests beautifulsoup4
"""
import requests
from bs4 import BeautifulSoup
from pathlib import Path
import time

OUTPUT_DIR = Path("data/policies")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Key Airbnb Help Center pages covering common customer service topics
POLICY_URLS = {
    "cancellation_flexible":    "https://www.airbnb.com/help/article/149",
    "cancellation_moderate":    "https://www.airbnb.com/help/article/150",
    "cancellation_strict":      "https://www.airbnb.com/help/article/151",
    "refunds_guests":           "https://www.airbnb.com/help/article/544",
    "refunds_hosts":            "https://www.airbnb.com/help/article/279",
    "guest_refund_policy":      "https://www.airbnb.com/help/article/2062",
    "extenuating_circumstances":"https://www.airbnb.com/help/article/1320",
    "how_to_cancel":            "https://www.airbnb.com/help/article/796",
    "check_in_instructions":    "https://www.airbnb.com/help/article/1249",
    "resolution_center":        "https://www.airbnb.com/help/article/767",
    "house_rules":              "https://www.airbnb.com/help/article/2517",
    "service_fees":             "https://www.airbnb.com/help/article/1857",
    "payment_methods":          "https://www.airbnb.com/help/article/126",
    "security_deposit":         "https://www.airbnb.com/help/article/140",
    "review_policy":            "https://www.airbnb.com/help/article/2673",
}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; AirbnbPolicyBot/1.0)"
}


def fetch_page(url: str) -> str | None:
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        # Remove nav, header, footer, scripts
        for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
            tag.decompose()

        # Extract main content
        main = soup.find("main") or soup.find("article") or soup.body
        text = main.get_text(separator="\n", strip=True) if main else ""
        return text
    except Exception as e:
        print(f"  ⚠️  Failed to fetch {url}: {e}")
        return None


def main():
    print("Fetching Airbnb Help Center pages…\n")
    saved = 0
    for name, url in POLICY_URLS.items():
        print(f"  → {name}")
        text = fetch_page(url)
        if text and len(text) > 100:
            out_path = OUTPUT_DIR / f"{name}.txt"
            out_path.write_text(f"Source: {url}\n\n{text}", encoding="utf-8")
            saved += 1
        time.sleep(1.0)    # polite crawl rate

    print(f"\n✅ Saved {saved}/{len(POLICY_URLS)} policy files to {OUTPUT_DIR}")

    # Build FAISS index
    print("\nBuilding FAISS vector store…")
    from app.rag.ingest import build_vector_store
    build_vector_store()


if __name__ == "__main__":
    main()
