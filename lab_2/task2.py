import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

SOURCE_A = {
    "name": "blog-a",
    "urls": [
        'https://www.jpmorganchase.com/about/technology/blog/ai-is-making-social-engineering-easier-than-ever',
        'https://www.jpmorganchase.com/about/technology/blog/the-future-engineer-is-a-problem-framer',
        'https://www.jpmorganchase.com/about/technology/blog/one-qubit-and-a-few-springs',
        'https://www.jpmorganchase.com/about/technology/blog/scaling-community-led-learning-at-enterprise-level-in-the-age-of-ai',
        'https://www.jpmorganchase.com/about/technology/blog/ambient-payments',
        'https://www.jpmorganchase.com/about/technology/blog/key-takeaways-from-innovation-week-2026',
        'https://www.jpmorganchase.com/about/technology/blog/oqc-jpmc-amd-quantum-ai-platform',
        'https://www.jpmorganchase.com/about/technology/blog/ai-is-changing-cyber-threats',
        'https://www.jpmorganchase.com/about/technology/blog/quantum-resistant-distributed-ledger',
        'https://www.jpmorganchase.com/about/technology/blog/fence-framework'
    ],
    "title_selector":   ("h3", "post-title"),
    "content_selector": ("div", "post-body entry-content"),
}

SOURCE_B = {
    "name": "blog-b",
    "urls": [
        'https://rightmove.blog/13-things-we-learned-from-two-ai-developer-tools-workshops-using-just-github-copilot-agent-mode/',
        'https://rightmove.blog/from-legacy-to-resilience-how-event-driven-systems-improved-efficiency-at-rightmove/',
        'https://rightmove.blog/how-to-make-your-react-tests-resilient-to-change/',
        'https://rightmove.blog/learning-new-skills-in-my-10-time/',
        'https://rightmove.blog/rightmoves-journey-to-cloud/',
        'https://rightmove.blog/our-journey-into-the-world-of-containers-so-far/',
        'https://rightmove.blog/design-cheat-sheet-for-devs/',
        'https://rightmove.blog/how-we-optimise-browser-performance-using-differential-serving-and-conditional-polyfilling/',
        'https://rightmove.blog/improving-password-integrity-with-the-haveibeenpwned-api/',
        'https://rightmove.blog/making-android-ui-tests-more-readable/'
    ],
    "title_selector":   ("h1", None),
    "content_selector": ("div", None),
}

def scrape_source(source):
    """Scrape one source and return a labelled DataFrame."""
    rows = []

    for url in source["urls"]:
        try:
            res = requests.get(url, headers=headers, timeout=15)
            res.raise_for_status()

            soup = BeautifulSoup(res.content, "lxml")

            title = soup.title.get_text(strip=True) if soup.title else "No title"

            block = (
                soup.find("div", class_="text")
                or soup.find("article")
                or soup.find("main")
            )



            if not block:
                print(f"[NO CONTENT] {url}")
                continue

            paragraphs = block.find_all("p")
            text = " ".join(
                p.get_text(" ", strip=True)
                for p in paragraphs
            )

            if len(text) > 100:
                rows.append({
                    "source": source["name"],
                    "url": url,
                    "title": title,
                    "text": text
                })

                print(f"[OK] {source['name']} -> {title}")

            else:
                print(f"[SHORT CONTENT] {url} ({len(text)} characters)")

        except Exception as e:
            print(f"[ERROR] {url} -> {e}")

        time.sleep(1)

    return pd.DataFrame(rows)

corpus = pd.concat([scrape_source(SOURCE_A), scrape_source(SOURCE_B)], ignore_index=True)
corpus.to_csv("two_source_corpus.csv", index=False)
print(corpus)