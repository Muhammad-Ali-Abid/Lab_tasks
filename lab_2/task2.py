import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import spacy
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import stats
import os

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

nlp = spacy.load("en_core_web_sm")

def extract_features(text):
    """Word/sentence/POS counts for one article."""
    doc = nlp(str(text))
    sentences = list(doc.sents)
    num_tokens = len(doc)
    num_nouns = len([t for t in doc if t.pos_ == 'NOUN'])
    return pd.Series({
        'num_tokens':          num_tokens,                             
        'num_sentences':       len(sentences),
        'num_entities':        len(doc.ents),                          
        'num_nouns':           num_nouns,
        'num_verbs':           len([t for t in doc if t.pos_ == 'VERB']),
        'avg_sentence_length': (num_tokens / len(sentences)) if sentences else 0.0,
        'noun_density':        (num_nouns / num_tokens) if num_tokens else 0.0,
    })


features = corpus['text'].apply(extract_features)
corpus = pd.concat([corpus, features], axis=1)
corpus['title_length'] = corpus['title'].apply(lambda t: len(str(t)))

print("\nLinguistic features:")
print(corpus[['source', 'title', 'num_tokens', 'num_sentences', 'num_entities',
              'num_nouns', 'num_verbs', 'avg_sentence_length', 'noun_density']])

def get_sentiment(text):
    blob = TextBlob(str(text))
    return pd.Series({
        'sentiment_polarity':     blob.sentiment.polarity,
        'sentiment_subjectivity': blob.sentiment.subjectivity
    })


sentiment_features = corpus['text'].apply(get_sentiment)
corpus = pd.concat([corpus, sentiment_features], axis=1)

corpus.to_csv("two_source_corpus_features.csv", index=False)
print(f"\nSaved two_source_corpus_features.csv with {corpus.shape[1]} columns")
print(corpus[['source', 'title', 'sentiment_polarity', 'sentiment_subjectivity']])

os.makedirs("figures", exist_ok=True)
plt.style.use('ggplot')

count_features = ['num_tokens', 'num_sentences', 'num_nouns', 'num_verbs']
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for ax, feat in zip(axes.flat, count_features):
    sns.histplot(data=corpus, x=feat, hue='source', kde=True, bins=8, ax=ax)
    ax.set_title(f'Distribution of {feat}')
plt.tight_layout()
plt.savefig('figures/01_count_distributions.png', dpi=150)
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.histplot(data=corpus, x='sentiment_polarity', hue='source', kde=True, bins=8, ax=axes[0])
axes[0].set_title('Sentiment Polarity by Source')
axes[0].set_xlabel('Polarity Score (-1.0 to 1.0)')
sns.histplot(data=corpus, x='sentiment_subjectivity', hue='source', kde=True, bins=8, ax=axes[1])
axes[1].set_title('Sentiment Subjectivity by Source')
axes[1].set_xlabel('Subjectivity Score (0.0 to 1.0)')
plt.tight_layout()
plt.savefig('figures/02_sentiment_distributions.png', dpi=150)
plt.show()

numeric_cols = ['title_length', 'num_tokens', 'num_sentences', 'num_entities',
                'num_nouns', 'num_verbs', 'noun_density', 'sentiment_polarity',
                'sentiment_subjectivity']
sns.pairplot(corpus[numeric_cols + ['source']], hue='source')
plt.suptitle('Pair Plot of All Derived Features, by Source', y=1.02)
plt.savefig('figures/03_pairplot_all_features.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nCorrelation matrix (all articles pooled):")
print(corpus[numeric_cols].corr().round(2))

means = corpus.groupby('source')[['sentiment_polarity', 'noun_density']].mean().reset_index()
sems = corpus.groupby('source')[['sentiment_polarity', 'noun_density']].sem().reset_index()
plot_df = means.melt(id_vars='source', var_name='metric', value_name='mean').merge(
    sems.melt(id_vars='source', var_name='metric', value_name='sem'), on=['source', 'metric'])

fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(data=plot_df, x='metric', y='mean', hue='source', ax=ax)
for i, bar in enumerate(ax.patches):
    if i < len(plot_df):
        row = plot_df.iloc[i]
        ax.errorbar(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     yerr=row['sem'], color='black', capsize=4, fmt='none')
ax.set_title('Mean Sentiment Polarity & Noun Density by Source (±1 SEM)')
ax.set_ylabel('Mean value')
ax.set_xlabel('')
plt.tight_layout()
plt.savefig('figures/04_grouped_bar_comparison.png', dpi=150)
plt.show()

vectorizer = TfidfVectorizer(stop_words='english', max_features=15)
for src in corpus['source'].unique():
    subset = corpus.loc[corpus['source'] == src, 'text']
    X = vectorizer.fit_transform(subset)
    tfidf_df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
    top_terms = tfidf_df.sum().sort_values(ascending=False)

    plt.figure(figsize=(9, 4))
    sns.barplot(x=top_terms.index, y=top_terms.values)
    plt.title(f"Top TF-IDF Terms — {src}")
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("TF-IDF score sum")
    plt.tight_layout()
    plt.savefig(f'figures/05_tfidf_{src}.png', dpi=150)
    plt.show()

a = corpus[corpus['source'] == 'blog-a']
b = corpus[corpus['source'] == 'blog-b']


def welch_test(col):
    t_stat, p_val = stats.ttest_ind(a[col], b[col], equal_var=False)
    pooled_sd = ((a[col].std() ** 2 + b[col].std() ** 2) / 2) ** 0.5
    cohens_d = (a[col].mean() - b[col].mean()) / pooled_sd if pooled_sd else float('nan')
    return {
        'feature': col,
        'mean_blog_a': a[col].mean(), 'mean_blog_b': b[col].mean(),
        'mean_diff (a-b)': a[col].mean() - b[col].mean(),
        't_stat': t_stat, 'p_value': p_val, 'cohens_d': cohens_d,
    }


results = pd.DataFrame([welch_test('sentiment_polarity'), welch_test('noun_density')]).round(4)
print("\nStatistical comparison (blog-a vs blog-b):")
print(results)
results.to_csv("stat_comparison.csv", index=False)