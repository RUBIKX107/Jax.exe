# 🤖 JAX.exe — Search & Summarize Agent

A terminal-based AI agent that searches the web via DuckDuckGo and 
summarizes articles using a JAX/Flax powered NLP model.

---

## ✨ Features

- 🔍 DuckDuckGo search — no API key needed
- 📋 Interactive result picker in the terminal
- 📄 Auto article extraction from any webpage
- ⚡ Summarization powered by JAX/Flax (BART-large-CNN)
- 🔁 Continuous loop — search as many times as you want

---

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/Jax.exe.git
cd Jax.exe
```

### 2. Install dependencies
```bash
/usr/bin/python3 -m pip install -r requirements.txt
```

### 3. Run the agent
```bash
/usr/bin/python3 agent.py
```

---

## 📋 Requirements

| Package | Version |
|---|---|
| jax | 0.4.23 |
| jaxlib | 0.4.23 |
| flax | 0.7.5 |
| transformers | 4.38.2 |
| requests | latest |
| beautifulsoup4 | latest |

> ⚠️ Version pinning is important — JAX breaks between minor versions.

---

## 💻 How It Works

```
You type a query
      ↓
DuckDuckGo returns 5 results
      ↓
You pick one [1-5]
      ↓
Agent fetches the full article
      ↓
JAX/BART summarizes it
      ↓
Summary printed in terminal
```

---

## 🔮 Roadmap

- [ ] GPU support via `jax[cuda12_pip]`
- [ ] Web UI instead of terminal
- [ ] Support multiple search engines
- [ ] Summarize multiple results at once
- [ ] Export summaries to .txt or .md

---

## 🛠️ Built With

- [JAX](https://github.com/google/jax) — high performance numerical computing
- [Flax](https://github.com/google/flax) — neural networks on top of JAX
- [HuggingFace Transformers](https://huggingface.co/docs/transformers) — BART model
- [DuckDuckGo](https://duckduckgo.com) — privacy-friendly search
- [BeautifulSoup4](https://www.crummy.com/software/BeautifulSoup/) — web scraping

---

## ⚠️ Known Issues

- Some websites block scraping — try a different result if extraction fails
- First run is slow while BART model downloads (~1.6GB)
- Use `/usr/bin/python3` explicitly on GitHub Codespaces to avoid Python path conflicts

---

## 📄 License

MIT License — free to use and modify.