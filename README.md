# LangServe Translator API

### FastAPI + LangServe chain that translates text with Groq Gemma2-9b-It.

[![GitHub](https://img.shields.io/badge/repo-LangServe-Translator-API-181717?logo=github)](https://github.com/ArchanaChetan07/LangServe-Translator-API)
[![Language](https://img.shields.io/badge/language-Jupyter%20Notebook-3572A5)](https://github.com/ArchanaChetan07/LangServe-Translator-API)
[![License](https://img.shields.io/badge/license-See%20repository-yellow)](https://github.com/ArchanaChetan07/LangServe-Translator-API)
[![CI](https://img.shields.io/badge/CI-GitHub%20Actions-2088FF?logo=githubactions&logoColor=white)](https://github.com/ArchanaChetan07/LangServe-Translator-API/actions)

---

## Overview

Need a simple deployable translation HTTP API wrapping an LLM prompt chain.

LangChain ChatPromptTemplate | ChatGroq | StrOutputParser exposed via langserve.add_routes on FastAPI path /chain; uvicorn on port 8001; companion simplellm.ipynb.

Minimal translator microservice pattern with CI/pytest scaffold.

This repository is maintained as **production-minded portfolio work**: clear architecture, automated checks where present, and metrics that are **traceable to committed artifacts** (never invented).

---

## Architecture

Client â†’ FastAPI/LangServe /chain â†’ ChatPromptTemplate (system: Translate into {language}) â†’ ChatGroq Gemma2-9b-It â†’ string output.

```mermaid
sequenceDiagram
  participant C as Client
  participant API as FastAPI+LangServe
  participant G as ChatGroq Gemma2-9b-It
  C->>API: POST /chain (language, text)
  API->>G: prompt | model | parse
  G-->>API: translated text
  API-->>C: response
```

```mermaid
flowchart LR
  In[Inputs] --> Core[Processing]
  Core --> Out[Outputs / metrics]
  Out --> CI[CI artifacts]
```

---

## Results & repository facts

> Only values found in code, configs, tests, or generated reports are listed. Absence of a clinical/ML accuracy number means it was **not** published in-repo.

| Metric | Value | Source |
|---|---|---|
| Tracked repository files | **6** | `git tree` |
| Default serve port | **8001** | `server.py` |
| Tracked files | **6** | `git tree` |
| Python modules | **2** | `git tree` |
| Test-related paths | **1** | `git tree` |
| CI workflows | **Yes** | `.github/workflows` |
| Docker present | **No** | `repo root` |

```mermaid
%%{init: {'theme':'base'}}%%
pie showData title Language composition (bytes)
    "Jupyter Notebook" : 70
    "Python" : 30
```

---

## Key features

- Prompted multilingual translation chain
- LangServe /chain routes on FastAPI
- GROQ_API_KEY via dotenv
- Notebook experiment file

---

## Tech stack

| Layer | Technology |
|---|---|
| api | FastAPI |
| serving | LangServe |
| llm | LangChain |
| llm | ChatGroq |
| model | Gemma2-9b-It |
| server | Uvicorn |
| ci | GitHub Actions |

---

## Skills demonstrated

Jupyter Notebook · F · a · s · t · A · P · CI/CD · testing · automation

Keyword surface: **Python · Jupyter Notebook · machine-learning · CI/CD · testing · API · Docker · automation · data-science · software-engineering · system-design · observability · LLM · cloud**

---

## Project structure

```text
LangServe-Translator-API/
â”œâ”€â”€ server.py
â”œâ”€â”€ simplellm.ipynb
â”œâ”€â”€ requirements.txt
â”œâ”€â”€ tests/test_langserve_translator_api.py
â””â”€â”€ .github/workflows/ci.yml
```

---

## Installation & usage

```bash
git clone https://github.com/ArchanaChetan07/LangServe-Translator-API.git
cd LangServe-Translator-API
pip install -r requirements.txt
echo GROQ_API_KEY=... > .env
python server.py
```

---

## How it works

On startup, server.py loads GROQ_API_KEY, builds a translate-into-{language} chat prompt chain with Gemma2-9b-It, and mounts it with LangServe. Clients invoke the /chain endpoints; missing key fails startup.

---

## Future improvements

- Pin dependency versions
- Add sample curl/OpenAPI usage to a real README

---

## License

See repository.

---

<p align="center">
  <b>LangServe Translator API</b><br/>
  <a href="https://github.com/ArchanaChetan07/LangServe-Translator-API">github.com/ArchanaChetan07/LangServe-Translator-API</a>
</p>
