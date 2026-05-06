<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=200&section=header&text=IndiaDoc%20RAG&fontSize=60&fontColor=fff&animation=fadeIn&fontAlignY=38&desc=Ask%20anything%20about%20Indian%20Government%20Documents&descAlignY=60&descSize=18" width="100%"/>

<br/>

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Groq](https://img.shields.io/badge/Groq-LLaMA3-F55036?style=for-the-badge&logo=groq&logoColor=white)](https://groq.com)
[![FAISS](https://img.shields.io/badge/FAISS-Vector_Search-0467DF?style=for-the-badge&logo=meta&logoColor=white)](https://faiss.ai)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Render](https://img.shields.io/badge/Deployed_on-Render-46E3B7?style=for-the-badge&logo=render&logoColor=white)](https://render.com)

<br/>

> **🇮🇳 Ask questions about the DPDP Act, RTI Act, RBI Circulars, and more — get grounded answers with exact source citations.**
> 
> *No hallucination. Every answer is backed by the actual document.*

<br/>



</div>

---

## 🎬 Demo

<div align="center">

<!-- Replace with your actual GIF/screenshot -->
```
┌─────────────────────────────────────────────────────────────────┐
│  🇮🇳  IndiaDoc RAG                              [● Live on Render] │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Ask a question:                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ What are the penalties under the DPDP Act 2023?         │    │
│  └─────────────────────────────────────────────────────────┘    │
│                          [Ask IndiaDoc →]                         │
│                                                                   │
│  ✅ Answer (1.2s · Groq LLaMA3 · 4 sources)                     │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ Under Section 33 of the DPDP Act 2023, the Data         │    │
│  │ Protection Board may impose penalties up to ₹250 crore  │    │
│  │ for significant data breaches...                         │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                   │
│  📄 Sources: DPDP_Act_2023.pdf (p.12) · DPDP_Act_2023.pdf (p.8)│
└─────────────────────────────────────────────────────────────────┘
```

*Record a Loom and paste the GIF here for 10x more GitHub stars*

</div>

---

## ⚡ What Makes This Different

| Feature | IndiaDoc RAG | Generic Chatbot |
|---|---|---|
| **Hallucination** | ❌ Impossible — answers from docs only | ✅ Makes things up |
| **Source citations** | ✅ Doc name + page number | ❌ None |
| **Indian context** | ✅ DPDP, RTI, RBI, Budget | ❌ Generic |
| **Speed** | ✅ ~1s with Groq inference | 🐢 3–5s with OpenAI |
| **Cost** | ✅ Free tier on Groq | 💰 Pay per token |
| **Deployment** | ✅ Live on Render | 🤷 Localhost only |

---

## 🏗️ Architecture

```
                        User Question
                             │
                    ┌────────▼────────┐
                    │   Streamlit UI   │
                    │  (Render Cloud)  │
                    └────────┬────────┘
                             │ POST /query
                    ┌────────▼────────┐
                    │   FastAPI API    │
                    │  (Render Cloud)  │
                    └────┬───────┬────┘
                         │       │
             ┌───────────▼──┐ ┌──▼──────────────┐
             │  retriever.py │ │    chain.py      │
             │               │ │                  │
             │  FAISS Index  │ │  Groq LLaMA3-70B │
             │  (SEMANTIC    │ │  Temperature=0   │
             │   SEARCH)     │ │  Max tokens=1024 │
             └───────┬───────┘ └──────────────────┘
                     │
          ┌──────────▼──────────┐
          │     ingest.py        │
          │                      │
          │  PDF → Chunks →      │
          │  all-MiniLM-L6-v2 → │
          │  FAISS Index         │
          └──────────────────────┘
                     │
          ┌──────────▼──────────┐
          │    data/docs/        │
          │                      │
          │  📄 DPDP_Act.pdf     │
          │  📄 RTI_Act.pdf      │
          │  📄 RBI_Circular.pdf │
          └──────────────────────┘
```

---

## 🚀 Run Locally in 5 Minutes

### Prerequisites
- Python 3.11+
- [Groq API key](https://console.groq.com) (free)
- Indian govt PDFs (links below)

### 1. Clone & Install

```bash
git clone https://github.com/YOUR_USERNAME/indiadoc-rag.git
cd indiadoc-rag
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
```

```env
# .env
GROQ_API_KEY=gsk_your_groq_key_here
```

### 3. Add Documents

Download these free PDFs and place in `data/docs/`:

| Document | Download |
|---|---|
| 🔐 DPDP Act 2023 | [meity.gov.in](https://www.meity.gov.in/writereaddata/files/Digital%20Personal%20Data%20Protection%20Act%202023.pdf) |
| 📋 RTI Act 2005 | [rtionline.gov.in](https://rtionline.gov.in/documents/rti_act.pdf) |
| 🏦 RBI Master Circular | [rbi.org.in](https://www.rbi.org.in) |

### 4. Build Index (once)

```bash
python src/ingest.py
```

```
✓ Loaded 3 PDFs (142 pages)
✓ Created 1,847 chunks
✓ Built FAISS index (all-MiniLM-L6-v2)
✓ Index saved to data/faiss_index/
```

### 5. Launch

```bash
# Terminal 1 — API
uvicorn src.api:app --reload --port 8000

# Terminal 2 — UI
streamlit run ui/app.py
```

Open **http://localhost:8501** 🎉

---

## 🌐 Deploy to Render (Free)

This project is already deployed on Render. Fork and deploy your own:

### API Service
```yaml
# render.yaml (API)
services:
  - type: web
    name: indiadoc-api
    runtime: python
    buildCommand: pip install -r requirements.txt && python src/ingest.py
    startCommand: uvicorn src.api:app --host 0.0.0.0 --port $PORT
    envVars:
      - key: GROQ_API_KEY
        sync: false
```

### Streamlit Service
```yaml
  - type: web
    name: indiadoc-ui
    runtime: python
    buildCommand: pip install -r requirements.txt
    startCommand: streamlit run ui/app.py --server.port $PORT --server.address 0.0.0.0
    envVars:
      - key: API_URL
        value: https://indiadoc-api.onrender.com
```

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy)

---

## 📡 API Reference

**Base URL:** `https://your-api.onrender.com`

### `POST /query`

Ask a question about any indexed document.

**Request:**
```json
{
  "question": "What are the rights of a data principal under DPDP Act?",
  "k": 4
}
```

**Response:**
```json
{
  "question": "What are the rights of a data principal under DPDP Act?",
  "answer": "Under Section 11-14 of the DPDP Act 2023, a Data Principal has the following rights: (1) Right to access information about personal data processing, (2) Right to correction and erasure of personal data, (3) Right to grievance redressal...",
  "sources": [
    {
      "source_file": "DPDP_Act_2023.pdf",
      "page": 8,
      "excerpt": "Section 11 — Right of Data Principal to obtain information..."
    }
  ],
  "latency_ms": 987
}
```

### `GET /health`

```json
{
  "status": "ok",
  "index_ready": true,
  "docs_count": 3,
  "message": "IndiaDoc RAG is ready to answer questions."
}
```

### `GET /docs-list`

```json
{
  "documents": ["DPDP_Act_2023.pdf", "RTI_Act_2005.pdf", "RBI_Circular.pdf"],
  "total": 3
}
```

---

## 💡 Example Questions

<details>
<summary><b>🔐 DPDP Act 2023</b></summary>

- What are the penalties for data breach under the DPDP Act?
- Who qualifies as a Data Fiduciary?
- What are the consent requirements under DPDP?
- How does the Data Protection Board work?
- What are the obligations of a Significant Data Fiduciary?

</details>

<details>
<summary><b>📋 RTI Act 2005</b></summary>

- How do I file an RTI application?
- Within how many days must an RTI be answered?
- What information is exempt from RTI disclosure?
- What is the fee for filing an RTI?
- How do I appeal a rejected RTI?

</details>

<details>
<summary><b>🏦 RBI Circulars</b></summary>

- What are RBI's digital lending guidelines?
- What is the KYC requirement for bank accounts?
- What are the rules for prepaid payment instruments?

</details>

---

## 🧠 Technical Deep Dive

<details>
<summary><b>Why Groq instead of OpenAI?</b></summary>

Groq's LPU (Language Processing Unit) inference hardware delivers **10x faster** response times compared to standard GPU inference:

- OpenAI GPT-4o-mini: ~3–5 seconds per query
- Groq LLaMA3-70B: ~0.8–1.5 seconds per query

For a RAG system where UX depends on perceived speed, this matters enormously. Plus the free tier is generous enough for portfolio demos.

</details>

<details>
<summary><b>Why FAISS over a managed vector DB?</b></summary>

For this use case (static government documents that change rarely), a managed vector DB like Pinecone or Weaviate adds cost and latency with no real benefit. FAISS runs in-process, has zero network overhead, and handles millions of vectors on a laptop. It's saved to disk and loaded once at startup.

</details>

<details>
<summary><b>Why all-MiniLM-L6-v2 for embeddings?</b></summary>

- **Free**: No API cost — runs locally
- **Fast**: 384-dimensional vectors (vs 1536 for OpenAI ada-002)
- **Good enough**: Cosine similarity on legal/policy text works well at this dimension
- **Consistent**: Same model used for ingestion and query — vectors are in the same space

</details>

<details>
<summary><b>Chunk size reasoning</b></summary>

500 characters with 50-character overlap was chosen because:
- Indian legal documents have dense clauses that fit well in 500 chars
- Overlap ensures a clause split across two chunks is still retrievable
- Larger chunks (1000+) reduce retrieval precision; smaller chunks (<200) lose context

</details>

---

## 📁 Project Structure

```
indiadoc-rag/
├── data/
│   ├── docs/               ← Add your PDFs here
│   └── faiss_index/        ← Auto-generated by ingest.py
├── src/
│   ├── ingest.py           ← PDF → chunks → FAISS index
│   ├── retriever.py        ← Semantic search over FAISS
│   ├── chain.py            ← RAG chain (Groq LLaMA3 + prompt)
│   └── api.py              ← FastAPI REST endpoints
├── ui/
│   └── app.py              ← Streamlit frontend
├── tests/
│   └── test_retriever.py   ← Retrieval accuracy tests
├── Dockerfile
├── render.yaml
├── requirements.txt
└── .env.example
```

---

## 🛠️ Tech Stack

| Layer | Technology | Why |
|---|---|---|
| **Embeddings** | `all-MiniLM-L6-v2` | Free, fast, local |
| **Vector Search** | FAISS (ANN) | Zero-cost, in-process |
| **LLM** | Groq LLaMA3-70B | 10x faster than OpenAI |
| **Orchestration** | LangChain | RAG chains + retrievers |
| **API** | FastAPI + Pydantic | Type-safe, auto-docs |
| **UI** | Streamlit | Rapid deployment |
| **Deployment** | Render.com | Free tier, auto-deploy |
| **Containerization** | Docker | Portable, reproducible |

---

## 🗺️ Roadmap

- [x] PDF ingestion pipeline
- [x] FAISS semantic search
- [x] FastAPI REST API
- [x] Streamlit UI
- [x] Groq integration
- [x] Render deployment
- [ ] Hindi language support (Hinglish queries)
- [ ] Multi-document comparison mode
- [ ] Add Supreme Court judgments dataset
- [ ] Streaming responses
- [ ] User feedback loop for retrieval quality

---

## 🤝 Contributing

PRs welcome! Please open an issue first for major changes.

```bash
git checkout -b feature/your-feature
git commit -m "Add: your feature"
git push origin feature/your-feature
```

---

## 📜 License

MIT — use freely, attribution appreciated.

---

<div align="center">

**Built by [Isha Vishwakarma](https://linkedin.com/in/isha-vishwakarma)**

*Day 1 of 7 · AI Engineering Sprint · May 2026*

*If this helped you, drop a ⭐ — it helps more people find it.*

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=100&section=footer" width="100%"/>

</div>
