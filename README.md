# Agentic RAG Engine 🤖📄

A professional-grade Retrieval-Augmented Generation (RAG) pipeline that integrates autonomous agentic reasoning and cross-encoder re-ranking.

## 🌟 Advanced Features
- **Autonomous Agentic Flow**: Multi-step reasoning for document analysis.
- **Cross-Encoder Re-Ranking**: Precision scoring using `ms-marco-MiniLM-L-6-v2`.
- **Vector Search Optimization**: Designed for high-speed semantic retrieval.

## 🛠️ Installation
```bash
git clone https://github.com/vishalmandaki/agentic-rag-engine.git
cd agentic-rag-engine
pip install -r requirements.txt
```

## 🚀 Usage
```python
from src.agentic_rag_core import AgenticRAG
rag = AgenticRAG()
result = rag.execute_agentic_flow("Explain RAG architectures.")
```
