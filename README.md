# Research Navigator - Multi-Agent RAG for Research Synthesis

This project was conducted as part of the **CAS Generative AI** program at the **University of Zurich**. It investigates how multi-agent architectures with specialized retrieval, synthesis, and validation agents compare to single LLM baselines for academic literature synthesis tasks.

![](/docs/infographic.png)


---

## Project Context

**Thesis:** Multi-Agent Retrieval-Augmented Generation for Academic Literature Review Synthesis  
**Question:** *"How does a multi-agent RAG architecture compare to a single LLM approach for academic literature synthesis in terms of response quality?"*

**Author:** Elson Felix Mendes Filho  
**Program:** CAS Generative AI (Fall 2025)  
**Institution:** University of Zurich  
**Supervisor:** Prof. Dr. Jannis Vamvas  
**Submission Date:** February 13, 2026

---

## Project Overview

This repository contains two main components:

### 1. Evaluation Experiment

A rigorous comparative evaluation framework that systematically assessed multi-agent RAG against a single LLM baseline across diverse research queries.

**Key Components:**

**Corpus Collection** - Gradio-based interface developed to facilitate and automate paper selection, corpus collection and the processing pipeline.

**RAG Setup** - Text Processing and Vector Storage - Gradio interface was implemented to enable interactive dataset selection and to automate the ensuing text processing workflow, comprising four sequential stages: chunking, embedding generation, vector storage, and validation.

**Baseline System** - A single LLM, relying entirely on the model's parametric knowledge.

**Multi-Agent System** - The multi-agent system consisted of six specialized agents.

**Evaluation Interface** - A seven-step Gradio web interface guiding the user through systematic evaluation, employing a LLM as a judge, and showing report on results.

**Results Summary:**
- **Win Rate:** Multi-Agent 83% vs Baseline 8% (Ties 9%)
- **Largest Improvement:** Citation Quality (+1.17 points, Cohen's d = 1.43 - large effect)

**What You Can Do:**
- Run complete evaluations on custom query sets
- Compare multi-agent vs baseline responses side-by-side
- Generate statistical reports with visualizations
- Export results for further analysis

📄 **[→ Detailed Evaluation Documentation](docs/EVALUATION.md)**

---

### 2. Multi-Agent RAG System - User Interface

An extended interface is under development and will enable researchers to upload custom paper collections for domain-specific research, with automated RAG setup requiring no technical expertise. This system will incorporate blind A/B comparison between baseline and multi-agent approaches, capturing user preferences to validate the architectural findings with human feedback. An interactive multi-query workflow will address iterative refinement rather than single queries, while integrated reference access will allow users to check citations and read the papers or documents directly. This tool will provide both human evaluation of the comparative results and demonstrate practical viability for self-service research assistance.

**What You Can Do:**
- Build custom corpora using the integrated paper collection interface, integrating papers search and uploading your own document collections for domain-specific research
- Receive ~800-word synthesized responses with validated citations
- Interactive queries
- Compare multi-agent responses against baseline (single LLM) (blindly)

📄 **[→ Detailed System Documentation](docs/SYSTEM.md)**

---

## Multi-Agent Architecture

The multi-agent RAG system comprises 6 specialized agents working in coordination:

| | Agent | Primary Role | Key Capabilities |
|---|-------|--------------|------------------|
| 1. | Coordinator | Workflow orchestration | Phase management, routing, parallel execution |
| 2. | Query Decomposition | Complex query handling | Temporal analysis, multi-aspect breakdown, sub-query generation |
| 3. | Retrieval | Corpus search | Semantic search, hybrid retrieval (dense and sparse), reranking, relevance filtering, full-text access |
| 4. | Parametric | LLM knowledge access | GPT-5.2 synthesis, reference extraction, contextual integration |
| 5. | Citation Validator | Reference verification | Multiple source verification, two-tier validation |
| 6. | Synthesis | Multi-source integration | Combines Retrieval and Parametric sources into natural prose, source-tagged citations |


![](/docs/infographic2.png)
---

## Technical Stack

**Core Framework:**
- Python 3.11 - Base language
- LangGraph - Multi-agent orchestration
- LangChain-core - Agent abstractions

**Language Models:**
- GPT-5.2 - Agents, baseline, synthesis, LLM-as-judge
- text-embedding-3-small - Corpus embeddings

**Vector Database:**
- ChromaDB - Persistent vector storage (HNSW indexing, cosine similarity)

**External APIs:**
- OpenAI - Language models and embeddings
- ArXiv - Recent ML/AI papers
- OpenAlex - Broad academic coverage (rate-limit free!)
- Google Scholar - Comprehensive fallback

**Analysis & Visualization:**
- Gradio - Web interfaces
- Plotly - Interactive visualizations
- scipy/numpy/pandas - Statistical analysis

**PDF Processing:**
- PyPDF2 - Primary extraction
- pdfminer.six - Fallback for complex PDFs

---

## Documentation

**Technical Documentation:**
- **[Evaluation Experiment Details](docs/EVALUATION.md)** - Complete evaluation methodology, test set, statistical analysis, results
- **[System Architecture Details](docs/SYSTEM.md)** - Agent descriptions, workflow, configuration, usage examples

**Project Documentation:**
- **[Thesis Paper](docs/CAS_Gen_AI_SeminarPaper_ELSON_FILHO.docx)** - Complete thesis document
- **[Complete Query Set](data/test_queries_100.csv)** - 100 evaluation queries with metadata

---

## Contributing

This is a thesis project (completed February 2026), but feedback and suggestions are welcome!

- **Issues:** Report bugs or suggest improvements
- **Discussions:** Share ideas for extensions or applications to other domains
- **Pull Requests:** Documentation improvements, bug fixes

---

## License

MIT License - See [LICENSE](LICENSE) for details

---

## Acknowledgments

- **Prof. Dr. Jannis Vamvas**
- **CAS Generative AI Program**
- **University of Zurich**
- **OpenAI, ArXiv, OpenAlex, Google Scholar**

---

## Citation

If you use this work in your research, please cite:

**BibTeX:**
```bibtex
@casthesis{mendesfilho2026multiagent,
  title={Multi-Agent Retrieval-Augmented Generation for Academic Literature Review Synthesis},
  author={Mendes Filho, Elson Felix},
  year={2026},
  school={University of Zurich},
  type={CAS Generative AI}
}
```

**APA:**
```
Mendes Filho, E. F. (2026). Multi-Agent Retrieval-Augmented Generation for Academic Literature Review Synthesis [CAS thesis]. University of Zurich.
```

---

**⭐ If you find this work useful, please star this repository!**

---

**[⬆ Back to Top](#research-navigator---multi-agent-rag-for-research-synthesis)**