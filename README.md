# 🔍 Multilingual and Crosslingual Fact-Checked Claim Retrieval  

Ensuring the accuracy of online information is crucial in today's digital landscape. Automated fact-checking consists of multiple stages, including **claim detection, retrieval of evidence, veracity prediction, and explanation generation**.  

A **key component** of this process is **retrieving previously fact-checked claims**, allowing fact-checking systems to efficiently match new claims with existing fact-checks.  

## 🚀 Overview  
In this work, we present a **hybrid retrieval pipeline** that combines:  
- **Lexical Retrieval (BM25)** – Fast but limited to exact word matches.  
- **Semantic Retrieval (Dense Retrieval Models)** – Captures contextual meaning.  

We evaluate various **retrieval and reranking strategies** and show that **hybrid ensembling consistently outperforms individual models**, while reranking provides only **marginal improvements**.  

Additionally, we analyze:  
- **Preprocessing steps** and their impact on retrieval performance.  
- **Comparison of retrieval models** in terms of performance, execution time, parameters, and memory usage.  
- **Error analysis** to identify key limitations and future improvements.  

This approach was applied to **[SemEval-2025 Task 7](https://disai.eu/semeval-2025/)**, where we present our findings and insights.  

## 📂 Dataset  

The dataset used for this research is a modified version of the **[MultiClaim dataset](https://zenodo.org/records/7737983)**.  

**Note:** Due to restrictions, we do **not** publish the used data.


## ⚙️ Installation  

To get started, install the required dependencies:  

```bash
pip install -r requirements.txt
```
Make sure you have Python 3.8+ installed.

## 🚀 Usage
To run the full pipeline, configure the following files:

1. *experiment_config.json* → Define languages, stages, and setup.
2. *pipeline_config.json* → Specify retrievers and rerankers, and ensemblers.
3. *model_config_bm25.json* → Set BM25 hyperparameters.

## Run the Full Pipeline
To execute the retrieval, reranking, and ensembling pipeline:

```bash
python ./src/pipeline.py
```

## ⚙️ Pipeline Workflow
The retrieval and reranking pipeline follows these steps:

**1. Retrieval Phase**
- Uses BM25 and semantic retrievers (Bi-Encoders, Cross-Encoders).
- Retrieves top N fact-checked claims for each query.

**2. Ensembling of Retrieval Results**
- Combines BM25 and neural retrievers results using a weighted ensembling approach and RRF.

**3. Reranking Phase**
- Uses transformer-based cross-encoders to refine the ranking of retrieved claims.

**4. Final Aggregation & Evaluation**
- The reranked results are ensembled again and evaluated using Success@10.

## 💡 Contributing
We welcome contributions! 🛠

### How to Contribute?
1. Fork the repository
2. Create a new branch (```git checkout -b feature-branch```)
3. Make your changes and commit (```git commit -m "Add new feature"```)
4. Push to your fork (```git push origin feature-branch```)
5. Create a Pull Request (PR)

For major changes, please open an issue first to discuss your proposal.


## License
This project is licensed under the MIT License. See the full license text [here](https://choosealicense.com/licenses/mit/).
