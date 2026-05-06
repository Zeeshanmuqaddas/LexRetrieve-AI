import { useState } from 'react';
import { Check, Copy, Download } from 'lucide-react';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';

const PYTHON_CODE = `import pandas as pd
import numpy as np
import re
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
import json
import logging
from typing import List, Dict

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LegalRetrievalSystem:
    def __init__(self, corpus_path: str):
        self.corpus_path = corpus_path
        self.df = None
        self.bm25 = None
        self.tokenized_corpus = []
        
        logger.info("Loading models...")
        # Dense Embedding Model (all-MiniLM-L6-v2)
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        # CrossEncoder for Reranking (ms-marco-MiniLM-L-6-v2)
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        self.index = None
        self.doc_embeddings = None
        
    def clean_text(self, text: str) -> str:
        """Normalize text: lowercase, remove extra whitespace and special chars."""
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r'\\s+', ' ', text)
        text = re.sub(r'[^\\w\\s]', '', text)
        return text.strip()

    def load_and_preprocess(self):
        """Load corpus and prepare BM25 and FAISS indices."""
        logger.info(f"Loading corpus from {self.corpus_path}")
        self.df = pd.read_csv(self.corpus_path)
        
        # Ensure correct columns
        assert 'doc_id' in self.df.columns and 'text' in self.df.columns and 'citation' in self.df.columns
        
        logger.info("Cleaning text...")
        self.df['clean_text'] = self.df['text'].apply(self.clean_text)
        
        logger.info("Initializing BM25...")
        self.tokenized_corpus = [doc.split(" ") for doc in self.df['clean_text']]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        
        logger.info("Generating dense embeddings and building FAISS index...")
        self.doc_embeddings = self.embedder.encode(self.df['clean_text'].tolist(), show_progress_bar=True, batch_size=32)
        
        dim = self.doc_embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim) # Cosine similarity
        faiss.normalize_L2(self.doc_embeddings)
        self.index.add(self.doc_embeddings)
        logger.info("Preprocessing complete.")

    def expand_query(self, query: str) -> str:
        """
        Expand query using synonyms and multilingual expansion (e.g. German terms).
        """
        expansion_dict = {
            "divorce": "divorce scheidung separation",
            "contract": "contract vertrag agreement",
            "liability": "liability haftung responsibility",
        }
        clean_q = self.clean_text(query)
        expanded_terms = []
        for word in clean_q.split():
            if word in expansion_dict:
                expanded_terms.append(expansion_dict[word])
            else:
                expanded_terms.append(word)
        return " ".join(expanded_terms)

    def hybrid_search(self, query: str, top_k: int = 50) -> List[int]:
        """Perform BM25 and FAISS retrieval and fuse the results."""
        expanded_query = self.expand_query(query)
        tokenized_query = expanded_query.split(" ")
        
        # 1. BM25 Search (Sparse)
        bm25_scores = self.bm25.get_scores(tokenized_query)
        bm25_top_indices = np.argsort(bm25_scores)[::-1][:top_k]
        
        # 2. FAISS Search (Dense)
        query_embedding = self.embedder.encode([expanded_query])
        faiss.normalize_L2(query_embedding)
        _, faiss_top_indices = self.index.search(query_embedding, top_k)
        faiss_top_indices = faiss_top_indices[0]
        
        # Fusion Strategy: Union of sets
        fused_indices = list(set(bm25_top_indices).union(set(faiss_top_indices)))
        return fused_indices

    def rerank(self, query: str, candidate_indices: List[int], top_k: int = 10) -> pd.DataFrame:
        """Rerank candidates using CrossEncoder."""
        candidates = self.df.iloc[candidate_indices]
        
        # Prepare pairs for cross-encoder scoring
        pairs = [[query, text] for text in candidates['clean_text'].tolist()]
        scores = self.reranker.predict(pairs)
        
        candidates = candidates.copy()
        candidates['relevance_score'] = scores
        candidates = candidates.sort_values(by='relevance_score', ascending=False).head(top_k)
        
        return candidates

    def select_citations(self, reranked_docs: pd.DataFrame) -> str:
        """
        Extract unique citations. Prioritize relevance.
        Return 5 to 7 citations based on confidence scores.
        """
        citations = []
        for dict_val in reranked_docs.to_dict('records'):
            cit = dict_val['citation']
            if cit not in citations:
                citations.append(cit)
            
            # Dynamic cutoff logic (up to 7 max)
            if len(citations) >= 7:
                break
        
        # Ensure at least up to 5 if available, max 7
        final_citations = citations[:7] if len(citations) > 5 else citations[:5]
        return ";".join(final_citations)

    def predict(self, query: str) -> Dict[str, str]:
        """End-to-End Pipeline prediction."""
        logger.info(f"Processing query: {query}")
        candidates = self.hybrid_search(query, top_k=50)
        reranked = self.rerank(query, candidates, top_k=10)
        final_citations = self.select_citations(reranked)
        return {"predicted_citations": final_citations}

# === CSV Submission Generator ===
def generate_submission(system: LegalRetrievalSystem, test_path: str, output_path: str):
    logger.info(f"Loading test queries from {test_path}")
    test_df = pd.read_csv(test_path)
    
    predictions = []
    for _, row in test_df.iterrows():
        res = system.predict(row['query'])
        predictions.append({
            "query_id": row['query_id'],
            "predicted_citations": res['predicted_citations']
        })
        
    out_df = pd.DataFrame(predictions)
    out_df.to_csv(output_path, index=False)
    logger.info(f"Submission saved to {output_path}")

if __name__ == "__main__":
    # Example usage for Kaggle / Local Execution
    # system = LegalRetrievalSystem("corpus.csv")
    # system.load_and_preprocess()
    # print(system.predict("what are the grounds for divorce?"))
    # generate_submission(system, "test_queries.csv", "submission.csv")
    pass
`;

export function PythonPipeline() {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    await navigator.clipboard.writeText(PYTHON_CODE);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleDownload = () => {
    const blob = new Blob([PYTHON_CODE], { type: 'text/x-python' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'legal_retrieval.py';
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="w-full h-full flex flex-col bg-[#1E1E1E] rounded-xl overflow-hidden border border-slate-800 shadow-2xl">
      <div className="flex items-center justify-between px-4 py-3 bg-[#2D2D2D] border-b border-slate-700">
        <div className="flex items-center space-x-2">
          <div className="flex space-x-1.5">
            <div className="w-3 h-3 rounded-full bg-red-500"></div>
            <div className="w-3 h-3 rounded-full bg-yellow-500"></div>
            <div className="w-3 h-3 rounded-full bg-green-500"></div>
          </div>
          <span className="ml-2 font-mono text-xs text-slate-300">legal_retrieval.py</span>
        </div>
        <div className="flex items-center space-x-2">
          <button 
            onClick={handleCopy}
            className="flex items-center space-x-1.5 px-3 py-1.5 rounded-md bg-slate-700 hover:bg-slate-600 transition-colors text-xs font-medium text-slate-200"
          >
            {copied ? <Check className="w-3.5 h-3.5" /> : <Copy className="w-3.5 h-3.5" />}
            <span>{copied ? 'Copied!' : 'Copy Code'}</span>
          </button>
          <button 
            onClick={handleDownload}
            className="flex items-center space-x-1.5 px-3 py-1.5 rounded-md bg-blue-600 hover:bg-blue-500 transition-colors text-xs font-medium text-white shadow-sm"
          >
            <Download className="w-3.5 h-3.5" />
            <span>Download</span>
          </button>
        </div>
      </div>
      <div className="flex-1 overflow-auto bg-[#1E1E1E] p-4 custom-scrollbar">
        <SyntaxHighlighter
          language="python"
          style={vscDarkPlus}
          customStyle={{
            margin: 0,
            padding: 0,
            background: 'transparent',
            fontSize: '13px',
            lineHeight: '1.5',
            fontFamily: 'var(--font-mono)'
          }}
          showLineNumbers={true}
          wrapLines={true}
        >
          {PYTHON_CODE}
        </SyntaxHighlighter>
      </div>
    </div>
  );
}
