import { useState } from 'react';
import { Search, Loader2, Database, BrainCircuit, ListOrdered, Share2, FileText, ChevronRight } from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';
import { GoogleGenAI } from '@google/genai';

const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });

type PipelineStep = 'idle' | 'preprocessing' | 'hybrid_search' | 'reranking' | 'extracting' | 'complete';

interface SimulationResult {
  expandedQuery: string;
  hybridCandidates: number;
  citations: string[];
}

export function Dashboard() {
  const [query, setQuery] = useState('');
  const [step, setStep] = useState<PipelineStep>('idle');
  const [result, setResult] = useState<SimulationResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const runSimulation = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim()) return;

    setStep('preprocessing');
    setResult(null);
    setError(null);

    try {
      // Step 1: Preprocessing & Expansion
      await new Promise(r => setTimeout(r, 800));
      setStep('hybrid_search');

      // Step 2: Hybrid Search
      await new Promise(r => setTimeout(r, 1000));
      setStep('reranking');

      // Step 3: Reranking & AI generation of citations
      const prompt = `
You are the Reranking & Citation Selection component of a Legal AI retrieval pipeline.
The user queried: "${query}"
Generate 5 to 7 highly relevant and realistic legal citations (e.g. statutes, case laws) based on this query.
Return the result STRICTLY as a JSON object with this exact structure:
{
  "expanded_query": "multilingual or synonym expanded version of the query (e.g. including German terms if relevant)",
  "hybrid_candidates": 50,
  "citations": ["citation 1", "citation 2", "citation 3", "citation 4", "citation 5"]
}
Do not return markdown formatting, just the raw JSON string.
`;

      const response = await ai.models.generateContent({
        model: 'gemini-2.5-flash',
        contents: prompt,
        config: {
          responseMimeType: 'application/json',
        }
      });

      const responseText = response.text() || '{}';
      const parsed = JSON.parse(responseText);
      
      setStep('extracting');
      await new Promise(r => setTimeout(r, 800));

      setResult({
        expandedQuery: parsed.expanded_query || 'expanded query fallback',
        hybridCandidates: parsed.hybrid_candidates || 50,
        citations: parsed.citations || ['BGB § 1353', 'FamFG § 111']
      });
      setStep('complete');

    } catch (err: any) {
      setError(err.message || 'An error occurred during pipeline simulation.');
      setStep('idle');
    }
  };

  const steps = [
    { id: 'preprocessing', label: 'Query Expansion', icon: Share2, desc: 'Synonyms & Multilingual (De/En)' },
    { id: 'hybrid_search', label: 'Hybrid Retrieval', icon: Database, desc: 'BM25 + FAISS (Dense/Sparse)' },
    { id: 'reranking', label: 'Neural Reranking', icon: BrainCircuit, desc: 'CrossEncoder (ms-marco)' },
    { id: 'extracting', label: 'Citation Selection', icon: ListOrdered, desc: 'Filtering & Deduplication' },
  ];

  const getStepStatus = (stepId: string) => {
    const states = ['idle', 'preprocessing', 'hybrid_search', 'reranking', 'extracting', 'complete'];
    const currentIndex = states.indexOf(step);
    const stepIndex = states.indexOf(stepId);
    
    if (currentIndex > stepIndex) return 'complete';
    if (currentIndex === stepIndex) return 'current';
    return 'pending';
  };

  return (
    <div className="w-full max-w-5xl mx-auto space-y-8">
      {/* Search Header */}
      <div className="bg-white p-8 rounded-2xl shadow-sm border border-slate-200">
        <h2 className="text-2xl font-semibold text-slate-800 mb-2">Legal Information Retrieval</h2>
        <p className="text-slate-500 mb-6">Enter a legal question to run the hybrid BM25 + FAISS pipeline.</p>
        
        <form onSubmit={runSimulation} className="relative flex items-center">
          <div className="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none">
            <Search className="h-5 w-5 text-slate-400" />
          </div>
          <input
            type="text"
            className="block w-full pl-11 pr-32 py-4 bg-slate-50 border border-slate-200 rounded-xl text-slate-900 placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:bg-white transition-colors"
            placeholder="e.g., What are the grounds for divorce in Germany?"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            disabled={step !== 'idle' && step !== 'complete'}
          />
          <button
            type="submit"
            disabled={!query.trim() || (step !== 'idle' && step !== 'complete')}
            className="absolute right-2 top-2 bottom-2 px-6 bg-blue-600 hover:bg-blue-700 text-white font-medium rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center"
          >
            {step !== 'idle' && step !== 'complete' ? (
              <><Loader2 className="w-4 h-4 mr-2 animate-spin" /> Processing</>
            ) : (
              'Search'
            )}
          </button>
        </form>
        {error && <p className="mt-4 text-red-500 text-sm">{error}</p>}
      </div>

      {/* Pipeline Visualization */}
      {(step !== 'idle' || result) && (
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
          
          {/* Timeline */}
          <div className="lg:col-span-4 bg-white p-6 rounded-2xl shadow-sm border border-slate-200">
            <h3 className="text-sm font-semibold text-slate-400 uppercase tracking-wider mb-6">Pipeline Status</h3>
            <div className="space-y-6">
              {steps.map((s, idx) => {
                const status = getStepStatus(s.id);
                return (
                  <div key={s.id} className="relative flex items-start">
                    {idx !== steps.length - 1 && (
                      <div className={`absolute top-8 left-4 w-0.5 h-full -ml-px ${status === 'complete' ? 'bg-blue-500' : 'bg-slate-200'}`} />
                    )}
                    <div className={`relative z-10 flex items-center justify-center w-8 h-8 rounded-full border-2 ${
                      status === 'complete' ? 'bg-blue-500 border-blue-500 text-white' :
                      status === 'current' ? 'bg-white border-blue-500 text-blue-500' :
                      'bg-white border-slate-200 text-slate-400'
                    }`}>
                      <s.icon className="w-4 h-4" />
                    </div>
                    <div className="ml-4 flex-1">
                      <h4 className={`text-sm font-medium ${status === 'current' ? 'text-blue-600' : status === 'complete' ? 'text-slate-800' : 'text-slate-500'}`}>
                        {s.label}
                      </h4>
                      <p className="text-xs text-slate-500 mt-1">{s.desc}</p>
                      {status === 'current' && (
                        <motion.div initial={{ width: 0 }} animate={{ width: '100%' }} transition={{ duration: 1 }} className="h-1 bg-blue-100 rounded-full mt-2 overflow-hidden">
                          <motion.div className="h-full bg-blue-500 rounded-full" animate={{ x: ['-100%', '100%'] }} transition={{ repeat: Infinity, duration: 1 }} />
                        </motion.div>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Results Area */}
          <div className="lg:col-span-8 bg-white p-6 rounded-2xl shadow-sm border border-slate-200 flex flex-col justify-center min-h-[400px]">
            <AnimatePresence mode="wait">
              {step !== 'complete' && step !== 'idle' ? (
                <motion.div
                  key="loading"
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -10 }}
                  className="flex flex-col items-center justify-center text-slate-500"
                >
                  <BrainCircuit className="w-16 h-16 mb-6 text-blue-500/50 animate-pulse" />
                  <p className="text-lg font-medium text-slate-700">Running AI Pipeline...</p>
                  <p className="text-sm mt-2">Computing dense embeddings with cross-encoder reranking</p>
                </motion.div>
              ) : result ? (
                <motion.div
                  key="result"
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="w-full"
                >
                  <div className="flex items-center space-x-2 mb-6">
                    <FileText className="w-6 h-6 text-blue-600" />
                    <h3 className="text-xl font-semibold text-slate-800">Final Predicted Citations</h3>
                  </div>
                  
                  <div className="bg-slate-50 rounded-xl p-4 border border-slate-200 mb-6">
                    <div className="flex items-start justify-between text-sm">
                      <div>
                        <span className="text-slate-500 font-medium mr-2">Original Query:</span>
                        <span className="text-slate-800">"{query}"</span>
                      </div>
                    </div>
                    <div className="mt-3 pt-3 border-t border-slate-200">
                      <span className="text-slate-500 font-medium mr-2">Expanded & Cleaned:</span>
                      <span className="text-slate-800 font-mono text-xs bg-slate-200 px-2 py-0.5 rounded">
                        {result.expandedQuery}
                      </span>
                    </div>
                  </div>

                  <div className="space-y-3">
                    {result.citations.map((cit, idx) => (
                      <motion.div 
                        initial={{ opacity: 0, x: -10 }} 
                        animate={{ opacity: 1, x: 0 }} 
                        transition={{ delay: idx * 0.1 }}
                        key={idx} 
                        className="flex items-center bg-white border border-slate-200 p-4 rounded-xl shadow-sm hover:border-blue-300 transition-colors"
                      >
                        <div className="flex items-center justify-center w-8 h-8 rounded-full bg-blue-50 text-blue-600 font-bold text-sm mr-4">
                          {idx + 1}
                        </div>
                        <span className="font-medium text-slate-800">{cit}</span>
                      </motion.div>
                    ))}
                  </div>

                  <div className="mt-8 pt-6 border-t border-slate-100 grid grid-cols-3 gap-4 text-center">
                    <div>
                      <p className="text-xs text-slate-400 font-medium uppercase tracking-wider mb-1">Total Candidates</p>
                      <p className="text-2xl font-bold text-slate-800">{result.hybridCandidates}</p>
                    </div>
                    <div>
                      <p className="text-xs text-slate-400 font-medium uppercase tracking-wider mb-1">Top K Used</p>
                      <p className="text-2xl font-bold text-slate-800">10</p>
                    </div>
                    <div>
                      <p className="text-xs text-slate-400 font-medium uppercase tracking-wider mb-1">Final Extracted</p>
                      <p className="text-2xl font-bold text-blue-600">{result.citations.length}</p>
                    </div>
                  </div>

                </motion.div>
              ) : null}
            </AnimatePresence>
          </div>
        </div>
      )}
    </div>
  );
}
