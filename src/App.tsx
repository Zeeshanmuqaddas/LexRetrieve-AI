import { useState } from 'react';
import { Scale, Code2, LayoutDashboard, Settings } from 'lucide-react';
import { cn } from './lib/utils';
import { Dashboard } from './components/Dashboard';
import { PythonPipeline } from './components/PythonPipeline';

function App() {
  const [activeTab, setActiveTab] = useState<'dashboard' | 'code'>('dashboard');

  return (
    <div className="min-h-screen bg-[#F8FAFC] flex flex-col font-sans">
      {/* Header */}
      <header className="bg-white border-b border-slate-200 sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 h-16 flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="bg-blue-600 p-2 rounded-lg">
              <Scale className="w-5 h-5 text-white" />
            </div>
            <div>
              <h1 className="text-xl font-bold text-slate-900 tracking-tight leading-none">LexRetrieve</h1>
              <p className="text-[10px] font-medium text-slate-500 tracking-wider uppercase mt-1">Hybrid Legal AI Search</p>
            </div>
          </div>
          
          <nav className="flex space-x-1 border border-slate-200 rounded-lg p-1 bg-slate-50">
            <button
              onClick={() => setActiveTab('dashboard')}
              className={cn(
                "flex items-center space-x-2 px-4 py-2 rounded-md text-sm font-medium transition-all transition-colors duration-200",
                activeTab === 'dashboard' 
                  ? "bg-white text-blue-600 shadow-sm border border-slate-200" 
                  : "text-slate-500 hover:text-slate-800 hover:bg-slate-100 border border-transparent"
              )}
            >
              <LayoutDashboard className="w-4 h-4" />
              <span>Live Demo</span>
            </button>
            <button
              onClick={() => setActiveTab('code')}
              className={cn(
                "flex items-center space-x-2 px-4 py-2 rounded-md text-sm font-medium transition-all transition-colors duration-200",
                activeTab === 'code' 
                  ? "bg-white text-blue-600 shadow-sm border border-slate-200" 
                  : "text-slate-500 hover:text-slate-800 hover:bg-slate-100 border border-transparent"
              )}
            >
              <Code2 className="w-4 h-4" />
              <span>Python Source Code</span>
            </button>
          </nav>

          <div className="flex items-center px-3 py-1.5 bg-slate-100 rounded-full text-xs font-medium text-slate-500 border border-slate-200">
            <div className="w-2 h-2 rounded-full bg-green-500 mr-2 animate-pulse"></div>
            System Online
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="flex-1 max-w-7xl w-full mx-auto p-4 md:p-8 flex flex-col items-center justify-start">
        {activeTab === 'dashboard' ? (
          <div className="w-full flex-1">
            <div className="text-center mb-10 mt-4">
              <h2 className="text-3xl font-bold text-slate-900">Agentic Citation Retrieval</h2>
              <p className="text-slate-500 mt-3 max-w-2xl mx-auto leading-relaxed">
                A hybrid retrieval engine utilizing <strong>BM25</strong> sparse search and <strong>FAISS</strong> dense embeddings (all-MiniLM-L6-v2), reranked via a neural cross-encoder (ms-marco) to surface highly relevant and deterministic legal citations.
              </p>
            </div>
            <Dashboard />
          </div>
        ) : (
          <div className="w-full h-[85vh] flex flex-col">
            <div className="mb-6 flex justify-between items-end">
              <div>
                <h2 className="text-2xl font-bold text-slate-900">End-to-End Pipeline Code</h2>
                <p className="text-slate-500 mt-2">Production-ready Python implementation using <code className="bg-slate-200 px-1 py-0.5 rounded text-sm text-slate-800">sentence-transformers</code> & <code className="bg-slate-200 px-1 py-0.5 rounded text-sm text-slate-800">faiss</code>.</p>
              </div>
            </div>
            <PythonPipeline />
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
