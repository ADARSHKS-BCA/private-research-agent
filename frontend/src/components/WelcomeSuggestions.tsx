import React from 'react';
import { Sparkles, Search, Compass, Cpu, Database, ArrowUpRight } from 'lucide-react';

interface WelcomeSuggestionsProps {
  onSelectPrompt: (prompt: string) => void;
}

const SUGGESTIONS = [
  {
    icon: Compass,
    title: 'Agentic RAG Approaches',
    prompt: 'What are the latest approaches to agentic RAG?',
    category: 'Architecture',
  },
  {
    icon: Database,
    title: 'Vector Search & Embeddings',
    prompt: 'How do dense vector embeddings and hybrid search work together in modern retrieval pipelines?',
    category: 'Retrieval',
  },
  {
    icon: Cpu,
    title: 'LangGraph Autonomous Loops',
    prompt: 'Explain how state graphs enable self-corrective and iterative research agents.',
    category: 'Agent Workflows',
  },
  {
    icon: Search,
    title: 'Citation & Grounding',
    prompt: 'What are the best methods for citation validation and preventing hallucinations in RAG systems?',
    category: 'Safety & Grounding',
  },
];

export const WelcomeSuggestions: React.FC<WelcomeSuggestionsProps> = ({ onSelectPrompt }) => {
  return (
    <div className="flex-1 flex flex-col items-center justify-center max-w-3xl mx-auto px-4 py-8 text-center my-auto">
      {/* Hero Badge */}
      <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-indigo-500/10 border border-indigo-500/20 text-indigo-300 text-xs font-semibold mb-6">
        <Sparkles className="w-3.5 h-3.5" />
        <span>Autonomous Research & Grounded Synthesis</span>
      </div>

      {/* Main Title */}
      <h2 className="text-2xl sm:text-4xl font-extrabold text-slate-100 tracking-tight mb-3">
        What would you like to research today?
      </h2>

      <p className="text-sm sm:text-base text-slate-400 max-w-xl mb-8 leading-relaxed">
        The agent searches the live web, scrapes authoritative sources, builds local vector representations in Qdrant, and synthesizes grounded answers with validated citations.
      </p>

      {/* Suggestion Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 w-full text-left">
        {SUGGESTIONS.map((item, idx) => {
          const Icon = item.icon;
          return (
            <button
              key={idx}
              type="button"
              onClick={() => onSelectPrompt(item.prompt)}
              className="group p-4 rounded-2xl border border-slate-800 bg-slate-900/50 hover:bg-slate-800/70 hover:border-slate-700 transition-all duration-200 flex flex-col justify-between text-left shadow-sm hover:shadow-md"
            >
              <div className="flex items-start justify-between w-full mb-2">
                <div className="flex items-center gap-2">
                  <div className="p-1.5 rounded-lg bg-slate-800 text-indigo-400 group-hover:bg-indigo-500/10 group-hover:text-indigo-300 transition-colors">
                    <Icon className="w-4 h-4" />
                  </div>
                  <span className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider">
                    {item.category}
                  </span>
                </div>
                <ArrowUpRight className="w-4 h-4 text-slate-600 group-hover:text-indigo-400 transition-colors" />
              </div>

              <span className="text-sm font-medium text-slate-200 group-hover:text-white leading-snug">
                {item.prompt}
              </span>
            </button>
          );
        })}
      </div>
    </div>
  );
};
