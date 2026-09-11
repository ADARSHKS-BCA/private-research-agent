import React from 'react';
import { Plus, Sparkles, Layers, ShieldCheck } from 'lucide-react';

interface HeaderProps {
  onNewChat: () => void;
  disabled?: boolean;
}

export const Header: React.FC<HeaderProps> = ({ onNewChat, disabled = false }) => {
  return (
    <header className="sticky top-0 z-20 w-full border-b border-slate-800/80 bg-slate-950/80 backdrop-blur-xl px-4 py-3 sm:px-6">
      <div className="max-w-5xl mx-auto flex items-center justify-between">
        {/* Brand & Logo */}
        <div className="flex items-center gap-3">
          <div className="flex items-center justify-center w-9 h-9 rounded-xl bg-gradient-to-tr from-indigo-600 to-violet-500 text-white shadow-lg shadow-indigo-950/50 border border-indigo-400/20">
            <Sparkles className="w-5 h-5" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-base sm:text-lg font-bold text-slate-100 tracking-tight">
                Private Research Agent
              </h1>
              <span className="hidden sm:inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-indigo-500/10 border border-indigo-500/20 text-indigo-300 text-[10px] font-semibold tracking-wide uppercase">
                <ShieldCheck className="w-3 h-3" />
                Autonomous RAG
              </span>
            </div>
            <p className="text-xs text-slate-400 hidden sm:block">
              LangGraph Multi-Agent • Firecrawl • Qdrant • Grounded Citations
            </p>
          </div>
        </div>

        {/* Action Controls */}
        <div className="flex items-center gap-2.5">
          <button
            type="button"
            onClick={onNewChat}
            disabled={disabled}
            className="flex items-center gap-1.5 px-3.5 py-1.5 rounded-xl bg-slate-900 hover:bg-slate-800 border border-slate-700/80 hover:border-slate-600 text-xs sm:text-sm font-medium text-slate-200 transition-all shadow-sm active:scale-95 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <Plus className="w-4 h-4 text-indigo-400" />
            <span>New Chat</span>
          </button>
        </div>
      </div>
    </header>
  );
};
