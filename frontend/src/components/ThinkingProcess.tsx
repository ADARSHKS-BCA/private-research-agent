import React, { useState, useEffect } from 'react';
import {
  ChevronDown,
  ChevronRight,
  CheckCircle2,
  Loader2,
  CircleDashed,
  AlertCircle,
  Sparkles,
  Compass,
} from 'lucide-react';
import { ResearchStep } from '../types/chat';

interface ThinkingProcessProps {
  steps: ResearchStep[];
  isStreaming?: boolean;
  elapsedSeconds?: number;
}

export const ThinkingProcess: React.FC<ThinkingProcessProps> = ({
  steps = [],
  isStreaming = false,
  elapsedSeconds,
}) => {
  // Start expanded when running, keep user toggle preference afterwards
  const [isOpen, setIsOpen] = useState<boolean>(true);

  // Auto-collapse when finished if there are answers, but default to open during generation
  useEffect(() => {
    if (!isStreaming && steps.length > 0) {
      // Optional: keep it collapsed or open based on preference
    }
  }, [isStreaming, steps.length]);

  if (!steps || steps.length === 0) {
    if (isStreaming) {
      return (
        <div className="flex items-center gap-2.5 px-4 py-3 rounded-xl bg-slate-900/80 border border-slate-800/80 text-sm text-slate-300 animate-pulse my-3">
          <Loader2 className="w-4 h-4 text-indigo-400 animate-spin" />
          <span className="font-medium">Initializing research engine...</span>
        </div>
      );
    }
    return null;
  }

  const completedCount = steps.filter((s) => s.state === 'completed').length;
  const isFinished = !isStreaming && completedCount === steps.length;

  return (
    <div className="my-3 rounded-xl border border-slate-800/90 bg-slate-900/60 backdrop-blur-md overflow-hidden transition-all duration-200 shadow-sm">
      {/* Accordion Header */}
      <button
        type="button"
        onClick={() => setIsOpen((prev) => !prev)}
        className="w-full px-4 py-3 flex items-center justify-between text-left hover:bg-slate-800/40 transition-colors group"
      >
        <div className="flex items-center gap-2.5">
          {isStreaming ? (
            <div className="flex items-center justify-center w-5 h-5 rounded-full bg-indigo-500/10 text-indigo-400">
              <Loader2 className="w-3.5 h-3.5 animate-spin" />
            </div>
          ) : (
            <div className="flex items-center justify-center w-5 h-5 rounded-full bg-emerald-500/10 text-emerald-400">
              <Sparkles className="w-3.5 h-3.5" />
            </div>
          )}

          <div className="flex items-center gap-2">
            <span className="text-sm font-semibold text-slate-200 tracking-wide">
              {isStreaming ? 'Researching...' : 'Research Process'}
            </span>
            <span className="text-xs px-2 py-0.5 rounded-full bg-slate-800 text-slate-400 font-mono">
              {completedCount}/{steps.length} steps
            </span>
            {elapsedSeconds !== undefined && elapsedSeconds > 0 && !isStreaming && (
              <span className="text-xs text-slate-500 font-mono">({elapsedSeconds}s)</span>
            )}
          </div>
        </div>

        <div className="flex items-center gap-1.5 text-slate-400 group-hover:text-slate-200 text-xs">
          <span>{isOpen ? 'Hide process' : 'Show process'}</span>
          {isOpen ? <ChevronDown className="w-4 h-4" /> : <ChevronRight className="w-4 h-4" />}
        </div>
      </button>

      {/* Accordion Body */}
      {isOpen && (
        <div className="px-4 pb-3.5 pt-1 border-t border-slate-800/60 bg-slate-950/30">
          <div className="relative pl-3 border-l-2 border-slate-800/80 space-y-2.5 my-2">
            {steps.map((step, idx) => {
              const isLast = idx === steps.length - 1;
              const isActive = isStreaming && isLast;

              return (
                <div key={`${step.step}-${idx}`} className="flex items-start gap-3 relative group">
                  {/* Step status indicator icon */}
                  <div className="mt-0.5 shrink-0 -ml-[19px] bg-slate-900 ring-2 ring-slate-900 rounded-full">
                    {step.state === 'completed' ? (
                      <CheckCircle2 className="w-4 h-4 text-emerald-400 fill-emerald-950/40" />
                    ) : step.state === 'running' || isActive ? (
                      <Loader2 className="w-4 h-4 text-indigo-400 animate-spin" />
                    ) : step.state === 'error' ? (
                      <AlertCircle className="w-4 h-4 text-rose-400" />
                    ) : (
                      <CircleDashed className="w-4 h-4 text-slate-600" />
                    )}
                  </div>

                  {/* Step label & details */}
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center justify-between gap-2">
                      <span
                        className={`text-xs font-medium ${
                          step.state === 'completed'
                            ? 'text-slate-300'
                            : isActive
                            ? 'text-indigo-300 font-semibold'
                            : 'text-slate-400'
                        }`}
                      >
                        {step.title}
                      </span>
                    </div>

                    {step.details && (
                      <p className="text-[11px] text-slate-400/90 font-mono mt-0.5 truncate">
                        {step.details}
                      </p>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
};
