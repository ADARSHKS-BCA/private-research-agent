import React, { useState, useRef, useEffect } from 'react';
import { Send, Square, Sparkles } from 'lucide-react';

interface ChatInputProps {
  onSend: (message: string) => void;
  onStop?: () => void;
  isLoading: boolean;
  disabled?: boolean;
}

export const ChatInput: React.FC<ChatInputProps> = ({
  onSend,
  onStop,
  isLoading,
  disabled = false,
}) => {
  const [input, setInput] = useState('');
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Auto-resize textarea height
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${Math.min(
        textareaRef.current.scrollHeight,
        200
      )}px`;
    }
  }, [input]);

  // Focus on mount
  useEffect(() => {
    if (!isLoading && textareaRef.current) {
      textareaRef.current.focus();
    }
  }, [isLoading]);

  const handleSubmit = (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    if (!input.trim() || isLoading || disabled) return;

    onSend(input.trim());
    setInput('');
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  return (
    <div className="w-full max-w-4xl mx-auto px-4 pb-6 pt-2">
      <form
        onSubmit={handleSubmit}
        className="relative flex items-end rounded-2xl bg-slate-900/90 border border-slate-700/80 shadow-xl shadow-black/40 focus-within:border-indigo-500/80 focus-within:ring-2 focus-within:ring-indigo-500/20 transition-all backdrop-blur-lg"
      >
        <textarea
          ref={textareaRef}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={
            isLoading
              ? 'Autonomous research workflow is running...'
              : 'Ask any research question (e.g. "What are the latest approaches to agentic RAG?")...'
          }
          disabled={isLoading || disabled}
          rows={1}
          className="w-full resize-none bg-transparent px-4 py-3.5 text-sm md:text-base text-slate-100 placeholder-slate-500 focus:outline-none disabled:opacity-60 max-h-48 leading-relaxed font-normal"
        />

        <div className="flex items-center gap-2 p-2 shrink-0">
          {isLoading ? (
            <button
              type="button"
              onClick={onStop}
              className="flex items-center justify-center w-9 h-9 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-300 hover:text-white border border-slate-700 transition-colors shadow-sm"
              title="Stop research"
            >
              <Square className="w-4 h-4 fill-current text-rose-400" />
            </button>
          ) : (
            <button
              type="submit"
              disabled={!input.trim() || disabled}
              className={`flex items-center justify-center w-9 h-9 rounded-xl transition-all duration-200 shadow-sm ${
                input.trim() && !disabled
                  ? 'bg-indigo-600 hover:bg-indigo-500 text-white shadow-indigo-950/50'
                  : 'bg-slate-800 text-slate-500 cursor-not-allowed border border-slate-800'
              }`}
              title="Send question (Enter)"
            >
              <Send className="w-4 h-4" />
            </button>
          )}
        </div>
      </form>

      <div className="flex items-center justify-between px-2 pt-2 text-[11px] text-slate-400">
        <span>Press <kbd className="px-1.5 py-0.5 rounded bg-slate-800 border border-slate-700 text-slate-300 font-mono text-[10px]">Enter</kbd> to submit, <kbd className="px-1.5 py-0.5 rounded bg-slate-800 border border-slate-700 text-slate-300 font-mono text-[10px]">Shift + Enter</kbd> for newline</span>
        <span className="flex items-center gap-1">
          <Sparkles className="w-3 h-3 text-indigo-400" />
          Autonomous LangGraph Engine
        </span>
      </div>
    </div>
  );
};
