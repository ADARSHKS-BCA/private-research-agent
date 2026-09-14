import React, { useState, useRef, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import {
  User,
  Copy,
  Check,
  AlertTriangle,
  RotateCcw,
  Sparkles,
  Download,
  ChevronDown,
  FileText,
  FileCode,
  File,
} from 'lucide-react';
import { ChatMessage } from '../types/chat';
import { ThinkingProcess } from './ThinkingProcess';
import { SourcesSection } from './SourcesSection';
import { exportResearchReport } from '../services/api';

interface MessageItemProps {
  message: ChatMessage;
  onRetry?: (question: string) => void;
  isLastAssistant?: boolean;
  conversationId?: string | null;
}

/**
 * Helper to highlight citations like [S1], [S2] within text strings.
 */
function renderTextWithCitationBadges(text: string): React.ReactNode {
  if (!text) return text;

  // Split on [S1], [S2], [S12], etc.
  const parts = text.split(/(\[S\d+\])/g);
  if (parts.length === 1) return text;

  return parts.map((part, i) => {
    const match = part.match(/^\[(S\d+)\]$/);
    if (match) {
      return (
        <span
          key={i}
          className="inline-flex items-center px-1.5 py-0.5 mx-0.5 rounded bg-indigo-500/15 text-indigo-300 font-mono text-[11px] font-bold border border-indigo-500/30 shadow-xs cursor-default select-none align-baseline hover:bg-indigo-500/25 transition-colors"
          title={`Source ${match[1]}`}
        >
          {part}
        </span>
      );
    }
    return part;
  });
}

/**
 * Recursively apply citation badge styling to string children in React nodes.
 */
function enhanceChildrenWithCitations(children: React.ReactNode): React.ReactNode {
  return React.Children.map(children, (child) => {
    if (typeof child === 'string') {
      return renderTextWithCitationBadges(child);
    }
    return child;
  });
}

export const MessageItem: React.FC<MessageItemProps> = ({
  message,
  onRetry,
  isLastAssistant = false,
  conversationId = null,
}) => {
  const [copied, setCopied] = useState(false);
  const [showExportMenu, setShowExportMenu] = useState(false);
  const [isExporting, setIsExporting] = useState(false);
  const exportMenuRef = useRef<HTMLDivElement>(null);

  const isUser = message.role === 'user';
  const targetConvId = message.conversationId || conversationId;

  // Close export menu when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (exportMenuRef.current && !exportMenuRef.current.contains(event.target as Node)) {
        setShowExportMenu(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const handleCopy = async () => {
    if (!message.content) return;
    try {
      await navigator.clipboard.writeText(message.content);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error('Failed to copy text:', err);
    }
  };

  const handleExport = async (format: 'markdown' | 'json' | 'pdf') => {
    if (!targetConvId) return;
    try {
      setIsExporting(true);
      setShowExportMenu(false);
      await exportResearchReport(targetConvId, format);
    } catch (err) {
      console.error(`Export as ${format} failed:`, err);
    } finally {
      setIsExporting(false);
    }
  };

  if (isUser) {
    return (
      <div className="flex items-start justify-end gap-3 max-w-4xl mx-auto my-4 group">
        <div className="max-w-[85%] md:max-w-[75%] rounded-2xl px-4 py-3 bg-indigo-600 text-white shadow-md shadow-indigo-950/30">
          <p className="text-sm md:text-base whitespace-pre-wrap leading-relaxed">
            {message.content}
          </p>
        </div>
        <div className="w-8 h-8 rounded-full bg-slate-800 border border-slate-700 flex items-center justify-center text-slate-300 shrink-0 mt-0.5 shadow-sm">
          <User className="w-4 h-4" />
        </div>
      </div>
    );
  }

  return (
    <div className="flex items-start gap-3 max-w-4xl mx-auto my-6">
      {/* Assistant Avatar */}
      <div className="w-8 h-8 rounded-full bg-gradient-to-tr from-indigo-600 to-violet-500 flex items-center justify-center text-white shrink-0 mt-0.5 shadow-md shadow-indigo-950/40 border border-indigo-400/20">
        <Sparkles className="w-4 h-4" />
      </div>

      {/* Assistant Message Body */}
      <div className="flex-1 min-w-0 rounded-2xl p-5 bg-slate-900/70 border border-slate-800/90 backdrop-blur-sm shadow-sm relative group">
        {/* Thinking / Research Process Section */}
        {message.steps && message.steps.length > 0 && (
          <ThinkingProcess
            steps={message.steps}
            isStreaming={message.isStreaming}
            elapsedSeconds={message.elapsedSeconds}
          />
        )}

        {/* Error Notification */}
        {message.error ? (
          <div className="mt-3 p-4 rounded-xl bg-rose-950/40 border border-rose-800/60 text-rose-200">
            <div className="flex items-center gap-2.5">
              <AlertTriangle className="w-4 h-4 text-rose-400 shrink-0" />
              <span className="text-sm font-medium">
                {message.error || 'Something went wrong while researching this question.'}
              </span>
            </div>
            {onRetry && (
              <button
                type="button"
                onClick={() => onRetry(message.content || '')}
                className="mt-3 inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-rose-900/60 hover:bg-rose-900 border border-rose-700/50 text-xs font-semibold text-rose-100 transition-colors"
              >
                <RotateCcw className="w-3.5 h-3.5" />
                Retry Question
              </button>
            )}
          </div>
        ) : (
          <>
            {/* Streaming / Final Markdown Answer */}
            {message.content ? (
              <div className="prose prose-invert max-w-none prose-p:leading-relaxed prose-p:text-slate-200 prose-headings:text-slate-100 prose-pre:bg-slate-950 prose-pre:border prose-pre:border-slate-800 prose-code:text-indigo-300 prose-a:text-indigo-400 prose-a:no-underline hover:prose-a:underline text-sm md:text-base">
                <ReactMarkdown
                  remarkPlugins={[remarkGfm]}
                  components={{
                    p: ({ node, children, ...props }) => {
                      return <p {...props}>{enhanceChildrenWithCitations(children)}</p>;
                    },
                    li: ({ node, children, ...props }) => {
                      return <li {...props}>{enhanceChildrenWithCitations(children)}</li>;
                    },
                    code: ({ node, className, children, ...props }) => {
                      return (
                        <code className="bg-slate-950/80 px-1.5 py-0.5 rounded text-xs font-mono text-indigo-300 border border-slate-800" {...props}>
                          {children}
                        </code>
                      );
                    },
                  }}
                >
                  {message.content}
                </ReactMarkdown>

                {/* Pulsing cursor while streaming */}
                {message.isStreaming && (
                  <span className="inline-block w-2 h-4 ml-1 bg-indigo-400 animate-pulse rounded-sm align-middle" />
                )}
              </div>
            ) : message.isStreaming ? (
              <div className="text-sm text-slate-400 italic flex items-center gap-2">
                <span className="inline-block w-2 h-2 rounded-full bg-indigo-400 animate-ping" />
                Synthesizing research findings...
              </div>
            ) : null}

            {/* Verified Sources Section */}
            {message.sources && message.sources.length > 0 && (
              <SourcesSection sources={message.sources} />
            )}

            {/* Action Bar (Copy & Export buttons) */}
            {!message.isStreaming && message.content && (
              <div className="flex items-center justify-end gap-2 mt-4 pt-3 border-t border-slate-800/40">
                {/* Export Dropdown */}
                {targetConvId && (
                  <div className="relative" ref={exportMenuRef}>
                    <button
                      type="button"
                      onClick={() => setShowExportMenu((prev) => !prev)}
                      disabled={isExporting}
                      className="flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs text-slate-400 hover:text-slate-200 hover:bg-slate-800/60 transition-colors disabled:opacity-50"
                      title="Export research report"
                    >
                      <Download className="w-3.5 h-3.5" />
                      <span>Export</span>
                      <ChevronDown className="w-3 h-3 text-slate-500" />
                    </button>

                    {showExportMenu && (
                      <div className="absolute right-0 bottom-full mb-1 w-44 rounded-xl bg-slate-900 border border-slate-700/80 shadow-xl py-1 z-30 backdrop-blur-md animate-fade-in">
                        <button
                          type="button"
                          onClick={() => handleExport('markdown')}
                          className="w-full flex items-center gap-2 px-3 py-2 text-xs text-slate-200 hover:bg-slate-800 text-left transition-colors"
                        >
                          <FileCode className="w-3.5 h-3.5 text-indigo-400" />
                          <span>Markdown (.md)</span>
                        </button>
                        <button
                          type="button"
                          onClick={() => handleExport('json')}
                          className="w-full flex items-center gap-2 px-3 py-2 text-xs text-slate-200 hover:bg-slate-800 text-left transition-colors"
                        >
                          <FileText className="w-3.5 h-3.5 text-emerald-400" />
                          <span>JSON (.json)</span>
                        </button>
                        <button
                          type="button"
                          onClick={() => handleExport('pdf')}
                          className="w-full flex items-center gap-2 px-3 py-2 text-xs text-slate-200 hover:bg-slate-800 text-left transition-colors"
                        >
                          <File className="w-3.5 h-3.5 text-rose-400" />
                          <span>PDF Document (.pdf)</span>
                        </button>
                      </div>
                    )}
                  </div>
                )}

                {/* Copy Button */}
                <button
                  type="button"
                  onClick={handleCopy}
                  className="flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs text-slate-400 hover:text-slate-200 hover:bg-slate-800/60 transition-colors"
                  title="Copy answer"
                >
                  {copied ? (
                    <>
                      <Check className="w-3.5 h-3.5 text-emerald-400" />
                      <span className="text-emerald-400 font-medium">Copied</span>
                    </>
                  ) : (
                    <>
                      <Copy className="w-3.5 h-3.5" />
                      <span>Copy</span>
                    </>
                  )}
                </button>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
};
