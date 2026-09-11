import React from 'react';
import { ExternalLink, Globe, BookOpen, Layers } from 'lucide-react';
import { CitationSource } from '../types/chat';

interface SourcesSectionProps {
  sources: CitationSource[];
}

export const SourcesSection: React.FC<SourcesSectionProps> = ({ sources = [] }) => {
  if (!sources || sources.length === 0) {
    return null;
  }

  // Deduplicate sources by URL while maintaining order
  const uniqueSources: CitationSource[] = [];
  const seenUrls = new Set<string>();

  sources.forEach((src) => {
    if (src.url && src.url !== 'N/A' && !seenUrls.has(src.url)) {
      seenUrls.add(src.url);
      uniqueSources.push(src);
    }
  });

  const getDomain = (url: string, defaultDomain?: string): string => {
    if (defaultDomain) return defaultDomain;
    try {
      const parsed = new URL(url);
      return parsed.hostname.replace(/^www\./, '');
    } catch {
      return 'Web Source';
    }
  };

  return (
    <div className="mt-6 pt-5 border-t border-slate-800/80">
      <div className="flex items-center justify-between gap-2 mb-3.5">
        <div className="flex items-center gap-2">
          <BookOpen className="w-4 h-4 text-indigo-400" />
          <h4 className="text-xs font-bold uppercase tracking-wider text-slate-300">
            Verified Sources & Research Papers ({uniqueSources.length})
          </h4>
        </div>
        <span className="text-[11px] text-slate-500 font-mono hidden sm:inline-block">
          All considered references
        </span>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-2 gap-3">
        {uniqueSources.map((src, index) => {
          const sourceId = src.source_id || `S${index + 1}`;
          const domain = getDomain(src.url, src.domain);

          return (
            <a
              key={`${src.url}-${index}`}
              href={src.url}
              target="_blank"
              rel="noopener noreferrer"
              className="group flex flex-col justify-between p-3.5 rounded-xl border border-slate-800/90 bg-slate-900/50 hover:bg-slate-800/70 hover:border-slate-700 transition-all duration-200 shadow-sm hover:shadow-md"
            >
              <div>
                <div className="flex items-start gap-2.5 mb-1.5">
                  <span className="shrink-0 px-2 py-0.5 rounded-md bg-indigo-500/15 text-indigo-300 font-mono text-xs font-bold border border-indigo-500/30 group-hover:bg-indigo-500/25 transition-colors">
                    [{sourceId}]
                  </span>
                  <span className="text-xs font-semibold text-slate-200 line-clamp-2 leading-relaxed group-hover:text-indigo-300 transition-colors">
                    {src.title || 'Untitled Research Source'}
                  </span>
                </div>

                {src.snippet && (
                  <p className="text-[11px] text-slate-400 line-clamp-2 mt-1 leading-relaxed pl-1 font-normal">
                    {src.snippet}
                  </p>
                )}
              </div>

              <div className="flex items-center justify-between mt-3 pt-2 border-t border-slate-800/60 text-[11px] text-slate-400">
                <div className="flex items-center gap-1.5 truncate max-w-[85%]">
                  <Globe className="w-3.5 h-3.5 text-slate-500 shrink-0" />
                  <span className="truncate text-slate-400 font-mono">{domain}</span>
                </div>
                <ExternalLink className="w-3.5 h-3.5 text-slate-500 group-hover:text-indigo-400 shrink-0 transition-colors" />
              </div>
            </a>
          );
        })}
      </div>
    </div>
  );
};
