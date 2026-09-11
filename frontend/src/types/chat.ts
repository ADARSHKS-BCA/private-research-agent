export interface CitationSource {
  source_id: string;
  title: string;
  url: string;
  domain?: string;
  document_id?: string;
  chunk_id?: string;
  score?: number;
  snippet?: string;
}

export interface ResearchStep {
  step: string;
  title: string;
  state: 'pending' | 'running' | 'completed' | 'error';
  details?: string;
  timestamp?: number;
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  steps?: ResearchStep[];
  sources?: CitationSource[];
  isStreaming?: boolean;
  error?: string;
  elapsedSeconds?: number;
  timestamp: number;
}

export type SSEEventType = 'status' | 'token' | 'sources' | 'done' | 'error';

export interface SSEStatusData {
  step: string;
  title: string;
  state?: 'pending' | 'running' | 'completed' | 'error';
  details?: string;
}

export interface SSETokenData {
  text: string;
}

export interface SSESourcesData {
  sources: CitationSource[];
}

export interface SSEDoneData {
  elapsed_seconds?: number;
}

export interface SSEErrorData {
  message: string;
}
