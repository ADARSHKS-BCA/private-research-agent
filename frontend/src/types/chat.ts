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
  conversationId?: string;
}

export type SSEEventType = 'status' | 'token' | 'sources' | 'done' | 'error' | 'conversation';

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
  conversation_id?: string;
}

export interface SSEErrorData {
  message: string;
}

export interface SSEConversationData {
  conversation_id: string;
}

export interface UploadResult {
  status: string;
  filename: string;
  document_id?: string;
  document_type?: string;
  total_pages?: number;
  chunks_indexed: number;
  message: string;
}

export interface ConversationSummary {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
}
