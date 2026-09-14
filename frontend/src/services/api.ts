import {
  CitationSource,
  ConversationSummary,
  SSEConversationData,
  SSEDoneData,
  SSEErrorData,
  SSESourcesData,
  SSEStatusData,
  SSETokenData,
  UploadResult,
} from '../types/chat';

export interface StreamCallbacks {
  onStatus: (data: SSEStatusData) => void;
  onToken: (text: string) => void;
  onSources: (sources: CitationSource[]) => void;
  onDone: (data: SSEDoneData) => void;
  onError: (errorMsg: string) => void;
  onConversation?: (data: SSEConversationData) => void;
}

/**
 * Stream research execution from the FastAPI backend over SSE using Fetch & ReadableStream.
 */
export async function streamResearch(
  question: string,
  callbacks: StreamCallbacks,
  conversationId?: string | null,
  signal?: AbortSignal
): Promise<void> {
  const API_URL = '/api/research/stream';

  try {
    const payload: { question: string; max_iterations: number; conversation_id?: string } = {
      question,
      max_iterations: 3,
    };
    if (conversationId) {
      payload.conversation_id = conversationId;
    }

    const response = await fetch(API_URL, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Accept: 'text/event-stream',
      },
      body: JSON.stringify(payload),
      signal,
    });

    if (!response.ok) {
      const errorText = await response.text().catch(() => 'Network error');
      throw new Error(`Server returned ${response.status}: ${errorText}`);
    }

    if (!response.body) {
      throw new Error('ReadableStream not supported by browser or empty body returned.');
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder('utf-8');
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });

      // Split by double newline (SSE event boundary)
      const eventBlocks = buffer.split('\n\n');
      buffer = eventBlocks.pop() || '';

      for (const block of eventBlocks) {
        if (!block.trim()) continue;

        let eventType = 'message';
        let eventData = '';

        const lines = block.split('\n');
        for (const line of lines) {
          if (line.startsWith('event:')) {
            eventType = line.substring(6).trim();
          } else if (line.startsWith('data:')) {
            eventData += line.substring(5).trim();
          }
        }

        if (!eventData) continue;

        try {
          const parsed = JSON.parse(eventData);

          switch (eventType) {
            case 'conversation':
              if (callbacks.onConversation) {
                callbacks.onConversation(parsed as SSEConversationData);
              }
              break;
            case 'status':
              callbacks.onStatus(parsed as SSEStatusData);
              break;
            case 'token':
              callbacks.onToken((parsed as SSETokenData).text);
              break;
            case 'sources':
              callbacks.onSources((parsed as SSESourcesData).sources || []);
              break;
            case 'done':
              callbacks.onDone(parsed as SSEDoneData);
              break;
            case 'error':
              callbacks.onError((parsed as SSEErrorData).message || 'An error occurred during research.');
              break;
            default:
              break;
          }
        } catch (jsonErr) {
          console.warn('Failed to parse SSE payload:', eventData, jsonErr);
        }
      }
    }
  } catch (err: any) {
    if (err.name === 'AbortError') {
      console.log('Research stream was cancelled by user.');
      return;
    }
    console.error('SSE Stream Error:', err);
    callbacks.onError(err.message || 'Something went wrong while researching this question.');
  }
}

/**
 * Upload a local document (.pdf, .docx, .txt, .md) to be parsed and indexed into Qdrant.
 */
export async function uploadDocument(file: File): Promise<UploadResult> {
  const formData = new FormData();
  formData.append('file', file);

  const response = await fetch('/api/documents/upload', {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const errText = await response.text().catch(() => 'Upload failed');
    throw new Error(`Upload error (${response.status}): ${errText}`);
  }

  return response.json();
}

/**
 * Fetch past conversation sessions from SQLite store.
 */
export async function fetchConversations(limit: number = 30): Promise<ConversationSummary[]> {
  const response = await fetch(`/api/conversations?limit=${limit}`);
  if (!response.ok) {
    throw new Error('Failed to fetch conversation history');
  }
  const data = await response.json();
  return data.conversations || [];
}

/**
 * Fetch a single conversation session with full message history.
 */
export async function fetchConversation(conversationId: string): Promise<any> {
  const response = await fetch(`/api/conversations/${conversationId}`);
  if (!response.ok) {
    throw new Error('Failed to fetch conversation');
  }
  return response.json();
}

/**
 * Delete a conversation session.
 */
export async function deleteConversation(conversationId: string): Promise<void> {
  const response = await fetch(`/api/conversations/${conversationId}`, {
    method: 'DELETE',
  });
  if (!response.ok) {
    throw new Error('Failed to delete conversation');
  }
}

/**
 * Export research report for a conversation session and trigger browser download.
 */
export async function exportResearchReport(
  conversationId: string,
  format: 'markdown' | 'json' | 'pdf' = 'markdown'
): Promise<void> {
  const url = `/api/research/${conversationId}/export?format=${format}`;
  const response = await fetch(url);
  if (!response.ok) {
    const errorText = await response.text().catch(() => 'Export failed');
    throw new Error(`Export failed (${response.status}): ${errorText}`);
  }

  const blob = await response.blob();
  const downloadUrl = window.URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = downloadUrl;

  const ext = format === 'markdown' ? 'md' : format;
  a.download = `research_report_${conversationId.slice(0, 8)}.${ext}`;
  document.body.appendChild(a);
  a.click();
  window.URL.revokeObjectURL(downloadUrl);
  document.body.removeChild(a);
}
