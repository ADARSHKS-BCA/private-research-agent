import {
  CitationSource,
  ResearchStep,
  SSEStatusData,
  SSETokenData,
  SSESourcesData,
  SSEDoneData,
  SSEErrorData,
} from '../types/chat';

export interface StreamCallbacks {
  onStatus: (data: SSEStatusData) => void;
  onToken: (text: string) => void;
  onSources: (sources: CitationSource[]) => void;
  onDone: (data: SSEDoneData) => void;
  onError: (errorMsg: string) => void;
}

/**
 * Stream research execution from the FastAPI backend over SSE using Fetch & ReadableStream.
 */
export async function streamResearch(
  question: string,
  callbacks: StreamCallbacks,
  signal?: AbortSignal
): Promise<void> {
  const API_URL = '/api/research/stream';

  try {
    const response = await fetch(API_URL, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Accept: 'text/event-stream',
      },
      body: JSON.stringify({ question, max_iterations: 3 }),
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
