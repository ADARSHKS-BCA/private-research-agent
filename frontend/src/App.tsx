import React, { useState, useRef } from 'react';
import { Header } from './components/Header';
import { ChatContainer } from './components/ChatContainer';
import { ChatInput } from './components/ChatInput';
import { ChatMessage, ResearchStep, CitationSource, SSEStatusData, SSEDoneData } from './types/chat';
import { streamResearch } from './services/api';

export const App: React.FC = () => {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const abortControllerRef = useRef<AbortController | null>(null);

  const handleSend = async (question: string) => {
    if (!question.trim() || isLoading) return;

    const userMessageId = `user-${Date.now()}`;
    const assistantMessageId = `assistant-${Date.now()}`;

    const userMsg: ChatMessage = {
      id: userMessageId,
      role: 'user',
      content: question.trim(),
      timestamp: Date.now(),
    };

    const initialAssistantMsg: ChatMessage = {
      id: assistantMessageId,
      role: 'assistant',
      content: '',
      steps: [],
      sources: [],
      isStreaming: true,
      timestamp: Date.now(),
    };

    setMessages((prev) => [...prev, userMsg, initialAssistantMsg]);
    setIsLoading(true);

    const abortController = new AbortController();
    abortControllerRef.current = abortController;

    await streamResearch(
      question.trim(),
      {
        onStatus: (statusData: SSEStatusData) => {
          setMessages((prev) =>
            prev.map((msg) => {
              if (msg.id !== assistantMessageId) return msg;

              const existingSteps = msg.steps || [];
              const stepIndex = existingSteps.findIndex((s) => s.step === statusData.step);

              let updatedSteps: ResearchStep[];
              if (stepIndex >= 0) {
                updatedSteps = existingSteps.map((s, idx) =>
                  idx === stepIndex
                    ? {
                        ...s,
                        title: statusData.title || s.title,
                        state: statusData.state || 'completed',
                        details: statusData.details !== undefined ? statusData.details : s.details,
                      }
                    : s
                );
              } else {
                updatedSteps = [
                  ...existingSteps,
                  {
                    step: statusData.step,
                    title: statusData.title,
                    state: statusData.state || 'completed',
                    details: statusData.details,
                    timestamp: Date.now(),
                  },
                ];
              }

              return {
                ...msg,
                steps: updatedSteps,
              };
            })
          );
        },

        onToken: (tokenText: string) => {
          setMessages((prev) =>
            prev.map((msg) => {
              if (msg.id !== assistantMessageId) return msg;
              return {
                ...msg,
                content: (msg.content || '') + tokenText,
              };
            })
          );
        },

        onSources: (sourcesList: CitationSource[]) => {
          setMessages((prev) =>
            prev.map((msg) => {
              if (msg.id !== assistantMessageId) return msg;
              return {
                ...msg,
                sources: sourcesList,
              };
            })
          );
        },

        onDone: (doneData: SSEDoneData) => {
          setMessages((prev) =>
            prev.map((msg) => {
              if (msg.id !== assistantMessageId) return msg;
              return {
                ...msg,
                isStreaming: false,
                elapsedSeconds: doneData.elapsed_seconds,
              };
            })
          );
          setIsLoading(false);
          abortControllerRef.current = null;
        },

        onError: (errorMessage: string) => {
          setMessages((prev) =>
            prev.map((msg) => {
              if (msg.id !== assistantMessageId) return msg;
              return {
                ...msg,
                isStreaming: false,
                error: errorMessage,
              };
            })
          );
          setIsLoading(false);
          abortControllerRef.current = null;
        },
      },
      abortController.signal
    );
  };

  const handleStop = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
    setIsLoading(false);
    setMessages((prev) =>
      prev.map((msg) => (msg.isStreaming ? { ...msg, isStreaming: false } : msg))
    );
  };

  const handleNewChat = () => {
    handleStop();
    setMessages([]);
  };

  const handleRetry = (questionToRetry: string) => {
    if (questionToRetry) {
      handleSend(questionToRetry);
    }
  };

  return (
    <div className="flex flex-col h-screen w-full bg-slate-950 text-slate-100 font-sans antialiased overflow-hidden">
      {/* App Header */}
      <Header onNewChat={handleNewChat} disabled={isLoading} />

      {/* Main Chat Area */}
      <main className="flex-1 flex flex-col min-h-0 relative overflow-hidden">
        <ChatContainer
          messages={messages}
          isLoading={isLoading}
          onSelectPrompt={handleSend}
          onRetry={handleRetry}
        />

        {/* Input Bar */}
        <ChatInput
          onSend={handleSend}
          onStop={handleStop}
          isLoading={isLoading}
        />
      </main>
    </div>
  );
};

export default App;
