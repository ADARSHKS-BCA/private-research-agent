import React, { useState, useRef, useEffect } from 'react';
import { Header } from './components/Header';
import { ChatContainer } from './components/ChatContainer';
import { ChatInput } from './components/ChatInput';
import {
  ChatMessage,
  ResearchStep,
  CitationSource,
  SSEStatusData,
  SSEDoneData,
  SSEConversationData,
} from './types/chat';
import {
  streamResearch,
  uploadDocument,
  fetchConversation,
} from './services/api';

export const App: React.FC = () => {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [conversationId, setConversationId] = useState<string | null>(() => {
    return localStorage.getItem('pra_conversation_id') || null;
  });
  const [isUploading, setIsUploading] = useState<boolean>(false);
  const [uploadStatus, setUploadStatus] = useState<string | null>(null);

  const abortControllerRef = useRef<AbortController | null>(null);

  // Load existing session history on initial mount if conversationId exists
  useEffect(() => {
    if (conversationId && messages.length === 0) {
      fetchConversation(conversationId)
        .then((data) => {
          if (data && Array.isArray(data.messages) && data.messages.length > 0) {
            const formattedMessages: ChatMessage[] = data.messages.map((m: any) => ({
              id: m.id || `msg-${Date.now()}`,
              role: m.role,
              content: m.content || '',
              steps: m.steps || [],
              sources: m.sources || [],
              isStreaming: false,
              timestamp: m.timestamp || Date.now(),
              conversationId: data.id,
            }));
            setMessages(formattedMessages);
          }
        })
        .catch(() => {
          // If conversation expired or not found, reset
          localStorage.removeItem('pra_conversation_id');
          setConversationId(null);
        });
    }
  }, []);

  const handleSend = async (question: string) => {
    if (!question.trim() || isLoading) return;

    const userMessageId = `user-${Date.now()}`;
    const assistantMessageId = `assistant-${Date.now()}`;

    const userMsg: ChatMessage = {
      id: userMessageId,
      role: 'user',
      content: question.trim(),
      timestamp: Date.now(),
      conversationId: conversationId || undefined,
    };

    const initialAssistantMsg: ChatMessage = {
      id: assistantMessageId,
      role: 'assistant',
      content: '',
      steps: [],
      sources: [],
      isStreaming: true,
      timestamp: Date.now(),
      conversationId: conversationId || undefined,
    };

    setMessages((prev) => [...prev, userMsg, initialAssistantMsg]);
    setIsLoading(true);

    const abortController = new AbortController();
    abortControllerRef.current = abortController;

    await streamResearch(
      question.trim(),
      {
        onConversation: (convData: SSEConversationData) => {
          setConversationId(convData.conversation_id);
          localStorage.setItem('pra_conversation_id', convData.conversation_id);
        },

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
          if (doneData.conversation_id) {
            setConversationId(doneData.conversation_id);
            localStorage.setItem('pra_conversation_id', doneData.conversation_id);
          }
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
      conversationId,
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
    setConversationId(null);
    localStorage.removeItem('pra_conversation_id');
    setMessages([]);
  };

  const handleRetry = (questionToRetry: string) => {
    if (!questionToRetry) return;
    handleSend(questionToRetry);
  };

  const handleUploadFile = async (file: File) => {
    try {
      setIsUploading(true);
      setUploadStatus(`Parsing and indexing ${file.name}...`);
      const result = await uploadDocument(file);
      setUploadStatus(
        `Indexed ${result.chunks_indexed} chunks from ${file.name} into knowledge base.`
      );
      setTimeout(() => {
        setUploadStatus(null);
      }, 5000);
    } catch (err: any) {
      setUploadStatus(`Upload failed: ${err.message || 'Error parsing document'}`);
      setTimeout(() => {
        setUploadStatus(null);
      }, 6000);
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <div className="flex flex-col h-screen bg-slate-950 text-slate-100 font-sans selection:bg-indigo-500/30 selection:text-indigo-200">
      {/* Top Header */}
      <Header onNewChat={handleNewChat} />

      {/* Main Research Chat Area */}
      <ChatContainer
        messages={messages}
        isLoading={isLoading}
        onSelectPrompt={handleSend}
        onRetry={handleRetry}
        conversationId={conversationId}
      />

      {/* Bottom Chat & File Upload Bar */}
      <ChatInput
        onSend={handleSend}
        onStop={handleStop}
        onUploadFile={handleUploadFile}
        isLoading={isLoading}
        isUploading={isUploading}
        uploadStatus={uploadStatus}
      />
    </div>
  );
};
