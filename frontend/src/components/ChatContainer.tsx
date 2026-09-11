import React, { useRef, useEffect } from 'react';
import { ChatMessage } from '../types/chat';
import { MessageItem } from './MessageItem';
import { WelcomeSuggestions } from './WelcomeSuggestions';

interface ChatContainerProps {
  messages: ChatMessage[];
  isLoading: boolean;
  onSelectPrompt: (prompt: string) => void;
  onRetry: (question: string) => void;
}

export const ChatContainer: React.FC<ChatContainerProps> = ({
  messages,
  isLoading,
  onSelectPrompt,
  onRetry,
}) => {
  const bottomRef = useRef<HTMLDivElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom when messages or content updates
  useEffect(() => {
    if (bottomRef.current) {
      bottomRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages, isLoading]);

  if (messages.length === 0) {
    return <WelcomeSuggestions onSelectPrompt={onSelectPrompt} />;
  }

  return (
    <div
      ref={containerRef}
      className="flex-1 overflow-y-auto px-4 py-6 md:px-6 custom-scrollbar"
    >
      <div className="max-w-4xl mx-auto space-y-6">
        {messages.map((message, index) => (
          <MessageItem
            key={message.id || index}
            message={message}
            onRetry={onRetry}
            isLastAssistant={
              message.role === 'assistant' && index === messages.length - 1
            }
          />
        ))}
        <div ref={bottomRef} className="h-4" />
      </div>
    </div>
  );
};
