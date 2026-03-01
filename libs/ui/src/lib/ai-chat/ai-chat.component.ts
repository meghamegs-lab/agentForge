/**
 * GfAiChatComponent — Fortio AI Chat Widget
 *
 * A floating chat button (bottom-right corner) that opens a full chat panel.
 * Connects to the Fortio FastAPI agent at /api/chat.
 *
 * Usage in any Ghostfolio page template:
 *   <gf-ai-chat [fortioApiUrl]="'http://localhost:8001'" />
 */
import { CommonModule } from '@angular/common';
import {
  ChangeDetectionStrategy,
  ChangeDetectorRef,
  Component,
  Input,
  OnDestroy,
  OnInit
} from '@angular/core';
import { FormsModule } from '@angular/forms';
import { MatButtonModule } from '@angular/material/button';
import { MatIconModule } from '@angular/material/icon';
import { MatProgressSpinnerModule } from '@angular/material/progress-spinner';
import { MatTooltipModule } from '@angular/material/tooltip';

// ── Types ────────────────────────────────────────────────────────────────────

export interface ToolCallInfo {
  tool_name: string;
  status: string; // "ok" | "error" | "empty"
  error?: string | null;
}

interface FortioApiResponse {
  answer: string;
  confidence: 'HIGH' | 'MEDIUM' | 'LOW';
  flags: { type: string; severity: string; message: string }[];
  tool_calls: ToolCallInfo[];
  conversation_id: string;
}

export interface AiChatMessage {
  role: 'user' | 'assistant';
  content: string;
  confidence?: 'HIGH' | 'MEDIUM' | 'LOW'; // Only on assistant messages
  hasWarning?: boolean; // True if any HIGH/MEDIUM flags
  timestamp: Date;
  // Full structured response — shown in the debug panel
  rawResponse?: FortioApiResponse;
  showRaw?: boolean; // Toggle state for debug panel
  suggestions?: string[]; // Clickable prompt chips (welcome msg)
  // Streaming state
  isStreaming?: boolean; // True while tokens are still arriving
  activeTools?: string[]; // Tool names currently running (shown as pills)
}

export interface PromptItem {
  text: string;
  note?: string; // "Forces: ..." shown for Hard prompts, "Expect: ..." for Edge cases
}

export interface PromptGroup {
  tool?: string; // tool name header (Simple & Advanced sections)
  prompts: PromptItem[];
}

export interface PromptCategory {
  emoji: string;
  level: string;
  description: string;
  groups: PromptGroup[];
}

// ── Component ────────────────────────────────────────────────────────────────

@Component({
  changeDetection: ChangeDetectionStrategy.OnPush,
  imports: [
    CommonModule,
    FormsModule,
    MatButtonModule,
    MatIconModule,
    MatProgressSpinnerModule,
    MatTooltipModule
  ],
  selector: 'gf-ai-chat',
  styleUrls: ['./ai-chat.scss'],
  templateUrl: './ai-chat.html'
})
export class GfAiChatComponent implements OnInit, OnDestroy {
  /**
   * URL of the Fortio FastAPI service.
   * Default: localhost:8001 for local dev.
   * Override with your Railway URL in production.
   */
  @Input() public fortioApiUrl = 'http://localhost:8001';

  // Chat panel open/close state
  public isPanelOpen = false;

  // All messages in the current conversation
  public messages: AiChatMessage[] = [];

  // The text currently typed in the input box
  public inputText = '';

  // True while waiting for agent response (disables input / send button)
  public isLoading = false;

  // True once the first streaming token has arrived (switches spinner → live text)
  public isStreamingActive = false;

  // Prompt tips panel open/close state
  public isPromptTipsOpen = false;

  // The conversation_id returned by Fortio (lets agent remember context)
  private conversationId = '';

  /** AbortController for the current in-flight streaming fetch; null when idle. */
  private currentAbortController: AbortController | null = null;

  /** requestAnimationFrame handle used to throttle change-detection during streaming. */
  private pendingRaf: number | null = null;

  // ── Prompt Tips Data ─────────────────────────────────────────────────────
  public readonly promptCategories: PromptCategory[] = [
    {
      emoji: '🟢',
      level: 'Simple',
      description: 'Single Tool',
      groups: [
        {
          tool: 'get_portfolio_summary',
          prompts: [
            { text: 'What does my portfolio look like right now?' },
            { text: 'What are my top 5 holdings by value?' },
            { text: 'How much is my portfolio worth in total?' }
          ]
        },
        {
          tool: 'get_performance',
          prompts: [
            { text: 'How has my portfolio performed this year?' },
            { text: 'What are my returns over the last 5 years?' },
            { text: 'How did I do last month vs year-to-date?' }
          ]
        },
        {
          tool: 'get_transactions',
          prompts: [
            {
              text: 'Show me my last 10 trades',
              note: 'Uses limit=10 parameter'
            },
            { text: 'Show me my fee transactions for this year' },
            { text: 'Show me all my dividend payments' },
            { text: 'What did I buy and sell in 2024?' }
          ]
        },
        {
          tool: 'get_market_data',
          prompts: [
            { text: 'What is the current price of AAPL?' },
            { text: 'Give me the 52-week range for MSFT, GOOGL, and NVDA' }
          ]
        },
        {
          tool: 'analyze_diversification',
          prompts: [
            { text: 'How diversified is my portfolio?' },
            { text: 'Am I too concentrated in any single stock or sector?' }
          ]
        }
      ]
    },
    {
      emoji: '🟡',
      level: 'Advanced',
      description: 'Multi-Step Tools',
      groups: [
        {
          tool: 'get_portfolio_health_scorecard',
          prompts: [
            {
              text: 'Give me an overall health assessment of my portfolio with a grade'
            },
            { text: 'What should I fix first in my portfolio?' }
          ]
        },
        {
          tool: 'get_fee_drag_analysis',
          prompts: [
            {
              text: 'How much have fees cost me as a percentage of my returns?'
            },
            {
              text: 'Which of my holdings have the highest fee drag on returns?'
            }
          ]
        },
        {
          tool: 'get_rebalancing_plan',
          prompts: [
            {
              text: 'I want 45% US stocks, 15% international, 30% bonds, 10% cash — what do I need to buy and sell?',
              note: 'Uses all 4 rebalancing parameters explicitly'
            },
            {
              text: 'How do I rebalance my portfolio? Give me specific dollar amounts'
            }
          ]
        },
        {
          tool: 'get_market_context_overlay',
          prompts: [
            { text: 'How would my portfolio hold up in a recession?' },
            { text: 'Am I hedged against rising interest rates?' },
            { text: 'Which of my holdings are most exposed to inflation?' }
          ]
        },
        {
          tool: 'get_transaction_pattern_intelligence',
          prompts: [
            {
              text: 'Am I a good investor? What patterns do you see in my trading history?'
            },
            { text: 'Do I tend to buy high and sell low?' }
          ]
        },
        {
          tool: 'get_proactive_risk_monitor',
          prompts: [
            {
              text: 'Do I have any urgent risks I should know about right now?'
            },
            { text: "What's changed in my portfolio since last time?" }
          ]
        }
      ]
    },
    {
      emoji: '🔴',
      level: 'Hard',
      description: 'Chain Multiple Tools',
      groups: [
        {
          prompts: [
            {
              text: 'My portfolio is down this year. Is it because of fees, bad diversification, or just market conditions?',
              note: 'Forces: get_performance + get_fee_drag_analysis + analyze_diversification'
            },
            {
              text: "Should I rebalance now, and if so, what's the macro environment I'm rebalancing into?",
              note: 'Forces: get_rebalancing_plan + get_market_context_overlay'
            },
            {
              text: "Give me a complete picture — my portfolio grade, what it's worth, and how it's positioned for a bull market",
              note: 'Forces: get_portfolio_health_scorecard + get_portfolio_summary + get_market_context_overlay'
            },
            {
              text: "I'm a buy-and-hold investor. Have my fees been worth it given my actual returns?",
              note: 'Forces: get_transaction_pattern_intelligence + get_fee_drag_analysis + get_performance'
            }
          ]
        }
      ]
    },
    {
      emoji: '⚠️',
      level: 'Edge Cases',
      description: 'Test the Verification Layer',
      groups: [
        {
          prompts: [
            {
              text: 'Should I sell AAPL and put it all in Bitcoin?',
              note: 'Expect: DISCLAIMER_ADDED flag, LOW confidence'
            },
            {
              text: 'Will my portfolio be worth $1 million next year?',
              note: 'Expect: LOW_CONFIDENCE flag (prediction language)'
            },
            {
              text: "What's the price of FAKESYMBOL123?",
              note: 'Expect: error from get_market_data, graceful fallback'
            }
          ]
        }
      ]
    }
  ];

  public constructor(private readonly changeDetectorRef: ChangeDetectorRef) {}

  public ngOnInit(): void {
    // Push a welcome message when the widget loads
    this.messages.push({
      role: 'assistant',
      content:
        "👋 Hi! I'm **Fortio**, your AI portfolio assistant.\n" +
        'Click any suggestion below or type your own question:',
      confidence: 'HIGH',
      hasWarning: false,
      timestamp: new Date(),
      suggestions: [
        // ── Core tools (1 chip each) ──────────────────────────────────────────
        'What does my portfolio look like right now?',
        'How has my portfolio performed this year?',
        'Show me my recent transactions',
        'Am I too concentrated in any single stock or sector?',
        'Get me the current price of AAPL and MSFT',
        // ── Advanced multi-step tools ─────────────────────────────────────────
        'How much have fees cost me as a percentage of my returns?',
        'Give me an overall health assessment of my portfolio with a grade',
        'How do I rebalance my portfolio? Give me specific dollar amounts',
        'How would my portfolio hold up in a recession?',
        'What patterns do you see in my trading history?',
        'Do I have any urgent risks I should know about right now?'
      ]
    });
  }

  // ── Panel Toggle ────────────────────────────────────────────────────────────

  public togglePanel(): void {
    this.isPanelOpen = !this.isPanelOpen;
    if (!this.isPanelOpen) {
      this.isPromptTipsOpen = false;
    }
    this.changeDetectorRef.markForCheck();
  }

  public closePanel(): void {
    this.isPanelOpen = false;
    this.isPromptTipsOpen = false;
    this.changeDetectorRef.markForCheck();
  }

  // ── Prompt Tips ─────────────────────────────────────────────────────────────

  public togglePromptTips(): void {
    this.isPromptTipsOpen = !this.isPromptTipsOpen;
    this.changeDetectorRef.markForCheck();
  }

  /** Select a prompt from the tips panel — fills input and closes tips */
  public onPromptSelect(text: string): void {
    this.inputText = text;
    this.isPromptTipsOpen = false;
    this.changeDetectorRef.markForCheck();
    setTimeout(() => {
      const ta =
        document.querySelector<HTMLTextAreaElement>('.gf-ai-chat-input');
      if (ta) {
        ta.focus();
      }
    }, 0);
  }

  /** Start a fresh conversation */
  public startNewChat(): void {
    this.conversationId = '';
    this.messages = [];
    this.inputText = '';
    this.isPromptTipsOpen = false;
    this.ngOnInit(); // re-push the welcome message
    this.changeDetectorRef.markForCheck();
  }

  // ── Send Message ────────────────────────────────────────────────────────────

  /**
   * Send a message to Fortio using the streaming `/api/chat/stream` SSE endpoint.
   *
   * UX flow:
   *   1. User message pushed immediately.
   *   2. Empty assistant "streaming" message pushed — shows a blinking cursor.
   *   3. `token` events append text character-by-character.
   *   4. `tool_start/tool_done` events update the active-tools pill list.
   *   5. `done` event finalises the message (confidence, flags, debug panel).
   *
   * Change detection is throttled to one requestAnimationFrame per batch of
   * tokens so OnPush stays snappy even at high streaming rates.
   */
  public async onSend(): Promise<void> {
    const message = this.inputText.trim();
    if (!message || this.isLoading) {
      return;
    }

    // Abort any prior in-flight stream (shouldn't normally happen)
    this.currentAbortController?.abort();
    this.currentAbortController = new AbortController();

    // Add user message to the chat
    this.messages.push({
      role: 'user',
      content: message,
      timestamp: new Date()
    });
    this.inputText = '';
    this.isLoading = true;
    this.isStreamingActive = false;
    this.changeDetectorRef.markForCheck();

    // Push empty streaming assistant message — it will fill in as tokens arrive
    const streamingMsg: AiChatMessage = {
      role: 'assistant',
      content: '',
      timestamp: new Date(),
      isStreaming: true,
      activeTools: []
    };
    this.messages.push(streamingMsg);
    const streamingIdx = this.messages.length - 1;

    try {
      const response = await fetch(`${this.fortioApiUrl}/api/chat/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message,
          conversation_id: this.conversationId,
          user_id: 'ghostfolio-user'
        }),
        signal: this.currentAbortController.signal
      });

      if (!response.ok || !response.body) {
        throw new Error(`HTTP ${response.status}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let lineBuffer = '';

      // eslint-disable-next-line no-constant-condition
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        lineBuffer += decoder.decode(value, { stream: true });
        const lines = lineBuffer.split('\n');
        lineBuffer = lines.pop() ?? ''; // keep the last incomplete line

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;
          const raw = line.slice(6).trim();
          if (!raw) continue;

          try {
            const evt = JSON.parse(raw);
            this.handleStreamEvent(evt, streamingIdx);
          } catch {
            /* ignore malformed SSE lines */
          }
        }
      }
    } catch (err: unknown) {
      const isAbort = err instanceof Error && err.name === 'AbortError';
      if (!isAbort) {
        const msg = this.messages[streamingIdx];
        if (msg) {
          msg.isStreaming = false;
          msg.content =
            '⚠️ Could not reach the Fortio agent. Make sure it is running on port 8001.';
        }
      }
    } finally {
      this.isLoading = false;
      this.isStreamingActive = false;
      this.currentAbortController = null;
      // Ensure final state is rendered
      this.scheduleUpdate(true);
      setTimeout(() => this.scrollToBottom(), 50);
    }
  }

  /**
   * Process one parsed SSE event and mutate the in-flight streaming message.
   * Schedules a batched change-detection update via requestAnimationFrame.
   */
  private handleStreamEvent(
    event: Record<string, unknown>,
    msgIdx: number
  ): void {
    const msg = this.messages[msgIdx];
    if (!msg) return;

    switch (event['type']) {
      case 'token': {
        const text = (event['content'] as string) ?? '';
        if (text) {
          msg.content += text;
          if (!this.isStreamingActive) {
            this.isStreamingActive = true;
          }
        }
        break;
      }

      case 'tool_start': {
        const name = (event['name'] as string) ?? 'tool';
        msg.activeTools = [...(msg.activeTools ?? []), name];
        break;
      }

      case 'tool_done': {
        const name = (event['name'] as string) ?? '';
        msg.activeTools = (msg.activeTools ?? []).filter((t) => t !== name);
        break;
      }

      case 'done': {
        msg.isStreaming = false;
        msg.activeTools = [];
        // Use the verified final answer (may have disclaimer prepended by verification)
        const answer = (event['answer'] as string) ?? msg.content;
        msg.content = answer;
        msg.confidence =
          (event['confidence'] as 'HIGH' | 'MEDIUM' | 'LOW') ?? 'MEDIUM';
        const flags = (event['flags'] as { severity: string }[]) ?? [];
        msg.hasWarning = flags.some(
          (f) => f.severity === 'HIGH' || f.severity === 'MEDIUM'
        );
        msg.rawResponse = {
          answer,
          confidence: msg.confidence,
          flags: flags as FortioApiResponse['flags'],
          tool_calls: (event['tool_calls'] as ToolCallInfo[]) ?? [],
          conversation_id: (event['conversation_id'] as string) ?? ''
        };
        msg.showRaw = false;
        this.conversationId =
          (event['conversation_id'] as string) ?? this.conversationId;
        break;
      }

      case 'error': {
        msg.isStreaming = false;
        msg.activeTools = [];
        msg.content = `⚠️ Error: ${(event['message'] as string) ?? 'Unknown error'}`;
        break;
      }
    }

    this.scheduleUpdate();
  }

  /**
   * Throttle change detection to one call per animation frame.
   * Prevents excessive re-renders when tokens arrive faster than 60 fps.
   * Pass `force = true` to skip throttling (e.g. on final done/error).
   */
  private scheduleUpdate(force = false): void {
    if (force) {
      if (this.pendingRaf !== null) {
        cancelAnimationFrame(this.pendingRaf);
        this.pendingRaf = null;
      }
      this.changeDetectorRef.markForCheck();
      return;
    }
    if (this.pendingRaf !== null) return; // already scheduled
    this.pendingRaf = requestAnimationFrame(() => {
      this.pendingRaf = null;
      this.changeDetectorRef.markForCheck();
    });
  }

  // Allow pressing Enter to send (Shift+Enter = newline)
  public onKeydown(event: KeyboardEvent): void {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      this.onSend();
    }
  }

  // ── Helpers ─────────────────────────────────────────────────────────────────

  /** Map confidence level to an emoji badge */
  public confidenceEmoji(confidence: string): string {
    const map: Record<string, string> = {
      HIGH: '🟢',
      MEDIUM: '🟡',
      LOW: '🔴'
    };
    return map[confidence] ?? '🟡';
  }

  /** Copy a suggestion chip text into the input box and focus the textarea */
  public onSuggestionClick(suggestion: string): void {
    this.inputText = suggestion;
    this.changeDetectorRef.markForCheck();
    setTimeout(() => {
      const ta =
        document.querySelector<HTMLTextAreaElement>('.gf-ai-chat-input');
      if (ta) {
        ta.focus();
      }
    }, 0);
  }

  /** Toggle the raw structured debug panel for a specific message */
  public toggleRaw(message: AiChatMessage): void {
    message.showRaw = !message.showRaw;
    this.changeDetectorRef.markForCheck();
  }

  /** Pretty-print the structured response for the debug panel */
  public formatRaw(response: FortioApiResponse): string {
    // Show the structured data without the full answer text to keep it compact
    const structured = { ...response } as Partial<FortioApiResponse>;
    delete structured.answer;
    return JSON.stringify(structured, null, 2);
  }

  /** Scroll the message list to the bottom after a new message arrives */
  private scrollToBottom(): void {
    const container = document.querySelector('.gf-ai-chat-messages');
    if (container) {
      container.scrollTop = container.scrollHeight;
    }
  }

  public ngOnDestroy(): void {
    // Abort any in-flight streaming fetch
    this.currentAbortController?.abort();
    if (this.pendingRaf !== null) {
      cancelAnimationFrame(this.pendingRaf);
    }
  }
}
