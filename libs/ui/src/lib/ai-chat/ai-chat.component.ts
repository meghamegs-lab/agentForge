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
import { HttpClient, HttpClientModule } from '@angular/common/http';
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
import { Subject } from 'rxjs';
import { of } from 'rxjs';
import { takeUntil, catchError } from 'rxjs/operators';

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

/** A saved conversation snapshot stored in localStorage */
export interface ConversationSnapshot {
  id: string;
  savedAt: string; // ISO string
  title: string; // First user message, truncated to 60 chars
  conversationId: string; // Fortio conversation_id for resuming context
  messages: AiChatMessage[];
}

// ── Component ────────────────────────────────────────────────────────────────

@Component({
  changeDetection: ChangeDetectionStrategy.OnPush,
  imports: [
    CommonModule,
    FormsModule,
    HttpClientModule,
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

  // True while waiting for agent response
  public isLoading = false;

  // Prompt tips panel open/close state
  public isPromptTipsOpen = false;

  // Conversation history panel open/close state
  public isHistoryOpen = false;

  // Loaded history snapshots from localStorage
  public historySnapshots: ConversationSnapshot[] = [];

  // The conversation_id returned by Fortio (lets agent remember context)
  private conversationId = '';

  private readonly unsubscribeSubject = new Subject<void>();
  private readonly HISTORY_KEY = 'fortio-chat-history';
  private readonly MAX_HISTORY = 10;

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
            { text: 'How did I do last month compared to year-to-date?' }
          ]
        },
        {
          tool: 'get_transactions',
          prompts: [
            { text: 'Show me my last 10 trades' },
            { text: 'How much did I pay in fees this year?' },
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
              text: 'Which stocks are costing me the most in transaction fees?'
            }
          ]
        },
        {
          tool: 'get_rebalancing_plan',
          prompts: [
            {
              text: 'I want a 60/30/10 stocks/bonds/cash split — what exactly do I need to buy and sell?'
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

  public constructor(
    private readonly changeDetectorRef: ChangeDetectorRef,
    private readonly http: HttpClient
  ) {}

  public ngOnInit(): void {
    this.historySnapshots = this.loadHistory();

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
        'What does my portfolio look like?',
        'How has my portfolio performed this year?',
        'Am I too concentrated in any sector?',
        'What fees are dragging down my returns?',
        'Give me a full portfolio health scorecard',
        'How is my portfolio positioned if rates keep rising?',
        'What rebalancing trades should I make?',
        'What patterns do you see in my trading behaviour?',
        'What are my top concentration or volatility risks?',
        "What's the current price of NVDA?"
      ]
    });
  }

  // ── Panel Toggle ────────────────────────────────────────────────────────────

  public togglePanel(): void {
    if (this.isPanelOpen) {
      this.saveCurrentSession();
    }
    this.isPanelOpen = !this.isPanelOpen;
    if (!this.isPanelOpen) {
      this.isPromptTipsOpen = false;
      this.isHistoryOpen = false;
    }
    this.changeDetectorRef.markForCheck();
  }

  public closePanel(): void {
    this.saveCurrentSession();
    this.isPanelOpen = false;
    this.isPromptTipsOpen = false;
    this.isHistoryOpen = false;
    this.changeDetectorRef.markForCheck();
  }

  // ── Prompt Tips ─────────────────────────────────────────────────────────────

  public togglePromptTips(): void {
    this.isPromptTipsOpen = !this.isPromptTipsOpen;
    if (this.isPromptTipsOpen) {
      this.isHistoryOpen = false; // close history when tips open
    }
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

  // ── History ─────────────────────────────────────────────────────────────────

  public toggleHistory(): void {
    this.isHistoryOpen = !this.isHistoryOpen;
    if (this.isHistoryOpen) {
      this.isPromptTipsOpen = false; // close tips when history opens
      this.historySnapshots = this.loadHistory();
    }
    this.changeDetectorRef.markForCheck();
  }

  /** Start a fresh conversation — saves the current one first */
  public startNewChat(): void {
    this.saveCurrentSession();
    this.conversationId = '';
    this.messages = [];
    this.inputText = '';
    this.isHistoryOpen = false;
    this.isPromptTipsOpen = false;
    this.ngOnInit(); // re-push the welcome message
    this.changeDetectorRef.markForCheck();
  }

  /** Restore a past conversation into the current view */
  public resumeConversation(snapshot: ConversationSnapshot): void {
    this.saveCurrentSession();
    // Deserialise Date objects (stored as ISO strings in JSON)
    this.messages = snapshot.messages.map((m) => ({
      ...m,
      timestamp: new Date(m.timestamp)
    }));
    this.conversationId = snapshot.conversationId;
    this.isHistoryOpen = false;
    this.changeDetectorRef.markForCheck();
    setTimeout(() => this.scrollToBottom(), 50);
  }

  /** Delete a single history entry without opening the conversation */
  public deleteSnapshot(id: string, event: Event): void {
    event.stopPropagation();
    const snapshots = this.loadHistory().filter((s) => s.id !== id);
    localStorage.setItem(this.HISTORY_KEY, JSON.stringify(snapshots));
    this.historySnapshots = snapshots;
    this.changeDetectorRef.markForCheck();
  }

  /** Human-readable relative time for a snapshot date */
  public formatSnapshotDate(isoString: string): string {
    const d = new Date(isoString);
    const diffMs = Date.now() - d.getTime();
    const diffMins = Math.floor(diffMs / 60000);
    if (diffMins < 1) return 'Just now';
    if (diffMins < 60) return `${diffMins}m ago`;
    const diffHrs = Math.floor(diffMins / 60);
    if (diffHrs < 24) return `${diffHrs}h ago`;
    const diffDays = Math.floor(diffHrs / 24);
    if (diffDays < 7) return `${diffDays}d ago`;
    return d.toLocaleDateString();
  }

  // ── Send Message ────────────────────────────────────────────────────────────

  public onSend(): void {
    const message = this.inputText.trim();
    if (!message || this.isLoading) {
      return;
    }

    // Add user message to the chat
    this.messages.push({
      role: 'user',
      content: message,
      timestamp: new Date()
    });
    this.inputText = '';
    this.isLoading = true;
    this.changeDetectorRef.markForCheck();

    // Call Fortio FastAPI
    this.http
      .post<FortioApiResponse>(`${this.fortioApiUrl}/api/chat`, {
        message,
        conversation_id: this.conversationId,
        user_id: 'ghostfolio-user'
      })
      .pipe(
        takeUntil(this.unsubscribeSubject),
        catchError(() => {
          // Network error — Fortio service might be down
          return of({
            answer:
              '⚠️ Could not reach the Fortio agent. Make sure it is running on port 8001.',
            confidence: 'LOW' as const,
            flags: [],
            tool_calls: [],
            conversation_id: this.conversationId
          });
        })
      )
      .subscribe((response) => {
        // Store conversation_id for continuity
        this.conversationId = response.conversation_id;

        // Check if any high/medium severity flags exist
        const hasWarning = response.flags.some(
          (f) => f.severity === 'HIGH' || f.severity === 'MEDIUM'
        );

        // Add Fortio's answer to the chat (with full structured response stored)
        this.messages.push({
          role: 'assistant',
          content: response.answer,
          confidence: response.confidence,
          hasWarning,
          timestamp: new Date(),
          rawResponse: response,
          showRaw: false
        });

        this.isLoading = false;
        this.changeDetectorRef.markForCheck();

        // Scroll to bottom after Angular renders the new message
        setTimeout(() => {
          this.scrollToBottom();
        }, 50);
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

  /** Save the current session to localStorage (only if it has user messages) */
  private saveCurrentSession(): void {
    const userMessages = this.messages.filter((m) => m.role === 'user');
    if (userMessages.length === 0) return;

    const rawTitle = userMessages[0].content;
    const title = rawTitle.length > 60 ? rawTitle.slice(0, 60) + '…' : rawTitle;

    const snapshot: ConversationSnapshot = {
      id: this.conversationId || `local-${Date.now()}`,
      savedAt: new Date().toISOString(),
      title,
      conversationId: this.conversationId,
      // Strip rawResponse to keep localStorage storage compact
      messages: this.messages.map((msg) => {
        const copy = { ...msg } as Partial<AiChatMessage>;
        delete copy.rawResponse;
        return copy as AiChatMessage;
      })
    };

    const existing = this.loadHistory().filter((s) => s.id !== snapshot.id);
    const updated = [snapshot, ...existing].slice(0, this.MAX_HISTORY);
    localStorage.setItem(this.HISTORY_KEY, JSON.stringify(updated));
    this.historySnapshots = updated;
  }

  /** Read conversation snapshots from localStorage */
  private loadHistory(): ConversationSnapshot[] {
    try {
      const raw = localStorage.getItem(this.HISTORY_KEY);
      return raw ? (JSON.parse(raw) as ConversationSnapshot[]) : [];
    } catch {
      return [];
    }
  }

  public ngOnDestroy(): void {
    this.unsubscribeSubject.next();
    this.unsubscribeSubject.complete();
  }
}
