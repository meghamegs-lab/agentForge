/**
 * GfAiChatComponent — Fortio AI Chat Widget
 *
 * A floating chat button (bottom-right corner) that opens a full chat panel.
 * Connects to the Fortio FastAPI agent at /api/chat.
 *
 * Usage in any Ghostfolio page template:
 *   <gf-ai-chat [fortioApiUrl]="'http://localhost:8001'" />
 */
import {
  ChangeDetectionStrategy,
  ChangeDetectorRef,
  Component,
  Input,
  OnDestroy,
  OnInit
} from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { HttpClient, HttpClientModule } from '@angular/common/http';
import { MatButtonModule } from '@angular/material/button';
import { MatIconModule } from '@angular/material/icon';
import { MatProgressSpinnerModule } from '@angular/material/progress-spinner';
import { MatTooltipModule } from '@angular/material/tooltip';
import { Subject } from 'rxjs';
import { takeUntil, catchError } from 'rxjs/operators';
import { of } from 'rxjs';

// ── Types ────────────────────────────────────────────────────────────────────

export interface ToolCallInfo {
  tool_name: string;
  status: string;         // "ok" | "error" | "empty"
  error?: string | null;
}

interface FortioApiResponse {
  answer: string;
  confidence: 'HIGH' | 'MEDIUM' | 'LOW';
  flags: Array<{ type: string; severity: string; message: string }>;
  tool_calls: ToolCallInfo[];
  conversation_id: string;
}

export interface AiChatMessage {
  role: 'user' | 'assistant';
  content: string;
  confidence?: 'HIGH' | 'MEDIUM' | 'LOW';   // Only on assistant messages
  hasWarning?: boolean;                        // True if any HIGH/MEDIUM flags
  timestamp: Date;
  // Full structured response — shown in the debug panel
  rawResponse?: FortioApiResponse;
  showRaw?: boolean;                           // Toggle state for debug panel
  suggestions?: string[];                      // Clickable prompt chips (welcome msg)
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

  // The conversation_id returned by Fortio (lets agent remember context)
  private conversationId = '';

  private readonly unsubscribeSubject = new Subject<void>();

  public constructor(
    private readonly changeDetectorRef: ChangeDetectorRef,
    private readonly http: HttpClient
  ) {}

  public ngOnInit(): void {
    // Push a welcome message when the widget loads
    this.messages.push({
      role: 'assistant',
      content:
        '👋 Hi! I\'m **Fortio**, your AI portfolio assistant.\n' +
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
    this.isPanelOpen = !this.isPanelOpen;
    this.changeDetectorRef.markForCheck();
  }

  public closePanel(): void {
    this.isPanelOpen = false;
    this.changeDetectorRef.markForCheck();
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
        catchError((_error) => {
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
      const ta = document.querySelector<HTMLTextAreaElement>('.gf-ai-chat-input');
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
    const { answer: _answer, ...structured } = response;
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
    this.unsubscribeSubject.next();
    this.unsubscribeSubject.complete();
  }
}
