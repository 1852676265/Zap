import { memo } from 'preact/compat';
import { useMemo, useState } from 'preact/hooks';
import type { VNode } from 'preact';
import { useT } from '../../settings';
import { formatTurnElapsed } from '../../lib/turnTimer';
import type { TodoItem } from '../../lib/todos';
import { copyTextToClipboard } from '../../lib/clipboard';
import { showToast } from '../Toast';
import { Markdown } from '../Markdown';
import { ReasoningBlock } from './ReasoningBlock';
import { SessionTodoPanel } from './SessionTodoPanel';
import { NoticeCard } from './NoticeCard';
import { ElapsedBadge } from './ElapsedBadge';
import { LiveElapsedText } from './LiveElapsedText';
import { highlightText } from './highlight';
import { messageHasTools, messageText } from './messageText';
import { usePluginActions, zapHostSDK } from '../../custom/plugin-kernel';
import type { CanvasRecord as Message } from '../../lib/canvas/records';
import type { MsgPart } from '../../lib/toolRows';

/** Modern display-only chip: this assistant round runs on a model that was
 *  switched in at the round boundary. Never part of the model context. */
function ModelSwitchBadge({ model }: { model: string }) {
  const t = useT();
  return (
    <div class="model-switch-badge notranslate" translate={false} role="note">
      <svg
        class="model-switch-icon"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        stroke-width="1.8"
        stroke-linecap="round"
        stroke-linejoin="round"
        aria-hidden="true"
      >
        <path d="M4 7h11l-3-3" />
        <path d="M20 17H9l3 3" />
      </svg>
      <span class="model-switch-label">{t('chat.modelSwitch')}</span>
      <code class="model-switch-model">{model}</code>
    </div>
  );
}

function isIgnoredChatNotice(text?: string | null): boolean {
  if (!text) return false;
  const s = text.trim();
  return (
    s.includes('已折叠工具输出') ||
    s.includes('Folded tool output') ||
    s.includes('Đã thu gọn đầu ra công cụ') ||
    (s.includes('已折叠') && s.includes('tok')) ||
    (s.includes('Folded') && s.includes('tok'))
  );
}

function assistantMessagePropsEqual(prev: any, next: any): boolean {
  const affectsActiveTurn = prev.isLast || prev.isLastInTurn || next.isLast || next.isLastInTurn;
  const affectsTurnTail = prev.isLastInTurn || next.isLastInTurn;
  return prev.msg === next.msg
    && prev.isLast === next.isLast
    && (!affectsActiveTurn || prev.busy === next.busy)
    && prev.isLastInTurn === next.isLastInTurn
    && prev.isCompacting === next.isCompacting
    && prev.isActiveSearchMatch === next.isActiveSearchMatch
    && prev.timeLabel === next.timeLabel
    && prev.timeFull === next.timeFull
    && prev.liveAnchorMs === next.liveAnchorMs
    && prev.liveElapsedMs === next.liveElapsedMs
    && prev.turnTotalMs === next.turnTotalMs
    && prev.search === next.search
    && (!affectsTurnTail || prev.turnLastText === next.turnLastText)
    && (!affectsTurnTail || prev.turnAllText === next.turnAllText)
    && (!affectsTurnTail || prev.turnReasoning === next.turnReasoning)
    && prev.cwd === next.cwd;
}

/** Render an assistant message cleanly: aggregated reasoning capsule at the top of
 *  the turn's final card, markdown prose in chronological order, and tool rows
 *  hidden from the main chat timeline to avoid multi-card stacking. */
export const AssistantMessageView = memo(function AssistantMessageView({
  msg,
  isLast,
  busy,
  lastIdx: _lastIdx,
  isLastInTurn,
  turnLastText,
  turnAllText,
  turnReasoning,
  searchRef,
  timeLabel,
  timeFull,
  liveAnchorMs,
  liveElapsedMs,
  turnTotalMs,
  isCompacting,
  search,
  isActiveSearchMatch,
  onRegenerate,
  onDelete,
  cwd,
}: {
  msg: Message;
  isLast: boolean;
  busy: boolean;
  lastIdx: number;
  isLastInTurn: boolean;
  turnLastText: string;
  turnAllText: string;
  turnReasoning?: string;
  searchRef?: (el: HTMLElement | null) => void;
  timeLabel?: string;
  timeFull?: string;
  liveAnchorMs?: number;
  liveElapsedMs?: number;
  /** User-bubble → this final answer. Only set on the last assistant of the turn. */
  turnTotalMs?: number;
  isCompacting?: boolean;
  search: string;
  isActiveSearchMatch: boolean;
  onRegenerate?: () => void;
  onDelete?: () => void;
  cwd?: string;
}) {
  const t = useT();
  const text = messageText(msg);
  const resolvedActionText = turnLastText || turnAllText || text;
  const pluginContext = useMemo(
    () => ({
      message: msg,
      isLast,
      isLastInTurn,
      text: resolvedActionText,
    }),
    [msg, isLast, isLastInTurn, resolvedActionText],
  );
  const pluginActions = usePluginActions('chat.message:actions', pluginContext);
  const trimmed = text.trim();
  const isError =
    !messageHasTools(msg) &&
    !msg.parts.some((p) => p.kind === 'reasoning') &&
    (trimmed.startsWith('[错误:') ||
      trimmed.startsWith('[连接错误:') ||
      trimmed.startsWith('[Error:') ||
      trimmed.startsWith('[Connection error:'));
  const streaming = isLast && busy;
  // 终条且简短（无工具、单行）时，去掉多余 of "时间线末端"橙点，只留一个起始点。
  const terse =
    isLast && !streaming && !messageHasTools(msg) && !text.includes('\n');
  const dotClass = isError ? 'dot-error' : 'dot-brand';
  const cls =
    'timeline-message ' +
    dotClass +
    (streaming ? ' dot-blink' : '') +
    (isLast ? ' is-last' : '') +
    (terse ? ' is-terse' : '') +
    (isActiveSearchMatch ? ' is-active-search-match' : '');
  // 本轮还在跑：画布上**所有**本轮助手行都不能被离屏优化跳过绘制。
  // 旧实现只护 `is-last`：模型在代码块闭合后继续说话时会新开一行，前一行
  // 立刻失去 is-last → `content-visibility` 回到 auto → 代码框被跳过绘制，
  // 表现就是“闭合那一瞬代码块消失，stop 后才又出现”。
  // `is-last-in-turn` 比 `is-last` 更贴近“这一轮还在输出”的语义。
  const liveCls = busy && isLastInTurn ? ' is-live' : '';

  // Copy buttons: only shown on the last assistant message in a turn.
  // 1. 复制最后正文回答 (Copy final response)
  // 2. 复制所有正文 (Copy all responses)
  const [copiedLast, setCopiedLast] = useState(false);
  const [copiedAll, setCopiedAll] = useState(false);

  function handleCopyLast() {
    const target = turnLastText || turnAllText;
    if (!target) return;
    void copyTextToClipboard(target).then((ok) => {
      if (ok) {
        setCopiedLast(true);
        setTimeout(() => setCopiedLast(false), 2000);
      } else {
        // Surface failure — silent .catch previously made the button look dead.
        showToast(t('copy.failed'), 'error');
      }
    });
  }

  function handleCopyAll() {
    const target = turnAllText || turnLastText;
    if (!target) return;
    void copyTextToClipboard(target).then((ok) => {
      if (ok) {
        setCopiedAll(true);
        setTimeout(() => setCopiedAll(false), 2000);
      } else {
        showToast(t('copy.failed'), 'error');
      }
    });
  }

  const copyBtn = isLastInTurn && !isError && !streaming && (turnLastText || turnAllText || onRegenerate || onDelete) ? (
    <div class="msg-actions msg-actions-left assistant-footer-actions">
      {(turnLastText || turnAllText) && (
        <>
          <button
            class={'msg-copy-btn' + (copiedLast ? ' copied' : '')}
            onClick={handleCopyLast}
            title={copiedLast ? t('copy.copiedLast') : t('copy.copyLast')}
            aria-label={copiedLast ? t('copy.copiedLast') : t('copy.copyLast')}
          >
            {copiedLast ? (
              <svg width="14" height="14" viewBox="0 0 16 16" fill="none" aria-hidden="true">
                <path d="M3.5 8.5 6.5 11.5 12.5 4.5" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" />
              </svg>
            ) : (
              <svg width="14" height="14" viewBox="0 0 16 16" fill="none" aria-hidden="true">
                <rect x="5" y="5" width="8.5" height="8.5" rx="1.5" stroke="currentColor" stroke-width="1.2" />
                <path d="M2.5 10.5V3.5A1.5 1.5 0 0 1 4 2h7" stroke="currentColor" stroke-width="1.2" stroke-linecap="round" />
              </svg>
            )}
          </button>
          <button
            class={'msg-copy-btn' + (copiedAll ? ' copied' : '')}
            onClick={handleCopyAll}
            title={copiedAll ? t('copy.copiedAll') : t('copy.copyAll')}
            aria-label={copiedAll ? t('copy.copiedAll') : t('copy.copyAll')}
          >
            {copiedAll ? (
              <svg width="14" height="14" viewBox="0 0 16 16" fill="none" aria-hidden="true">
                <path d="M3.5 8.5 6.5 11.5 12.5 4.5" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" />
              </svg>
            ) : (
              <svg width="14" height="14" viewBox="0 0 16 16" fill="none" aria-hidden="true">
                <rect x="4.5" y="4.5" width="9" height="9" rx="1.5" stroke="currentColor" stroke-width="1.2" />
                <path d="M2.5 10.5V3A1.5 1.5 0 0 1 4 1.5h6.5" stroke="currentColor" stroke-width="1.2" stroke-linecap="round" />
                <path d="M7 7.5h4M7 9.5h4M7 11.5h2.5" stroke="currentColor" stroke-width="1.1" stroke-linecap="round" />
              </svg>
            )}
          </button>
        </>
      )}
      {onRegenerate && (
        <button
          type="button"
          class="msg-action-btn btn-regenerate"
          onClick={onRegenerate}
          title={t('chat.regenerate')}
          aria-label={t('chat.regenerate')}
        >
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <polyline points="1 4 1 10 7 10" />
            <path d="M3.51 15a9 9 0 1 0 2.13-9.36L1 10" />
          </svg>
        </button>
      )}
      {onDelete && (
        <button
          type="button"
          class="msg-action-btn btn-delete"
          onClick={onDelete}
          title={t('chat.deleteMessage')}
          aria-label={t('chat.deleteMessage')}
        >
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
            <polyline points="3 6 5 6 21 6" />
            <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
          </svg>
        </button>
      )}
    </div>
  ) : null;

  const effectiveReasoning = isLastInTurn ? (turnReasoning || '') : '';
  const hasAnswerText = msg.parts.some((p) => p.kind === 'text' && p.text.trim().length > 0);
  const hasPendingTool = msg.parts.some((p) => p.kind === 'tool' && p.tool.status === 'pending');
  let lastMeaningfulKind: MsgPart['kind'] | null = null;
  for (let idx = msg.parts.length - 1; idx >= 0; idx--) {
    const p = msg.parts[idx]!;
    if (p.kind === 'text' && p.text.trim().length > 0) {
      lastMeaningfulKind = 'text';
      break;
    }
    if (p.kind === 'reasoning' && p.text.trim().length > 0) {
      lastMeaningfulKind = 'reasoning';
      break;
    }
    if (p.kind === 'tool') {
      lastMeaningfulKind = 'tool';
      break;
    }
    if (p.kind === 'notice' && p.text.trim().length > 0 && !isIgnoredChatNotice(p.text)) {
      lastMeaningfulKind = 'notice';
      break;
    }
    if (p.kind === 'rate_limited' && p.text.trim().length > 0) {
      lastMeaningfulKind = 'rate_limited';
      break;
    }
  }
  const thinkingLive =
    streaming &&
    !isCompacting &&
    (!hasAnswerText || hasPendingTool || (lastMeaningfulKind !== null && lastMeaningfulKind !== 'text'));

  const hasContent =
    effectiveReasoning.trim().length > 0 ||
    hasAnswerText ||
    msg.parts.some(
      (p) =>
        (p.kind === 'reasoning' && p.text.trim().length > 0) ||
        (p.kind === 'notice' && p.text.trim().length > 0 && !isIgnoredChatNotice(p.text)) ||
        (p.kind === 'rate_limited' && p.text.trim().length > 0) ||
        (p.kind === 'todo_list' && p.items && p.items.length > 0),
    );

  // 纯工具调用或无可见正文的消息在回合结束后不再渲染成堆叠卡片；
  // 整轮思考已聚合到终条消息，中间无正文的轮次直接跳过。
  if (!isLastInTurn && !streaming && !hasAnswerText) {
    return null;
  }
  if (!hasContent && !streaming) {
    return null;
  }

  const showDoneTime = isLastInTurn && timeLabel && !streaming && !isError;
  const showDoneFallbackTime =
    isLastInTurn && !timeLabel && !streaming && turnTotalMs != null && !isError;
  const showLiveTime =
    isLastInTurn && streaming && (liveAnchorMs != null || liveElapsedMs != null);
  const hasFooterRow = Boolean(
    showDoneTime || showDoneFallbackTime || showLiveTime || msg.readOnly === true || copyBtn,
  );

  return (
    <div class={cls + liveCls} ref={searchRef}>
      {msg.switchTo ? <ModelSwitchBadge model={msg.switchTo} /> : null}
      {/* Error turns are pure injected text — render flat. */}
      {isError ? (
        <div class="error-message-content">
          {highlightText(text, search)}
          {streaming && !hasContent && (
            <span class="codex-shimmer-status" aria-live="polite">
              {t('chat.statusWorking')}
            </span>
          )}
        </div>
      ) : (
        <>
          {renderAssistantParts(
            msg.parts,
            search,
            cwd,
            isLastInTurn ? turnReasoning : undefined,
            thinkingLive,
            hasPendingTool,
          )}
          {streaming && !hasContent && !thinkingLive && (
            <span class="codex-shimmer-status" aria-live="polite">
              {isCompacting ? t('chat.statusCompacting') : t('chat.statusWorking')}
            </span>
          )}
          {streaming && (hasContent || thinkingLive) && isCompacting && (
            <span class="codex-shimmer-status" aria-live="polite" style={{ display: 'block', marginTop: '6px' }}>
              {t('chat.statusCompacting')}
            </span>
          )}
        </>
      )}
      {hasFooterRow && (
        <div class="msg-footer-row assistant-footer-row">
          {/* 只在本轮最后一条助手消息上显示时刻和用时。中间的工具/思考/分段正文不渲染空脚栏，避免工具块下方留白过大。 */}
          {showDoneTime && (
            <span class="msg-time" title={timeFull}>
              <span class="msg-clock">{timeLabel}</span>
              {turnTotalMs != null && (
                <ElapsedBadge
                  ms={turnTotalMs}
                  label={t('chat.turnElapsedDone', { time: formatTurnElapsed(turnTotalMs) })}
                />
              )}
            </span>
          )}
          {showDoneFallbackTime && (
            <span class="msg-time">
              <ElapsedBadge
                ms={turnTotalMs!}
                label={t('chat.turnElapsedDone', { time: formatTurnElapsed(turnTotalMs!) })}
              />
            </span>
          )}
          {showLiveTime && (
            liveAnchorMs != null ? (
              <LiveElapsedText anchorMs={liveAnchorMs} />
            ) : (
              <span class="msg-time is-live">
                <ElapsedBadge
                  ms={liveElapsedMs!}
                  live
                  label={t('chat.turnElapsedLive', { time: formatTurnElapsed(liveElapsedMs!) })}
                />
              </span>
            )
          )}
          {msg.readOnly === true && (
            <span class="msg-readonly-badge" title={t('chat.readOnlyArchiveHint')}>
              {t('chat.readOnlyArchive')}
            </span>
          )}
          {pluginActions.map((act) => (
            <button
              key={act.id}
              type="button"
              class="msg-action-btn plugin-msg-action-btn"
              onClick={() =>
                void act.execute(
                  { message: msg, isLast, isLastInTurn, text: turnLastText || turnAllText || text },
                  zapHostSDK,
                )
              }
              title={act.title}
              aria-label={act.title}
            >
              {typeof act.icon === 'function' ? act.icon() : act.icon}
            </button>
          ))}
          {copyBtn}
        </div>
      )}
    </div>
  );
}, assistantMessagePropsEqual);

function renderAssistantParts(
  parts: MsgPart[],
  search: string,
  cwd?: string,
  overrideReasoning?: string,
  thinkingLive = false,
  hasPendingTool = false,
): VNode[] {
  const out: VNode[] = [];
  let trailingTodos: { kind: 'todo_list'; items: TodoItem[] } | null = null;

  // 整轮所有思考聚合为一个单一思考胶囊，消除多轮卡片堆叠
  let combinedReasoning = (overrideReasoning || '').trim();
  if (!combinedReasoning) {
    const list: string[] = [];
    for (const part of parts) {
      if (part.kind === 'reasoning' && part.text && part.text.trim()) {
        list.push(part.text.trim());
      }
    }
    combinedReasoning = list.join('\n\n---\n\n');
  }

  if (combinedReasoning || thinkingLive) {
    out.push(
      <ReasoningBlock
        key="rs-combined"
        text={combinedReasoning}
        search={search}
        live={thinkingLive}
        hasPendingTool={hasPendingTool}
      />,
    );
  }

  let i = 0;
  while (i < parts.length) {
    const p = parts[i];
    if (p.kind === 'reasoning' || p.kind === 'tool') {
      // reasoning 已在顶部聚合渲染，tool 在干净会话视图中隐藏
      i++;
      continue;
    }
    if (p.kind === 'todo_list') {
      trailingTodos = p;
      i++;
      continue;
    } else if (p.kind === 'notice') {
      if (isIgnoredChatNotice(p.text)) {
        i++;
        continue;
      }
      out.push(
        <NoticeCard
          key={`nt-${i}`}
          text={p.text}
          kind={p.noticeKind}
          severity={p.severity}
          status={p.status}
          search={search}
        />,
      );
      i++;
    } else if (p.kind === 'rate_limited') {
      out.push(
        <div class="rate-limited-notice" key={`rl-${i}`}>
          {highlightText(p.text, search)}
        </div>,
      );
      i++;
    } else if (p.kind === 'text') {
      if (p.text) out.push(<Markdown key={`tx-${i}`} content={p.text} search={search} cwd={cwd} targetContext="chat" />);
      i++;
    } else {
      i++;
    }
  }

  // 待办清单是整轮任务的最终成果核对单，始终且必然沉淀在助手回复的最末尾（所有思考、工具卡片、最终文本的最下方）！
  if (trailingTodos && trailingTodos.items.length > 0) {
    out.push(
      <SessionTodoPanel key="td-trailing" items={trailingTodos.items} embedded />,
    );
  }

  return out;
}

/** Collapsible thinking / reasoning block (chain-of-thought stream). */
