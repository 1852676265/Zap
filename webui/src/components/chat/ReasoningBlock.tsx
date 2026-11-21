import { useEffect, useMemo, useRef, useState } from 'preact/hooks';
import { useT } from '../../settings';
import { textMatchesSearch } from '../../lib/canvasSearch';
import { highlightText } from './highlight';

function cleanReasoningPreviewLine(rawLine: string): string {
  const trimmed = rawLine.trim();
  if (!trimmed || /^---+$/.test(trimmed)) return '';
  return trimmed
    .replace(/^([#*>-]|\d+\.)\s*/, '')
    .replace(/[*_`]/g, '')
    .replace(/\.\.\.$/, '')
    .trim();
}

export function ReasoningBlock({
  text,
  search,
  live = false,
  hasPendingTool = false,
}: {
  text: string;
  search: string;
  live?: boolean;
  hasPendingTool?: boolean;
}) {
  const t = useT();
  const [open, setOpen] = useState(false);
  const [idleTick, setIdleTick] = useState(0);
  const lastTextUpdateAtRef = useRef<number>(Date.now());
  const prevLatestLineRef = useRef<string>('');
  const bodyRef = useRef<HTMLDivElement>(null);

  // 搜索命中思考块时必须展开：折叠态下命中文本不在 DOM 上，会出现“计数有、页面上却找不到”。
  useEffect(() => {
    if (textMatchesSearch(text, search)) setOpen(true);
  }, [search, text]);

  const followBottomRef = useRef(true);
  const REASONING_BOTTOM_SLACK = 48;
  const recomputeFollow = () => {
    const el = bodyRef.current;
    if (!el) return;
    followBottomRef.current =
      el.scrollHeight - el.scrollTop - el.clientHeight <= REASONING_BOTTOM_SLACK;
  };
  useEffect(() => {
    if (!open) return;
    const el = bodyRef.current;
    if (!el || !followBottomRef.current) return;
    el.scrollTop = el.scrollHeight;
  }, [text, open]);

  // 仅从首尾局部扫行提取首行（完成后静止摘要）与最新有效行（流式生成时实时滚动展示）
  const { firstLine, latestLine } = useMemo(() => {
    if (!text) return { firstLine: '', latestLine: '' };
    let first = '';
    let start = 0;
    while (start <= text.length) {
      const nl = text.indexOf('\n', start);
      const end = nl === -1 ? text.length : nl;
      const cleaned = cleanReasoningPreviewLine(text.slice(start, end));
      if (cleaned) {
        first = cleaned;
        break;
      }
      if (nl === -1) break;
      start = nl + 1;
    }
    let latest = '';
    let end = text.length;
    while (end >= 0) {
      const nl = text.lastIndexOf('\n', end - 1);
      const segStart = nl === -1 ? 0 : nl + 1;
      const cleaned = cleanReasoningPreviewLine(text.slice(segStart, end));
      if (cleaned) {
        latest = cleaned;
        break;
      }
      if (nl <= 0) break;
      end = nl;
    }
    return { firstLine: first, latestLine: latest };
  }, [text]);

  useEffect(() => {
    if (latestLine && latestLine !== prevLatestLineRef.current) {
      prevLatestLineRef.current = latestLine;
      lastTextUpdateAtRef.current = Date.now();
      setIdleTick(0);
    }
  }, [latestLine]);

  useEffect(() => {
    if (!live) {
      setIdleTick(0);
      return;
    }
    const timer = window.setInterval(() => {
      if (Date.now() - lastTextUpdateAtRef.current >= 1900) {
        setIdleTick((n) => n + 1);
      }
    }, 2200);
    return () => window.clearInterval(timer);
  }, [live]);

  const hasReasoningText = text.trim().length > 0;

  const liveDisplay = useMemo(() => {
    if (!live) return firstLine;
    const stage1 = t('chat.thinkingStage1');
    const stage2 = t('chat.thinkingStage2');
    const stage3 = t('chat.thinkingStage3');
    const stageTool = t('chat.thinkingStageTool');
    const pool = hasPendingTool
      ? [...(latestLine ? [latestLine] : []), stageTool, stage2, stage3]
      : [...(latestLine ? [latestLine] : []), stage1, stage2, stage3];
    if (pool.length === 0) return stage1;
    return pool[idleTick % pool.length]!;
  }, [live, firstLine, latestLine, hasPendingTool, idleTick, t]);

  if (!hasReasoningText && !live) return null;

  const previewText = live ? liveDisplay : (!open ? firstLine : '');
  const tickerKey = live ? `${idleTick}:${previewText.slice(0, 48)}` : 'settled';

  return (
    <div class={'reasoning-block' + (open && hasReasoningText ? ' is-open' : '') + (live ? ' is-live' : '')}>
      <button
        type="button"
        class={'reasoning-toggle' + (!hasReasoningText ? ' is-non-expandable' : '')}
        onClick={() => {
          if (hasReasoningText) setOpen((v) => !v);
        }}
        aria-expanded={open && hasReasoningText}
      >
        <span class={'reasoning-icon' + (live ? ' is-live' : '')} aria-hidden="true">
          ✦
        </span>
        <span class="reasoning-label">{t('chat.thinking')}</span>
        {previewText && (
          <>
            <span class="reasoning-separator" aria-hidden="true">·</span>
            <span class="reasoning-summary" title={previewText} aria-live={live ? 'polite' : undefined}>
              <span key={tickerKey} class={'reasoning-summary-inner' + (live ? ' is-live' : '')}>
                {previewText}
              </span>
              {live && (
                <span class="reasoning-live-dots" aria-hidden="true">
                  <span>.</span>
                  <span>.</span>
                  <span>.</span>
                </span>
              )}
            </span>
          </>
        )}
        {hasReasoningText && (
          <span class={'reasoning-chevron' + (open ? ' expanded' : '')}>▾</span>
        )}
      </button>
      {open && hasReasoningText && (
        <div class="reasoning-body" ref={bodyRef} onScroll={recomputeFollow}>
          {highlightText(text, search)}
        </div>
      )}
    </div>
  );
}
