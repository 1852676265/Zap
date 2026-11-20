/** Live turn elapsed-time helpers for the WebUI chat timeline. */

function pad2(n: number): string {
  return n < 10 ? '0' + n : '' + n;
}

/**
 * Format a duration for the in-turn stopwatch.
 * Seconds stay in seconds until 60, then the unit rolls to m:ss (and h:mm:ss).
 */
export function formatTurnElapsed(ms: number): string {
  const totalSec = Math.max(0, Math.floor(ms / 1000));
  const h = Math.floor(totalSec / 3600);
  const m = Math.floor((totalSec % 3600) / 60);
  const s = totalSec % 60;
  if (h > 0) return `${h}:${pad2(m)}:${pad2(s)}`;
  if (m > 0) return `${m}:${pad2(s)}`;
  return `${s}s`;
}

/** Resume the live stopwatch from the current user-bubble timestamp.
 *  Falls back to `now` when the send time is missing or in the future so a
 *  refresh during an in-flight turn does not restart at 0s when history has
 *  the original `created_at`. */
export function resumeTurnClockEpoch(now: number, lastUserTs?: number): number {
  if (lastUserTs == null || !Number.isFinite(lastUserTs) || lastUserTs <= 0) return now;
  if (lastUserTs > now) return now;
  return lastUserTs;
}

/** Wall-clock from the user bubble to `endTs` (final answer or now). */
export function turnDurationMs(userTs?: number, endTs?: number): number | undefined {
  if (userTs == null || endTs == null) return undefined;
  if (!Number.isFinite(userTs) || !Number.isFinite(endTs)) return undefined;
  if (endTs < userTs) return undefined;
  return endTs - userTs;
}

/** Done-turn "用时": take the larger of user-bubble → final answer span and
 *  stamped/persisted full-turn elapsed so neither a lone per-round kernel
 *  duration nor a mid-turn steer / trailing tool wait truncates the total. */
export function turnTotalElapsedMs(
  userTs?: number,
  endTs?: number,
  elapsedMs?: number,
): number | undefined {
  const fromUser = turnDurationMs(userTs, endTs);
  const stamped =
    elapsedMs != null && Number.isFinite(elapsedMs) && elapsedMs >= 0
      ? elapsedMs
      : undefined;
  if (fromUser && fromUser > 0) {
    return stamped != null && stamped > fromUser ? stamped : fromUser;
  }
  return stamped;
}

/** Stamp `elapsedMs` (and optional completion time) onto the latest assistant
 *  message of the current turn (never crossing back over a `user` bubble).
 *  Idempotent by default; pass `{ force: true }` on an authoritative live turn
 *  terminal so a stale mid-turn round stamp cannot freeze the final bubble. */
export function stampLastAssistantElapsed<T extends { role: string; elapsedMs?: number; ts?: number }>(
  msgs: T[],
  elapsedMs: number,
  finishedAt?: number,
  opts?: { force?: boolean },
): T[] {
  const force = opts?.force === true;
  const safeElapsed = Math.max(0, Number.isFinite(elapsedMs) ? elapsedMs : 0);
  for (let i = msgs.length - 1; i >= 0; i--) {
    if (msgs[i]!.role === 'user') return msgs;
    if (msgs[i]!.role !== 'assistant') continue;
    if (!force && msgs[i]!.elapsedMs != null) return msgs;
    const next = msgs.slice();
    next[i] = {
      ...msgs[i]!,
      elapsedMs: safeElapsed,
      ts: finishedAt ?? Date.now(),
    };
    return next;
  }
  return msgs;
}

/** Strip stale per-round `ts` / `elapsedMs` from the trailing assistant of an
 *  active (still-running) turn — e.g. after reattaching from an inflight disk
 *  snapshot whose intermediate round carried its own `created_at` / kernel
 *  per-round `elapsed_ms`. */
export function clearInFlightAssistantTiming<
  T extends { role: string; elapsedMs?: number; ts?: number },
>(msgs: T[]): T[] {
  for (let i = msgs.length - 1; i >= 0; i--) {
    if (msgs[i]!.role === 'user') return msgs;
    if (msgs[i]!.role !== 'assistant') continue;
    if (msgs[i]!.elapsedMs == null && msgs[i]!.ts == null) return msgs;
    const next = msgs.slice();
    const { elapsedMs: _elapsed, ts: _ts, ...rest } = msgs[i]!;
    next[i] = rest as T;
    return next;
  }
  return msgs;
}

