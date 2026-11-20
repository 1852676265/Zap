import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  clearInFlightAssistantTiming,
  formatTurnElapsed,
  resumeTurnClockEpoch,
  stampLastAssistantElapsed,
  turnDurationMs,
  turnTotalElapsedMs,
} from './turnTimer.ts';

test('formatTurnElapsed stays in seconds until 60, then rolls to m:ss', () => {
  assert.equal(formatTurnElapsed(0), '0s');
  assert.equal(formatTurnElapsed(999), '0s');
  assert.equal(formatTurnElapsed(1000), '1s');
  assert.equal(formatTurnElapsed(59_000), '59s');
  assert.equal(formatTurnElapsed(60_000), '1:00');
  assert.equal(formatTurnElapsed(61_500), '1:01');
  assert.equal(formatTurnElapsed(12 * 60_000 + 3_000), '12:03');
  assert.equal(formatTurnElapsed(3600_000), '1:00:00');
  assert.equal(formatTurnElapsed(3600_000 + 65_000), '1:01:05');
});

test('stampLastAssistantElapsed writes the latest unstamped assistant only', () => {
  const msgs = [
    { role: 'user' as const },
    { role: 'assistant' as const },
    { role: 'system' as const },
  ];
  const next = stampLastAssistantElapsed(msgs, 4200, 1_700_000_000_000);
  assert.equal(next[1]!.elapsedMs, 4200);
  assert.equal(next[1]!.ts, 1_700_000_000_000);
  assert.equal(next[0]!.elapsedMs, undefined);
  const again = stampLastAssistantElapsed(next, 9999);
  assert.equal(again[1]!.elapsedMs, 4200);
  assert.equal(again, next);
});

test('stampLastAssistantElapsed is a no-op when there is no assistant', () => {
  const msgs = [{ role: 'user' as const }];
  assert.equal(stampLastAssistantElapsed(msgs, 1000), msgs);
});

test('resumeTurnClockEpoch keeps the original user-send time across refresh', () => {
  assert.equal(resumeTurnClockEpoch(1_000_000, 995_000), 995_000);
  assert.equal(resumeTurnClockEpoch(1_000_000, undefined), 1_000_000);
  assert.equal(resumeTurnClockEpoch(1_000_000, 0), 1_000_000);
  assert.equal(resumeTurnClockEpoch(1_000_000, 1_000_500), 1_000_000);
});

test('turnDurationMs is user-bubble to end, not per-agent-round', () => {
  assert.equal(turnDurationMs(1000, 46000), 45000);
  assert.equal(turnDurationMs(undefined, 46000), undefined);
  assert.equal(turnDurationMs(5000, 4000), undefined);
});

test('turnTotalElapsedMs prefers the larger of user-bubble span and full-turn elapsed', () => {
  assert.equal(turnTotalElapsedMs(1000, 1_201_000, 18_000), 1_200_000);
  assert.equal(turnTotalElapsedMs(1_000_000, 1_020_000, 5_400_000), 5_400_000);
  assert.equal(turnTotalElapsedMs(undefined, undefined, 1_200_000), 1_200_000);
  assert.equal(turnTotalElapsedMs(5000, 5000, 1_200_000), 1_200_000);
  assert.equal(turnTotalElapsedMs(undefined, 9, undefined), undefined);
});

test('stampLastAssistantElapsed never crosses a user boundary and supports force overwrite on live terminal', () => {
  const withoutReply = [
    { role: 'user' as const, ts: 1000 },
    { role: 'assistant' as const, ts: 2000 },
    { role: 'user' as const, ts: 3000 },
  ];
  assert.equal(stampLastAssistantElapsed(withoutReply, 5000, 8000), withoutReply);

  const staleInFlight = [
    { role: 'user' as const, ts: 1_000_000 },
    { role: 'assistant' as const, ts: 2_015_000, elapsedMs: 1_015_000 },
  ];
  const finalized = stampLastAssistantElapsed(staleInFlight, 5_700_000, 6_700_000, { force: true });
  assert.equal(finalized[1]!.elapsedMs, 5_700_000);
  assert.equal(finalized[1]!.ts, 6_700_000);
});

test('clearInFlightAssistantTiming strips stale round timing only from the current turn', () => {
  const inflight = [
    { role: 'user' as const, ts: 1_000_000 },
    { role: 'assistant' as const, ts: 2_015_000, elapsedMs: 15_000 },
  ];
  const cleared = clearInFlightAssistantTiming(inflight);
  assert.equal(cleared[1]!.ts, undefined);
  assert.equal(cleared[1]!.elapsedMs, undefined);
  assert.equal(clearInFlightAssistantTiming(cleared), cleared);

  const openAfterOldTurn = [
    { role: 'user' as const, ts: 1000 },
    { role: 'assistant' as const, ts: 5000, elapsedMs: 4000 },
    { role: 'user' as const, ts: 9000 },
  ];
  assert.equal(clearInFlightAssistantTiming(openAfterOldTurn), openAfterOldTurn);
});
