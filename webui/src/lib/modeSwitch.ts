export type ApprovalMode = 'build' | 'plan' | 'bypass' | 'accept_edits';

export interface ApprovalModeState {
  confirmedMode: ApprovalMode;
  displayMode: ApprovalMode;
  pendingMode?: ApprovalMode;
}

export function initModeState(mode: ApprovalMode): ApprovalModeState {
  return {
    confirmedMode: mode,
    displayMode: mode,
  };
}

/** WebUI pill for a loaded session. Both protocol (OpenAI/Anthropic) and
 *  new / manual sessions default to Auto (`bypass`). */
export function modeForSessionOrigin(_origin?: string | null): ApprovalMode {
  return 'bypass';
}

export function beginModeSwitch(
  state: ApprovalModeState,
  requested: ApprovalMode,
): ApprovalModeState {
  if (state.pendingMode || requested === state.displayMode) return state;
  return {
    confirmedMode: state.confirmedMode,
    displayMode: requested,
    pendingMode: requested,
  };
}

export function completeModeSwitch(
  state: ApprovalModeState,
  confirmed: ApprovalMode,
): ApprovalModeState {
  if (!state.pendingMode) return state;
  return {
    confirmedMode: confirmed,
    displayMode: confirmed,
  };
}

export function failModeSwitch(state: ApprovalModeState): ApprovalModeState {
  if (!state.pendingMode) return state;
  return {
    confirmedMode: state.confirmedMode,
    displayMode: state.confirmedMode,
  };
}
