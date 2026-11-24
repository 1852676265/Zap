/**
 * Zap Plugin Micro-Kernel & Typed Slot System
 * -----------------------------------------------------------------------------
 * Provides:
 * - Typed UI Slot Registry (`settings.section`, `settings.action`,
 *   `settings.general.item`, `settings.plugins.tab`, `plugins.tab`,
 *   `shell.overlay`, `sidebar.nav`, `workbench.view`)
 * - Generic Action & Inspector Contribution Registries + Standard Host SDK
 * - Manifest-driven External Plugin Package Lifecycle
 */

import { ComponentChildren } from 'preact';
import { useEffect, useRef, useState } from 'preact/hooks';
import { fetchFsRead } from '../api';

// =============================================================================
// 0. Generic Contribution System & Standard Host SDK
// =============================================================================

export interface ActionContribution<TContext = any> {
  id: string;
  order: number;
  title: string;
  icon?: string | ComponentChildren;
  pluginPackage?: string;
  when?: (ctx: TContext) => boolean;
  execute: (ctx: TContext, host: ZapHostSDK) => void | Promise<void>;
  render?: (props: { ctx: TContext; host: ZapHostSDK; action: ActionContribution<TContext> }) => ComponentChildren;
}

export type ActionScope =
  | 'chat.message:actions'
  | 'chat.turn:actions'
  | 'chat.input:actions'
  | 'chat.selection:actions'
  | string;

export interface InspectorContribution<TTarget = any> {
  id: string;
  title: string;
  icon?: string | ComponentChildren;
  order?: number;
  pluginPackage?: string;
  supports?: (target: TTarget) => boolean;
  render: (props: { target: TTarget; onClose: () => void; host: ZapHostSDK }) => ComponentChildren;
}

export type InspectorScope =
  | 'chat.turn:inspectors'
  | 'chat.message:inspectors'
  | 'workbench:inspectors'
  | string;

export interface ZapHostSDK {
  session: {
    getActiveSessionId(): string | null;
    create(options?: { prefillPrompt?: string; autoSend?: boolean }): Promise<string | null>;
    rollback(turnIndex: number, options?: { promptOverride?: string }): Promise<void>;
    send(text: string): Promise<void>;
  };
  history: {
    getTurn(turnIndex: number): Promise<any | null>;
    getRawPrompt(turnIndex: number): Promise<{ system: string; user: string } | null>;
  };
  storage: {
    get<T = any>(key: string): Promise<T | null>;
    set<T = any>(key: string, value: T): Promise<void>;
    delete(key: string): Promise<void>;
  };
  ui: {
    toast(message: string, type?: 'info' | 'success' | 'warn' | 'error'): void;
    copy(text: string): Promise<boolean>;
    openInspector(inspectorId: string, targetPayload: any, scope?: InspectorScope): void;
    closeInspector(): void;
  };
  events: {
    on(event: string, handler: (data: any) => void): () => void;
    emit(event: string, data?: any): void;
  };
}

export interface ZapHostDriver {
  getActiveSessionId?: () => string | null;
  createSession?: (options?: { prefillPrompt?: string; autoSend?: boolean }) => Promise<string | null>;
  rollbackTurn?: (turnIndex: number, options?: { promptOverride?: string }) => Promise<void>;
  sendMessage?: (text: string) => Promise<void>;
  getTurnSnapshot?: (turnIndex: number) => Promise<any | null>;
  getRawPromptSnapshot?: (turnIndex: number) => Promise<{ system: string; user: string } | null>;
  showToast?: (message: string, type?: 'info' | 'success' | 'warn' | 'error') => void;
  copyText?: (text: string) => Promise<boolean>;
}

// =============================================================================
// 1. Typed Slot Contracts
// =============================================================================

export interface SettingsSectionRuntimeProps {
  close: () => void;
  navigate: (sectionId: string) => void;
}

export interface SettingsSectionContribution {
  id: string;
  order: number;
  icon: ComponentChildren;
  label: string;
  badge?: string | number;
  title?: string;
  description?: string;
  pluginPackage?: string;
  render: (props: SettingsSectionRuntimeProps) => ComponentChildren;
}

export interface SettingsActionContribution {
  id: string;
  order: number;
  pluginPackage?: string;
  render: (props: SettingsSectionRuntimeProps) => ComponentChildren;
}

export interface SettingsGeneralItemContribution {
  id: string;
  order: number;
  pluginPackage?: string;
  render: () => ComponentChildren;
}

export interface SettingsPluginsTabContribution {
  id: string;
  order: number;
  label: string;
  pluginPackage?: string;
  render: () => ComponentChildren;
}

export interface PluginsManagerTabContribution {
  id: string;
  order: number;
  label: string;
  pluginPackage?: string;
  render: (props: {
    openPackage: (packageName: string) => void;
    openInstall: (spec?: string) => void;
  }) => ComponentChildren;
}

export interface ShellOverlayContribution {
  id: string;
  order: number;
  pluginPackage?: string;
  render: () => ComponentChildren;
}

export interface SidebarNavContribution {
  id: string;
  order: number;
  label: string;
  icon?: ComponentChildren;
  badge?: string | number;
  active?: boolean;
  pluginPackage?: string;
  viewId?: string;
  onClick?: () => void;
  render?: () => ComponentChildren;
}

export interface WorkbenchViewContribution {
  id: string;
  order: number;
  title: string;
  icon?: ComponentChildren;
  pluginPackage?: string;
  type?: 'component' | 'iframe';
  src?: string;
  render?: () => ComponentChildren;
}

export interface SlotMap {
  'settings.section': SettingsSectionContribution;
  'settings.action': SettingsActionContribution;
  'settings.general.item': SettingsGeneralItemContribution;
  'settings.plugins.tab': SettingsPluginsTabContribution;
  'plugins.tab': PluginsManagerTabContribution;
  'shell.overlay': ShellOverlayContribution;
  'sidebar.nav': SidebarNavContribution;
  'workbench.view': WorkbenchViewContribution;
}

export type SlotKey = keyof SlotMap;

// =============================================================================
// 2. External Plugin Package & Manifest Models
// =============================================================================

export type PluginRuntimePhase = 'active' | 'unobserved' | 'pending' | 'failed';

export interface ExternalPluginRow {
  rowId: string;
  title: string;
  description: string;
  moduleName: string;
  enabled: boolean;
  phase: PluginRuntimePhase;
  hasConfig?: boolean;
  sectionId?: string;
}

export interface ExternalPluginPackage {
  name: string;
  title: string;
  description: string;
  version: string;
  author?: string;
  icon?: string;
  official: boolean;
  pinnedByDefault?: boolean;
  installed: boolean;
  enabled: boolean;
  removable: boolean;
  updatable: boolean;
  beta?: boolean;
  source: string;
  installDir: string;
  readmeMarkdown: string;
  versions: string[];
  rows: ExternalPluginRow[];
  configValues?: Record<string, any>;
  configSchema?: {
    key: string;
    label: string;
    description: string;
    type: 'boolean' | 'string' | 'number' | 'select' | 'model';
    default?: any;
    options?: { label: string; value: string }[];
  }[];
  contributesSidebarNav?: {
    id?: string;
    label: string;
    badge?: string | number;
    order?: number;
    viewId?: string;
    eventName?: string;
    sectionId?: string;
  };
  contributesWorkbenchView?: {
    id: string;
    title?: string;
    order?: number;
    type?: 'component' | 'iframe';
    src?: string;
  };
  contributesSettingsSection?: {
    id: string;
    label: string;
    title?: string;
    description?: string;
  };
  entry?: string;
  entryDir?: string;
}

export interface ZapPluginManifest {
  id: string;
  name: string;
  version: string;
  title?: string;
  description?: string;
  author?: string;
  icon?: string;
  entry?: string;
  contributes?: {
    sidebarNav?: {
      id?: string;
      label: string;
      badge?: string | number;
      order?: number;
      viewId?: string;
      sectionId?: string;
      eventName?: string;
    };
    workbenchView?: {
      id: string;
      title?: string;
      order?: number;
      type?: 'iframe' | 'component';
      src?: string;
    };
    settingsSection?: {
      id?: string;
      label: string;
      title?: string;
      description?: string;
      schema?: {
        key: string;
        label: string;
        description?: string;
        type: 'boolean' | 'string' | 'number' | 'select' | 'model';
        options?: { label: string; value: string }[];
        default?: any;
      }[];
    };
  };
  defaultConfig?: Record<string, any>;
}

// =============================================================================
// 3. Persistence & Legacy Mock Purge
// =============================================================================

const STORAGE_KEY_PACKAGES = 'zap.plugin_kernel.packages.v1';

const LEGACY_STORAGE_KEYS = [
  'zap.plugin_kernel.builtin.v1',
  'zap.plugin_kernel.presets.v1',
  'zap.plugin_kernel.shell_limits.v1',
  'zap.plugin_kernel.dev_tools.v1',
  'zap_dsh_pet_full_config',
];

const LEGACY_MOCK_PACKAGE_NAMES = new Set([
  '@deepseek-ai/plugin-mermaid-enhanced',
  '@deepseek-ai/plugin-code-canvas',
  '@deepseek-ai/plugin-git-worktree-guard',
  '@dsh-community/prompt-vault',
  '@dsh-community/theme-customizer',
  '@dsh-community/context-telemetry',
  '@dsh-community/webhook-notifier',
]);

function isLegacyMockPackage(pkg: Partial<ExternalPluginPackage> | null | undefined): boolean {
  if (!pkg || typeof pkg.name !== 'string') return true;
  if (LEGACY_MOCK_PACKAGE_NAMES.has(pkg.name)) return true;
  const source = typeof pkg.source === 'string' ? pkg.source : '';
  if (source.startsWith('builtin:@deepseek-ai/')) return true;
  if (source.startsWith('registry:@dsh-community/')) return true;
  if (pkg.name === '@dsh-community/pet' && source.includes('dsh-pet')) return true;
  return false;
}

// =============================================================================
// 4. Kernel Implementation
// =============================================================================

type Listener = () => void;

const EMPTY_ACTIONS: ActionContribution<any>[] = [];
const EMPTY_INSPECTORS: InspectorContribution<any>[] = [];

function sameContributionList<T extends { id: string }>(a: readonly T[], b: readonly T[]): boolean {
  if (a === b) return true;
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i += 1) {
    if (a[i] !== b[i]) return false;
  }
  return true;
}

export class ZapPluginKernel {
  private slots: { [K in SlotKey]: Map<string, SlotMap[K]> } = {
    'settings.section': new Map(),
    'settings.action': new Map(),
    'settings.general.item': new Map(),
    'settings.plugins.tab': new Map(),
    'plugins.tab': new Map(),
    'shell.overlay': new Map(),
    'sidebar.nav': new Map(),
    'workbench.view': new Map(),
  };

  private externalPackages: ExternalPluginPackage[] = [];
  private listeners = new Set<Listener>();
  // 单调递增的内核版本号：订阅方在挂载时比对，补偿「渲染→订阅」之间错过的变更通知
  private version = 0;
  private dynamicDisposers = new Map<string, () => void>();
  // 每个插件包的运行时同步代际，用于作废停用/重同步之前发起的异步脚本加载
  private runtimeGenerations = new Map<string, number>();
  private styleElements = new Map<string, HTMLStyleElement>();

  // Generic Extension Registries (Action & Inspector models)
  private actionRegistries = new Map<string, Map<string, ActionContribution>>();
  private inspectorRegistries = new Map<string, Map<string, InspectorContribution>>();
  private activeInspectorState: { scope: string; id: string; target: any } | null = null;
  private hostDriver: ZapHostDriver = {};
  private hostEventListeners = new Map<string, Set<(data: any) => void>>();
  public readonly hostSDK: ZapHostSDK;

  constructor() {
    this.hostSDK = this.createHostSDK();
    this.loadPersistedState();
  }

  private loadPersistedState() {
    try {
      for (const key of LEGACY_STORAGE_KEYS) {
        localStorage.removeItem(key);
      }
    } catch {
      /* ignore */
    }

    try {
      const savedPackages = localStorage.getItem(STORAGE_KEY_PACKAGES);
      if (savedPackages) {
        const parsed = JSON.parse(savedPackages);
        if (Array.isArray(parsed)) {
          const filtered = parsed.filter((p) => !isLegacyMockPackage(p));
          this.externalPackages = filtered;
          if (filtered.length !== parsed.length) {
            this.persistPackages();
          }
          return;
        }
      }
    } catch {
      /* ignore */
    }
    this.externalPackages = [];
  }

  private persistPackages() {
    try {
      localStorage.setItem(STORAGE_KEY_PACKAGES, JSON.stringify(this.externalPackages));
    } catch {
      /* ignore */
    }
  }

  public subscribe(listener: Listener): () => void {
    this.listeners.add(listener);
    return () => {
      this.listeners.delete(listener);
    };
  }

  public getVersion(): number {
    return this.version;
  }

  public notify() {
    this.version += 1;
    for (const listener of this.listeners) {
      try {
        listener();
      } catch (err) {
        console.error('[ZapPluginKernel] Listener error:', err);
      }
    }
  }

  // ---------------------------------------------------------------------------
  // Slot Registration & Query API
  // ---------------------------------------------------------------------------

  public registerSlot<K extends SlotKey>(slotName: K, contribution: SlotMap[K]): () => void {
    const map = this.slots[slotName];
    map.set(contribution.id, contribution);
    this.notify();
    return () => {
      if (map.get(contribution.id) === contribution) {
        map.delete(contribution.id);
        this.notify();
      }
    };
  }

  public getSlotEntries<K extends SlotKey>(slotName: K): SlotMap[K][] {
    const list = Array.from(this.slots[slotName].values());
    return list.sort((a, b) => a.order - b.order);
  }

  // ---------------------------------------------------------------------------
  // Generic Action Provider API
  // ---------------------------------------------------------------------------

  public registerAction<TContext = any>(
    scope: ActionScope,
    action: ActionContribution<TContext>,
  ): () => void {
    let registry = this.actionRegistries.get(scope);
    if (!registry) {
      registry = new Map();
      this.actionRegistries.set(scope, registry);
    }
    registry.set(action.id, action);
    this.notify();
    return () => {
      const reg = this.actionRegistries.get(scope);
      if (reg && reg.get(action.id) === action) {
        reg.delete(action.id);
        this.notify();
      }
    };
  }

  public getActions<TContext = any>(
    scope: ActionScope,
    context?: TContext,
  ): ActionContribution<TContext>[] {
    const registry = this.actionRegistries.get(scope);
    if (!registry || registry.size === 0) return EMPTY_ACTIONS;
    const list = Array.from(registry.values()) as ActionContribution<TContext>[];
    const filtered = context !== undefined
      ? list.filter((act) => !act.when || act.when(context))
      : list;
    return filtered.length === 0 ? EMPTY_ACTIONS : filtered.sort((a, b) => a.order - b.order);
  }

  public hasRegisteredActions(scope: ActionScope): boolean {
    const registry = this.actionRegistries.get(scope);
    return Boolean(registry && registry.size > 0);
  }

  // ---------------------------------------------------------------------------
  // Generic Inspector & View Drawer API
  // ---------------------------------------------------------------------------

  public registerInspector<TTarget = any>(
    scope: InspectorScope,
    inspector: InspectorContribution<TTarget>,
  ): () => void {
    let registry = this.inspectorRegistries.get(scope);
    if (!registry) {
      registry = new Map();
      this.inspectorRegistries.set(scope, registry);
    }
    registry.set(inspector.id, inspector);
    this.notify();
    return () => {
      const reg = this.inspectorRegistries.get(scope);
      if (reg && reg.get(inspector.id) === inspector) {
        reg.delete(inspector.id);
        if (this.activeInspectorState?.id === inspector.id) {
          this.activeInspectorState = null;
        }
        this.notify();
      }
    };
  }

  public getInspectors<TTarget = any>(
    scope: InspectorScope,
    target?: TTarget,
  ): InspectorContribution<TTarget>[] {
    const registry = this.inspectorRegistries.get(scope);
    if (!registry || registry.size === 0) return EMPTY_INSPECTORS;
    const list = Array.from(registry.values()) as InspectorContribution<TTarget>[];
    const filtered = target !== undefined
      ? list.filter((ins) => !ins.supports || ins.supports(target))
      : list;
    return filtered.length === 0 ? EMPTY_INSPECTORS : filtered.sort((a, b) => (a.order ?? 50) - (b.order ?? 50));
  }

  public hasRegisteredInspectors(scope: InspectorScope): boolean {
    const registry = this.inspectorRegistries.get(scope);
    return Boolean(registry && registry.size > 0);
  }

  public openInspector(inspectorId: string, targetPayload: any, scope: InspectorScope = 'chat.turn:inspectors') {
    const registry = this.inspectorRegistries.get(scope);
    const item = registry?.get(inspectorId);
    if (!item) {
      console.warn(`[ZapPluginKernel] Inspector "${inspectorId}" not found in scope "${scope}"`);
      return;
    }
    this.activeInspectorState = { scope, id: inspectorId, target: targetPayload };
    this.notify();
  }

  public closeInspector() {
    if (this.activeInspectorState) {
      this.activeInspectorState = null;
      this.notify();
    }
  }

  public getActiveInspector(): {
    scope: string;
    id: string;
    target: any;
    inspector: InspectorContribution;
  } | null {
    if (!this.activeInspectorState) return null;
    const { scope, id, target } = this.activeInspectorState;
    const reg = this.inspectorRegistries.get(scope);
    const inspector = reg?.get(id);
    if (!inspector) return null;
    return { scope, id, target, inspector };
  }

  // ---------------------------------------------------------------------------
  // Host Driver Binding & Standard Host SDK
  // ---------------------------------------------------------------------------

  public bindHostDriver(driver: Partial<ZapHostDriver>) {
    this.hostDriver = { ...this.hostDriver, ...driver };
  }

  private createHostSDK(): ZapHostSDK {
    const self = this;
    return {
      session: {
        getActiveSessionId: () => self.hostDriver.getActiveSessionId?.() ?? null,
        create: async (options) => {
          if (self.hostDriver.createSession) {
            return await self.hostDriver.createSession(options);
          }
          console.warn('[ZapHostSDK] createSession handler is not bound yet');
          return null;
        },
        rollback: async (turnIndex, options) => {
          if (self.hostDriver.rollbackTurn) {
            await self.hostDriver.rollbackTurn(turnIndex, options);
            return;
          }
          console.warn('[ZapHostSDK] rollbackTurn handler is not bound yet');
        },
        send: async (text) => {
          if (self.hostDriver.sendMessage) {
            await self.hostDriver.sendMessage(text);
            return;
          }
          console.warn('[ZapHostSDK] sendMessage handler is not bound yet');
        },
      },
      history: {
        getTurn: async (turnIndex) => {
          if (self.hostDriver.getTurnSnapshot) {
            return await self.hostDriver.getTurnSnapshot(turnIndex);
          }
          return null;
        },
        getRawPrompt: async (turnIndex) => {
          if (self.hostDriver.getRawPromptSnapshot) {
            return await self.hostDriver.getRawPromptSnapshot(turnIndex);
          }
          return null;
        },
      },
      storage: {
        get: async <T = any>(key: string): Promise<T | null> => {
          try {
            const raw = localStorage.getItem(`zap.plugin_storage.${key}`);
            return raw ? JSON.parse(raw) : null;
          } catch {
            return null;
          }
        },
        set: async <T = any>(key: string, value: T): Promise<void> => {
          try {
            localStorage.setItem(`zap.plugin_storage.${key}`, JSON.stringify(value));
          } catch (e) {
            console.error('[ZapHostSDK] storage set error:', e);
          }
        },
        delete: async (key: string): Promise<void> => {
          try {
            localStorage.removeItem(`zap.plugin_storage.${key}`);
          } catch {
            /* ignore */
          }
        },
      },
      ui: {
        toast: (message, type = 'info') => {
          if (self.hostDriver.showToast) {
            self.hostDriver.showToast(message, type);
            return;
          }
          console.log(`[Toast ${type.toUpperCase()}]: ${message}`);
        },
        copy: async (text) => {
          if (self.hostDriver.copyText) {
            return await self.hostDriver.copyText(text);
          }
          if (typeof navigator !== 'undefined' && navigator.clipboard) {
            await navigator.clipboard.writeText(text);
            return true;
          }
          return false;
        },
        openInspector: (id, target, scope) => {
          self.openInspector(id, target, scope);
        },
        closeInspector: () => {
          self.closeInspector();
        },
      },
      events: {
        on: (event, handler) => {
          let set = self.hostEventListeners.get(event);
          if (!set) {
            set = new Set();
            self.hostEventListeners.set(event, set);
          }
          set.add(handler);
          return () => {
            const s = self.hostEventListeners.get(event);
            if (s) s.delete(handler);
          };
        },
        emit: (event, data) => {
          const s = self.hostEventListeners.get(event);
          if (s) {
            for (const h of s) {
              try {
                h(data);
              } catch (e) {
                console.error(`[ZapHostSDK] event ${event} error:`, e);
              }
            }
          }
        },
      },
    };
  }

  // ---------------------------------------------------------------------------
  // Dynamic Style & Theme Token Injection
  // ---------------------------------------------------------------------------

  public insertStyle(id: string, cssText: string): () => void {
    if (typeof document === 'undefined') return () => {};
    let styleEl = this.styleElements.get(id);
    if (!styleEl) {
      styleEl = document.createElement('style');
      styleEl.setAttribute('data-zap-plugin-style', id);
      document.head.appendChild(styleEl);
      this.styleElements.set(id, styleEl);
    }
    styleEl.textContent = cssText;
    return () => {
      const el = this.styleElements.get(id);
      if (el) {
        el.remove();
        this.styleElements.delete(id);
      }
    };
  }

  // ---------------------------------------------------------------------------
  // External Packages Lifecycle
  // ---------------------------------------------------------------------------

  public getExternalPackages(): ExternalPluginPackage[] {
    return this.externalPackages;
  }

  public getPackage(packageName: string): ExternalPluginPackage | undefined {
    return this.externalPackages.find((p) => p.name === packageName);
  }

  public togglePackage(packageName: string): boolean {
    const pkg = this.getPackage(packageName);
    if (!pkg) return false;
    this.setPackageEnabled(packageName, !pkg.enabled);
    return !pkg.enabled;
  }

  public setPackageEnabled(packageName: string, enabled: boolean) {
    this.externalPackages = this.externalPackages.map((pkg) => {
      if (pkg.name !== packageName) return pkg;
      return {
        ...pkg,
        enabled,
        rows: pkg.rows.map((r) => ({
          ...r,
          enabled,
          phase: enabled ? 'active' : 'unobserved',
        })),
      };
    });
    this.persistPackages();
    this.syncPackageRuntimeEffects(packageName);
    this.notify();
  }

  public setPackageRowEnabled(packageName: string, rowId: string, enabled: boolean) {
    this.externalPackages = this.externalPackages.map((pkg) => {
      if (pkg.name !== packageName) return pkg;
      const nextRows = pkg.rows.map((r) =>
        r.rowId === rowId ? { ...r, enabled, phase: (enabled ? 'active' : 'unobserved') as PluginRuntimePhase } : r,
      );
      const anyEnabled = nextRows.some((r) => r.enabled);
      return {
        ...pkg,
        enabled: anyEnabled,
        rows: nextRows,
      };
    });
    this.persistPackages();
    this.syncPackageRuntimeEffects(packageName);
    this.notify();
  }

  public updatePackageConfig(packageName: string, key: string, value: any) {
    this.externalPackages = this.externalPackages.map((pkg) => {
      if (pkg.name !== packageName) return pkg;
      return {
        ...pkg,
        configValues: {
          ...(pkg.configValues || {}),
          [key]: value,
        },
      };
    });
    this.persistPackages();
    this.syncPackageRuntimeEffects(packageName);
    this.notify();
  }

  public uninstallPackage(packageName: string) {
    this.runtimeGenerations.set(packageName, (this.runtimeGenerations.get(packageName) ?? 0) + 1);
    const disposer = this.dynamicDisposers.get(packageName);
    if (disposer) {
      disposer();
      this.dynamicDisposers.delete(packageName);
    }
    this.purgePackageContributions(packageName);
    if (typeof document !== 'undefined') {
      const cleanId = packageName.replace(/^@/, '').replace(/\//g, '-');
      document.querySelectorAll(`script[src*="${cleanId}"]`).forEach((s) => s.remove());
      window.dispatchEvent(new CustomEvent('zap:plugin-uninstalled', { detail: { package: packageName } }));
    }
    this.externalPackages = this.externalPackages.filter((pkg) => pkg.name !== packageName || !pkg.removable);
    this.persistPackages();
    this.notify();
  }

  public installFromManifest(manifest: ZapPluginManifest, source?: string): ExternalPluginPackage {
    const rawId = manifest.id || manifest.name;
    const cleanId = rawId.toLowerCase().replace(/[^a-z0-9-_]/g, '-');
    const pkgName = manifest.name.startsWith('@') || manifest.name.includes('/')
      ? manifest.name
      : `@plugins/${cleanId}`;
    const sectionId = manifest.contributes?.settingsSection?.id || `ext-${cleanId}`;

    const newPkg: ExternalPluginPackage = {
      name: pkgName,
      title: manifest.title || manifest.name,
      description: manifest.description || '外部扩展插件包',
      version: manifest.version || '1.0.0',
      author: manifest.author || 'Plugin Contributor',
      official: false,
      installed: true,
      enabled: true,
      removable: true,
      updatable: false,
      source: source || `manifest:${cleanId}`,
      installDir: `~/.zap/plugins/${cleanId}`,
      entry: manifest.entry,
      entryDir: source && (source.startsWith('/') || /^[A-Za-z]:/.test(source))
        ? source.replace(/[\\/]+[^\\/]*\.json$/, '').replace(/[\\/]+$/, '')
        : undefined,
      readmeMarkdown: `### ${manifest.title || manifest.name}\n\n${manifest.description || ''}`,
      versions: [manifest.version || '1.0.0'],
      configSchema: manifest.contributes?.settingsSection?.schema?.map((s) => ({
        key: s.key,
        label: s.label,
        description: s.description || '',
        type: s.type,
        options: s.options,
      })) || [],
      configValues: { ...(manifest.defaultConfig || {}) },
      contributesSidebarNav: manifest.contributes?.sidebarNav
        ? {
            id: manifest.contributes.sidebarNav.id || `sidebar-nav-${cleanId}`,
            label: manifest.contributes.sidebarNav.label,
            badge: manifest.contributes.sidebarNav.badge,
            order: manifest.contributes.sidebarNav.order,
            viewId: manifest.contributes.sidebarNav.viewId || manifest.contributes.workbenchView?.id,
            sectionId: manifest.contributes.sidebarNav.sectionId || sectionId,
            eventName: manifest.contributes.sidebarNav.eventName,
          }
        : undefined,
      contributesWorkbenchView: manifest.contributes?.workbenchView
        ? {
            id: manifest.contributes.workbenchView.id,
            title: manifest.contributes.workbenchView.title || manifest.title,
            order: manifest.contributes.workbenchView.order,
            type: manifest.contributes.workbenchView.type || 'iframe',
            src: manifest.contributes.workbenchView.src,
          }
        : undefined,
      contributesSettingsSection: manifest.contributes?.settingsSection
        ? {
            id: sectionId,
            label: manifest.contributes.settingsSection.label,
            title: manifest.contributes.settingsSection.title,
            description: manifest.contributes.settingsSection.description,
          }
        : undefined,
      rows: [
        {
          rowId: `${cleanId}-main`,
          title: manifest.title || manifest.name,
          description: manifest.description || '插件核心功能与工作台插槽',
          moduleName: `${pkgName}/index.js`,
          enabled: true,
          phase: 'active',
          hasConfig: Boolean(manifest.contributes?.settingsSection?.schema?.length),
          sectionId,
        },
      ],
    };

    const existsInList = this.externalPackages.some((p) => p.name === pkgName);
    if (existsInList) {
      this.externalPackages = this.externalPackages.map((p) => (p.name === pkgName ? newPkg : p));
    } else {
      this.externalPackages = [...this.externalPackages, newPkg];
    }
    this.persistPackages();
    this.syncPackageRuntimeEffects(newPkg.name);
    this.notify();
    return newPkg;
  }

  public syncAllPackagesRuntimeEffects() {
    for (const pkg of this.externalPackages) {
      this.syncPackageRuntimeEffects(pkg.name);
    }
  }

  /**
   * 清理插件脚本通过全局内核 API 自行注册的插槽 / Action / Inspector。
   */
  private purgePackageContributions(packageName: string) {
    for (const map of Object.values(this.slots) as Map<string, { pluginPackage?: string }>[]) {
      for (const [id, entry] of map) {
        if (entry.pluginPackage === packageName) map.delete(id);
      }
    }
    const registries = [...this.actionRegistries.values(), ...this.inspectorRegistries.values()] as Map<
      string,
      { pluginPackage?: string }
    >[];
    for (const registry of registries) {
      for (const [id, entry] of registry) {
        if (entry.pluginPackage === packageName) registry.delete(id);
      }
    }
  }

  private syncPackageRuntimeEffects(packageName: string) {
    const generation = (this.runtimeGenerations.get(packageName) ?? 0) + 1;
    this.runtimeGenerations.set(packageName, generation);

    const prevDisposer = this.dynamicDisposers.get(packageName);
    if (prevDisposer) {
      prevDisposer();
      this.dynamicDisposers.delete(packageName);
    }

    const pkg = this.externalPackages.find((p) => p.name === packageName);
    if (!pkg || !pkg.enabled) {
      this.purgePackageContributions(packageName);
      if (typeof window !== 'undefined') {
        window.dispatchEvent(new CustomEvent('zap:plugin-deactivated', { detail: { package: packageName } }));
      }
      return;
    }

    const cleanups: (() => void)[] = [];

    // If plugin contributes a workbench view, register into `workbench.view`
    if (pkg.contributesWorkbenchView && pkg.contributesWorkbenchView.id) {
      const wb = pkg.contributesWorkbenchView;
      const unregWb = this.registerSlot('workbench.view', {
        id: wb.id,
        order: wb.order ?? 50,
        title: wb.title || pkg.title,
        pluginPackage: pkg.name,
        type: wb.type || 'iframe',
        src: wb.src,
      });
      cleanups.push(unregWb);
    }

    // If plugin declares a sidebar navigation entry, mount into `sidebar.nav`
    if (pkg.contributesSidebarNav && pkg.contributesSidebarNav.label) {
      const navCfg = pkg.contributesSidebarNav;
      const navId = navCfg.id || `sidebar-nav-${pkg.name}`;
      const targetViewId = navCfg.viewId || pkg.contributesWorkbenchView?.id;
      const unregNav = this.registerSlot('sidebar.nav', {
        id: navId,
        order: navCfg.order ?? 50,
        label: navCfg.label,
        badge: navCfg.badge,
        pluginPackage: pkg.name,
        viewId: targetViewId,
        onClick: () => {
          if (targetViewId) {
            window.dispatchEvent(
              new CustomEvent('zap:switch-workbench-view', {
                detail: { viewId: targetViewId, pluginPackage: pkg.name },
              }),
            );
          } else if (navCfg.eventName) {
            window.dispatchEvent(new CustomEvent(navCfg.eventName, { detail: { pkg } }));
          } else if (navCfg.sectionId) {
            window.dispatchEvent(
              new CustomEvent('zap:open-deepseek-settings-center', {
                detail: { section: navCfg.sectionId },
              }),
            );
          }
        },
      });
      cleanups.push(unregNav);
    }

    // If plugin declares a settings section or has configSchema, mount into `settings.section`
    if (pkg.contributesSettingsSection || (pkg.configSchema && pkg.configSchema.length > 0)) {
      const secCfg = pkg.contributesSettingsSection;
      const secId = secCfg?.id || `ext-${pkg.name.replace(/^@/, '').replace(/\//g, '-')}`;
      const unregSec = this.registerSlot('settings.section', {
        id: secId,
        order: 85,
        icon: null,
        label: secCfg?.label || pkg.title || pkg.name,
        title: secCfg?.title || `${pkg.title || pkg.name} 配置`,
        description: secCfg?.description || pkg.description,
        pluginPackage: pkg.name,
        render: () => null,
      });
      cleanups.push(unregSec);
    }

    // Dynamically load plugin bundle script when an entry is declared
    if (pkg.entry && pkg.entryDir) {
      const normalizedDir = pkg.entryDir.replace(/\\/g, '/');
      const normalizedEntry = pkg.entry.replace(/\\/g, '/').replace(/^\.\//, '');
      const fullEntryPath = `${normalizedDir}/${normalizedEntry}`;
      fetchFsRead(fullEntryPath)
        .then((res) => {
          if (this.runtimeGenerations.get(packageName) !== generation) return;
          if (res.success && res.content) {
            const scriptEl = document.createElement('script');
            scriptEl.type = 'module';
            scriptEl.textContent = res.content;
            document.head.appendChild(scriptEl);
            cleanups.push(() => scriptEl.remove());
          }
        })
        .catch(() => {});
    } else if (pkg.entry) {
      const cleanId = pkg.name.replace(/^@plugins\//, '').replace(/^@/, '').replace(/\//g, '-');
      const normalizedEntry = pkg.entry.replace(/\\/g, '/').replace(/^\.\//, '');
      const scriptUrl = `/plugins/${cleanId}/${normalizedEntry}`;
      const scriptEl = document.createElement('script');
      scriptEl.type = 'module';
      scriptEl.src = scriptUrl;
      scriptEl.onerror = () => {
        scriptEl.remove();
      };
      document.head.appendChild(scriptEl);
      cleanups.push(() => scriptEl.remove());
    }

    this.dynamicDisposers.set(packageName, () => {
      for (const fn of cleanups) fn();
    });
  }
}

export const zapPluginKernel = new ZapPluginKernel();
export const zapHostSDK = zapPluginKernel.hostSDK;

// =============================================================================
// Preact Hooks for Plugin System Consumption
// =============================================================================

export function usePluginActions<TContext = any>(
  scope: ActionScope,
  context?: TContext,
): ActionContribution<TContext>[] {
  const contextRef = useRef(context);
  contextRef.current = context;
  const [actions, setActions] = useState<ActionContribution<TContext>[]>(() =>
    zapPluginKernel.getActions(scope, context),
  );

  // 订阅插件注册表变化只按 scope 绑定；无注册动作时直接跳过 context 变化带来的额外状态刷新，
  // 避免调用方每次传入新对象引发逐气泡双倍重渲染。
  useEffect(() => {
    const sync = () => {
      const next = zapPluginKernel.getActions(scope, contextRef.current);
      setActions((prev) => (sameContributionList(prev, next) ? prev : next));
    };
    sync();
    return zapPluginKernel.subscribe(sync);
  }, [scope]);

  useEffect(() => {
    if (!zapPluginKernel.hasRegisteredActions(scope)) return;
    const next = zapPluginKernel.getActions(scope, context);
    setActions((prev) => (sameContributionList(prev, next) ? prev : next));
  }, [scope, context]);

  return actions;
}

export function usePluginInspectors<TTarget = any>(
  scope: InspectorScope,
  target?: TTarget,
): InspectorContribution<TTarget>[] {
  const targetRef = useRef(target);
  targetRef.current = target;
  const [inspectors, setInspectors] = useState<InspectorContribution<TTarget>[]>(() =>
    zapPluginKernel.getInspectors(scope, target),
  );

  useEffect(() => {
    const sync = () => {
      const next = zapPluginKernel.getInspectors(scope, targetRef.current);
      setInspectors((prev) => (sameContributionList(prev, next) ? prev : next));
    };
    sync();
    return zapPluginKernel.subscribe(sync);
  }, [scope]);

  useEffect(() => {
    if (!zapPluginKernel.hasRegisteredInspectors(scope)) return;
    const next = zapPluginKernel.getInspectors(scope, target);
    setInspectors((prev) => (sameContributionList(prev, next) ? prev : next));
  }, [scope, target]);

  return inspectors;
}

export function useActiveInspector() {
  const [active, setActive] = useState(() => zapPluginKernel.getActiveInspector());

  useEffect(() => {
    return zapPluginKernel.subscribe(() => {
      setActive(zapPluginKernel.getActiveInspector());
    });
  }, []);

  return active;
}

/**
 * 订阅插件内核变更并返回当前版本号；任何插槽、插件包状态变化都会触发使用方重渲染。
 */
export function usePluginKernelVersion(): number {
  const [version, setVersion] = useState(() => zapPluginKernel.getVersion());

  useEffect(() => {
    const sync = () => setVersion(zapPluginKernel.getVersion());
    const unsub = zapPluginKernel.subscribe(sync);
    sync();
    return unsub;
  }, []);

  return version;
}

export function useZapHostSDK(): ZapHostSDK {
  return zapPluginKernel.hostSDK;
}

// Expose global runtime hooks for external user scripts & plugins
if (typeof window !== 'undefined') {
  (window as any).__ZAP_PLUGIN_KERNEL__ = zapPluginKernel;
  (window as any).__ZAP_HOST_SDK__ = zapPluginKernel.hostSDK;
}
