#!/usr/bin/env node
/**
 * Zap 上游同步守卫脚本 (Upstream Sync Guard)
 * -----------------------------------------------------------------------------
 * 每次从官方 JeikCode 同步上游后运行 `node scripts/zap-sync-guard.mjs`，
 * 自动拦截品牌回退、`.jeikcode/codegraph` 残留、`Build` 默认模式覆盖、
 * CI 包名倒退、插件微内核耦合与关键 UI 入口丢失等问题。
 */

import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const read = (rel) => readFileSync(join(root, rel), 'utf8');

const checks = [];
function check(name, fn) {
  checks.push({ name, fn });
}

check('Cargo workspace points to 1852676265/Zap and avoids jeikcode-core', () => {
  const cargo = read('Cargo.toml');
  assert.match(cargo, /repository\s*=\s*"https:\/\/github\.com\/1852676265\/Zap"/);
  assert.doesNotMatch(cargo, /jeikcode-core/);

  const kernelCargo = read('crates/zap-kernel/Cargo.toml');
  const codingCargo = read('crates/zap-coding/Cargo.toml');
  assert.doesNotMatch(kernelCargo, /zap-neural-memory|zap-agent-teams/);
  assert.doesNotMatch(codingCargo, /zap-neural-memory|zap-agent-teams/);
});

check('Endpoints use 1852676265/Zap update repo', () => {
  const ep = read('crates/zap-config/src/endpoints.rs');
  assert.match(ep, /DEFAULT_UPDATE_REPO:\s*&str\s*=\s*"1852676265\/Zap"/);
  assert.match(ep, /https:\/\/github\.com\/1852676265\/Zap\/releases\/latest\/download\/latest\.json/);
  assert.match(ep, /https:\/\/github\.com\/1852676265\/Zap\/releases\/download/);
});

check('CI & Release workflows use zap-* package names and zap CLI artifacts', () => {
  const ci = read('.github/workflows/ci.yml');
  const port = read('.github/workflows/portability.yml');
  const build = read('.github/workflows/build.yml');
  assert.doesNotMatch(ci, /-p\s+jeikcode-/);
  assert.doesNotMatch(port, /-p\s+jeikcode-/);
  assert.match(build, /dist\/zap-\$\{TAG_NAME\}-windows-/);
});

check('CLI & CodeGraph default to zap and .zap/codegraph', () => {
  const cliCargo = read('crates/zap-cli/Cargo.toml');
  const cliMain = read('crates/zap-cli/src/main.rs');
  const indexDb = read('crates/zap-capabilities/src/codeintel/index_db.rs');
  const codeintelMod = read('crates/zap-capabilities/src/codeintel/mod.rs');
  const chat = read('webui/src/components/Chat.tsx');
  assert.match(cliCargo, /default-run\s*=\s*"zap"/);
  assert.match(cliMain, /#\[command\(name\s*=\s*"zap"/);
  assert.match(cliMain, /init_process_file_logging\("zap\.log",\s*"zap"\)/);
  assert.match(indexDb, /DISK_CACHE_REL_DB:\s*&str\s*=\s*"\.zap\/codegraph\/index\.v1\.db"/);
  assert.match(codeintelMod, /zap init --force/);
  assert.doesNotMatch(codeintelMod, /jeikcode init --force/);
  assert.match(chat, /dispatchOrQueueTerminalCommand\('zap init --force'\)/);
  assert.doesNotMatch(chat, /jeikcode init --force/);
});

check('Auto (bypass) default approval mode is preserved end-to-end', () => {
  const liveApi = read('crates/zap-daemon/src/live_api.rs');
  const apiCfg = read('crates/zap-daemon/src/api_config.rs');
  const modeSwitch = read('webui/src/lib/modeSwitch.ts');
  const sysDialog = read('webui/src/components/SystemSettingsDialog.tsx');

  const initFnStart = liveApi.indexOf('pub(crate) fn init_agent_mode_from_desktop_settings');
  const initFnEnd = liveApi.indexOf('pub(crate) async fn apply_persisted_agent_mode_policy');
  assert.ok(initFnStart >= 0 && initFnEnd > initFnStart);
  const initBody = liveApi.slice(initFnStart, initFnEnd);
  assert.doesNotMatch(initBody, /ApprovalMode::Build/);
  assert.match(initBody, /ApprovalMode::Auto/);

  assert.match(apiCfg, /fn default_agent_global_mode\(\)\s*->\s*String\s*\{\s*"bypass"\.to_string\(\)\s*\}/);
  assert.match(modeSwitch, /export function modeForSessionOrigin[\s\S]*return 'bypass';/);
  assert.match(sysDialog, /useState<ApprovalMode>\('bypass'\)/);
});

check('Prompts & builtin-tools reflect Zap identity and zap_config tool name', () => {
  const initYaml = read('crates/zap-coding/assets/prompts/init.yaml');
  const toolsTxt = read('crates/zap-capabilities/assets/builtin-tools.txt');
  assert.match(initYaml, /agent_name:\s*"Zap"/);
  assert.match(toolsTxt, /^zap_config\b/m);
  assert.doesNotMatch(toolsTxt, /\bjeikcode_config\b/);
});

check('WebUI critical entry points, context menus, and right panel resizer stay intact', () => {
  const appTsx = read('webui/src/app.tsx');
  const sidebar = read('webui/src/components/Sidebar.tsx');
  const sysDialog = read('webui/src/components/SystemSettingsDialog.tsx');
  const appCss = read('webui/src/styles/app.css');
  const pluginRegistry = read('webui/src/custom/custom-registry.tsx');
  const pluginKernel = read('webui/src/custom/plugin-kernel.ts');

  assert.match(appTsx, /settingsSection === 'system'[\s\S]*?initialTab="desktop"/);
  assert.match(appTsx, /onOpenUpdate=\{!updateDead \? openUpdateDialogWithFreshCheck : undefined\}/);
  assert.match(sysDialog, /about-app-title">Zap<\/strong>/);
  assert.match(sidebar, /openContextMenu/);
  assert.match(sidebar, /openProjectMenu/);
  assert.match(sidebar, /handleAnalyzeSession/);
  assert.match(sidebar, /setScheduleDialogOpen\(true\)/);
  assert.match(sidebar, /sidebar-brand-version-clickable/);
  assert.match(sidebar, /onClick=\{onOpenUpdate\}/);
  assert.match(sidebar, /session-group-header[\s\S]*sidebar\.recents[\s\S]*sidebar-recents-section[\s\S]*sidebar\.projects[\s\S]*sidebar-projects-list/);
  assert.ok(sidebar.indexOf('sidebar-recents-section') < sidebar.indexOf('sidebar-projects-list'));
  assert.match(appCss, /\.right-inspector-panel\s*\{[^}]*overflow:\s*visible;/s);
  assert.doesNotMatch(appCss, /\.right-inspector-panel\s*\{\s*--right-panel-width:\s*300px\s*!important;/);
  assert.doesNotMatch(pluginRegistry + pluginKernel, /agent-teams|neural-memory|teams:|neural:/);
});

let failed = 0;
for (const { name, fn } of checks) {
  try {
    fn();
    console.log(`PASS: ${name}`);
  } catch (err) {
    failed += 1;
    console.error(`FAIL: ${name}\n  ${err instanceof Error ? err.message : String(err)}`);
  }
}

if (failed > 0) {
  console.error(`\nzap-sync-guard: ${failed} check(s) failed.`);
  process.exit(1);
}
console.log(`\nzap-sync-guard: all ${checks.length} checks passed.`);
