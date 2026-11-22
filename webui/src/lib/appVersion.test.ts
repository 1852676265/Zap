import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { test } from 'node:test';
import { formatAppVersionLabel, normalizeAppVersion } from './appVersion.ts';

test('normalizeAppVersion strips a leading v and trims', () => {
  assert.equal(normalizeAppVersion('v7.0.13'), '7.0.13');
  assert.equal(normalizeAppVersion('7.0.13'), '7.0.13');
  assert.equal(normalizeAppVersion('  V7.0.13  '), '7.0.13');
  assert.equal(normalizeAppVersion(''), '');
  assert.equal(normalizeAppVersion(undefined), '');
});

test('formatAppVersionLabel always shows a single v prefix', () => {
  assert.equal(formatAppVersionLabel('7.0.13'), 'v7.0.13');
  assert.equal(formatAppVersionLabel('v7.0.13'), 'v7.0.13');
  assert.equal(formatAppVersionLabel(''), '');
});

test('Sidebar brand version and About pane expose clickable update check entries', () => {
  const root = process.cwd();
  const sidebar = readFileSync(join(root, 'src/components/Sidebar.tsx'), 'utf8');
  const app = readFileSync(join(root, 'src/app.tsx'), 'utf8');
  const sysDialog = readFileSync(join(root, 'src/components/SystemSettingsDialog.tsx'), 'utf8');

  assert.match(sidebar, /class=\{`sidebar-brand-version sidebar-brand-version-clickable/);
  assert.match(sidebar, /onClick=\{onOpenUpdate\}/);
  assert.match(app, /onOpenUpdate=\{!updateDead \? openUpdateDialogWithFreshCheck : undefined\}/);
  assert.match(sysDialog, /onClick=\{onOpenUpdate\}/);
});
