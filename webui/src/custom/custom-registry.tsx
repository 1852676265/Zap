/**
 * Zap 统一设置中心与插件覆盖层宿主 (Plugin & Settings Host)
 */

import { h } from 'preact';
import { useEffect, useState } from 'preact/hooks';
import { DeepSeekSettingsCenterModal } from './plugin-settings-center';
import { zapPluginKernel, usePluginKernelVersion } from './plugin-kernel';
import './custom.css';

const OPEN_CENTER_EVENT = 'zap:open-deepseek-settings-center';

export function openDeepSeekSettingsCenter(initialSection: string = 'general') {
  window.dispatchEvent(
    new CustomEvent(OPEN_CENTER_EVENT, { detail: { section: initialSection } }),
  );
}

export function PluginCenterHostApp() {
  const [open, setOpen] = useState(false);
  const [initialSection, setInitialSection] = useState<string>('general');
  usePluginKernelVersion();

  useEffect(() => {
    zapPluginKernel.syncAllPackagesRuntimeEffects();
    const handler = (e: Event) => {
      const detail = (e as CustomEvent<{ section?: string }>).detail;
      setInitialSection(detail?.section || 'general');
      setOpen(true);
    };
    window.addEventListener(OPEN_CENTER_EVENT, handler);
    return () => {
      window.removeEventListener(OPEN_CENTER_EVENT, handler);
    };
  }, []);

  const overlays = zapPluginKernel.getSlotEntries('shell.overlay');

  return (
    <>
      {overlays.map((ov) => (
        <div key={ov.id} class="zap-plugin-overlay-slot">
          {ov.render()}
        </div>
      ))}
      <DeepSeekSettingsCenterModal
        open={open}
        initialSection={initialSection}
        onClose={() => setOpen(false)}
      />
    </>
  );
}
