import DOMPurify from 'dompurify';
import { useMemo, useRef, useEffect, useState } from 'preact/hooks';
import { render } from 'preact';
import { createPortal, memo } from 'preact/compat';
import { markdownToHtml } from '../lib/markdownRender';
import { MermaidDiagram } from './MermaidDiagram';
import { useSettings } from '../settings';
import { showToast } from './Toast';
import { parseFilePathAndLines } from '../lib/displayPath';

export { preprocessMarkdown, markdownToHtml } from '../lib/markdownRender';

export function highlightHtml(html: string, search: string, activeIdx?: number): { html: string; count: number } {
  if (!search.trim()) return { html, count: 0 };
  const parser = new DOMParser();
  const doc = parser.parseFromString(html, 'text/html');
  const walk = doc.createTreeWalker(doc.body, NodeFilter.SHOW_TEXT, null);
  const nodes: Text[] = [];
  let node: Node | null;
  while ((node = walk.nextNode())) {
    nodes.push(node as Text);
  }
  const searchLower = search.toLowerCase();
  let matchCount = 0;
  for (const textNode of nodes) {
    const text = textNode.nodeValue ?? '';
    if (!text.toLowerCase().includes(searchLower)) continue;

    const parent = textNode.parentNode;
    if (!parent) continue;

    const parentTag = (parent as HTMLElement).tagName?.toLowerCase();
    if (parentTag === 'script' || parentTag === 'style') continue;

    const newFragment = doc.createDocumentFragment();
    let lastIndex = 0;
    let index = text.toLowerCase().indexOf(searchLower);
    while (index !== -1) {
      if (index > lastIndex) {
        newFragment.appendChild(doc.createTextNode(text.substring(lastIndex, index)));
      }
      const mark = doc.createElement('mark');
      const isCurrent = activeIdx !== undefined && matchCount === activeIdx;
      mark.className = 'msg-search-highlight' + (isCurrent ? ' is-current-match' : '');
      mark.setAttribute('data-match-idx', String(matchCount));
      mark.textContent = text.substring(index, index + search.length);
      newFragment.appendChild(mark);
      matchCount++;
      lastIndex = index + search.length;
      index = text.toLowerCase().indexOf(searchLower, lastIndex);
    }
    if (lastIndex < text.length) {
      newFragment.appendChild(doc.createTextNode(text.substring(lastIndex)));
    }
    parent.replaceChild(newFragment, textNode);
  }
  return { html: doc.body.innerHTML, count: matchCount };
}

// 路径归一化函数：兼容 Windows 盘符路径与 POSIX 路径
function normalizePath(rawPath: string): string {
  const norm = rawPath.trim().replace(/^['"`]|['"`]$/g, '').replace(/\\/g, '/');
  const lineCleaned = norm.replace(/:\d+(?::\d+)?$/, '');
  const hasDrive = /^[a-zA-Z]:/.test(lineCleaned);
  const parts = lineCleaned.split('/');
  const stack: string[] = [];
  for (const p of parts) {
    if (!p || p === '.') continue;
    if (p === '..') stack.pop();
    else stack.push(p);
  }
  return (hasDrive ? '' : lineCleaned.startsWith('/') ? '/' : '') + stack.join('/');
}

const ALL_FILE_EXTS = 'kt|kts|java|c|cpp|cc|cxx|h|hpp|hxx|cs|swift|m|mm|scala|clj|cljs|rs|go|py|pyw|rb|php|pl|pm|lua|r|dart|ts|tsx|js|jsx|mjs|cjs|json|json5|jsonc|toml|yaml|yml|xml|html|htm|xhtml|css|scss|sass|less|styl|sh|bash|zsh|fish|ps1|psm1|bat|cmd|sql|prisma|proto|graphql|gql|vue|svelte|astro|md|markdown|mdx|txt|text|log|diff|patch|env|properties|ini|conf|cfg|lock|svg|png|jpg|jpeg|gif|webp|ico|bmp|avif|gradle|cmake|dockerfile|containerfile|nix';
const EXT_REGEX = new RegExp(`\\.(${ALL_FILE_EXTS})$`, 'i');
const ROOT_DIRS_REGEX = /^(crates|src|webui|packages|desktop|docs|scripts|extensions|functions|plans|tests|assets|public|internal|cmd|api|lib)\/[\w.\-\\/]+$/i;

const CLI_COMMAND_STARTS = new Set([
  'npm', 'npx', 'pnpm', 'yarn', 'bun', 'node', 'deno',
  'cargo', 'rustc', 'rustup',
  'git', 'gh', 'svn', 'hg',
  'python', 'python3', 'py', 'pip', 'pip3', 'pytest',
  'go', 'gofmt',
  'docker', 'docker-compose', 'podman', 'kubectl', 'helm',
  'bash', 'sh', 'zsh', 'fish', 'source', 'export',
  'cat', 'grep', 'find', 'awk', 'sed', 'cd', 'ls', 'dir',
  'cp', 'mv', 'rm', 'mkdir', 'touch', 'chmod', 'chown',
  'curl', 'wget', 'ssh', 'scp', 'tar', 'zip', 'unzip',
  'make', 'cmake', 'ninja', 'mvn', 'gradle',
  'vite', 'tsc', 'eslint', 'prettier', 'jest', 'vitest',
]);

/**
 * 智能识别一段文本是否是 CLI 终端命令行，防止命令代码被误当成文件路径
 */
export function isCommandSnippet(text: string): boolean {
  const t = (text || '').trim().replace(/^['"`]|['"`]$/g, '');
  if (!t) return false;

  // 1. 包含命令行连接符、管道符、分号或重定向符
  if (/(\s+&&\s+|\s+\|\|\s+|\s+\|\s+|\s+>>?\s+|\s+<<?\s+|;\s*)/.test(t)) return true;

  // 2. 包含典型的命令行参数标志 (如 --test, -v, -p, --save-dev, -rf)
  if (/\s+--?[a-zA-Z0-9]/.test(t)) return true;

  // 3. 以常见 CLI 命令关键字开头且附带参数 (如 npm run, git status, node --test)
  const firstWord = t.split(/[\s/\\:]/)[0]?.toLowerCase();
  if (firstWord && CLI_COMMAND_STARTS.has(firstWord) && /\s+/.test(t)) {
    return true;
  }

  // 4. 包含空格且含常见命令行动词 (run build, test, install 等)
  if (/\s+(?:run|build|test|start|install|add|commit|checkout|push|pull|revert|fetch|merge|diff|status|serve|watch)\b/i.test(t)) {
    return true;
  }

  return false;
}

/**
 * 智能识别一段文本是否是 Web 网址或域名 (如 https://..., //..., www...., viewer.autodesk.com)
 */
export function isWebUrlSnippet(text: string): boolean {
  const t = (text || '').trim().replace(/^['"`]|['"`]$/g, '');
  if (!t || /\s/.test(t)) return false;
  if (/^(https?:|\/\/)/i.test(t) || /^www\./i.test(t)) return true;
  return /^([a-zA-Z0-9-]+\.)+(com|cn|org|net|edu|gov|io|ai|dev|co|me|app|top|xyz|info|vip|cc|tv|site|online|tech|cloud|icu|club|fun|art|link|biz)(?::\d+)?(?:[/?#]|$)/i.test(t);
}

/**
 * 智能识别一段文本是否是项目源码/资源文件路径（支持健壮提取附带的行号与行范围）
 */
export function isFilePathSnippet(text: string): boolean {
  if (isCommandSnippet(text)) return false;

  const { filePath } = parseFilePathAndLines(text);
  const t = filePath.trim().replace(/^['"`]|['"`]$/g, '');
  if (t.length < 2 || t.length > 350) return false;
  if (/^(https?:|\/\/)/i.test(t) || /^www\./i.test(t) || isWebUrlSnippet(t)) return false;

  // 1. 标准扩展名匹配 (全语言源码、脚本、配置、文本及资源)
  if (EXT_REGEX.test(t)) return true;

  // 2. 绝对路径特征
  if (/^[a-zA-Z]:[\\/][\w.\-\\/]+?\.[a-zA-Z0-9]+$/i.test(t)) return true;
  if (/^\/[\w.\-\\/]+?\.[a-zA-Z0-9]+$/i.test(t)) return true;

  // 3. 常见工程目录层级结构
  if (ROOT_DIRS_REGEX.test(t)) return true;

  return false;
}

// 解析目标文件路径 (全面支持 Windows/POSIX 绝对路径与基于 basePath / cwd 的相对路径)
function resolveTargetFile(href: string, basePath?: string, cwd?: string): string {
  const clean = href.trim().replace(/^['"`]|['"`]$/g, '').replace(/\\/g, '/');
  const hasDrive = /^[a-zA-Z]:/.test(clean);
  const isPosixAbs = clean.startsWith('/') && !clean.startsWith('//');

  // 1. 如果是绝对路径
  if (hasDrive || isPosixAbs) {
    return normalizePath(clean);
  }

  // 2. 如果是相对路径
  let anchorDir = '';
  if (basePath) {
    const normBase = basePath.replace(/\\/g, '/');
    anchorDir = normBase.substring(0, normBase.lastIndexOf('/'));
  } else if (cwd) {
    anchorDir = cwd.replace(/\\/g, '/');
  }

  if (anchorDir) {
    return normalizePath(anchorDir + '/' + clean.replace(/^\.\//, ''));
  }
  return normalizePath(clean);
}

// 在 HTML 注入 DOM 之前，统一将相对图片 src 预重写为 /fs/raw 流式接口，根治离开久了图片 404 与休眠裂图问题！
function rewriteImageSources(html: string, basePath?: string, cwd?: string): string {
  if (!/<img\b/i.test(html)) return html;
  const parser = new DOMParser();
  const doc = parser.parseFromString(html, 'text/html');
  const imgNodes = doc.querySelectorAll('img');
  if (imgNodes.length === 0) return html;

  imgNodes.forEach((img) => {
    const src = img.getAttribute('src');
    if (!src || /^(https?:|data:|\/\/|\/fs\/raw|fs\/raw)/i.test(src)) return;
    const resolvedAbs = resolveTargetFile(src, basePath, cwd);
    img.setAttribute('src', `/fs/raw?path=${encodeURIComponent(resolvedAbs)}`);
    img.setAttribute('loading', 'lazy');
  });
  return doc.body.innerHTML;
}

function escapeHtmlAttr(val: string): string {
  return val
    .replace(/&/g, '&amp;')
    .replace(/"/g, '&quot;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}

function decodeHtmlEntities(raw: string): string {
  if (!raw.includes('&')) return raw;
  return raw
    .replace(/&quot;/g, '"')
    .replace(/&#39;|&apos;/g, "'")
    .replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>')
    .replace(/&amp;/g, '&');
}

const inlineCodeKindCache = new Map<string, 'none' | 'web' | 'file'>();
const MAX_INLINE_CODE_CACHE = 2000;

function classifyInlineCodeText(text: string): 'none' | 'web' | 'file' {
  if (!text || text.length > 360 || text.includes('\n')) return 'none';
  const hit = inlineCodeKindCache.get(text);
  if (hit !== undefined) return hit;
  let kind: 'none' | 'web' | 'file' = 'none';
  if (isWebUrlSnippet(text)) {
    kind = 'web';
  } else if (isFilePathSnippet(text)) {
    kind = 'file';
  }
  if (inlineCodeKindCache.size >= MAX_INLINE_CODE_CACHE) {
    const firstKey = inlineCodeKindCache.keys().next().value;
    if (firstKey !== undefined) inlineCodeKindCache.delete(firstKey);
  }
  inlineCodeKindCache.set(text, kind);
  return kind;
}

// 在字符串阶段预注入行内代码徽章与超链接胶囊样式，彻底免除挂载后 querySelectorAll 全子树扫描与二次样式重算
function decorateBadgesAndLinks(
  html: string,
  openExternalLabel: string,
  clickPreviewLabel: string,
  openInMainLabel: string,
): string {
  if (!html.includes('<code') && !html.includes('<a ')) return html;
  return html.replace(
    /<pre\b[\s\S]*?<\/pre>|<a\s+([^>]*?\bhref="[^"]*"[^>]*)>([\s\S]*?)<\/a>|<code>([\s\S]*?)<\/code>/gi,
    (match, anchorAttrs?: string, anchorInner?: string, codeInner?: string) => {
      if (anchorAttrs !== undefined) {
        let attrs = anchorAttrs;
        if (/\bclass="/i.test(attrs)) {
          attrs = attrs.replace(/\bclass="([^"]*)"/i, (_m, cls: string) =>
            cls.includes('markdown-link-pill') ? `class="${cls}"` : `class="${cls} markdown-link-pill"`,
          );
        } else {
          attrs += ' class="markdown-link-pill"';
        }
        if (!/\bdraggable=/i.test(attrs)) {
          attrs += ' draggable="false"';
        }
        return `<a ${attrs}>${anchorInner ?? ''}</a>`;
      }
      if (codeInner !== undefined) {
        if (codeInner.includes('<')) return match;
        const plain = decodeHtmlEntities(codeInner).trim();
        const kind = classifyInlineCodeText(plain);
        if (kind === 'web') {
          const title = escapeHtmlAttr(`${plain} (${openExternalLabel})`);
          return `<code class="file-path-badge web-link-badge" title="${title}" tabindex="0" role="link" draggable="false">${codeInner}</code>`;
        }
        if (kind === 'file') {
          const title = escapeHtmlAttr(`${plain} (${clickPreviewLabel}, Ctrl+Click: ${openInMainLabel})`);
          return `<code class="file-path-badge" title="${title}" tabindex="0" role="button" draggable="false">${codeInner}</code>`;
        }
        return match;
      }
      return match;
    },
  );
}

const MAX_RENDERED_MD_CACHE = 600;
const renderedMarkdownCache = new Map<string, string>();

function compileBaseMarkdownHtml(
  content: string,
  basePath: string | undefined,
  cwd: string | undefined,
  lang: string,
  openExternalLabel: string,
  clickPreviewLabel: string,
  openInMainLabel: string,
): string {
  const text = content ?? '';
  // 超长单篇文档（>120KB）不占长驻缓存，常规对话消息走 O(1) LRU 命中
  const cacheable = text.length <= 120_000;
  const cacheKey = cacheable ? `${lang}\0${basePath ?? ''}\0${cwd ?? ''}\0${text}` : '';
  if (cacheable) {
    const cached = renderedMarkdownCache.get(cacheKey);
    if (cached !== undefined) {
      renderedMarkdownCache.delete(cacheKey);
      renderedMarkdownCache.set(cacheKey, cached);
      return cached;
    }
  }

  const raw = markdownToHtml(text, { highlight: true });
  // SECURITY: model output is untrusted — sanitize before injecting as HTML.
  const sanitized = DOMPurify.sanitize(raw, {
    ADD_TAGS: ['mark', 'math', 'annotation', 'semantics', 'mrow', 'mi', 'mo', 'mn', 'ms', 'mtext', 'mspace', 'mfrac', 'msqrt', 'mroot', 'msub', 'msup', 'msubsup', 'munder', 'mover', 'munderover', 'mtable', 'mtr', 'mtd', 'mstyle'],
    ADD_ATTR: ['data-match-idx', 'id', 'data-alt-id', 'data-copy', 'data-mermaid-code', 'class', 'checked', 'disabled', 'type', 'align', 'start', 'colspan', 'rowspan', 'style', 'aria-hidden', 'encoding', 'target', 'rel'],
  });
  const withImages = rewriteImageSources(sanitized, basePath, cwd);
  const decorated = decorateBadgesAndLinks(
    withImages,
    openExternalLabel,
    clickPreviewLabel,
    openInMainLabel,
  );

  if (cacheable) {
    if (renderedMarkdownCache.size >= MAX_RENDERED_MD_CACHE) {
      const oldestKey = renderedMarkdownCache.keys().next().value;
      if (oldestKey !== undefined) renderedMarkdownCache.delete(oldestKey);
    }
    renderedMarkdownCache.set(cacheKey, decorated);
  }
  return decorated;
}

/** 供后台空闲预热会话 Markdown 渲染缓存，使切入历史会话时直接 O(1) 命中已编译 HTML */
export function warmMarkdownCache(
  content: string,
  cwd: string | undefined,
  lang: string,
  labels: { openExternal: string; clickPreview: string; openInMain: string },
): void {
  if (!content || content.length > 120_000) return;
  const cacheKey = `${lang}\0\0${cwd ?? ''}\0${content}`;
  if (renderedMarkdownCache.has(cacheKey)) return;
  compileBaseMarkdownHtml(
    content,
    undefined,
    cwd,
    lang,
    labels.openExternal,
    labels.clickPreview,
    labels.openInMain,
  );
}

export interface MarkdownProps {
  content: string;
  search?: string;
  basePath?: string;
  cwd?: string;
  activeSearchIndex?: number;
  onMatchCountChange?: (count: number) => void;
  targetContext?: 'main' | 'right' | 'chat';
}

export const Markdown = memo(function Markdown({
  content,
  search,
  basePath,
  cwd,
  activeSearchIndex,
  onMatchCountChange,
  targetContext = 'chat',
}: MarkdownProps) {
  const { theme, lang, t } = useSettings();
  const isDark = (theme === 'dark') ||
    (theme === 'system' && typeof window !== 'undefined' && window.matchMedia('(prefers-color-scheme: dark)').matches) ||
    (typeof document !== 'undefined' && document.documentElement.dataset.theme === 'dark');

  const rootRef = useRef<HTMLDivElement>(null);
  const openExternalLabel = t('preview.clickToOpenExternal');
  const clickPreviewLabel = t('preview.clickToPreview');
  const openInMainLabel = t('preview.openInMain');

  const { renderedHtml, matchCount } = useMemo(() => {
    const prepped = compileBaseMarkdownHtml(
      content ?? '',
      basePath,
      cwd,
      lang,
      openExternalLabel,
      clickPreviewLabel,
      openInMainLabel,
    );

    if (search && search.trim()) {
      const res = highlightHtml(prepped, search);
      return { renderedHtml: res.html, matchCount: res.count };
    }
    return { renderedHtml: prepped, matchCount: 0 };
  }, [content, search, basePath, cwd, lang, openExternalLabel, clickPreviewLabel, openInMainLabel]);

  useEffect(() => {
    if (!onMatchCountChange) return;
    onMatchCountChange(matchCount);
  }, [matchCount, onMatchCountChange]);

  // 当前激活选中的匹配项切换时，添加 .is-current-match 并平滑滚动居中定位！
  useEffect(() => {
    if (!rootRef.current || !renderedHtml.includes('msg-search-highlight')) return;
    const marks = rootRef.current.querySelectorAll<HTMLElement>('.msg-search-highlight');
    marks.forEach((m, idx) => {
      if (activeSearchIndex !== undefined && idx === activeSearchIndex) {
        m.classList.add('is-current-match');
        m.scrollIntoView({ behavior: 'smooth', block: 'center' });
      } else {
        m.classList.remove('is-current-match');
      }
    });
  }, [activeSearchIndex, renderedHtml]);

  // 水合挂载 MermaidDiagram 交互矢量图组件
  useEffect(() => {
    if (!rootRef.current || !renderedHtml.includes('mermaid-diagram-mount')) return;

    const mountNodes = rootRef.current.querySelectorAll<HTMLElement>('.mermaid-diagram-mount');
    const mountedElements: HTMLElement[] = [];

    mountNodes.forEach((node) => {
      const rawCode = node.getAttribute('data-mermaid-code');
      if (!rawCode) return;
      try {
        const decoded = decodeURIComponent(rawCode);
        render(
          <MermaidDiagram code={decoded} isDark={isDark} t={t} />,
          node,
        );
        mountedElements.push(node);
      } catch (err) {
        console.warn('[Markdown] Failed to mount MermaidDiagram:', err);
      }
    });

    return () => {
      for (const node of mountedElements) {
        try {
          render(null, node);
        } catch {}
      }
    };
  }, [renderedHtml, isDark, lang, basePath]);

  const lastDragTimeRef = useRef(0);
  const dragCleanupRef = useRef<(() => void) | null>(null);

  useEffect(() => {
    return () => {
      dragCleanupRef.current?.();
      dragCleanupRef.current = null;
    };
  }, []);

  // 按需监听鼠标长按拖拽划词事件：仅在点中徽章/链接时临时挂载 window 监听，松开即卸载，消除常驻监听器风暴
  function onMouseDown(e: MouseEvent) {
    if (e.button !== 0) return;
    const badge = (e.target as HTMLElement)?.closest('.file-path-badge, a[href]') as HTMLElement | null;
    if (!badge) return;

    dragCleanupRef.current?.();
    let didDrag = false;
    const downX = e.clientX;
    const downY = e.clientY;

    const handleMouseMove = (moveEv: MouseEvent) => {
      const dx = Math.abs(moveEv.clientX - downX);
      const dy = Math.abs(moveEv.clientY - downY);
      if (dx > 3 || dy > 3) {
        didDrag = true;
        badge.classList.add('is-selecting');
      }
    };

    const cleanup = () => {
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
      badge.classList.remove('is-selecting');
      if (dragCleanupRef.current === cleanup) {
        dragCleanupRef.current = null;
      }
    };

    const handleMouseUp = () => {
      if (didDrag) {
        lastDragTimeRef.current = Date.now();
      }
      cleanup();
    };

    dragCleanupRef.current = cleanup;
    window.addEventListener('mousemove', handleMouseMove, { passive: true });
    window.addEventListener('mouseup', handleMouseUp, { passive: true });
  }

  function onClick(e: MouseEvent) {
    const targetEl = e.target as HTMLElement;

    // 1. 处理内部锚点平滑跳转（如 [中文更新日志](#-中文更新日志-chinese)）
    const anchor = targetEl?.closest('a[href^="#"]') as HTMLAnchorElement | null;
    if (anchor) {
      const rawHash = anchor.getAttribute('href')?.slice(1);
      if (rawHash) {
        e.preventDefault();
        const hash = decodeURIComponent(rawHash);
        const altHash = hash.startsWith('-') ? hash.slice(1) : `-${hash}`;
        const container = (e.currentTarget as HTMLElement)?.closest('.messages-container') || document;
        let target = container.querySelector(`[id="${CSS.escape(hash)}"]`) ||
          container.querySelector(`[id="${CSS.escape(altHash)}"]`) ||
          container.querySelector(`[data-alt-id="${CSS.escape(hash)}"]`) ||
          container.querySelector(`[data-alt-id="${CSS.escape(altHash)}"]`);
        if (!target && container !== document) {
          target = document.querySelector(`[id="${CSS.escape(hash)}"]`) ||
            document.querySelector(`[id="${CSS.escape(altHash)}"]`);
        }
        if (target) {
          target.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
      }
      return;
    }

    // 2. 拦截文件引用链接、Markdown超链接与行内代码路径/链接识别
    const clickable = targetEl?.closest('a[href], .file-preview-link, code') as HTMLElement | null;
    if (clickable) {
      let href = clickable.getAttribute('href') || clickable.dataset.filePath || '';
      // 如果点击的是行内 code 标签，智能检测其内容是否是 Web 链接或文件路径
      if (!href && clickable.tagName?.toLowerCase() === 'code') {
        const text = (clickable.textContent || '').trim();
        if (isWebUrlSnippet(text)) {
          href = text;
        } else if (isFilePathSnippet(text)) {
          href = text;
        }
      }

      // 如果用户刚刚进行过拖拽划词选择（250ms 内），或者当前窗口已有文本被划选，绝不触发跳转/打开
      if (Date.now() - lastDragTimeRef.current < 250) {
        return;
      }
      const sel = window.getSelection()?.toString().trim();
      if (sel && sel.length > 0) {
        return;
      }

      // (A) 外部 Web 链接：在系统默认原始浏览器中弹出打开 (Tauri 桌面壳自动由 on_new_window 接管)
      if (href && (isWebUrlSnippet(href) || /^(https?:|\/\/)/i.test(href) || /^www\./i.test(href))) {
        e.preventDefault();
        e.stopPropagation();

        let targetUrl = href.trim();
        if (targetUrl.startsWith('//')) {
          targetUrl = 'https:' + targetUrl;
        } else if (!/^https?:\/\//i.test(targetUrl)) {
          targetUrl = 'https://' + targetUrl;
        }

        try {
          window.open(targetUrl, '_blank', 'noopener,noreferrer');
        } catch {}
        return;
      }

      // (B) 本地文件路径引用：在预览或主编辑区打开
      if (href && !href.startsWith('#') && !/^(https?:|mailto:|\/\/)/i.test(href)) {
        e.preventDefault();
        e.stopPropagation();

        const { filePath: cleanHref, startLine, endLine } = parseFilePathAndLines(href);
        const resolved = resolveTargetFile(cleanHref, basePath, cwd);
        const fileName = resolved.split(/[\\/]/).pop() || resolved;

        if (e.ctrlKey || e.metaKey || e.altKey) {
          // 按住 Ctrl / Cmd / Alt 点击：直接在中间主编辑区打开并定位行号！
          window.dispatchEvent(new CustomEvent('jeikcode:open_file', {
            detail: { path: resolved, startLine, endLine }
          }));
          return;
        }

        if (targetContext === 'right') {
          // 右侧预览区点击：在右侧标签区打开，绝不跳去中间主舞台，互不干扰！
          window.dispatchEvent(new CustomEvent('jeikcode:preview:open_file', {
            detail: { filePath: resolved, fileName, startLine, endLine }
          }));
        } else if (targetContext === 'main') {
          // 中间主预览区点击：在中间主标签区打开，绝不跳去右侧，互不干扰！
          window.dispatchEvent(new CustomEvent('jeikcode:open_file', {
            detail: { path: resolved, startLine, endLine }
          }));
        } else {
          // AI 对话主画布中点击：默认在右侧区域预览出来，复用异地浏览文件能力，不遮盖AI对话！
          window.dispatchEvent(new CustomEvent('jeikcode:preview:open_file', {
            detail: { filePath: resolved, fileName, startLine, endLine }
          }));
          window.dispatchEvent(new CustomEvent('jeikcode:switch_right_tab', {
            detail: 'preview'
          }));
        }
        return;
      }
    }

    // 2. 处理代码块 Copy 按钮
    const t = targetEl?.closest('.copy-button') as HTMLElement | null;
    if (t?.dataset.copy) {
      const text = decodeURIComponent(t.dataset.copy);
      const prev = t.textContent;
      const mark = (ok: boolean) => {
        t.textContent = ok ? 'Copied' : 'Failed';
        setTimeout(() => {
          t.textContent = prev;
        }, 1200);
      };
      void (async () => {
        try {
          if (navigator.clipboard?.writeText) {
            await navigator.clipboard.writeText(text);
            mark(true);
            return;
          }
        } catch {
          /* fall through */
        }
        try {
          const ta = document.createElement('textarea');
          ta.value = text;
          ta.setAttribute('readonly', '');
          ta.style.position = 'fixed';
          ta.style.left = '-9999px';
          document.body.appendChild(ta);
          ta.select();
          const ok = document.execCommand('copy');
          document.body.removeChild(ta);
          mark(ok);
        } catch {
          mark(false);
        }
      })();
    }
  }

  function onKeyDown(e: KeyboardEvent) {
    if (e.key === 'Enter') {
      const targetEl = e.target as HTMLElement;
      if (targetEl?.classList.contains('file-path-badge')) {
        targetEl.click();
      }
    }
  }

  const [badgeMenu, setBadgeMenu] = useState<{
    x: number;
    y: number;
    type: 'file' | 'link';
    url?: string;
    text: string;
    filePath?: string;
    fileName?: string;
    startLine?: number;
    endLine?: number;
  } | null>(null);

  useEffect(() => {
    if (!badgeMenu) return;
    const close = () => setBadgeMenu(null);
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setBadgeMenu(null);
    };
    window.addEventListener('click', close);
    window.addEventListener('contextmenu', close);
    window.addEventListener('keydown', onKey);
    return () => {
      window.removeEventListener('click', close);
      window.removeEventListener('contextmenu', close);
      window.removeEventListener('keydown', onKey);
    };
  }, [badgeMenu]);

  function onContextMenu(e: MouseEvent) {
    const targetEl = e.target as HTMLElement;
    const clickable = targetEl?.closest('a[href], .file-path-badge, .file-preview-link, code') as HTMLElement | null;
    if (clickable) {
      let href = clickable.getAttribute('href') || clickable.dataset.filePath || '';
      const text = (clickable.textContent || '').trim();
      if (!href && clickable.tagName?.toLowerCase() === 'code') {
        if (isWebUrlSnippet(text)) {
          href = text;
        } else if (isFilePathSnippet(text)) {
          href = text;
        }
      }

      if (!href) return;

      const menuWidth = 180;
      const menuHeight = 160;
      let x = e.clientX;
      let y = e.clientY;
      if (typeof window !== 'undefined') {
        if (x + menuWidth > window.innerWidth) x = Math.max(10, window.innerWidth - menuWidth - 10);
        if (y + menuHeight > window.innerHeight) y = Math.max(10, window.innerHeight - menuHeight - 10);
      }

      // (A) Web 链接
      if (isWebUrlSnippet(href) || /^(https?:|\/\/)/i.test(href) || /^www\./i.test(href)) {
        e.preventDefault();
        e.stopPropagation();
        let targetUrl = href.trim();
        if (targetUrl.startsWith('//')) {
          targetUrl = 'https:' + targetUrl;
        } else if (!/^https?:\/\//i.test(targetUrl)) {
          targetUrl = 'https://' + targetUrl;
        }

        setBadgeMenu({
          x,
          y,
          type: 'link',
          url: targetUrl,
          text: text || targetUrl,
        });
        return;
      }

      // (B) 本地文件路径引用
      if (!href.startsWith('#') && !/^(https?:|mailto:|\/\/)/i.test(href)) {
        e.preventDefault();
        e.stopPropagation();
        const { filePath: cleanHref, startLine, endLine } = parseFilePathAndLines(href);
        const resolved = resolveTargetFile(cleanHref, basePath, cwd);
        const fileName = resolved.split(/[\\/]/).pop() || resolved;
        setBadgeMenu({
          x,
          y,
          type: 'file',
          filePath: resolved,
          fileName,
          text: text || fileName,
          startLine,
          endLine,
        });
        return;
      }
    }
  }

  return (
    <>
      <div
        ref={rootRef}
        class="markdown-root assistant-message-content"
        onMouseDown={onMouseDown}
        onClick={onClick}
        onContextMenu={onContextMenu}
        onKeyDown={onKeyDown}
        dangerouslySetInnerHTML={{ __html: renderedHtml }}
      />
      {badgeMenu && typeof document !== 'undefined' && createPortal(
        <div
          class="file-context-menu"
          style={{ top: `${badgeMenu.y}px`, left: `${badgeMenu.x}px` }}
          onClick={(e) => e.stopPropagation()}
        >
          {badgeMenu.type === 'link' ? (
            <>
              <button
                type="button"
                class="context-menu-btn"
                onClick={() => {
                  if (badgeMenu.url) {
                    try {
                      window.open(badgeMenu.url, '_blank', 'noopener,noreferrer');
                    } catch {}
                  }
                  setBadgeMenu(null);
                }}
              >
                <span>🌐 {t('preview.openInBrowser')}</span>
              </button>
              <button
                type="button"
                class="context-menu-btn"
                onClick={() => {
                  if (badgeMenu.url) {
                    void navigator.clipboard?.writeText(badgeMenu.url);
                    showToast(t('copy.copied'), 'success');
                  }
                  setBadgeMenu(null);
                }}
              >
                <span>🔗 {t('preview.copyLink')}</span>
              </button>
              <button
                type="button"
                class="context-menu-btn"
                onClick={() => {
                  if (badgeMenu.text) {
                    void navigator.clipboard?.writeText(badgeMenu.text);
                    showToast(t('copy.copied'), 'success');
                  }
                  setBadgeMenu(null);
                }}
              >
                <span>📋 {t('preview.copyText')}</span>
              </button>
            </>
          ) : (
            <>
              <button
                type="button"
                class="context-menu-btn"
                onClick={() => {
                  window.dispatchEvent(new CustomEvent('jeikcode:preview:open_file', {
                    detail: {
                      filePath: badgeMenu.filePath,
                      fileName: badgeMenu.fileName,
                      startLine: badgeMenu.startLine,
                      endLine: badgeMenu.endLine,
                    }
                  }));
                  window.dispatchEvent(new CustomEvent('jeikcode:switch_right_tab', {
                    detail: 'preview'
                  }));
                  setBadgeMenu(null);
                }}
              >
                <span>👁️ {t('preview.openInRight')}</span>
              </button>
              <button
                type="button"
                class="context-menu-btn"
                onClick={() => {
                  window.dispatchEvent(new CustomEvent('jeikcode:open_file', {
                    detail: {
                      path: badgeMenu.filePath,
                      startLine: badgeMenu.startLine,
                      endLine: badgeMenu.endLine,
                    }
                  }));
                  setBadgeMenu(null);
                }}
              >
                <span>📑 {t('preview.openInMain')}</span>
              </button>
              <button
                type="button"
                class="context-menu-btn"
                onClick={() => {
                  if (badgeMenu.filePath) {
                    void navigator.clipboard?.writeText(badgeMenu.filePath);
                    showToast(t('explorer.copyFullPath'), 'success');
                  }
                  setBadgeMenu(null);
                }}
              >
                <span>📋 {t('explorer.copyFullPath')}</span>
              </button>
              <button
                type="button"
                class="context-menu-btn"
                onClick={() => {
                  if (badgeMenu.text) {
                    void navigator.clipboard?.writeText(badgeMenu.text);
                    showToast(t('copy.copied'), 'success');
                  }
                  setBadgeMenu(null);
                }}
              >
                <span>📋 {t('preview.copyText')}</span>
              </button>
            </>
          )}
        </div>,
        document.body
      )}
    </>
  );
});
