// "Playground" layout (optional, after Meta's UMA playground): the reaction
// graph fills the screen; everything else floats at its edges. Status and the
// primary action top left, the page switch top centre, level of theory and
// the structure list top right, and a bottom dock with every graph action.
// Selection details, calculations, profiles and adding structures open as
// dark sheets over the canvas. It reuses the classic layout's components;
// only their placement (and, inside sheets, the palette) changes.
import { html, useEffect, useState } from '../lib.js';
import { deleteSelection } from '../api.js';
import { clearSelection, confirmProfileEdits, openTab, prefs, select, set, state, useStore } from '../store.js';
import { cls, levelStatus, playgroundFitMargins } from '../util.js';
import { Graph } from './Graph.js';
import { Inspector } from './Inspector.js';
import { JobView } from './JobView.js';
import { JobsView } from './Jobs.js';
import { AddBox, LevelBar } from './Library.js';
import { ProfilesView } from './Profiles.js';
import { ReferencesView } from './References.js';

// Small line icons (24px grid, stroke = currentColor).
const ICONS = {
  gear: 'M12 15a3 3 0 1 0 0-6 3 3 0 0 0 0 6Zm7.4-3a7.4 7.4 0 0 0-.1-1.2l2-1.6-2-3.4-2.4 1a7.5 7.5 0 0 0-2-1.2L14.5 3h-5l-.4 2.6a7.5 7.5 0 0 0-2 1.2l-2.4-1-2 3.4 2 1.6a7.4 7.4 0 0 0 0 2.4l-2 1.6 2 3.4 2.4-1a7.5 7.5 0 0 0 2 1.2l.4 2.6h5l.4-2.6a7.5 7.5 0 0 0 2-1.2l2.4 1 2-3.4-2-1.6c.1-.4.1-.8.1-1.2Z',
  plus: 'M12 5v14M5 12h14',
  arrange: 'M5 6h6M5 12h14M13 18h6M9 4v4M15 10v4M11 16v4',
  fit: 'M4 9V4h5M20 9V4h-5M4 15v5h5M20 15v5h-5',
  cursor: 'M5 3l14 7-6 2-2 6Z',
  link: 'M10 14a4 4 0 0 0 5.7 0l3-3a4 4 0 0 0-5.7-5.7l-1 1M14 10a4 4 0 0 0-5.7 0l-3 3a4 4 0 0 0 5.7 5.7l1-1',
  all: 'M4 4h6v6H4zM14 4h6v6h-6zM4 14h6v6H4zM14 14h6v6h-6z',
  trash: 'M4 7h16M9 7V4h6v3M6 7l1 13h10l1-13',
  help: 'M9.5 9a2.5 2.5 0 1 1 3.5 2.3c-.6.3-1 .9-1 1.6V14M12 17.5v.5M12 21a9 9 0 1 0 0-18 9 9 0 0 0 0 18Z',
  sparkle: 'M12 3l1.8 5.2L19 10l-5.2 1.8L12 17l-1.8-5.2L5 10l5.2-1.8Z',
  close: 'M6 6l12 12M18 6 6 18',
  chevron: 'M6 9l6 6 6-6',
};

export function Icon({ name, size = 16 }) {
  return html`<svg class="pg-icon" width=${size} height=${size} viewBox="0 0 24 24" fill="none" stroke="currentColor"
    stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d=${ICONS[name]} /></svg>`;
}

function Pill({ icon, label, onClick, active = false, title = '', className = '', disabled = false }) {
  return html`<button type="button" class=${cls('pg-pill', active && 'on', className)} onClick=${onClick}
    title=${title || label} aria-label=${label || title} disabled=${disabled}>
    ${icon && html`<${Icon} name=${icon} />`}${label && html`<span>${label}</span>`}</button>`;
}

// C6H12O -> C<sub>6</sub>H<sub>12</sub>O, as chemists write it.
function formula(text) {
  return String(text || '').split(/(\d+)/).map((part, i) => (i % 2 ? html`<sub>${part}</sub>` : part));
}

function TopLeft() {
  const connected = useStore((s) => s.connected);
  const root = useStore((s) => s.workspace.root);
  return html`
    <div class="pg-topleft">
      <button class="pg-status" onClick=${() => set({ modal: { kind: 'sessions' } })} title="Switch, open or start a session">
        <span class=${`conn ${connected ? 'on' : 'off'}`}></span>
        <span class="muted">Session:</span> <b>${root.split('/').pop()}</b>
      </button>
      <button class="pg-primary" onClick=${() => set({ modal: { kind: 'quick' } })}>
        <${Icon} name="sparkle" /> New calculation</button>
    </div>`;
}

function Tabs() {
  const view = useStore((s) => s.view);
  const active = useStore((s) => Object.values(s.jobs).filter((j) => j.status === 'running' || j.status === 'queued').length);
  const tabs = [['graph', 'Graph'], ['jobs', 'Calculations'], ['profiles', 'Profiles'], ['refs', 'References']];
  return html`
    <nav class="pg-tabs" aria-label="Pages">
      ${tabs.map(([k, l]) => html`<button class=${view.tab === k || (k === 'jobs' && view.tab === 'job') ? 'on' : ''}
        onClick=${() => openTab(k)}>${l}${k === 'jobs' && active > 0 && html` <span class="pg-count">${active}</span>`}</button>`)}
    </nav>`;
}

function TopRight() {
  const structures = useStore((s) => s.workspace.structures);
  const selected = useStore((s) => s.selection.structures);
  const levels = useStore((s) => s.levels);
  const levelProfile = useStore((s) => s.levelProfile);
  const [open, setOpen] = useState(() => ({ level: false, structures: !window.matchMedia('(max-width: 720px)').matches }));
  const [q, setQ] = useState('');
  const toggle = (k) => setOpen({ ...open, [k]: !open[k] });
  const all = Object.values(structures).sort((a, b) => b.created - a.created);
  const list = q ? all.filter((r) => `${r.name} ${r.smiles} ${r.formula}`.toLowerCase().includes(q.toLowerCase())) : all;
  const pick = (e, r) => {
    select({ structures: [r.id] }, e.shiftKey || e.metaKey || e.ctrlKey);
    window.dispatchEvent(new CustomEvent('mepd:graph', { detail: { cmd: 'center', id: r.id } }));
  };
  const level = levels[levelProfile ?? ''];
  return html`
    <div class="pg-topright">
      <button class="pg-section" onClick=${() => toggle('level')} aria-expanded=${open.level}>
        <span class=${cls('pg-caret', open.level && 'open')}><${Icon} name="chevron" size=${12} /></span> Level of theory</button>
      ${open.level ? html`<div class="pg-card"><${LevelBar} /></div>`
        : html`<div class="pg-chips"><span class="pg-chip">${level?.label ?? 'mepd defaults'} · ${levelProfile ?? 'defaults'}</span></div>`}
      <button class="pg-section" onClick=${() => toggle('structures')} aria-expanded=${open.structures}>
        <span class=${cls('pg-caret', open.structures && 'open')}><${Icon} name="chevron" size=${12} /></span> Structures
        <span class="muted">${all.length}</span></button>
      ${open.structures && all.length > 8 && html`<input class="pg-filter" type="search" placeholder="Filter structures"
        value=${q} onInput=${(e) => setQ(e.target.value)} />`}
      ${open.structures && html`<div class="pg-structs">
        ${list.map((r) => {
          const st = levelStatus(r);
          return html`<button key=${r.id} class=${cls('pg-struct', selected.includes(r.id) && 'on')}
            onClick=${(e) => pick(e, r)} title=${r.smiles || r.name}>
            <span class="pg-formula">${formula(r.formula)}</span><span class="pg-name">${r.name}</span>
            ${st.kind !== 'ok' && html`<span class=${`level-chip level-${st.kind}`}>${st.text}</span>`}
          </button>`;
        })}
        ${all.length === 0 && html`<p class="small muted">None yet. Add some from the dock below.</p>`}
        ${all.length > 0 && list.length === 0 && html`<p class="small muted">Nothing matches.</p>`}
      </div>`}
    </div>`;
}

function Dock() {
  const connectMode = useStore((s) => s.connectMode);
  const panel = useStore((s) => s.pgPanel);
  const nSel = useStore((s) => s.selection.structures.length + s.selection.edges.length);
  const counts = useStore((s) => {
    const js = Object.values(s.jobs);
    return { running: js.filter((j) => j.status === 'running').length, queued: js.filter((j) => j.status === 'queued').length };
  });
  const graph = (cmd) => { openTab('graph'); window.dispatchEvent(new CustomEvent('mepd:graph', { detail: cmd === 'fit' ? fitDetail() : cmd })); };
  const toggle = async (p) => {
    if (panel === p) { set({ pgPanel: null }); return; }
    if (state.view.tab !== 'graph') await openTab('graph');   // one sheet at a time
    set({ pgPanel: p });
  };
  const busy = counts.running + counts.queued;
  return html`
    <footer class="pg-dock">
      <div class="pg-dock-group">
        <${Pill} icon="gear" title="Settings" active=${panel === 'settings'} onClick=${() => toggle('settings')} className="round" />
        <${Pill} icon="plus" label="Add structures" active=${panel === 'add'} onClick=${() => toggle('add')} />
        <${Pill} icon="arrange" label="Arrange" onClick=${() => graph('arrange')} />
        <${Pill} icon="fit" label="Fit" onClick=${() => graph('fit')} />
      </div>
      <div class="pg-dock-group center">
        <button class="pg-pill" onClick=${() => openTab('jobs')} title="All calculations">
          <span class=${cls('pg-dot', counts.running > 0 && 'live')}></span>
          ${busy ? `${counts.running} running${counts.queued ? ` · ${counts.queued} queued` : ''}` : 'No calculations running'}
        </button>
      </div>
      <div class="pg-dock-group">
        <div class="pg-seg" role="group" aria-label="Graph mode">
          <button class=${!connectMode ? 'on' : ''} onClick=${() => set({ connectMode: false })} title="Select"><${Icon} name="cursor" /><span>Select</span></button>
          <button class=${connectMode ? 'on' : ''} onClick=${() => { openTab('graph'); set({ connectMode: true }); }}
            title="Connect: click a start, then an end structure (C)"><${Icon} name="link" /><span>Connect</span></button>
        </div>
        <${Pill} icon="all" title="Select every structure" onClick=${() => graph('select-all')} className="round" />
        <${Pill} icon="trash" title=${nSel ? `Delete ${nSel} selected` : 'Delete the selection'} onClick=${deleteSelection}
          disabled=${!nSel} className="round danger" />
        <${Pill} icon="help" title="How it works" active=${panel === 'help'} onClick=${() => toggle('help')} className="round" />
      </div>
    </footer>`;
}

// Where the graph fits: clear of the corner controls (and, on a wide
// screen, the structure list down the right).
function fitDetail() {
  return { cmd: 'fit', margins: playgroundFitMargins() };
}

function Sheet({ className = '', title, label, onClose, children }) {
  return html`
    <section class=${cls('pg-sheet', className)} role="dialog" aria-label=${title || label}>
      ${title ? html`<header class="pg-sheet-head"><h2>${title}</h2>
        <button class="btn-icon" onClick=${onClose} aria-label="Close"><${Icon} name="close" /></button></header>`
        : html`<button class="btn-icon pg-sheet-close" onClick=${onClose} aria-label="Close"><${Icon} name="close" /></button>`}
      <div class="pg-sheet-body">${children}</div>
    </section>`;
}

export async function setLayout(layout) {
  if (layout === state.layout) return;
  if (!(await confirmProfileEdits())) return;     // switching rebuilds the page
  prefs.set('layout', layout);
  set({ layout, pgPanel: null });
}

function Settings() {
  const layout = useStore((s) => s.layout);
  const bg = useStore((s) => s.pgBg);
  const seg = (value, options, onPick) => html`<div class="segmented">
    ${options.map(([k, l]) => html`<button class=${value === k ? 'on' : ''} onClick=${() => onPick(k)}>${l}</button>`)}</div>`;
  return html`
    <${Sheet} className="pg-settings" title="Settings" onClose=${() => set({ pgPanel: null })}>
      <div class="field"><span class="field-label">Layout</span>
        ${seg(layout, [['classic', 'Classic'], ['playground', 'Playground']], setLayout)}
        <span class="field-help">Classic: sidebars around the graph. Playground: the graph fills the screen, controls float.</span></div>
      <div class="field"><span class="field-label">Background</span>
        ${seg(bg, [['light', 'Light'], ['dark', 'Dark']], (v) => { prefs.set('pgBg', v); set({ pgBg: v }); })}</div>
    <//>`;
}

function Help() {
  return html`
    <${Sheet} className="pg-help" title="How it works" onClose=${() => set({ pgPanel: null })}>
      <ol class="steps">
        <li><b>Add structures</b><span>SMILES or XYZ, with "Add structures" in the dock. Each becomes a node.</span></li>
        <li><b>Select</b><span>One structure to explore around it; two, or an edge, to connect them. Shift-click for several.</span></li>
        <li><b>Run a calculation</b><span>The options for your selection open on the right. Results can be added back to the graph.</span></li>
      </ol>
    <//>`;
}

export function PlaygroundApp() {
  const view = useStore((s) => s.view);
  const panel = useStore((s) => s.pgPanel);
  const bg = useStore((s) => s.pgBg);
  const nSel = useStore((s) => s.selection.structures.length + s.selection.edges.length);
  const onGraph = view.tab === 'graph';
  // Saved positions come from whatever layout placed them: fit into the
  // free area once the graph has drawn.
  useEffect(() => {
    const t = setTimeout(() => window.dispatchEvent(new CustomEvent('mepd:graph', { detail: fitDetail() })), 700);
    return () => clearTimeout(t);
  }, []);
  const page = view.tab === 'jobs' ? html`<${JobsView} />`
    : view.tab === 'job' ? html`<${JobView} jobId=${view.jobId} />`
      : view.tab === 'profiles' ? html`<${ProfilesView} />`
        : view.tab === 'refs' ? html`<${ReferencesView} />` : null;
  return html`
    <div class=${`pg pg-${bg}`}>
      <div class="pg-canvas"><${Graph} /></div>
      <${TopLeft} />
      <${Tabs} />
      ${onGraph && html`<${TopRight} />`}
      ${onGraph && nSel > 0 && !panel && html`<div class="pg-sheet pg-inspector"><${Inspector} /></div>`}
      ${page && html`<${Sheet} className="pg-page" label=${{ profiles: 'Compute profiles', refs: 'References' }[view.tab] || 'Calculations'}
        onClose=${() => { clearSelection(); openTab('graph'); }}>${page}<//>`}
      ${panel === 'add' && html`<${Sheet} className="pg-add" title="Add structures" onClose=${() => set({ pgPanel: null })}>
        <${LevelBar} /><${AddBox} onDone=${() => set({ pgPanel: null })} /><//>`}
      ${panel === 'settings' && html`<${Settings} />`}
      ${panel === 'help' && html`<${Help} />`}
      <${Dock} />
    </div>`;
}

// For the classic layout's top bar.
export function layoutToggle() {
  return html`<button class="topbar-link small layout-switch" onClick=${() => setLayout(state.layout === 'playground' ? 'classic' : 'playground')}
    title="Try the playground layout: full-screen graph, floating controls">Playground layout</button>`;
}
