import { html, render, useEffect } from './lib.js';
import { connect, deleteSelection } from './api.js';
import { clearSelection, openTab, set, state, useStore } from './store.js';
import { Graph } from './components/Graph.js';
import { Inspector } from './components/Inspector.js';
import { JobView } from './components/JobView.js';
import { JobsView } from './components/Jobs.js';
import { Library } from './components/Library.js';
import { Modals } from './components/Modals.js';
import { ProfilesView } from './components/Profiles.js';
import { ReferencesView } from './components/References.js';
import { PlaygroundApp, layoutToggle } from './components/Playground.js';

function TopBar() {
  const view = useStore((s) => s.view);
  const connected = useStore((s) => s.connected);
  const root = useStore((s) => s.workspace.root);
  const counts = useStore((s) => {
    const js = Object.values(s.jobs);
    return { running: js.filter((j) => j.status === 'running').length, queued: js.filter((j) => j.status === 'queued').length };
  });
  const lastJob = useStore((s) => s.view.jobId);
  const tabs = [['graph', 'Graph'], ['jobs', 'Calculations'], ['profiles', 'Profiles'], ['refs', 'References']];
  const demo = useStore((s) => s.demo);
  const auth = useStore((s) => s.auth);
  const active = counts.running + counts.queued;
  return html`
    <header class="topbar">
      <div class="brand">
        <svg class="logo" viewBox="0 0 32 32" aria-hidden="true"><polygon points="16,3 27,9.5 27,22.5 16,29 5,22.5 5,9.5" fill="none" stroke="currentColor" stroke-width="2.6"/><circle cx="16" cy="16" r="3.2" fill="currentColor"/></svg>
        mepd
      </div>
      <nav class="main-tabs">
        ${tabs.map(([k, l]) => html`<button class=${view.tab === k || (k === 'jobs' && view.tab === 'job') ? 'on' : ''}
          onClick=${() => (k === 'jobs' && view.tab === 'jobs' && lastJob ? set({ view: { tab: 'job', jobId: lastJob } }) : openTab(k))}>
          ${l}${k === 'jobs' && active > 0 && html` <span class="tab-count" title=${`${counts.running} running, ${counts.queued} queued`}>${active}</span>`}
        </button>`)}
      </nav>
      <span class="spacer"></span>
      <button class="session-name" title=${`Session folder: ${root}\nSwitch, open or start a new session`}
        onClick=${() => set({ modal: { kind: 'sessions' } })}>
        <span class=${`conn ${connected ? 'on' : 'off'}`} title=${connected ? 'Connected' : 'Reconnecting…'}></span>
        <span class="session-label">${root.split('/').pop()}</span><span class="caret">▾</span>
      </button>
      ${demo && html`<span class="badge demo-badge" title="Public demo: private workspace, limited sizes and run times">Demo</span>`}
      ${layoutToggle()}
      ${auth && html`<a class="topbar-link small" href="/logout" title="Log this device out">Log out</a>`}
      <button class="btn primary" onClick=${() => set({ modal: { kind: 'quick' } })}>New calculation</button>
    </header>`;
}

function Toasts() {
  const toasts = useStore((s) => s.toasts);
  return html`<div class="toasts" aria-live="polite">
    ${toasts.map((t) => html`<div class=${`toast ${t.kind}`} onClick=${() => set({ toasts: state.toasts.filter((x) => x.id !== t.id) })}>
      ${t.message}${t.action && html` <button class="toast-action" onClick=${(e) => { e.stopPropagation(); t.action.run(); set({ toasts: state.toasts.filter((x) => x.id !== t.id) }); }}>${t.action.label}</button>`}</div>`)}
  </div>`;
}

function App() {
  const view = useStore((s) => s.view);
  const loaded = useStore((s) => s.loaded);
  const howToHidden = useStore((s) => s.howToHidden);
  const layout = useStore((s) => s.layout);
  const pgBg = useStore((s) => s.pgBg);
  useStore((s) => s.selection);
  // The layout's palette lives on <html>, so the graph (drawn from CSS
  // variables) and everything else pick it up.
  useEffect(() => {
    const root = document.documentElement;
    root.dataset.layout = layout;
    root.dataset.pgbg = pgBg;
    // The graph draws with the CSS colours: redraw it in the new palette.
    requestAnimationFrame(() => window.dispatchEvent(new CustomEvent('mepd:graph', { detail: 'restyle' })));
  }, [layout, pgBg]);

  useEffect(() => {
    const onKey = (e) => {
      const tag = e.target.tagName;
      if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || state.modal) return;
      if (e.key === 'Escape') {
        // Playground: Escape also closes a page (Calculations, Profiles, a job) back to the graph.
        if (state.layout === 'playground' && !state.pgPanel && state.view.tab !== 'graph') { openTab('graph'); return; }
        set({ connectMode: false, pgPanel: null });
        clearSelection();
      }
      if (e.key === 'c' && state.view.tab === 'graph') set({ connectMode: !state.connectMode });
      if ((e.key === 'Delete' || e.key === 'Backspace') && state.view.tab === 'graph') { e.preventDefault(); deleteSelection(); }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);

  if (!loaded) return html`<div class="boot">Connecting to mepd…</div>`;
  if (layout === 'playground') {
    return html`<${PlaygroundApp} /><${Modals} /><${Toasts} />`;
  }
  const demo = state.demo;
  const nothingSelected = !state.selection.structures.length && !state.selection.edges.length;
  const showInspector = (view.tab === 'graph' || view.tab === 'jobs') && !(nothingSelected && howToHidden);
  return html`
    <${TopBar} />
    ${demo && html`<div class="demo-banner small">${`Demo: this is your private workspace. Limits: ${demo.max_atoms} atoms per structure, ${demo.max_active_jobs} calculations at a time, ${Math.round(demo.job_timeout_s / 60)} min per calculation${demo.op_timeout_s?.vri ? ` (VRI searches ${Math.round(demo.op_timeout_s.vri / 60)} min)` : ''}. Compute profiles are fixed.`}</div>`}
    <main class=${`layout tab-${view.tab} ${showInspector ? '' : 'no-inspector'}`}>
      <${Library} />
      <section class="center">
        <div class="center-view" hidden=${view.tab !== 'graph'}><${Graph} /></div>
        ${view.tab === 'jobs' && html`<${JobsView} />`}
        ${view.tab === 'job' && html`<${JobView} jobId=${view.jobId} />`}
        ${view.tab === 'profiles' && html`<${ProfilesView} />`}
        ${view.tab === 'refs' && html`<${ReferencesView} />`}
      </section>
      ${showInspector && html`<${Inspector} />`}
    </main>
    <${Modals} />
    <${Toasts} />`;
}

connect();
render(html`<${App} />`, document.getElementById('app'));
