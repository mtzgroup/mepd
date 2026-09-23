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

function TopBar() {
  const view = useStore((s) => s.view);
  const connected = useStore((s) => s.connected);
  const root = useStore((s) => s.workspace.root);
  const counts = useStore((s) => {
    const js = Object.values(s.jobs);
    return { running: js.filter((j) => j.status === 'running').length, queued: js.filter((j) => j.status === 'queued').length };
  });
  const lastJob = useStore((s) => s.view.jobId);
  const tabs = [['graph', 'Graph'], ['jobs', 'Calculations'], ['profiles', 'Profiles']];
  return html`
    <header class="topbar">
      <div class="brand"><span class="logo">⌬</span> mepd</div>
      <nav class="main-tabs">
        ${tabs.map(([k, l]) => html`<button class=${view.tab === k || (k === 'jobs' && view.tab === 'job') ? 'on' : ''}
          onClick=${() => (k === 'jobs' && view.tab === 'jobs' && lastJob ? set({ view: { tab: 'job', jobId: lastJob } }) : openTab(k))}>
          ${l}${k === 'jobs' && counts.running + counts.queued > 0 ? html` <span class="badge accent">${counts.running}${counts.queued ? `+${counts.queued}` : ''}</span>` : ''}
        </button>`)}
      </nav>
      <span class="spacer"></span>
      <div class="session-buttons">
        <button class="btn ghost session-name" title=${`Session workspace: ${root}\nClick to switch sessions`}
          onClick=${() => set({ modal: { kind: 'sessions' } })}>
          <span class="muted small">Session</span> <strong>${root.split('/').pop()}</strong> ▾
        </button>
        <button class="btn" title="Start over in a new, empty session (the current one is kept)"
          onClick=${() => set({ modal: { kind: 'sessions', startWith: 'new' } })}>New session</button>
        <button class="btn" title="Open an existing session" onClick=${() => set({ modal: { kind: 'sessions' } })}>Open…</button>
      </div>
      ${useStore((s) => s.demo) && html`<span class="badge accent demo-badge" title="Public demo: private workspace, limited sizes and run times">demo</span>`}
      <span class=${`conn ${connected ? 'on' : 'off'}`} title=${connected ? 'Live' : 'Reconnecting…'}></span>
      ${useStore((s) => s.auth) && html`<a class="btn ghost small" href="/logout" title="Log this device out">Log out</a>`}
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

  useEffect(() => {
    const onKey = (e) => {
      const tag = e.target.tagName;
      if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || state.modal) return;
      if (e.key === 'Escape') { set({ connectMode: false }); clearSelection(); }
      if (e.key === 'c' && state.view.tab === 'graph') set({ connectMode: !state.connectMode });
      if ((e.key === 'Delete' || e.key === 'Backspace') && state.view.tab === 'graph') { e.preventDefault(); deleteSelection(); }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);

  if (!loaded) return html`<div class="boot">Connecting to mepd…</div>`;
  const demo = state.demo;
  const showInspector = view.tab === 'graph' || view.tab === 'jobs';
  return html`
    <${TopBar} />
    ${demo && html`<div class="demo-banner small">${`Demo: this is your private workspace. Limits: ${demo.max_atoms} atoms per structure, ${demo.max_active_jobs} calculations at a time, ${Math.round(demo.job_timeout_s / 60)} min per calculation. Compute profiles are fixed.`}</div>`}
    <main class=${`layout tab-${view.tab} ${showInspector ? '' : 'no-inspector'}`}>
      <${Library} />
      <section class="center">
        <div class="center-view" hidden=${view.tab !== 'graph'}><${Graph} /></div>
        ${view.tab === 'jobs' && html`<${JobsView} />`}
        ${view.tab === 'job' && html`<${JobView} jobId=${view.jobId} />`}
        ${view.tab === 'profiles' && html`<${ProfilesView} />`}
      </section>
      ${showInspector && html`<${Inspector} />`}
    </main>
    <${Modals} />
    <${Toasts} />`;
}

connect();
render(html`<${App} />`, document.getElementById('app'));
