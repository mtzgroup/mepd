// Quick start ("I just want a TS between these two") and opening an
// existing mepd output folder.
import { html, useEffect, useState } from '../lib.js';
import { api, attempt } from '../api.js';
import { openJob, prefs, set, state, useStore } from '../store.js';
import { fmtAgo, readFileText } from '../util.js';
import { clampToSchema, defaultsFor } from './ParamForm.js';
import { defaultProfile } from './Actions.js';

function Modal({ title, children, onClose }) {
  useEffect(() => {
    const onKey = (e) => e.key === 'Escape' && onClose();
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);
  return html`
    <div class="modal-backdrop" onMouseDown=${(e) => e.target === e.currentTarget && onClose()}>
      <div class="modal" role="dialog" aria-label=${title}>
        <div class="modal-head"><h2>${title}</h2><button class="btn-icon" onClick=${onClose}>✕</button></div>
        ${children}
      </div>
    </div>`;
}

function EndpointInput({ label, value, onChange }) {
  const [drag, setDrag] = useState(false);
  return html`
    <label class=${`endpoint ${drag ? 'drag' : ''}`}
      onDragOver=${(e) => { e.preventDefault(); setDrag(true); }} onDragLeave=${() => setDrag(false)}
      onDrop=${async (e) => { e.preventDefault(); setDrag(false); const f = e.dataTransfer.files[0]; if (f) onChange(await readFileText(f)); }}>
      <span class="field-label">${label}</span>
      <textarea rows="3" spellcheck="false" value=${value} placeholder="SMILES, pasted XYZ, or drop an .xyz file"
        onInput=${(e) => onChange(e.target.value)} />
    </label>`;
}

const QUICK_MODES = [
  { key: 'ts', title: 'Transition state', desc: 'MEP + TS optimization + IRC between two structures', pair: true },
  { key: 'channels', title: 'Reaction channels', desc: 'Sample conformers and mappings, find every distinct channel', pair: true },
  { key: 'hessian-sample', title: 'Explore nearby minima', desc: 'Hessian normal-mode sampling around one structure', pair: false },
];

function QuickStart({ onClose }) {
  const operations = useStore((s) => s.operations);
  const [mode, setMode] = useState('ts');
  const [a, setA] = useState('');
  const [b, setB] = useState('');
  const [charge, setCharge] = useState('');
  const [mult, setMult] = useState('');
  const [busy, setBusy] = useState(false);
  const [phase, setPhase] = useState('');
  const m = QUICK_MODES.find((x) => x.key === mode);
  const op = operations.find((o) => o.key === mode);
  const ready = a.trim() && (!m.pair || b.trim());

  const go = async () => {
    setBusy(true);
    const opts = { charge: charge === '' ? null : +charge, multiplicity: mult === '' ? null : +mult };
    const out = await attempt(async () => {
      const [sa] = await api.post('/api/structures', { text: a, ...opts, optimize: true });
      const ids = [sa.id];
      if (m.pair) { const [sb] = await api.post('/api/structures', { text: b, ...opts, optimize: true }); ids.push(sb.id); }
      // Every calculation starts from minima at the workspace level: wait for
      // the minimizations, and stop with an explanation if one of them turned
      // the input into something else.
      setPhase('Minimizing at the workspace level of theory…');
      const recs = [];
      for (const id of ids) {
        const t0 = Date.now();
        while (state.workspace.structures[id]?.status === 'optimizing' || !(id in state.workspace.structures)) {
          if (Date.now() - t0 > 30 * 60 * 1000) throw new Error('minimization is taking too long; run the calculation from the graph once it is done');
          await new Promise((r) => setTimeout(r, 500));
        }
        const rec = state.workspace.structures[id];
        if (rec.status === 'opt_failed') throw new Error(`minimizing ${rec.name} failed: ${rec.status_error || 'see its job'}`);
        recs.push(rec);
      }
      const reacted = recs.filter((r) => r.reacted);
      if (reacted.length) {
        const r = reacted[0];
        throw new Error(`${r.reacted.from} is not a stable minimum at this level of theory: it relaxed without a barrier to ${r.reacted.to}. `
          + 'Both structures are in the graph; nothing was run. Try a different level of theory, or start from a pre-reactive complex.');
      }
      if (m.pair && recs[0].smiles && recs[0].smiles === recs[1].smiles) {
        throw new Error(`After minimization both ends are ${recs[0].smiles}, so there is no reaction between them to search. Both structures are in the graph; nothing was run.`);
      }
      const params = clampToSchema(op.schema, { ...defaultsFor(op.schema), ...prefs.get(`params:${op.key}`, {}) });
      const profile = defaultProfile();
      return api.post('/api/jobs', { op: op.key, structures: ids, params, profile });
    });
    setBusy(false);
    setPhase('');
    if (out?.length) { onClose(); openJob(out[0].id); }
  };
  return html`
    <${Modal} title="Quick start" onClose=${onClose}>
      <div class="quick-modes">
        ${QUICK_MODES.map((x) => html`<button class=${`quick-mode ${mode === x.key ? 'on' : ''}`} onClick=${() => setMode(x.key)}>
          <strong>${x.title}</strong><span class="small muted">${x.desc}</span></button>`)}
      </div>
      <div class=${m.pair ? 'endpoint-pair' : ''}>
        <${EndpointInput} label=${m.pair ? 'Start (reactant)' : 'Structure'} value=${a} onChange=${setA} />
        ${m.pair && html`<${EndpointInput} label="End (product)" value=${b} onChange=${setB} />`}
      </div>
      <div class="add-row">
        <input type="number" class="tiny" placeholder="charge" value=${charge} onInput=${(e) => setCharge(e.target.value)} />
        <input type="number" class="tiny" placeholder="mult." min="1" value=${mult} onInput=${(e) => setMult(e.target.value)} />
        <span class="small muted">Structures are minimized at the workspace level of theory; the calculation then runs at that same level, with your last settings.</span>
      </div>
      <div class="modal-foot">
        <button class="btn" onClick=${onClose}>Cancel</button>
        <button class="btn primary" disabled=${!ready || busy || !op} onClick=${go}>${busy ? (phase || 'Setting up…') : `Run ${m.title.toLowerCase()}`}</button>
      </div>
    <//>`;
}

function ImportResult({ onClose }) {
  const operations = useStore((s) => s.operations);
  const [path, setPath] = useState('');
  const [op, setOp] = useState('');
  const [charge, setCharge] = useState(0);
  const [mult, setMult] = useState(1);
  const go = async () => {
    const job = await attempt(() => api.post('/api/jobs/import', { path, op: op || null, charge: +charge, multiplicity: +mult }),
      'Opened existing output');
    if (job) { onClose(); openJob(job.id); }
  };
  return html`
    <${Modal} title="Open an existing mepd output" onClose=${onClose}>
      <p class="small muted">Point at a directory written by <code>mepd run</code>, <code>channels</code>, <code>ts</code>, <code>network-splits</code> or <code>discovery</code> on this machine. It is read in place, never modified.</p>
      <div class="field"><label class="field-label">Directory</label>
        <input type="text" class="mono" value=${path} placeholder="/path/to/mepd_channels_output" onInput=${(e) => setPath(e.target.value)} /></div>
      <div class="add-row">
        <select value=${op} onChange=${(e) => setOp(e.target.value)}>
          <option value="">Detect command automatically</option>
          ${operations.filter((o) => o.available).map((o) => html`<option value=${o.key}>${o.title}</option>`)}
        </select>
        <input type="number" class="tiny" value=${charge} onInput=${(e) => setCharge(e.target.value)} title="Charge" />
        <input type="number" class="tiny" min="1" value=${mult} onInput=${(e) => setMult(e.target.value)} title="Multiplicity" />
      </div>
      <div class="modal-foot">
        <button class="btn" onClick=${onClose}>Cancel</button>
        <button class="btn primary" disabled=${!path.trim()} onClick=${go}>Open</button>
      </div>
    <//>`;
}

function Sessions({ onClose, startWith = 'open' }) {
  const demo = useStore((s) => s.demo);
  const [data, setData] = useState(null);
  const [name, setName] = useState('');
  const [path, setPath] = useState('');
  const [busy, setBusy] = useState(false);
  useEffect(() => { api.get('/api/sessions').then(setData).catch(() => setData({ sessions: [] })); }, []);

  const run = async (fn, ok) => {
    setBusy(true);
    const out = await attempt(fn, ok);
    setBusy(false);
    if (out) onClose();
  };
  const create = () => run(() => api.post('/api/sessions/new', !demo && name.includes('/') ? { path: name } : { name }),
    `Started ${name.split('/').pop()}`);
  const open = (p) => run(() => api.post('/api/sessions/open', { path: p }), `Opened ${p.split('/').pop()}`);

  return html`
    <${Modal} title="Sessions" onClose=${onClose}>
      <p class="small muted">A session is a workspace directory: its structures, graph, profiles and calculations.
        Switching leaves the other session's running jobs running; they are there when you come back.</p>
      <section class="session-new">
        <h3 class="section-title">Start a new session</h3>
        <div class="add-row">
          <input type="text" value=${name} autofocus=${startWith === 'new'} placeholder=${demo ? 'session name' : 'name, or a full path to a new directory'}
            onInput=${(e) => setName(e.target.value)} onKeyDown=${(e) => e.key === 'Enter' && name.trim() && create()} />
          <button class="btn primary" disabled=${busy || !name.trim()} onClick=${create}>Start</button>
        </div>
        ${data && !demo && html`<span class="small muted">A bare name is created in <code>${data.root}</code>. Your profiles are copied over.</span>`}
        ${demo && html`<span class="small muted">Up to ${demo.max_sessions} private sessions.</span>`}
      </section>
      <section>
        <h3 class="section-title">Recent sessions</h3>
        ${!data ? html`<p class="muted small">Loading…</p>`
          : data.sessions.length === 0 ? html`<p class="muted small">None yet.</p>`
          : html`<ul class="session-list">
              ${data.sessions.map((s) => html`
                <li class=${s.current ? 'current' : ''}>
                  <div class="session-main">
                    <strong>${s.name}</strong>
                    ${s.current && html`<span class="badge accent">current</span>`}
                    ${s.active_jobs > 0 && html`<span class="pill running">${s.active_jobs} running</span>`}
                    <div class="small muted mono session-path">${s.path}</div>
                    <div class="small muted">${s.structures} structures · ${s.edges} edges · ${s.jobs} calculations · opened ${fmtAgo(s.opened)}</div>
                  </div>
                  ${!s.current && html`<button class="btn" disabled=${busy} onClick=${() => open(s.path)}>Open</button>`}
                </li>`)}
            </ul>`}
      </section>
      ${!demo && html`<section>
        <h3 class="section-title">Open a workspace directory</h3>
        <div class="add-row">
          <input type="text" class="mono" value=${path} placeholder="/path/to/existing/workspace"
            onInput=${(e) => setPath(e.target.value)} onKeyDown=${(e) => e.key === 'Enter' && path.trim() && open(path.trim())} />
          <button class="btn" disabled=${busy || !path.trim()} onClick=${() => open(path.trim())}>Open</button>
        </div>
      </section>`}
    <//>`;
}

export function Modals() {
  const modal = useStore((s) => s.modal);
  const close = () => set({ modal: null });
  if (!modal) return null;
  if (modal.kind === 'quick') return html`<${QuickStart} onClose=${close} />`;
  if (modal.kind === 'import') return html`<${ImportResult} onClose=${close} />`;
  if (modal.kind === 'sessions') return html`<${Sessions} onClose=${close} startWith=${modal.startWith} />`;
  return null;
}
