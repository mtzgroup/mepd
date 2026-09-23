// Right panel: what is selected, what we know about it, what can be run on it.
import { html, useEffect, useState } from '../lib.js';
import { api, attempt, deleteSelection } from '../api.js';
import { clearSelection, openJob, select, set, useStore } from '../store.js';
import { PHONE_QUERY, STATUS_LABEL, edgeStatus, fmtAgo, fmtKcal, lastLine, levelStatus } from '../util.js';
import { ActionPanel } from './Actions.js';
import { Viewer3D } from './Viewer3D.js';
import { LevelChip } from './Library.js';

function useXyz(sid) {
  const [xyz, setXyz] = useState(null);
  useEffect(() => {
    let live = true;
    setXyz(null);
    if (sid) api.get(`/api/structures/${sid}/xyz`).then((t) => live && setXyz(t)).catch(() => {});
    return () => { live = false; };
  }, [sid]);
  return xyz;
}

function JobList({ jobs }) {
  useStore((s) => jobs.filter((j) => j.status === 'running').map((j) => s.progress[j.id]?.last_line));
  if (!jobs.length) return null;
  return html`
    <section>
      <h3 class="section-title">Calculations here</h3>
      <ul class="mini-jobs">
        ${jobs.map((j) => html`
          <li onClick=${() => openJob(j.id)} title=${j.command}>
            <span class=${`pill ${j.status}`}>${STATUS_LABEL[j.status]}</span>
            <span class="mini-title">${j.title}</span>
            <span class="mini-sub">${j.summary?.headline || lastLine(j) || fmtAgo(j.created)}</span>
          </li>`)}
      </ul>
    </section>`;
}

function Editable({ value, onSave, className = '', placeholder = '' }) {
  const [v, setV] = useState(value);
  useEffect(() => setV(value), [value]);
  return html`<input class=${`editable ${className}`} value=${v} placeholder=${placeholder} onInput=${(e) => setV(e.target.value)}
    onBlur=${() => v !== value && onSave(v)} onKeyDown=${(e) => e.key === 'Enter' && e.target.blur()} />`;
}

function StructureDetail({ rec }) {
  const xyz = useXyz(rec.id);
  const [labels, setLabels] = useState(false);
  const jobs = useStore((s) => Object.values(s.jobs).filter((j) => j.targets.structures.includes(rec.id))
    .sort((a, b) => b.created - a.created));
  const originJob = useStore((s) => (rec.origin?.job ? s.jobs[rec.origin.job] : null));
  const patch = (body) => attempt(() => api.patch(`/api/structures/${rec.id}`, body));
  const del = async () => {
    if (!confirm(`Delete ${rec.name} and its edges? Job outputs are kept.`)) return;
    await attempt(() => api.del(`/api/structures/${rec.id}`), 'Deleted');
  };
  return html`
    <section>
      <${Editable} className="title-input" value=${rec.name} onSave=${(name) => patch({ name })} />
      <div class="smiles" title="Connectivity perceived from the geometry">${rec.smiles || '—'}</div>
      <${Viewer3D} xyz=${xyz} labels=${labels} height=${240} />
      <div class="viewer-tools">
        <label class="small"><input type="checkbox" checked=${labels} onChange=${(e) => setLabels(e.target.checked)} /> atom indices</label>
        <a class="small" href=${`/api/structures/${rec.id}/xyz`} download=${`${rec.name}.xyz`}>Download xyz</a>
      </div>
      <dl class="props">
        <dt>Formula</dt><dd>${rec.formula} (${rec.natoms} atoms)</dd>
        <dt>Charge</dt><dd><input type="number" class="tiny" value=${rec.charge} onChange=${(e) => patch({ charge: +e.target.value })} /></dd>
        <dt>Multiplicity</dt><dd><input type="number" class="tiny" min="1" value=${rec.multiplicity} onChange=${(e) => patch({ multiplicity: +e.target.value })} /></dd>
        <dt>Level</dt><dd><${LevelChip} rec=${rec} />
          ${rec.level && html` <span class="small muted">${rec.level.profile ?? 'mepd defaults'}</span>`}
          ${levelStatus(rec).kind !== 'ok' && levelStatus(rec).kind !== 'busy' && html`
            <button class="btn-link small" onClick=${() => attempt(() => api.post('/api/structures/reoptimize', { structures: [rec.id] }), 'Re-optimizing at the workspace level')}>re-optimize</button>`}
          ${rec.status_error && html`<div class="small warn-text">${rec.status_error}</div>`}
          ${rec.validation && rec.validation.is_minimum && html`<div class="small muted">Hessian: lowest frequency ${rec.validation.min_frequency?.toFixed(1)} cm⁻¹${rec.validation.rescued ? ' (after a rescue push off a saddle point)' : ''}</div>`}</dd>
        ${rec.energy != null && html`<dt>Energy</dt><dd class="mono">${rec.energy.toFixed(6)} Eh</dd>`}
        <dt>Source</dt><dd>${rec.origin?.kind === 'job'
          ? html`<a href="#" onClick=${(e) => { e.preventDefault(); openJob(rec.origin.job); }}>${rec.origin.label} · ${originJob?.title || 'job'}</a>`
          : rec.origin?.kind === 'smiles' ? html`SMILES <code>${rec.origin.input}</code>` : 'XYZ'}</dd>
      </dl>
      <button class="btn-link danger small" onClick=${del}>Delete structure</button>
    </section>
    <${JobList} jobs=${jobs} />`;
}

function PairDetail({ a, b }) {
  const existing = useStore((s) => Object.values(s.workspace.edges).find((e) =>
    (e.source === a.id && e.target === b.id) || (e.source === b.id && e.target === a.id)));
  const [which, setWhich] = useState(0);
  const xyz = useXyz(which === 0 ? a.id : b.id);
  const mismatch = a.natoms !== b.natoms;
  return html`
    <section>
      <div class="pair-head">
        <span class="pair-role">start</span><strong>${a.name}</strong>
        <button class="btn-icon" title="Swap start and end" onClick=${() => select({ structures: [b.id, a.id] })}>⇄</button>
        <span class="pair-role">end</span><strong>${b.name}</strong>
      </div>
      <div class="segmented wide">
        <button class=${which === 0 ? 'on' : ''} onClick=${() => setWhich(0)}>Start</button>
        <button class=${which === 1 ? 'on' : ''} onClick=${() => setWhich(1)}>End</button>
      </div>
      <${Viewer3D} xyz=${xyz} height=${220} />
      ${mismatch && html`<p class="warn-box">Different atom counts (${a.natoms} vs ${b.natoms}): these cannot be reaction endpoints.</p>`}
      ${!mismatch && (existing
        ? html`<button class="btn" onClick=${() => select({ edges: [existing.id] })}>Open existing edge</button>`
        : html`<button class="btn" onClick=${() => attempt(() => api.post('/api/edges', { source: a.id, target: b.id }))
            .then((e) => e && select({ edges: [e.id] }))}>Connect as edge</button>`)}
    </section>`;
}

function EdgeDetail({ edge }) {
  const structures = useStore((s) => s.workspace.structures);
  const allJobs = useStore((s) => s.jobs);
  const a = structures[edge.source], b = structures[edge.target];
  const [which, setWhich] = useState(0);
  const xyz = useXyz(which === 0 ? edge.source : edge.target);
  const st = edgeStatus(edge, allJobs);
  const jobs = Object.values(allJobs).filter((j) => j.targets.edges.includes(edge.id)).sort((x, y) => y.created - x.created);
  const patch = (body) => attempt(() => api.patch(`/api/edges/${edge.id}`, body));
  return html`
    <section>
      <div class="pair-head">
        <span class="pair-role">start</span><strong>${a?.name}</strong>
        <button class="btn-icon" title="Reverse direction" onClick=${() => patch({ reverse: true })}>⇄</button>
        <span class="pair-role">end</span><strong>${b?.name}</strong>
      </div>
      <${Editable} value=${edge.label || ''} onSave=${(label) => patch({ label })} className="label-input" placeholder="Add a label…" />
      <div class="stat-row">
        <div class="stat" title=${st.barrier == null && st.barrierUnverified != null ? 'No TS/IRC found so far connects these two structures; this is only the highest point of the path' : 'Lowest barrier whose IRCs connect these two structures'}>
          <span class="stat-v">${st.barrier != null ? fmtKcal(st.barrier) : st.barrierUnverified != null ? `≈${fmtKcal(st.barrierUnverified)}?` : '—'}</span>
          <span class="stat-l">${st.barrier == null && st.barrierUnverified != null ? 'path max, not IRC-verified' : 'best ΔE‡ (kcal/mol)'}</span></div>
        <div class="stat"><span class="stat-v">${st.count}</span><span class="stat-l">calculations</span></div>
      </div>
      ${edge.origin?.kind === 'job' && html`<p class="small muted">From <a href="#" onClick=${(e) => { e.preventDefault(); openJob(edge.origin.job); }}>a job result</a>${edge.origin.headline ? ` · ${edge.origin.headline}` : ''}</p>`}
      <div class="segmented wide">
        <button class=${which === 0 ? 'on' : ''} onClick=${() => setWhich(0)}>Start</button>
        <button class=${which === 1 ? 'on' : ''} onClick=${() => setWhich(1)}>End</button>
      </div>
      <${Viewer3D} xyz=${xyz} height=${200} />
      <button class="btn-link danger small" onClick=${() => attempt(() => api.del(`/api/edges/${edge.id}`), 'Edge removed')}>Remove edge</button>
    </section>
    <${JobList} jobs=${jobs} />`;
}

function usePhone() {
  const [phone, setPhone] = useState(() => window.matchMedia(PHONE_QUERY).matches);
  useEffect(() => {
    const mq = window.matchMedia(PHONE_QUERY);
    const on = () => setPhone(mq.matches);
    mq.addEventListener('change', on);
    return () => mq.removeEventListener('change', on);
  }, []);
  return phone;
}

export function Inspector() {
  const phone = usePhone();
  // On phones the sheet opens on what you came for -- the calculations -- with
  // the viewer/properties one tap away; desktop shows both stacked.
  const [pane, setPane] = useState('run');
  const sel = useStore((s) => s.selection);
  // Phones show this panel as a bottom sheet; tapping its header folds it
  // down to a bar (to look at the graph) and back up. Re-opens on every new
  // selection.
  const [collapsed, setCollapsed] = useState(false);
  useEffect(() => { setCollapsed(false); setPane('run'); }, [sel]);
  const structures = useStore((s) => s.workspace.structures);
  const edges = useStore((s) => s.workspace.edges);
  const nS = sel.structures.length, nE = sel.edges.length;

  let head, body = null;
  if (!nS && !nE) {
    return html`<aside class="inspector empty">
      <div class="pane-head"><h2>Selection</h2></div>
      <div class="empty-hint">
        <p><strong>Nothing selected.</strong></p>
        <ul class="hint-list">
          <li><b>One structure</b>: explore around it (Hessian sampling, basin hopping, TS optimization).</li>
          <li><b>Two structures or an edge</b>: find the TS / MEP or sample reaction channels.</li>
          <li><b>Several edges</b>: run the same calculation on each.</li>
          <li><b>Several structures</b>: all-pairs network, or batch exploration.</li>
        </ul>
        <p class="small muted">Shift/⌘-click to multi-select. Shift-drag on the graph to box-select. The first structure you pick is the start.</p>
        <button class="btn" onClick=${() => set({ modal: { kind: 'quick' } })}>Quick start…</button>
      </div>
    </aside>`;
  }
  if (nS === 1 && !nE) {
    head = 'Structure';
    body = html`<${StructureDetail} rec=${structures[sel.structures[0]]} />`;
  } else if (nS === 2 && !nE) {
    head = 'Two structures';
    body = html`<${PairDetail} a=${structures[sel.structures[0]]} b=${structures[sel.structures[1]]} />`;
  } else if (nE === 1 && !nS) {
    head = 'Edge';
    body = html`<${EdgeDetail} edge=${edges[sel.edges[0]]} />`;
  } else if (nS && nE) {
    head = 'Mixed selection';
    body = html`<p class="warn-box">Select either structures or edges to see what can be run.</p>`;
  } else {
    head = nE ? `${nE} edges` : `${nS} structures`;
    body = html`<ul class="sel-list">
      ${(nE ? sel.edges.map((id) => edges[id]).filter(Boolean).map((e) => `${structures[e.source]?.name} → ${structures[e.target]?.name}`)
        : sel.structures.map((id) => structures[id]?.name)).map((t) => html`<li>${t}</li>`)}
    </ul>
    <div class="sel-actions">
      ${nS > 0 && html`<a class="btn" href=${`/api/structures-export?ids=${sel.structures.join(',')}`} download="structures.xyz">Download XYZ (${nS})</a>`}
      <button class="btn danger-outline" onClick=${deleteSelection}>Delete ${nS ? `${nS} structures` : `${nE} edges`}</button>
    </div>`;
  }
  if (sel.structures.some((id) => !structures[id]) || sel.edges.some((id) => !edges[id])) return null;
  return html`
    <aside class=${`inspector has-selection ${collapsed ? 'collapsed' : ''}`}>
      <div class="pane-head" onClick=${() => setCollapsed(!collapsed)}>
        <h2>${head}${collapsed ? html` <span class="small muted">· tap to show options</span>` : ''}</h2>
        <button class="btn-icon" title="Clear selection (Esc)" onClick=${(e) => { e.stopPropagation(); clearSelection(); }}>✕</button>
      </div>
      ${phone && html`<div class="segmented wide sheet-tabs">
        <button class=${pane === 'run' ? 'on' : ''} onClick=${() => setPane('run')}>Calculations</button>
        <button class=${pane === 'details' ? 'on' : ''} onClick=${() => setPane('details')}>Details</button>
      </div>`}
      <div class="inspector-scroll">
        ${(!phone || pane === 'details') && body}
        ${(!phone || pane === 'run') && html`<${ActionPanel} sel=${sel} />`}
      </div>
    </aside>`;
}
