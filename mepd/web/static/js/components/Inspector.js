// Right panel: what is selected, what we know about it, what can be run on it.
import { html, useEffect, useState } from '../lib.js';
import { api, attempt, deleteSelection } from '../api.js';
import { clearSelection, openJob, openTab, prefs, select, set, useStore } from '../store.js';
import { PHONE_QUERY, STATUS_LABEL, conformerLabel, conformerRows, edgeStatus, fmtAgo, fmtKcal, lastLine, levelStatus, reactionOfEdge } from '../util.js';

export { conformerLabel, conformerRows };
import { ActionPanel } from './Actions.js';
import { Viewer3D } from './Viewer3D.js';
import { LevelChip } from './Library.js';
import { ReactionCard, TsCell } from './Reactions.js';
import { openAnalyze } from './Analyze.js';

function useXyz(sid, conformer = null, version = '') {
  const [xyz, setXyz] = useState(null);
  useEffect(() => {
    let live = true;
    setXyz(null);
    const q = conformer ? `?conformer=${encodeURIComponent(conformer)}` : '';
    if (sid) api.get(`/api/structures/${sid}/xyz${q}`).then((t) => live && setXyz(t)).catch(() => {});
    return () => { live = false; };
  }, [sid, conformer, version]);
  return xyz;
}

function Conformers({ rec, viewing, onView }) {
  const rows = conformerRows(rec);
  if (rows.length <= 1) return html`<p class="small muted">One conformer. Run <b>Conformers</b> (RDKit or CREST) to sample more.</p>`;
  const del = (cid) => attempt(() => api.del(`/api/structures/${rec.id}/conformers/${cid}`), 'Conformer removed');
  return html`
    <section class="conformers">
      <h3 class="section-title">Conformers (${rows.length})</h3>
      <p class="small muted">The lowest-energy one represents this molecule and is used unless an edge picks another. Click one to view it.</p>
      <ul class="conf-list">
        ${rows.map((r) => html`
          <li class=${`${viewing === r.id || (!viewing && r.rep) ? 'on' : ''}`} onClick=${() => onView(r.rep ? null : r.id)}
              title=${r.source}>
            <span class="conf-name">#${r.index + 1}${r.rep ? html` <span class="badge">lowest</span>` : ''}</span>
            <span class="conf-e mono">${r.dE != null ? `${r.dE >= 0 ? '+' : ''}${r.dE.toFixed(1)}` : '—'}</span>
            <span class="conf-src small muted">${r.source}${r.level && r.dE == null ? ` · ${r.level.label}` : ''}${r.validation?.is_minimum === false ? ' · not a minimum' : ''}</span>
            ${!r.rep && html`<button class="btn-icon small" title="Remove this conformer"
              onClick=${(e) => { e.stopPropagation(); del(r.id); }}>✕</button>`}
          </li>`)}
      </ul>
      <p class="small muted">ΔE in kcal/mol from the lowest, at ${rows.find((r) => r.rep)?.level?.label || 'the same level'}; — = no energy at that level.</p>
    </section>`;
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
  const [viewing, setViewing] = useState(null);   // a conformer other than the representative
  useEffect(() => setViewing(null), [rec.id]);
  const shown = viewing && (rec.conformers || []).some((c) => c.id === viewing) ? viewing : null;
  const xyz = useXyz(rec.id, shown, rec.conformer);
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
      ${rec.smiles && rec.smiles !== rec.name && html`<div class="smiles" title="Connectivity perceived from the geometry">${rec.smiles}</div>`}
      <${Viewer3D} xyz=${xyz} labels=${labels} height=${240} />
      <div class="viewer-tools">
        <label class="small"><input type="checkbox" checked=${labels} onChange=${(e) => setLabels(e.target.checked)} /> atom indices</label>
        <a class="small" href=${`/api/structures/${rec.id}/xyz${shown ? `?conformer=${shown}` : ''}`} download=${`${rec.name}.xyz`}>Download xyz</a>
        ${shown && html`<span class="small muted">viewing conformer #${(rec.conformers || []).findIndex((c) => c.id === shown) + 1}</span>`}
      </div>
      <${Conformers} rec=${rec} viewing=${shown} onView=${setViewing} />
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
      <div class="row-actions">
        <button class="btn small" title="Open this structure (the conformer shown) in the Design tab to edit it"
          onClick=${() => attempt(() => api.post('/api/design/load', { structure: rec.id, conformer: shown }), 'Loaded into Design').then((d) => d && openTab('design'))}>Edit in Design</button>
        <button class="btn-link danger small" onClick=${del}>Delete structure</button>
      </div>
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

// A nanoreactor reaction: its equation, what it took (shuttles), and the
// subsystem its calculations run on.
function ReactionHead({ r, structures }) {
  const name = (id) => structures[id]?.name || '?';
  const side = (ids) => {
    const c = {};
    ids.forEach((id) => { c[id] = (c[id] || 0) + 1; });
    return Object.entries(c).map(([id, n]) => `${n > 1 ? `${n} ` : ''}${name(id)}`).join(' + ');
  };
  const shuttles = [...new Set(r.shuttles || [])].map(name);
  return html`<div class="rxn-head">
    <div class="rxn-eq">${side(r.reactants)} → ${side(r.products)}</div>
    <div class="small muted">
      ${r.delta_e_kcal != null && html`<span title="Sum over the products minus the reactants, each optimized alone">ΔE ${r.delta_e_kcal >= 0 ? '+' : ''}${fmtKcal(r.delta_e_kcal)} kcal/mol · </span>`}
      seen ${r.count || 0}×${r.reverse_count ? `, reverse ${r.reverse_count}×` : ''}
      ${shuttles.length ? html` · <span title="On both sides: takes part (e.g. relays a proton) and comes out unchanged">needs ${shuttles.join(', ')}</span>` : ''}
    </div>
    <div class="rx-ts"><${TsCell} r=${r} /></div>
  </div>`;
}

function EdgeDetail({ edge }) {
  const structures = useStore((s) => s.workspace.structures);
  const reaction = useStore((s) => reactionOfEdge(s.workspace, edge.id));
  const allJobs = useStore((s) => s.jobs);
  const a = structures[edge.source], b = structures[edge.target];
  const [which, setWhich] = useState(0);
  const picks = edge.conformers || {};
  const shownId = which === 0 ? edge.source : edge.target;
  const xyz = useXyz(shownId, picks[shownId] || null, structures[shownId]?.conformer);
  const st = edgeStatus(edge, allJobs);
  const jobs = Object.values(allJobs).filter((j) => j.targets.edges.includes(edge.id)).sort((x, y) => y.created - x.created);
  const patch = (body) => attempt(() => api.patch(`/api/edges/${edge.id}`, body));
  return html`
    <section>
      ${reaction ? html`<${ReactionHead} r=${reaction} structures=${structures} />` : html`<div class="pair-head">
        <span class="pair-role">start</span><strong>${a?.name}</strong>
        <button class="btn-icon" title="Reverse direction" onClick=${() => patch({ reverse: true })}>⇄</button>
        <span class="pair-role">end</span><strong>${b?.name}</strong>
      </div>
      <${Editable} value=${edge.label || ''} onSave=${(label) => patch({ label })} className="label-input" placeholder="Add a label…" />`}
      <div class="stat-row">
        <div class=${`stat ${st.barrierJob ? 'linked' : ''}`} onClick=${() => st.barrierJob && openJob(st.barrierJob)}
          title=${(st.barrier == null && st.barrierUnverified != null ? 'No TS/IRC found so far connects these two structures; this is only the highest point of the path' : 'Lowest barrier whose IRCs connect these two structures')
            + (st.barrierJob ? '. Click for its TS, IRC and follow-ups' : '')}>
          <span class="stat-v">${st.barrier != null ? fmtKcal(st.barrier) : st.barrierUnverified != null ? `≈${fmtKcal(st.barrierUnverified)}?` : '—'}</span>
          <span class="stat-l">${st.barrier == null && st.barrierUnverified != null ? 'path max, not IRC-verified' : 'best ΔE‡ (kcal/mol)'}</span></div>
        <div class="stat"><span class="stat-v">${st.count}</span><span class="stat-l">calculations</span></div>
      </div>
      ${[a, b].some((r) => (r?.conformers || []).length > 1) && html`<div class="conf-picks">
        ${[['start', a], ['end', b]].map(([role, r]) => r && html`<label class="field">
          <span class="field-label">${role} conformer · ${r.name}</span>
          <select value=${picks[r.id] || ''} disabled=${(r.conformers || []).length <= 1}
            onChange=${(e) => patch({ conformers: { [r.id]: e.target.value || null } })}>
            <option value="">Lowest energy (default)</option>
            ${conformerRows(r).map((c) => html`<option value=${c.id}>${conformerLabel(c)}</option>`)}
          </select></label>`)}
        <p class="small muted">Calculations on this edge start from these conformers.</p>
      </div>`}
      ${st.warning && html`<p class="warn-box small">⚠ ${st.warning}</p>`}
      ${edge.origin?.kind === 'job' && html`<p class="small muted">From <a href="#" onClick=${(e) => { e.preventDefault(); openJob(edge.origin.job); }}>a job result</a>${edge.origin.headline ? ` · ${edge.origin.headline}` : ''}</p>`}
      <div class="segmented wide">
        <button class=${which === 0 ? 'on' : ''} onClick=${() => setWhich(0)}>${reaction ? 'Reactants' : 'Start'}</button>
        <button class=${which === 1 ? 'on' : ''} onClick=${() => setWhich(1)}>${reaction ? 'Products' : 'End'}</button>
      </div>
      <${Viewer3D} xyz=${xyz} height=${200} />
      ${reaction && html`<p class="small muted">Only the molecules this reaction needs, optimized together; calculations here run on them.</p>`}
      <button class="btn-link danger small" onClick=${() => attempt(() => api.del(`/api/edges/${edge.id}`), reaction ? 'Reaction removed' : 'Edge removed')}>${reaction ? 'Remove reaction' : 'Remove edge'}</button>
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
  const workspace = useStore((s) => s.workspace);
  const nS = sel.structures.length, nE = sel.edges.length;

  let head, body = null, moreCalcs = false;
  if (!nS && !nE) {
    return html`<aside class="inspector empty">
      <div class="pane-head"><h2>How it works</h2>
        <button class="btn-icon" title="Hide this panel (bring it back with ? on the graph)" aria-label="Hide"
          onClick=${() => { prefs.set('hideHowTo', true); set({ howToHidden: true }); }}>✕</button></div>
      <div class="empty-hint">
        <ol class="steps">
          <li><b>Add structures</b><span>SMILES or XYZ, in the panel on the left. Each becomes a node.</span></li>
          <li><b>Select</b><span>One structure to explore around it; two, or an edge, to connect them.</span></li>
          <li><b>Run a calculation</b><span>The options for your selection appear here. Results can be added back to the graph.</span></li>
        </ol>
        ${Object.keys(workspace.reactions || {}).length > 0 && html`<p class="small rx-hint"><b>● Dots are reactions</b>${' '}
          (from the nanoreactor): click one to see its species, energies, reaction complex and MD event, and to find its TS.</p>`}
                <p class="small muted">Shift-click to select several. Shift-drag on the graph to box-select.</p>
      </div>
    </aside>`;
  }
  if (nS === 1 && !nE) {
    head = 'Structure';
    body = html`<${StructureDetail} rec=${structures[sel.structures[0]]} />`;
  } else if (nS === 2 && !nE) {
    head = 'Two structures';
    body = html`<${PairDetail} a=${structures[sel.structures[0]]} b=${structures[sel.structures[1]]} />`;
  } else if (nE === 1 && !nS && reactionOfEdge(workspace, sel.edges[0])) {
    head = 'Reaction';
    const r = reactionOfEdge(workspace, sel.edges[0]);
    body = html`<${ReactionCard} key=${r.id} r=${r} compact />
      <button class="btn-link danger small" onClick=${() => attempt(() => api.del(`/api/edges/${r.edge}`), 'Reaction removed')}>Remove reaction</button>`;
    moreCalcs = true;
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
  // Species selected: their reactions, or a new one with them, in Analyze.
  if (nS > 0 && !nE && sel.structures.every((id) => !['ts', 'complex'].includes(structures[id]?.role))) {
    const n = Object.values(workspace.reactions || {}).filter((r) => sel.structures.every((id) => [...r.reactants, ...r.products].includes(id))).length;
    body = html`<div class="an-links small">
      ${n > 0 && html`<a href="#" onClick=${(e) => { e.preventDefault(); openAnalyze({ species: [...sel.structures], reaction: null }); }}>${n} reaction${n > 1 ? 's' : ''} of ${nS > 1 ? 'these' : 'this'} →</a>`}
      <a href="#" onClick=${(e) => { e.preventDefault(); openAnalyze({ compose: true, composeReactants: [...sel.structures], reaction: null }); }}>New reaction with ${nS > 1 ? 'these' : 'this'} →</a>
    </div>${body}`;
  }
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
        ${(!phone || pane === 'run') && (moreCalcs
          ? html`<details class="more-calcs"><summary>More calculations on this reaction's molecules</summary><${ActionPanel} sel=${sel} /></details>`
          : html`<${ActionPanel} sel=${sel} />`)}
      </div>
    </aside>`;
}
