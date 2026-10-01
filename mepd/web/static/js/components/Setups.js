// Experimental setups on the Explore graph (a prototype; mepd/web/setups.py).
// A setup = solvent (or gas phase) + temperature + reaction time + the
// structures the experiment starts from. With one active, every edge shows
// its barrier under it (in that solvent when computed, else the gas-phase
// value, marked "gas") and is coloured by whether it runs within the
// reaction time; "Analyze network" gives its whole-network properties
// (mepd.network_model) and "Compare" how they change between two setups.
import { html, useMemo, useState } from '../lib.js';
import { api, attempt } from '../api.js';
import { select, state, toast, useStore } from '../store.js';
import { edgeStatus } from '../util.js';
import { NumberInput } from './ParamForm.js';

const KB_H = 2.083661912e10;   // kB/h, 1/(s K)
const R_KCAL = 1.98720425864083e-3;
const TIME_UNITS = [['min', 60], ['h', 3600], ['days', 86400]];

export function activeSetup(ws) {
  const id = ws?.active_setup;
  return id ? (ws.setups || []).find((s) => s.id === id) || null : null;
}

// Everything edge labels under a setup depend on (solvent jobs' summaries
// arrive after their 'done' status, so they are part of it).
export function setupKey(s) {
  const setup = activeSetup(s.workspace);
  if (!setup) return '';
  const solv = Object.values(s.jobs).filter((j) => j.op === 'solvent')
    .map((j) => `${j.id}:${j.status}:${j.summary?.conditions ? 1 : 0}`).sort().join(',');
  return JSON.stringify(setup) + solv;
}

export function halfLife(barrier, T) {
  const k = KB_H * T * Math.exp(-Math.max(barrier, 0) / (R_KCAL * T));
  return Math.LN2 / k;
}

export function fmtTime(s) {
  if (s == null || !isFinite(s)) return '∞';
  if (s > 3.15e7 * 1e4) return `10^${Math.round(Math.log10(s / 3.15e7))} y`;
  for (const [u, size] of [['y', 3.15e7], ['d', 86400], ['h', 3600], ['min', 60], ['s', 1]]) {
    if (s >= size) { const v = s / size; return `${v >= 10 ? v.toFixed(0) : v.toFixed(1)} ${u}`; }
  }
  return s >= 1e-3 ? `${(s * 1e3).toFixed(0)} ms` : '<1 ms';
}

function solventBarrier(edge, jobs, solvent) {
  const originJob = edge.origin?.job;
  let best = null;
  for (const j of Object.values(jobs)) {
    const cond = j.summary?.conditions;
    if (j.op !== 'solvent' || j.status !== 'done' || !cond) continue;
    if (!(j.targets.edges.includes(edge.id) || (originJob && j.source_job === originJob))) continue;
    const row = cond.solvents?.[solvent];
    if (row?.barrier_kcal == null) continue;
    const rank = [cond.mode === 'reoptimize' ? 1 : 0, j.finished || 0];
    if (!best || rank[0] > best.rank[0] || (rank[0] === best.rank[0] && rank[1] > best.rank[1])) {
      best = { rank, barrier: row.barrier_kcal, mode: cond.mode, job: j.id };
    }
  }
  return best;
}

// {barrier, phase: 'solvent'|'gas'|null, tHalf, runs: 'yes'|'slow'|'no'|'none'}
export function edgeUnderSetup(edge, jobs, setup) {
  const hit = setup.solvent ? solventBarrier(edge, jobs, setup.solvent) : null;
  const gas = edgeStatus(edge, jobs).barrier;
  const barrier = hit ? hit.barrier : gas;
  const phase = hit ? 'solvent' : gas != null ? 'gas' : null;
  if (barrier == null) return { barrier: null, phase, runs: 'none' };
  const tHalf = halfLife(barrier, setup.temperature);
  const runs = tHalf <= setup.time_s ? 'yes' : tHalf <= 100 * setup.time_s ? 'slow' : 'no';
  return { barrier, phase, tHalf, runs, mode: hit?.mode };
}

export function setupEdgeLabel(e, info, setup) {
  if (info.barrier == null) return setup.solvent ? '—' : '';
  const mark = setup.solvent && info.phase === 'gas' ? ' (gas)' : info.mode === 'single-point' ? '*' : '';
  return `${info.barrier.toFixed(1)}${mark} · t½ ${fmtTime(info.tHalf)}`;
}

function solventLabel(solvents, key) {
  return key ? (solvents.find((s) => s.key === key)?.label ?? key) : 'Gas phase';
}

function blankSetup() {
  return { id: null, name: '', solvent: null, temperature: 298.15, time_s: 3600, start: [], level: null, variant: null };
}

async function saveSetups(setups, active) {
  return attempt(() => api.put('/api/setups', { setups, active }));
}

function TimeField({ seconds, onChange }) {
  const unit = [...TIME_UNITS].reverse().find(([, f]) => seconds >= f && Number.isInteger(+(seconds / f).toFixed(6))) || TIME_UNITS[1];
  return html`<span class="time-field">
    <input type="number" class="tiny" min="0" step="any" value=${+(seconds / unit[1]).toFixed(3)}
      onInput=${(e) => { const v = parseFloat(e.target.value); if (v > 0) onChange(v * unit[1]); }} />
    <select value=${unit[0]} onChange=${(e) => onChange((seconds / unit[1]) * TIME_UNITS.find(([u]) => u === e.target.value)[1])}>
      ${TIME_UNITS.map(([u]) => html`<option value=${u}>${u}</option>`)}
    </select></span>`;
}

export function SetupPanel({ onClose }) {
  const ws = useStore((s) => s.workspace);
  const solvents = useStore((s) => s.solvents) || [];
  const selection = useStore((s) => s.selection);
  const setups = ws.setups || [];
  const active = activeSetup(ws);
  const [editId, setEditId] = useState(active?.id ?? setups[0]?.id ?? null);
  const current = setups.find((s) => s.id === editId);
  const [draft, setDraft] = useState(() => current ? { ...current } : blankSetup());
  const [prediction, setPrediction] = useState(null);
  const [comparison, setComparison] = useState(null);
  const [otherId, setOtherId] = useState('');
  const profiles = useStore((s) => s.profiles) || [];
  const groupLabels = useStore((s) => s.operations.find((o) => o.key === 'substituents')?.schema?.properties?.groups?.labels) || {};
  const [busy, setBusy] = useState(false);
  const dirty = !current || JSON.stringify(current) !== JSON.stringify(draft);
  const pick = (id) => {
    setEditId(id);
    const s = setups.find((x) => x.id === id);
    setDraft(s ? { ...s } : blankSetup());
    setPrediction(null);
    setComparison(null);
  };
  const up = (patch) => { setDraft({ ...draft, ...patch }); setPrediction(null); setComparison(null); };
  const apply = async () => {
    const made = draft.id ? draft : { ...draft, id: `s_${Math.random().toString(16).slice(2, 10)}` };
    const next = draft.id ? setups.map((x) => (x.id === draft.id ? made : x)) : [...setups, made];
    const out = await saveSetups(next, made.id);
    if (!out) return null;
    const saved = out.setups.find((x) => x.id === made.id);
    setEditId(saved.id);
    setDraft({ ...saved });
    return saved;
  };
  const hide = () => saveSetups(setups, null);
  const remove = async () => {
    if (!current) return;
    await saveSetups(setups.filter((s) => s.id !== current.id), ws.active_setup === current.id ? null : ws.active_setup);
    pick(null);
  };
  const runPredict = async () => {
    setBusy(true);
    const made = dirty ? await apply() : draft;
    if (made) {
      const out = await attempt(() => api.post(`/api/setups/${made.id}/analyze`));
      if (out) { setPrediction(out); setComparison(null); }
    }
    setBusy(false);
  };
  const runCompare = async () => {
    setBusy(true);
    const made = dirty ? await apply() : draft;
    if (made && otherId) {
      const out = await attempt(() => api.post('/api/setups/compare', { base: made.id, other: otherId }));
      if (out) { setComparison(out); setPrediction(null); }
    }
    setBusy(false);
  };
  const fill = async (mode) => {
    const made = dirty ? await apply() : draft;
    if (!made) return;
    const out = await attempt(() => api.post(`/api/setups/${made.id}/fill`, { mode }));
    if (out) toast(out.queued.length ? `Queued solvent effects on ${out.queued.length} reaction(s)` : 'Every edge with a TS already has this solvent', 'ok');
  };
  const edges = Object.values(ws.edges || {});
  const missing = draft.solvent ? edges.filter((e) => edgeStatus(e, state.jobs).barrier != null
    && !solventBarrier(e, state.jobs, draft.solvent)).length : 0;
  const names = (ids) => ids.map((id) => ws.structures[id]?.name || id);
  const tC = Math.round((draft.temperature - 273.15) * 10) / 10;
  return html`<div class="graph-view-panel setup-panel" onClick=${(e) => e.stopPropagation()}>
    <div class="gv-head"><b>Conditions</b>
      <span class="small muted">${active ? `showing: ${active.name}` : 'not shown on the graph'}</span>
      <button class="btn-icon" onClick=${onClose} title="Close">✕</button></div>
    <div class="gv-row">
      <select value=${editId ?? ''} onChange=${(e) => pick(e.target.value || null)}>
        ${setups.map((s) => html`<option value=${s.id}>${s.name}</option>`)}
        <option value="">＋ New setup</option>
      </select>
    </div>
    <label class="gv-row small"><span class="gv-label">Name</span>
      <input type="text" placeholder="automatic" value=${draft.name} onInput=${(e) => up({ name: e.target.value })} /></label>
    <label class="gv-row small"><span class="gv-label">Medium</span>
      <select value=${draft.solvent ?? ''} onChange=${(e) => up({ solvent: e.target.value || null })}>
        <option value="">Gas phase</option>
        ${solvents.map((s) => html`<option value=${s.key}>${s.label} (${s.kind}, ε ${s.epsilon})</option>`)}
      </select></label>
    <label class="gv-row small"><span class="gv-label">Temperature</span>
      <${NumberInput} className="tiny" value=${tC} onChange=${(v) => { if (v != null) up({ temperature: v + 273.15 }); }} /> °C</label>
    <div class="gv-row small"><span class="gv-label">Reaction time</span>
      <${TimeField} seconds=${draft.time_s} onChange=${(v) => up({ time_s: v })} /></div>
    <label class="gv-row small"><span class="gv-label">Level</span>
      <select value=${draft.level ?? ''} onChange=${(e) => up({ level: e.target.value || null })}
        title="Only barriers computed with this profile (compare levels of theory)">
        <option value="">any</option>
        ${profiles.map((p) => html`<option value=${p}>${p}</option>`)}
      </select></label>
    <div class="gv-row small"><span class="gv-label">Variant</span>
      <select value=${draft.variant?.group ?? ''} onChange=${(e) => up({ variant: e.target.value ? { site: draft.variant?.site ?? 0, group: e.target.value } : null })}
        title="A substituent on every step, from each edge's 'Substituent effects' result">
        <option value="">none</option>
        ${Object.entries(groupLabels).map(([k, l]) => html`<option value=${k}>${l.replace(/\s*\(.*\)$/, '')}</option>`)}
      </select>
      ${draft.variant && html`<span class="small">on atom</span>
        <input type="number" class="tiny" min="0" step="1" value=${draft.variant.site}
          onInput=${(e) => { const v = parseInt(e.target.value, 10); if (v >= 0) up({ variant: { ...draft.variant, site: v } }); }} />`}
    </div>
    <div class="gv-row small"><span class="gv-label">Starts from</span>
      <span class="setup-start">${draft.start.length ? names(draft.start).join(', ') : html`<span class="muted">nothing yet</span>`}</span>
      <button class="btn small ghost" disabled=${!selection.structures.length}
        title="Use the structures selected on the graph"
        onClick=${() => up({ start: [...selection.structures].filter((id) => ws.structures[id]?.role !== 'ts') })}>Use selection</button></div>
    ${draft.solvent && missing > 0 && html`<p class="small warn-box">${missing} reaction${missing > 1 ? 's have' : ' has'} a gas-phase
      barrier but none in ${solventLabel(solvents, draft.solvent)} yet (shown as "gas").
      <button class="btn small" onClick=${() => fill('single-point')} title="Solvent single points on each gas-phase path: seconds each">Compute (fast)</button>
      <button class="btn small ghost" onClick=${() => fill('reoptimize')} title="Search each TS again in the solvent: minutes each">Re-optimize</button></p>`}
    <div class="gv-row">
      <button class="btn small primary" onClick=${apply}>${active?.id === draft.id && !dirty ? 'Shown on graph' : 'Show on graph'}</button>
      <button class="btn small" disabled=${busy || !draft.start.length} onClick=${runPredict}
        title=${draft.start.length ? 'Run the graph forward in time from the starting structures' : 'Choose the starting structures first'}>
        ${busy ? 'Working…' : 'Analyze network'}</button>
      <span class="spacer"></span>
      ${active && html`<button class="btn small ghost" onClick=${hide}>Hide</button>`}
      ${current && html`<button class="btn small ghost danger-text" onClick=${remove}>Delete</button>`}
    </div>
    ${setups.length > 1 && html`<div class="gv-row small"><span class="gv-label">Compare with</span>
      <select value=${otherId} onChange=${(e) => setOtherId(e.target.value)}>
        <option value="">—</option>
        ${setups.filter((x) => x.id !== draft.id).map((x) => html`<option value=${x.id}>${x.name}</option>`)}
      </select>
      <button class="btn small" disabled=${busy || !otherId || !draft.start.length} onClick=${runCompare}>Compare</button></div>`}
    ${prediction && html`<${Analysis} a=${prediction} />`}
    ${comparison && html`<${Comparison} c=${comparison} />`}
    <p class="small muted">Edge labels: barrier ΔE‡ (kcal/mol) and half-life at this temperature; green runs within the
      reaction time, amber within 100×, grey not. * = single points on gas-phase geometries. Rates are Eyring's from
      electronic barriers (no entropy), so treat them as orders of magnitude.</p>
  </div>`;
}

function pct(x) { return x == null ? '—' : `${(x * 100).toFixed(x >= 0.1 || x === 0 ? 0 : 1)}%`; }

function stepName(names, steps, id) {
  const st = steps.find((x) => x.id === id);
  return st ? `${names[st.a] ?? st.a} → ${names[st.b] ?? st.b}` : id;
}

function Controls({ label, x, names, steps }) {
  const items = [...(x.ts || []).map((r) => ({ ...r, what: `TS ${stepName(names, steps, r.id)}` })),
    ...(x.species || []).map((r) => ({ ...r, what: names[r.id] ?? r.id }))]
    .sort((a, b) => Math.abs(b.x) - Math.abs(a.x)).slice(0, 4);
  if (!items.length) return null;
  return html`<div class="small ctrl"><span class="muted">${label}:</span>
    ${items.map((r) => html`<span class="ctrl-item" title="Degree of control: d ln P / d(−G/RT). +1: lowering it by RT raises P e-fold.">
      ${r.what} <b class=${r.x > 0 ? 'good' : 'bad'}>${r.x > 0 ? '+' : ''}${r.x.toFixed(2)}</b></span>`)}</div>`;
}

// Whole-network properties under one setup (mepd.network_model).
function Analysis({ a }) {
  const p = a.properties;
  const names = a.names;
  const ids = Object.keys(p.final).sort((x, y) => Math.max(p.final[y], p.equilibrium[y]) - Math.max(p.final[x], p.equilibrium[x])).slice(0, 6);
  const phases = {};
  for (const st of a.steps) {
    const k = st.phase === 'gas' ? 'gas' : `${st.phase}${st.mode === 'single-point' ? '*' : ''}`;
    phases[k] = (phases[k] || 0) + 1;
  }
  const levels = [...new Set(a.steps.map((st) => st.level).filter(Boolean))];
  const slow = p.timescales_s?.[0];
  return html`<div class="prediction">
    <div class="small"><b>After ${fmtTime(a.setup.time_s)} at ${Math.round(a.setup.temperature - 273.15)} °C</b>
      <span class="muted"> · ${p.control} control · conversion ${pct(p.conversion)}${slow ? ` · equilibrates in ~${fmtTime(slow)}` : ''}</span></div>
    <table class="data compact"><thead><tr><th>Species</th><th class="num">now</th><th class="num">equilibrium</th></tr></thead>
      <tbody>${ids.map((id) => html`<tr onClick=${() => select({ structures: [id] })} class="clickable">
        <td>${names[id] ?? id}</td><td class="num">${pct(p.final[id])}</td><td class="num muted">${pct(p.equilibrium[id])}</td></tr>`)}</tbody></table>
    <table class="data compact"><thead><tr><th>Product</th><th class="num" title="When it first reaches half its peak amount">forms in</th>
      <th class="num" title="Highest TS on the best route from the start">ΔE‡ eff</th><th>bottleneck</th></tr></thead>
      <tbody>${p.products.map((id) => {
        const b = p.bottleneck[id];
        return html`<tr><td>${names[id] ?? id}</td><td class="num">${fmtTime(p.formation_time_s[id])}</td>
          <td class="num">${b ? b.effective_barrier.toFixed(1) : '—'}</td>
          <td class="small">${b?.bottleneck_step ? stepName(names, a.steps, b.bottleneck_step) : '—'}</td></tr>`;
      })}</tbody></table>
    ${Object.entries(a.sensitivity).map(([id, s]) => html`
      <${Controls} label=${`${names[id] ?? id} yield`} x=${s.yield} names=${names} steps=${a.steps} />
      <${Controls} label=${`${names[id] ?? id} rate`} x=${s.rate} names=${names} steps=${a.steps} />`)}
    <p class="small muted">Data: ${a.steps.length} step${a.steps.length === 1 ? '' : 's'} (${Object.entries(phases).map(([k, n]) => `${n} ${k}`).join(', ')})${levels.length ? ` · ${levels.join(', ')}` : ''}. * single points.</p>
    ${a.warnings.map((w) => html`<p class="small warn-box">⚠ ${w}</p>`)}
  </div>`;
}

// Two setups of one network: how each property changes, and which steps'
// energy changes explain it (first order, from the base's sensitivities).
function Comparison({ c }) {
  const names = c.names;
  const rows = Object.entries(c.quantities);
  const label = (k) => { const [kind, id] = k.split(':'); return `${names[id] ?? id} ${kind === 'rate' ? 'rate' : 'yield'}`; };
  const fmtQ = (k, v) => (k.startsWith('rate') ? (v ? `1/${fmtTime(1 / v)}` : '—') : pct(v));
  return html`<div class="prediction">
    <div class="small"><b>${c.base.name} → ${c.other.name}</b></div>
    <table class="data compact"><thead><tr><th></th><th class="num">base</th><th class="num">other</th><th class="num">×</th><th>from</th></tr></thead>
      <tbody>${rows.map(([k, q]) => {
        const lin = q.contributions.reduce((s, x) => s + Math.abs(x.dlnP), 0) || 1;
        const top = q.contributions.slice(0, 2).map((x) => `${x.kind === 'ts' ? 'TS ' + stepName(names, c.steps, x.id) : names[x.id] ?? x.id} ${Math.round((100 * Math.abs(x.dlnP)) / lin)}%`);
        if (q.dlnP == null) {
          return html`<tr><td>${label(k)}</td><td class="num">${fmtQ(k, q.base)}</td><td class="num">${fmtQ(k, q.other)}</td>
            <td class="num muted">—</td><td class="small muted">${q.note}</td></tr>`;
        }
        const f = Math.exp(q.dlnP);
        return html`<tr><td>${label(k)}</td><td class="num">${fmtQ(k, q.base)}</td><td class="num">${fmtQ(k, q.other)}</td>
          <td class=${`num ${f > 1.5 ? 'good' : f < 0.67 ? 'bad' : ''}`}>${f >= 100 || f <= 0.01 ? `10^${Math.log10(f).toFixed(0)}` : f.toFixed(2)}</td>
          <td class="small" title="First-order split of the change over the energies that moved">${q.linear_ok === false
            ? html`<span class="muted">nonlinear (split explains ${Math.round(100 * (q.explained ?? 0))}%)</span>` : top.join(', ') || '—'}</td></tr>`;
      })}</tbody></table>
    ${c.temperature_changed && html`<p class="small muted">Temperature differs too: its effect is in the numbers, not in the split.</p>`}
    ${c.warnings.map((w) => html`<p class="small warn-box">⚠ ${w}</p>`)}
  </div>`;
}
