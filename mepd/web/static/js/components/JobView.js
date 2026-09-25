// One job: live progress while it runs, an explorable result when it has
// output (partial results work too), the raw log, and its files.
import { Component, html, useEffect, useMemo, useRef, useState } from '../lib.js';
import { api, attempt, refreshState } from '../api.js';
import { openJob, prefs, select, set, state, toast, update, useStore } from '../store.js';
import { STATUS_LABEL, copy, downloadText, entryToXyz, fmtDuration, fmtKcal, jobElapsed, safeName, structureName, tsFrameIndex } from '../util.js';
import { EnergyPlot } from './EnergyPlot.js';
import { ParamForm, clampToSchema, defaultsFor } from './ParamForm.js';
import { JobControls, useTick } from './Jobs.js';
import { NetworkLive } from './NetworkLive.js';
import { Viewer3D } from './Viewer3D.js';

// ------------------------------------------------------------ live
// Every live path of a job: one per stream (e.g. each `channels` pair) and,
// where a stream runs parallel MSMEP branches, one per branch.
function livePaths(progress, now) {
  const streams = { ...(progress?.streams || {}) };
  if (!Object.keys(streams).length && progress?.chain) streams.main = progress.chain;  // older jobs
  const out = [];
  for (const [name, st] of Object.entries(streams)) {
    const age = st.updated ? now - st.updated : Infinity;
    const state = st.finished ? (st.status === 'failed' ? 'failed' : 'done') : age < 30 ? 'running' : 'idle';
    if (st.kind === 'morph') {
      // One proposed reaction of a network expansion (queued until its
      // product guess starts optimizing).
      out.push({ id: name, kind: 'morph', label: st.label || name, plot: st.plot || { x: [], y: [] },
        geometry: st.geometry, caption: st.caption, outcome: st.outcome, updated: st.updated,
        state: st.finished ? state : st.status === 'queued' ? 'queued' : 'running' });
      continue;
    }
    if (st.kind === 'minimization') {
      // One geometry minimization (a Hessian-sampling candidate).
      out.push({ id: name, kind: 'minimization', label: st.label || name, plot: st.plot || { x: [], y: [] },
        geometry: st.geometry, caption: st.caption, state, outcome: st.outcome, reference: st.reference, updated: st.updated });
      continue;
    }
    const branches = Object.entries(st.monitors || {}).filter(([, m]) => m.geometry?.frames?.length && m.plot?.y?.length > 1);
    if (branches.length > 1) {
      for (const [mid, m] of branches) {
        out.push({ id: `${name}/${mid}`, label: name === 'main' ? mid : `${name} · ${mid}`, plot: m.plot, geometry: m.geometry,
          caption: m.caption, state: st.finished ? state : m.active ? 'running' : 'idle', updated: st.updated });
      }
    } else if (st.plot?.y?.length > 1) {
      out.push({ id: name, label: name, plot: st.plot, geometry: st.geometry, caption: st.plot.caption || st.caption,
        state, updated: st.updated });
    }
  }
  const rank = { running: 0, idle: 1, queued: 2, done: 3, failed: 4 };
  return out.sort((a, b) => rank[a.state] - rank[b.state] || a.label.localeCompare(b.label, undefined, { numeric: true }));
}

function lastEnergy(p) {
  const y = p.plot?.y || [];
  for (let i = y.length - 1; i >= 0; i -= 1) if (y[i] != null) return y[i];
  return null;
}

function fmtSigned(v) {
  return `${v >= 0 ? '+' : '−'}${Math.abs(v).toFixed(1)}`;
}

// One minimization, live: the newest optimizer step while it runs; once it
// is done, its whole trajectory to scrub or replay.
function MinimizationLive({ job, fg }) {
  const [full, setFull] = useState(null);     // finished: the replay, fetched on demand
  const [pick, setPick] = useState(null);     // chosen frame (null = newest)
  const [playing, setPlaying] = useState(false);
  const truncated = fg.geometry?.truncated;
  useEffect(() => { setFull(null); setPick(null); setPlaying(false); }, [fg.id]);
  useEffect(() => {
    if (!truncated) return undefined;
    let live = true;
    api.get(`/api/jobs/${job.id}/live/${encodeURIComponent(fg.id)}`).then((d) => live && setFull(d)).catch(() => {});
    return () => { live = false; };
  }, [fg.id, truncated]);
  const geometry = (truncated && full?.geometry) || fg.geometry || {};
  const frames = geometry.frames || [];
  const steps = geometry.frame_steps || [];
  const shown = pick == null ? frames.length - 1 : Math.min(pick, frames.length - 1);
  useEffect(() => {
    if (!playing || frames.length < 2) return undefined;
    const t = setInterval(() => setPick((i) => {
      const next = (i == null || i >= frames.length - 1) ? 0 : i + 1;
      if (next === frames.length - 1) setPlaying(false);
      return next;
    }), 120);
    return () => clearInterval(t);
  }, [playing, frames.length]);
  const y = fg.plot.y || [];
  const step = steps[shown] ?? (y.length ? y.length - 1 : 0);
  const e = y[step];
  const running = fg.state === 'running';
  const toStep = (k) => {                      // nearest stored frame to optimizer step k
    if (!steps.length) return;
    let best = 0;
    steps.forEach((st, i) => { if (Math.abs(st - k) < Math.abs(steps[best] - k)) best = i; });
    setPlaying(false);
    setPick(best);
  };
  return html`
    <div class="card-block live-path">
      <div class="live-head">
        <h4><span class="badge accent">${fg.label}</span>
          ${running ? ' Minimizing' : ` ${fg.outcome || fg.state}`}
          <span class="muted small"> · step ${y.length ? step + 1 : 0}${y.length ? ` of ${y.length}` : ''}${e != null ? ` · ${fmtSigned(e)} kcal/mol vs ${fg.reference === 'seed' ? 'the seed' : 'the start'}` : ''}</span>
          <span class=${`live-state ${fg.state}`}>${running ? 'minimizing' : fg.state}</span></h4>
        ${!running && frames.length > 1 && html`<button class="btn small" onClick=${() => { if (!playing && shown >= frames.length - 1) setPick(0); setPlaying(!playing); }}>
          ${playing ? 'Pause' : 'Replay'}</button>`}
      </div>
      ${frames.length
        ? html`<${Viewer3D} frames=${frames} frame=${Math.max(0, shown)} height=${280} />`
        : html`<p class="small muted">The geometry appears with the first optimizer step.</p>`}
      ${!running && frames.length > 1 && html`<div class="frame-bar">
        <input type="range" min="0" max=${frames.length - 1} value=${Math.max(0, shown)}
          onInput=${(ev) => { setPlaying(false); setPick(+ev.target.value); }} />
      </div>`}
      <${EnergyPlot} xs=${fg.plot.x} ys=${y} height=${150} current=${y.length ? step : null}
        onPick=${!running && steps.length > 1 ? toStep : null} />
      <p class="small muted">${fg.caption || ''}${running ? ' · the newest step is shown as it arrives' : ''}</p>
    </div>`;
}

function pathBarrier(p) {
  const y = p.plot?.y?.filter((v) => v != null) || [];
  return y.length > 2 ? Math.max(...y.slice(1, -1)) : null;
}

function LivePanel({ job }) {
  const progress = useStore((s) => s.progress[job.id]);
  // The foreground path is remembered per job and only changes when you
  // click another one -- new data never steals the view.
  const selected = useStore((s) => s.liveSelection?.[job.id] ?? null);
  const choose = (id) => update((s) => { s.liveSelection = { ...(s.liveSelection || {}), [job.id]: id }; });
  const [follow, setFollow] = useState(true);   // keep the viewer on the current TS guess
  const [frame, setFrame] = useState(null);     // null = the newest image
  useTick(5000, job.status === 'running');       // refresh running/idle badges
  useEffect(() => {
    // Seed with a full snapshot (SSE only sends deltas).
    api.get(`/api/jobs/${job.id}`).then((d) => {
      if (d.progress) set({ progress: { ...state.progress, [job.id]: { ...(state.progress[job.id] || {}), ...d.progress } } });
    }).catch(() => {});
  }, [job.id]);

  const allPaths = livePaths(progress, Date.now() / 1000);
  // A network expansion's proposed reactions have their own view; its path
  // searches (--connect) are listed below it as usual.
  const reactions = allPaths.filter((p) => p.kind === 'morph');
  const paths = allPaths.filter((p) => p.kind !== 'morph');
  const minimizing = paths.length > 0 && paths.every((p) => p.kind === 'minimization');
  // First time anything shows up, pin the first path; afterwards stay put.
  useEffect(() => { if (!selected && paths.length) choose(paths[0].id); }, [selected, paths.length]);
  // Sampling runs follow whichever candidate is minimizing now, until you
  // click one (then it stays in front).
  const [autoFollow, setAutoFollow] = useState(true);
  const runningNow = minimizing && autoFollow ? paths.find((p) => p.state === 'running') : null;
  const fg = runningNow || paths.find((p) => p.id === selected) || null;
  const stats = progress?.stats;

  const frames = useMemo(() => fg?.geometry?.frames, [fg?.geometry]);
  const tsIndex = fg?.geometry?.ts_index ?? null;
  const last = (frames?.length || 1) - 1;
  const shown = follow && tsIndex != null ? tsIndex : frame == null ? last : Math.min(frame, last);
  const energy = fg?.plot?.y?.[shown];

  return html`
    <div class="live">
      <div class="last-line mono">${progress?.last_line || job.last_line || (job.status === 'queued' ? 'Waiting for a free slot…' : '')}</div>
      ${fg && fg.kind === 'minimization' && html`<${MinimizationLive} job=${job} fg=${fg} />`}
      ${(reactions.length > 0 || job.op === 'graph-enumeration') && html`<${NetworkLive} job=${job} reactions=${reactions} />`}
      ${fg && fg.kind !== 'minimization' && html`
        <div class="card-block live-path">
          <div class="live-head">
            <h4><span class="badge accent">${fg.label}</span>
              ${follow && tsIndex != null ? ' Current TS guess' : ` Image ${shown + 1}`}
              ${frames && html`<span class="muted small"> · image ${shown + 1}/${frames.length}${energy != null ? ` · ${energy.toFixed(2)} kcal/mol` : ''}</span>`}
              <span class=${`live-state ${fg.state}`}>${fg.state}</span></h4>
            <label class="small"><input type="checkbox" checked=${follow} onChange=${(e) => setFollow(e.target.checked)} /> follow TS guess</label>
          </div>
          ${frames
            ? html`<${Viewer3D} frames=${frames} frame=${shown} height=${280} />`
            : html`<p class="small muted">Geometry appears with the next optimization step (restart jobs started before this feature to see it).</p>`}
          <${EnergyPlot} xs=${fg.plot.x} ys=${fg.plot.y} height=${160} current=${frames ? shown : null} tsIndex=${tsIndex}
            onPick=${frames ? (i) => { setFollow(false); setFrame(i); } : null} />
          <p class="small muted">${fg.caption || ''}${fg.plot.reactant_smiles ? html` · <span class="mono">${fg.plot.reactant_smiles} → ${fg.plot.product_smiles}</span>` : ''}</p>
        </div>`}
      ${paths.length > 1 && html`
        ${minimizing && html`<label class="small check follow-toggle"><input type="checkbox" class="switch" checked=${autoFollow}
          onChange=${(e) => setAutoFollow(e.target.checked)} /> Follow the candidate being minimized</label>`}
        <div class="live-list-head small muted">${minimizing
          ? `${paths.length} candidates · ${paths.filter((p) => p.state === 'running').length} minimizing · ${paths.filter((p) => p.outcome === 'new minimum').length} new minima · click one to watch it`
          : `${paths.length} paths · ${paths.filter((p) => p.state === 'running').length} running · click one to bring it to the front`}</div>
        <div class="monitors">
          ${paths.map((p) => html`<button type="button" key=${p.id}
              class=${`monitor ${p.state === 'running' ? 'active' : ''} ${p.id === fg?.id ? 'pinned' : ''}`}
              onClick=${() => { setAutoFollow(false); choose(p.id); }} title=${p.caption || p.label}>
            <div class="small monitor-head"><b>${p.label}</b>
              ${p.kind === 'minimization' && p.outcome
                ? html`<span class=${`live-state outcome-${p.outcome.replace(/\s+/g, '-')}`}>${p.outcome}</span>`
                : html`<span class=${`live-state ${p.state === 'running' && p.kind === 'minimization' ? 'running' : p.state}`}>${p.kind === 'minimization' && p.state === 'running' ? 'minimizing' : p.state}</span>`}
              ${p.kind === 'minimization'
                ? lastEnergy(p) != null && html`<span class="muted mono">${fmtSigned(lastEnergy(p))}</span>`
                : pathBarrier(p) != null && html`<span class="muted mono">${pathBarrier(p).toFixed(1)}</span>`}</div>
            <${EnergyPlot} xs=${p.plot.x} ys=${p.plot.y} compact />
          </button>`)}
        </div>`}
      ${stats && html`<${StatsBlock} stats=${stats} />`}
      ${!paths.length && !stats && job.status === 'running' && job.op !== 'graph-enumeration' && html`<p class="muted small">${['hessian-sample', 'hessian-global'].includes(job.op)
        ? 'Each candidate appears here as it starts minimizing (after the Hessian of the seed).'
        : 'The live paths appear once a path optimization starts.'}</p>`}
    </div>`;
}

function StatsBlock({ stats }) {
  const c = stats.conformers || {};
  const items = [
    ['Reactant conformers', c.start?.n_final],
    ['Product conformers', c.end?.n_final],
    ['Pairs', stats.n_pairs],
    ['Mechanisms', stats.n_mechanisms],
    ['Path searches', stats.n_path_searches],
    ['Elapsed', stats.total_seconds ? fmtDuration(stats.total_seconds) : null],
  ].filter(([, v]) => v != null);
  if (!items.length) return null;
  return html`<div class="stat-row wrap">
    ${items.map(([l, v]) => html`<div class="stat"><span class="stat-v">${v}</span><span class="stat-l">${l}</span></div>`)}
  </div>`;
}

// ------------------------------------------------------------ result
const TS_KINDS = new Set(['ts', 'channel', 'alternate', 'offtarget']);

// Every TS-bearing entry of a result: TS structures, each channel's
// representative TS, and (for channels) every individual TS search.
function tsEntries(result) {
  const out = [];
  for (const g of result.groups) {
    if (!TS_KINDS.has(g.kind)) continue;
    for (const e of g.entries) {
      const i = tsFrameIndex(e);
      if (i != null) out.push({ group: g.title, entry: e, index: i });
    }
  }
  return out;
}

function csvCell(v) {
  const s = v == null ? '' : String(v);
  return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
}

function downloadTsTable(job, result) {
  const rows = [['group', 'label', 'barrier_kcal_mol', 'ts_energy_hartree', 'ts_frame', 'n_frames', 'note']];
  for (const { group, entry, index } of tsEntries(result)) {
    rows.push([group, entry.label, entry.barrier_kcal, entry.frames[index].energy_hartree, index + 1, entry.frames.length, entry.note]);
  }
  downloadText(`${safeName(job.title)}_ts.csv`, rows.map((r) => r.map(csvCell).join(',')).join('\n') + '\n', 'text/csv');
}

function downloadAllTs(job, result) {
  const text = tsEntries(result).map(({ entry, index }) => entryToXyz(entry, [index])).join('');
  downloadText(`${safeName(job.title)}_ts.xyz`, text, 'chemical/x-xyz');
}

// The entry list only changes with the result or the selection -- not while
// a path is playing -- so it opts out of the per-frame re-render.
// Groups whose entries can be ticked and added to the Graph in bulk (each
// adds its structure: the only frame, or the TS of a path).
const PICKABLE = new Set(['minima', 'conformers', 'rejected', 'ts', 'ts_other', 'channel', 'alternate', 'offtarget']);

// Relative energy used by the bulk selectors (kcal/mol).
function entryEnergy(e) {
  if (e.frames.length === 1) return e.frames[0].energy_kcal;
  return e.barrier_kcal ?? e.frames[e.ts_index ?? 0]?.energy_kcal ?? null;
}

// The entry list only changes with the result, the selection, the ticks or
// what is already in the Graph -- not while a path is playing -- so it opts
// out of the per-frame re-render.
class EntryNav extends Component {
  shouldComponentUpdate(next) {
    const p = this.props;
    return next.groups !== p.groups || next.entryId !== p.entryId || next.picked !== p.picked || next.inGraph !== p.inGraph;
  }

  render({ groups, entryId, onSelect, picked, inGraph, onPick }) {
    return html`
      <nav class="entries">
        ${groups.map((g) => {
          const pickable = PICKABLE.has(g.kind);
          const free = g.entries.filter((e) => !inGraph.has(e.id));
          const nPicked = free.filter((e) => picked.has(e.id)).length;
          return html`
          <details open=${g.kind !== 'conformers' && g.entries.length < 40} class="entry-group">
            <summary>
              ${pickable && free.length > 0 && html`<input type="checkbox" class="pick" title="Select all in this group"
                checked=${nPicked === free.length} indeterminate=${nPicked > 0 && nPicked < free.length}
                onClick=${(e) => { e.stopPropagation(); onPick(free.map((x) => x.id), nPicked !== free.length); }} />`}
              ${g.title} <span class="count">${g.entries.length}</span></summary>
            <ul>
              ${g.entries.map((e) => html`
                <li class=${e.id === entryId ? 'on' : ''} onClick=${() => onSelect(e.id)} title=${e.note}>
                  ${pickable && (inGraph.has(e.id)
                    ? html`<span class="in-graph" title="Already in the Graph">✓</span>`
                    : html`<input type="checkbox" class="pick" checked=${picked.has(e.id)}
                        onClick=${(ev) => { ev.stopPropagation(); onPick([e.id], !picked.has(e.id)); }} />`)}
                  <span class="entry-label">${e.label}</span>
                  ${inGraph.has(e.id) && html`<span class="badge">in Graph</span>`}
                  ${e.barrier_kcal != null && html`<span class="barrier">${fmtKcal(e.barrier_kcal)}</span>`}
                </li>`)}
            </ul>
          </details>`;
        })}
      </nav>`;
  }
}

// Bulk "Add to Graph": energy-based selectors for one group, plus the button.
function BulkBar({ group, picked, inGraph, onSet, onAdd, busy }) {
  const [within, setWithin] = useState(5);
  const [lowest, setLowest] = useState(3);
  const free = group ? group.entries.filter((e) => !inGraph.has(e.id)) : [];
  const withE = free.filter((e) => entryEnergy(e) != null);
  const minE = withE.length ? Math.min(...withE.map(entryEnergy)) : null;
  const byEnergy = [...withE].sort((a, b) => entryEnergy(a) - entryEnergy(b));
  const n = picked.size;
  return html`
    <div class="bulk-bar">
      ${group && html`<div class="bulk-select small">
        <div class="bulk-row"><span class="bulk-what" title=${group.title}>Pick from <b>${group.title}</b></span>
          <span class="bulk-quick"><button class="btn-link small" onClick=${() => onSet(free.map((e) => e.id))}>All</button>
          <button class="btn-link small" onClick=${() => onSet([])}>None</button></span></div>
        ${minE != null && html`
          <div class="bulk-row">
            <span class="bulk-field">Within <input type="number" class="tiny" min="0" step="0.5" value=${within}
              onInput=${(e) => setWithin(+e.target.value)} /> kcal/mol of lowest</span>
            <button class="btn-link small" onClick=${() => onSet(withE.filter((e) => entryEnergy(e) - minE <= within + 1e-9).map((e) => e.id))}>Pick</button></div>
          <div class="bulk-row">
            <span class="bulk-field">The <input type="number" class="tiny" min="1" step="1" value=${lowest}
              onInput=${(e) => setLowest(Math.max(1, +e.target.value | 0))} /> lowest in energy</span>
            <button class="btn-link small" onClick=${() => onSet(byEnergy.slice(0, lowest).map((e) => e.id))}>Pick</button></div>`}
      </div>`}
      <button class="btn primary" disabled=${!n || busy} onClick=${onAdd}>
        ${busy ? 'Adding…' : n ? `Add ${n} to Graph` : 'Tick structures to add'}</button>
    </div>`;
}

function ResultPanel({ job }) {
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const [entryId, setEntryId] = useState(null);
  const [frame, setFrame] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [labels, setLabels] = useState(false);
  const running = job.status === 'running';

  const load = async () => {
    setLoading(true);
    setError(null);
    try {
      const r = await api.get(`/api/jobs/${job.id}/result`);
      setResult(r);
      const all = r.groups.flatMap((g) => g.entries);
      setEntryId((cur) => (all.some((e) => e.id === cur) ? cur : all[0]?.id ?? null));
    } catch (e) { setError(e.message); }
    setLoading(false);
  };
  // result_rev: a follow-up (e.g. 'Check the bifurcation') added to this job's output.
  useEffect(() => { load(); }, [job.id, job.status, job.finished, job.result_rev]);

  const entry = useMemo(() => result?.groups.flatMap((g) => g.entries).find((e) => e.id === entryId), [result, entryId]);
  const frameXyz = useMemo(() => entry?.frames.map((x) => x.xyz), [entry]);
  const plotXs = useMemo(() => entry?.frames.map((x) => x.path_length), [entry]);
  const plotYs = useMemo(() => entry?.frames.map((x) => x.energy_kcal) ?? [], [entry]);
  useEffect(() => {
    if (!entry) return;
    setFrame(entry.ts_index ?? 0);
    setPlaying(false);
  }, [entryId, result]);
  useEffect(() => {
    if (!playing || !entry) return undefined;
    const id = setInterval(() => setFrame((f) => (f + 1) % entry.frames.length), 90);
    return () => clearInterval(id);
  }, [playing, entry]);

  // Entries already pulled into the Graph from this job (by structure origin).
  const inGraphKey = useStore((s) => Object.values(s.workspace.structures)
    .filter((r) => r.origin?.job === job.id).map((r) => r.origin.entry).sort().join('|'));
  const inGraph = useMemo(() => new Set(inGraphKey ? inGraphKey.split('|') : []), [inGraphKey]);
  const [picked, setPicked] = useState(() => new Set());
  const [adding, setAdding] = useState(false);
  useEffect(() => setPicked(new Set()), [job.id]);
  const pick = (ids, on) => setPicked((prev) => {
    const next = new Set(prev);
    for (const id of ids) (on ? next.add(id) : next.delete(id));
    return next;
  });

  const reportAdded = (out) => {
    const n = out.added.length, r = out.reused.length;
    const msg = [n && `Added ${n} structure${n > 1 ? 's' : ''} to the Graph`, r && `${r} already there`, out.edge && 'edge added']
      .filter(Boolean).join(', ') || 'Nothing to add';
    const ids = [...out.added, ...out.reused].map((x) => x.id);
    const show = () => {
      if (out.edge) select({ edges: [out.edge.id] }); else select({ structures: ids });
      set({ view: { tab: 'graph', jobId: job.id } });
    };
    toast(msg, 'ok', 7000, ids.length || out.edge ? { label: 'Show in Graph', run: show } : null);
  };
  const importEntry = async (frames, extra = {}) => {
    const out = await attempt(() => api.post(`/api/jobs/${job.id}/import-entry`, { entry: entry.id, frames, ...extra }));
    if (out) reportAdded(out);
  };
  const addPicked = async () => {
    setAdding(true);
    const out = await attempt(() => api.post(`/api/jobs/${job.id}/import-entries`, { entries: [...picked] }));
    setAdding(false);
    if (out) { setPicked(new Set()); reportAdded(out); }
  };

  if (error) return html`<div class="warn-box">Could not read results: ${error} <button class="btn small" onClick=${load}>Retry</button></div>`;
  if (!result) return html`<p class="muted">Reading results…</p>`;
  if (!result.groups.length) {
    return html`<div class="empty-hint"><p>${result.headline}</p>
      ${running && html`<button class="btn small" onClick=${load} disabled=${loading}>${loading ? 'Reading…' : 'Check for partial results'}</button>`}</div>`;
  }
  const f = entry?.frames[Math.min(frame, entry.frames.length - 1)];
  // The bulk selectors act on the group of the entry being viewed (or the
  // first pickable group).
  const pickGroups = result.groups.filter((g) => PICKABLE.has(g.kind));
  const pickGroup = pickGroups.find((g) => g.entries.some((e) => e.id === entryId)) || pickGroups[0] || null;
  const multi = entry && entry.frames.length > 1;
  return html`
    <div class="result">
      <div class="result-head">
        <div class="headline">${result.headline}</div>
        ${running && html`<button class="btn small" onClick=${load} disabled=${loading}>${loading ? 'Reading…' : 'Refresh partial results'}</button>`}
        <span class="spacer"></span>
        <div class="downloads">
          <span class="small muted">Download</span>
          ${tsEntries(result).length > 0 && html`
            <button class="btn small" onClick=${() => downloadTsTable(job, result)} title="One row per TS: barrier, energy, what it connects">TS table (CSV)</button>
            <button class="btn small" onClick=${() => downloadAllTs(job, result)} title="Every TS geometry, one frame each, energies in the comment line">All TSs (XYZ)</button>`}
          <a class="btn small" href=${`/api/jobs/${job.id}/archive`} title="The full output folder, inputs and logs">All files (.zip)</a>
        </div>
      </div>
      ${result.summary.length > 0 && html`<dl class="summary">
        ${result.summary.map((s) => html`<div><dt>${s.label}</dt><dd>${s.value}</dd></div>`)}
      </dl>`}
      ${result.warnings.length > 0 && html`
        <details class="warn-box small warnings">
          <summary>${result.warnings.length} warning${result.warnings.length > 1 ? 's' : ''} reported by mepd — check before trusting every number</summary>
          <ul>${result.warnings.map((w) => html`<li class="mono">${w}</li>`)}</ul>
        </details>`}
      ${result.vri && html`<${VriFollowUps} job=${job} vri=${result.vri} />`}
      <div class="result-body">
        <div class="entries-col">
          ${pickGroup && html`<${BulkBar} group=${pickGroup} picked=${picked} inGraph=${inGraph} busy=${adding}
            onSet=${(ids) => setPicked(new Set(ids))} onAdd=${addPicked} />`}
          <${EntryNav} groups=${result.groups} entryId=${entryId} onSelect=${setEntryId}
            picked=${picked} inGraph=${inGraph} onPick=${pick} />
        </div>
        ${entry && html`
          <div class="entry-view">
            <div class="entry-title">
              <strong>${entry.label}</strong>
              ${entry.barrier_kcal != null && html`<span class="badge accent">ΔE‡ ${fmtKcal(entry.barrier_kcal)} kcal/mol</span>`}
            </div>
            ${entry.note && html`<div class="small mono muted note">${entry.note}</div>`}
            <${Viewer3D} frames=${frameXyz} frame=${frame} labels=${labels} height=${300} />
            ${multi && html`
              <div class="frame-bar">
                <button class="btn-icon" onClick=${() => setPlaying(!playing)} title=${playing ? 'Pause' : 'Play'}>${playing ? '❚❚' : '▶'}</button>
                <input type="range" min="0" max=${entry.frames.length - 1} value=${frame} onInput=${(e) => { setPlaying(false); setFrame(+e.target.value); }} />
                <span class="small mono nowrap">${frame + 1}/${entry.frames.length}${f?.energy_kcal != null ? ` · ${f.energy_kcal.toFixed(2)} kcal/mol` : ''}</span>
              </div>
              <${EnergyPlot} xs=${plotXs} ys=${plotYs} current=${frame}
                tsIndex=${entry.ts_index} onPick=${(i) => { setPlaying(false); setFrame(i); }} height=${150} />`}
            ${!multi && f?.energy_kcal != null && html`<p class="small muted">Relative energy ${f.energy_kcal.toFixed(2)} kcal/mol</p>`}
            <div class="entry-actions">
              <label class="small"><input type="checkbox" checked=${labels} onChange=${(e) => setLabels(e.target.checked)} /> atom indices</label>
              <span class="spacer"></span>
              <span class="small muted">XYZ:</span>
              ${multi ? html`
                <button class="btn small" onClick=${() => downloadText(`${safeName(entry.label)}.xyz`, entryToXyz(entry))} title="Every frame of this path">path</button>
                ${entry.ts_index != null && html`<button class="btn small" onClick=${() => downloadText(`${safeName(entry.label)}_ts.xyz`, entryToXyz(entry, [entry.ts_index]))}>TS</button>`}
                <button class="btn small" onClick=${() => downloadText(`${safeName(entry.label)}_frame${frame + 1}.xyz`, entryToXyz(entry, [frame]))}>frame</button>`
              : html`<button class="btn small" onClick=${() => downloadText(`${safeName(entry.label)}.xyz`, entryToXyz(entry))}>download</button>`}
              <span class="divider"></span>
              ${multi ? html`
                <button class="btn small primary" onClick=${() => importEntry('endpoints', { connect: true })}
                  title="Add both ends to the Graph and connect them with an edge carrying this result">Add ends + edge to Graph</button>
                ${entry.ts_index != null && html`<button class="btn small" onClick=${() => importEntry('ts')}>Add TS to Graph</button>`}
                <button class="btn small ghost" onClick=${() => importEntry('one', { frame })}>Add this frame to Graph</button>`
              : inGraph.has(entry.id)
                ? html`<span class="badge">✓ in Graph</span>`
                : html`<button class="btn small primary" onClick=${() => importEntry('one', { frame: 0 })}>Add to Graph</button>`}
            </div>
          </div>`}
      </div>
      ${result.vri?.explorer && html`<${VriExplorer} job=${job} />`}
    </div>`;
}

// ------------------------------------------------------------ VRI
const VRI_NEXT = {
  second_product_untested: 'A second product was found. Check the bifurcation to see whether paths from TS1 really split between P1 and P2.',
  bifurcation: 'Checked: sideways pushes off the IRC drain into both products. Map the surface to see TS1, the VRI, TS2, P1 and P2 on one picture.',
  second_product_no_split: 'Sideways pushes all end in P1, so the second product is probably not reached from TS1 (trajectories can still be run).',
};

function FollowUpCard({ op, job, runs, done }) {
  const [open, setOpen] = useState(false);
  const [values, setValues] = useState(() => clampToSchema(op.schema, { ...defaultsFor(op.schema), ...prefs.get(`params:${op.key}`, {}) }));
  const [busy, setBusy] = useState(false);
  const last = runs[0];
  const active = last && ['queued', 'running'].includes(last.status);
  const run = async () => {
    setBusy(true);
    prefs.set(`params:${op.key}`, values);
    const out = await attempt(() => api.post('/api/jobs', { op: op.key, params: values, source_job: job.id }),
      (js) => `Queued: ${js[0].title}`);
    setBusy(false);
    if (out) setOpen(false);
  };
  return html`
    <div class=${`op-card ${open ? 'open' : ''}`}>
      <button type="button" class="op-head" onClick=${() => setOpen(!open)} aria-expanded=${open}>
        <span class="op-text">
          <span class="op-title">${op.title}
            ${last ? html` <span class=${`pill ${last.status}`}>${STATUS_LABEL[last.status]}</span>`
              : done && html` <span class="pill done">Done</span>`}</span>
          <span class="op-summary">${op.summary.replace(/\s*\(`[^`]*`\)\s*$/, '')}</span>
        </span>
        <span class="op-chevron" aria-hidden="true">${open ? '−' : '+'}</span>
      </button>
      ${open && html`<div class="op-body">
        <${ParamForm} schema=${op.schema} values=${values} onChange=${setValues} />
        <div class="op-run">
          <button class="btn primary" disabled=${busy || active} onClick=${run}>
            ${active ? 'Running…' : busy ? 'Queuing…' : (last || done) ? 'Run again' : 'Run'}</button>
          ${last && html`<a href="#" class="small" onClick=${(e) => { e.preventDefault(); openJob(last.id); }}>its log</a>`}
          <span class="small muted">Runs at this job's level of theory; results appear on this page.</span>
        </div>
      </div>`}
    </div>`;
}

function VriFollowUps({ job, vri }) {
  const operations = useStore((s) => s.operations);
  const runsKey = useStore((s) => Object.values(s.jobs).filter((j) => j.source_job === job.id)
    .map((j) => `${j.id}:${j.status}`).sort().join('|'));
  const runs = useMemo(() => Object.values(state.jobs).filter((j) => j.source_job === job.id)
    .sort((a, b) => b.created - a.created), [runsKey]);
  const ops = operations.filter((o) => o.target === 'job' && o.available && (o.source_ops || []).includes(job.op));
  if (job.status !== 'done' || !ops.length || job.source_job) return null;
  const done = { 'vri-check': vri.checked, 'vri-surface': vri.surface };
  return html`
    <section class="followups">
      <h3 class="section-title">Next steps</h3>
      ${VRI_NEXT[vri.verdict] && html`<p class="small muted">${VRI_NEXT[vri.verdict]}</p>`}
      ${ops.map((op) => html`<${FollowUpCard} key=${op.key} op=${op} job=${job} done=${done[op.key]}
        runs=${runs.filter((r) => r.op === op.key)} />`)}
    </section>`;
}

function VriExplorer({ job }) {
  const src = `/api/jobs/${job.id}/vri-viewer?rev=${job.result_rev || 0}`;
  return html`
    <section class="vri-explorer">
      <div class="live-head">
        <h3 class="section-title">VRI explorer</h3>
        <a class="small" href=${src} target="_blank" rel="noopener">Open in a new tab</a>
      </div>
      <iframe class="vri-frame" src=${src} title="VRI explorer" loading="lazy"></iframe>
    </section>`;
}

// ------------------------------------------------------------ log
function LogPanel({ job }) {
  const [which, setWhich] = useState('stdout');
  const [text, setText] = useState('');
  const offset = useRef(0);
  const pre = useRef(null);
  const follow = useRef(true);
  const logSize = useStore((s) => s.progress[job.id]?.log_size);

  const fetchMore = async (reset = false) => {
    if (reset) { offset.current = 0; }
    const start = reset ? -200000 : offset.current;
    const res = await api.raw(`/api/jobs/${job.id}/log?which=${which}&offset=${start}`).catch(() => null);
    if (!res) return;
    const chunk = await res.text();
    offset.current = +(res.headers.get('X-Log-Size') || 0);
    setText((t) => (reset ? chunk : t + chunk));
  };
  useEffect(() => { fetchMore(true); }, [job.id, which]);
  useEffect(() => { if (job.status === 'running' || logSize) fetchMore(false); }, [logSize, job.status]);
  useEffect(() => {
    if (job.status !== 'running' || which !== 'progress') return undefined;
    const id = setInterval(() => fetchMore(false), 2000);
    return () => clearInterval(id);
  }, [job.status, which]);
  useEffect(() => { if (follow.current && pre.current) pre.current.scrollTop = pre.current.scrollHeight; }, [text]);

  // Carriage-return redraws: show only the final state of each line.
  const shown = text.split('\n').map((l) => l.split('\r').pop()).join('\n');
  return html`
    <div class="log-panel">
      <div class="segmented">
        <button class=${which === 'stdout' ? 'on' : ''} onClick=${() => setWhich('stdout')}>Output</button>
        <button class=${which === 'progress' ? 'on' : ''} onClick=${() => setWhich('progress')}>Branch progress</button>
      </div>
      <pre class="log" ref=${pre} onScroll=${(e) => { const el = e.target; follow.current = el.scrollHeight - el.scrollTop - el.clientHeight < 30; }}>${shown || '(empty)'}</pre>
    </div>`;
}

function FilesPanel({ job }) {
  const [files, setFiles] = useState(null);
  useEffect(() => { api.get(`/api/jobs/${job.id}/files`).then(setFiles).catch(() => setFiles([])); }, [job.id, job.status]);
  if (!files) return html`<p class="muted">Listing…</p>`;
  return html`
    <div class="files">
      <p class="small muted mono">${job.output_dir}</p>
      <ul>${files.map((f) => html`<li><a href=${`/api/jobs/${job.id}/files/${f}`} target="_blank" rel="noopener">${f}</a></li>`)}</ul>
      ${files.length === 0 && html`<p class="muted">No files yet.</p>`}
    </div>`;
}

// ------------------------------------------------------------ view
export function JobView({ jobId }) {
  const job = useStore((s) => s.jobs[jobId]);
  const [tab, setTab] = useState(null);
  useTick(1000, job?.status === 'running');
  useEffect(() => { setTab(null); }, [jobId]);
  // Opened before this page heard about the job (just queued): look it up
  // once before calling it gone.
  const [lookedUp, setLookedUp] = useState(false);
  useEffect(() => {
    setLookedUp(false);
    if (job) return;
    refreshState().catch(() => {}).finally(() => setLookedUp(true));
  }, [jobId]);
  if (!job) {
    return html`<div class="empty-hint"><p>${lookedUp ? 'This calculation no longer exists.' : 'Loading this calculation…'}</p></div>`;
  }
  const hasOutput = job.status !== 'queued';
  const current = tab || (['done'].includes(job.status) || job.external ? 'result' : job.status === 'failed' ? 'log' : 'live');
  const targets = job.targets.structures.map((id) => ({ id, label: structureName(id) }));
  return html`
    <div class="job-view">
      <div class="view-head">
        <button class="btn-icon" title="All calculations" onClick=${() => set({ view: { tab: 'jobs', jobId: null } })}>←</button>
        <div class="job-head-main">
          <h2>${job.title}</h2>
          <div class="small muted">
            <span class=${`pill ${job.status}`}>${STATUS_LABEL[job.status]}</span>
            ${job.external ? ' · existing output, read in place' : job.started && html` · ${fmtDuration(jobElapsed(job))}`}
            ${job.profile && html` · profile <b>${job.profile}</b>`}
            ${targets.length > 0 && html` · ${targets.map((t, i) => html`${i ? (job.op === 'ts' || job.op === 'channels' ? ' → ' : ', ') : ''}<a href="#"
              onClick=${(e) => { e.preventDefault(); select({ structures: [t.id] }); set({ view: { tab: 'graph', jobId } }); }}>${t.label}</a>`)}`}
          </div>
        </div>
        <${JobControls} job=${job} />
      </div>
      <details class="cmd-line">
        <summary>Command</summary>
        <div class="cmd-body" title="Exactly what ran; paste into a shell to reproduce">
          <code>${job.command}</code>
          <button class="btn-icon" onClick=${() => copy(job.command)} title="Copy">⧉</button>
        </div>
      </details>
      ${job.status === 'failed' && job.error && html`<pre class="error-box">${job.error}</pre>`}
      ${['cancelled', 'interrupted'].includes(job.status) && html`<p class="warn-box small">${job.error} ${!job.external && html`<button class="btn small" onClick=${() => attempt(() => api.post(`/api/jobs/${job.id}/retry`))}>Resume</button>`}</p>`}
      <div class="tabs">
        ${[['result', 'Results'], !job.external && ['live', 'Live'], !job.external && ['log', 'Log'], ['files', 'Files']].filter(Boolean)
          .map(([k, l]) => html`<button class=${current === k ? 'on' : ''} disabled=${!hasOutput && k !== 'live'} onClick=${() => setTab(k)}>${l}</button>`)}
      </div>
      <div class="tab-body">
        ${current === 'result' && html`<${ResultPanel} job=${job} />`}
        ${current === 'live' && html`<${LivePanel} job=${job} />`}
        ${current === 'log' && html`<${LogPanel} job=${job} />`}
        ${current === 'files' && html`<${FilesPanel} job=${job} />`}
      </div>
    </div>`;
}
