// One job: live progress while it runs, an explorable result when it has
// output (partial results work too), the raw log, and its files.
import { Component, html, useEffect, useMemo, useRef, useState } from '../lib.js';
import { api, attempt, refreshState } from '../api.js';
import { familyOf, openJob, pageJobId, prefs, select, set, state, toast, update, useStore } from '../store.js';
import { STATUS_LABEL, copy, downloadText, entryToXyz, fmtDuration, fmtKcal, jobElapsed, safeName, structureName, tsFrameIndex } from '../util.js';
import { EnergyPlot } from './EnergyPlot.js';
import { ParamForm, clampToSchema, defaultsFor } from './ParamForm.js';
import { JobControls, useTick } from './Jobs.js';
import { NetworkLive } from './NetworkLive.js';
import { ChannelsMap } from './ChannelsMap.js';
import { OptTree } from './OptTree.js';
import { ReactorLive } from './ReactorLive.js';
import { ComplexLive } from './ComplexLive.js';
import { ReactionTable, SpawnedSearches } from './Reactions.js';
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
                    ? html`<span class="in-graph" title="Already in Explore">✓</span>`
                    : html`<input type="checkbox" class="pick" checked=${picked.has(e.id)}
                        onClick=${(ev) => { ev.stopPropagation(); onPick([e.id], !picked.has(e.id)); }} />`)}
                  <span class="entry-label">${e.label}</span>
                  ${inGraph.has(e.id) && html`<span class="badge">in Explore</span>`}
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
        ${busy ? 'Adding…' : n ? `Add ${n} to Explore` : 'Tick structures to add'}</button>
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
    const msg = [n && `Added ${n} structure${n > 1 ? 's' : ''} to Explore`, r && `${r} already there`, out.edge && 'edge added']
      .filter(Boolean).join(', ') || 'Nothing to add';
    const ids = [...out.added, ...out.reused].map((x) => x.id);
    const show = () => {
      if (out.edge) select({ edges: [out.edge.id] }); else select({ structures: ids });
      set({ view: { tab: 'graph', jobId: job.id } });
    };
    toast(msg, 'ok', 7000, ids.length || out.edge ? { label: 'Show in Explore', run: show } : null);
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
  if (!result.groups.length && !result.mechano && !result.conditions && !result.substituents) {
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
      ${(result.barrier_warnings || []).map((w) => html`<p class="error-box small">⚠ ${w}</p>`)}
      ${result.warnings.length > 0 && html`
        <details class="warn-box small warnings">
          <summary>${result.warnings.length} warning${result.warnings.length > 1 ? 's' : ''} reported by mepd — check before trusting every number</summary>
          <ul>${result.warnings.map((w) => html`<li class="mono">${w}</li>`)}</ul>
        </details>`}
      ${job.op === 'nanoreactor' && html`<${SpawnedSearches} job=${job} />`}
      ${result.nanoreactor && html`<${ReactionTable} job=${job} reactions=${result.nanoreactor.reactions} />`}
      ${result.conditions && html`<${ConditionsPanel} cond=${result.conditions} />`}
      ${result.mechano && html`<p class="level-note small">Mechanical-force results are not shown in the web UI for now;
        this job's files are under Files.</p>`}
      ${result.substituents && html`<${SubstituentPanel} x=${result.substituents} />`}
      ${['ts', 'tsopt', 'design-tsopt', 'channels'].includes(job.op) && html`<${ConditionsFollowUps} job=${job} />`}
      ${result.vri && html`<${VriFollowUps} job=${job} vri=${result.vri} />`}
      ${job.op === 'channels' && html`<${ChannelsFollowUps} job=${job} />`}
      ${job.op === 'ts' && html`<${TsFollowUps} job=${job} />`}
      ${result.groups.length > 0 && html`<div class="result-body">
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
                  title="Add both ends to Explore and connect them with an edge carrying this result">Add ends + edge to Explore</button>
                ${entry.ts_index != null && html`<button class="btn small" onClick=${() => importEntry('ts')}>Add TS to Explore</button>`}
                <button class="btn small ghost" onClick=${() => importEntry('one', { frame })}>Add this frame to Explore</button>`
              : inGraph.has(entry.id)
                ? html`<span class="badge">✓ in Explore</span>`
                : html`<button class="btn small primary" onClick=${() => importEntry('one', { frame: 0 })}>Add to Explore</button>`}
            </div>
          </div>`}
      </div>`}
      ${result.vri?.explorer && html`<${VriExplorer} job=${job} />`}
    </div>`;
}

// ------------------------------------------------------------ VRI
const VRI_NEXT = {
  second_product_untested: 'A second product was found. Check the bifurcation to see whether paths from TS1 really split between P1 and P2.',
  bifurcation: 'Checked: sideways pushes off the IRC drain into both products. Map the surface to see TS1, the VRI, TS2, P1 and P2 on one picture.',
  second_product_no_split: 'Sideways pushes all end in P1, so the second product is probably not reached from TS1 (trajectories can still be run).',
};

function FollowUpCard({ op, job, runs, done, initial = {}, note = null }) {
  const [open, setOpen] = useState(false);
  const [values, setValues] = useState(() => clampToSchema(op.schema, { ...defaultsFor(op.schema), ...prefs.get(`params:${op.key}`, {}), ...initial }));
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
        ${note && html`<p class="small muted">${note}</p>`}
        <${ParamForm} schema=${op.schema} values=${values} onChange=${setValues} />
        <div class="op-run">
          <button class="btn primary" disabled=${busy || active} onClick=${run}>
            ${active ? 'Running…' : busy ? 'Queuing…' : (last || done) ? 'Run again' : 'Run'}</button>
          ${last && html`<a href="#" class="small" onClick=${(e) => { e.preventDefault(); openJob(last.id, 'log'); }}>its log</a>`}
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

// A finished channels run: search more conformer pairs per mechanism, reusing
// everything it computed (the follow-up reruns it in its own folder).
function ChannelsFollowUps({ job }) {
  const operations = useStore((s) => s.operations);
  const runsKey = useStore((s) => Object.values(s.jobs).filter((j) => j.source_job === job.id)
    .map((j) => `${j.id}:${j.status}`).sort().join('|'));
  const runs = useMemo(() => Object.values(state.jobs).filter((j) => j.source_job === job.id)
    .sort((a, b) => b.created - a.created), [runsKey]);
  const op = operations.find((o) => o.key === 'channels-more' && o.available);
  if (!op || job.status !== 'done' || job.op !== 'channels') return null;
  const done = runs.filter((r) => r.status === 'done').map((r) => r.params?.pairs_per_mechanism ?? 0);
  const cap = [job.params?.pairs_per_mechanism ?? 0, ...done].reduce((m, v) => (m === 0 || v === 0 ? 0 : Math.max(m, v)));
  if (cap === 0) return null;   // every pair of every mechanism is already searched
  return html`
    <section class="followups">
      <h3 class="section-title">Next steps</h3>
      <${FollowUpCard} op=${op} job=${job} runs=${runs.filter((r) => r.op === op.key)}
        initial=${{ pairs_per_mechanism: cap * 2 }}
        note=${`So far: the best ${cap} conformer pair${cap === 1 ? '' : 's'} per mechanism. Pairs already searched are skipped, and the result and path map then cover all of them.`} />
    </section>`;
}

// A finished single-pair TS search: sample more paths for the same pair --
// a Reaction channels run on the same two structures (and conformers), at
// the same level of theory, whose paths join this page's. Once there is
// one, sampling more extends that run (more conformer pairs per mechanism).
function TsFollowUps({ job }) {
  const operations = useStore((s) => s.operations);
  const op = operations.find((o) => o.key === 'channels' && o.available);
  const runsKey = useStore((s) => Object.values(s.jobs).filter((j) => j.extends === job.id)
    .map((j) => `${j.id}:${j.status}`).sort().join('|'));
  const runs = useMemo(() => Object.values(state.jobs).filter((j) => j.extends === job.id)
    .sort((a, b) => b.created - a.created), [runsKey]);
  const [open, setOpen] = useState(false);
  const [values, setValues] = useState(() => (op ? clampToSchema(op.schema, { ...defaultsFor(op.schema), ...prefs.get('params:channels', {}) }) : {}));
  const [busy, setBusy] = useState(false);
  if (!op || job.status !== 'done' || job.external || job.targets.structures.length !== 2) return null;
  const last = runs[0];
  const active = last && ['queued', 'running'].includes(last.status);
  const finished = runs.find((r) => r.status === 'done');
  if (finished && !active) return html`<${ChannelsFollowUps} job=${finished} />`;
  const conformers = Object.fromEntries(job.targets.structures.map((sid, i) => [sid, job.target_conformers?.[i] || null]));
  const run = async () => {
    setBusy(true);
    prefs.set('params:channels', values);
    const out = await attempt(() => api.post('/api/jobs', {
      op: 'channels', structures: job.targets.structures, edges: job.targets.edges || [], params: values,
      profile: job.profile, conformers, extends: job.id }), (js) => `Queued: ${js[0].title}`);
    setBusy(false);
    if (out?.length) { setOpen(false); openJob(out[0].id, 'live'); }
  };
  return html`
    <section class="followups">
      <h3 class="section-title">Next steps</h3>
      <div class=${`op-card ${open ? 'open' : ''}`}>
        <button type="button" class="op-head" onClick=${() => setOpen(!open)} aria-expanded=${open}>
          <span class="op-text">
            <span class="op-title">Sample more paths
              ${last && html` <span class=${`pill ${last.status}`}>${STATUS_LABEL[last.status]}</span>`}</span>
            <span class="op-summary">This search used one conformer of each end and one atom mapping. Sample both ends' conformers and
              the atom mappings, search every distinct mechanism, and keep the lowest channel (Reaction channels).</span>
          </span>
          <span class="op-chevron" aria-hidden="true">${open ? '−' : '+'}</span>
        </button>
        ${open && html`<div class="op-body">
          <${ParamForm} schema=${op.schema} values=${values} onChange=${setValues} />
          <div class="op-run">
            <button class="btn primary" disabled=${busy || active} onClick=${run}>
              ${active ? 'Running…' : busy ? 'Queuing…' : last ? 'Sample again' : 'Sample more paths'}</button>
            ${last && html`<a href="#" class="small" onClick=${(e) => { e.preventDefault(); openJob(last.id, 'log'); }}>its log</a>`}
            <span class="small muted">Same two structures and level of theory; results appear on this page.</span>
          </div>
        </div>`}
      </div>
    </section>`;
}

// ------------------------------------------------------------ conditions
const INSIGHT_ICON = { accelerates: '▼', slows: '▲', trend: '↗', caution: '⚠', note: 'ℹ' };

function fmtC(t) { return t == null ? '—' : `${Math.round(t)} °C`; }

// A solvent comparison (a `mepd solvent` result): what it means first, then
// the numbers. The single-point warning is always in view, never folded.
export function ConditionsPanel({ cond, compact = false }) {
  const T = cond.temperature ?? 298.15;
  const gas = cond.gas || {};
  const rows = cond.solvents || [];
  const reopt = cond.mode === 'reoptimize';
  const cautions = (cond.insights || []).filter((i) => i.level === 'caution');
  const rest = (cond.insights || []).filter((i) => i.level !== 'caution');
  // The single-point caveat leads; the composite-model note is quieter.
  const composite = (cond.warnings || []).filter((w) => /composite model/i.test(w));
  const lead = (cond.warnings || []).filter((w) => !composite.includes(w));
  return html`
    <section class="conditions">
      ${(lead.length > 0 || cautions.length > 0) && html`<div class="warn-box small conditions-warn">
        ${lead.map((w) => html`<p>⚠ ${w}</p>`)}
        ${cautions.length > 0 && html`<ul>${cautions.map((i) => html`<li>${i.text}</li>`)}</ul>`}
      </div>`}
      ${rest.length > 0 && html`<ul class="insights">
        ${rest.map((i) => html`<li class=${`insight ${i.level}`}><span class="insight-icon" aria-hidden="true">${INSIGHT_ICON[i.level] || 'ℹ'}</span>${i.text}</li>`)}
      </ul>`}
      ${!compact && html`<div class="table-scroll"><table class="data conditions-table">
        <thead><tr><th>Medium</th><th>Kind</th><th class="num" title="Barrier ΔE‡ (kcal/mol): electronic energy plus solvation free energy">ΔE‡</th>
          <th class="num" title="Change from the gas-phase barrier">vs gas</th>
          ${reopt && html`<th class="num" title="The same, from single points on the gas-phase path">single-point</th>`}
          <th title=${`First-order half-life at ${Math.round(T - 273.15)} °C (Eyring)`}>t½ at ${Math.round(T - 273.15)} °C</th>
          <th title="Temperature at which the half-life is 1 hour">1 h at</th><th>bp</th></tr></thead>
        <tbody>
          <tr class="gas"><td>Gas phase</td><td class="muted">—</td><td class="num">${fmtKcal(gas.barrier_kcal)}</td><td class="muted">—</td>
            ${reopt && html`<td class="muted">—</td>`}<td>${gas.t_half ?? '—'}</td><td>${fmtC(gas.t_1h_c)}</td><td class="muted">—</td></tr>
          ${rows.map((r) => html`<tr class=${r.error ? 'failed' : ''} title=${(r.notes || []).join('\n')}>
            <td>${r.label}</td><td class="small muted">${r.kind}</td>
            ${r.error ? html`<td class="num muted" title=${r.error}>no TS</td><td class="muted">—</td>
              ${reopt && html`<td class="num muted" title="From single points on the gas-phase path">${fmtKcal(r.single_point_barrier_kcal)}</td>`}
              <td class="muted">—</td><td class="muted">—</td><td class="small muted">${fmtC(r.bp_c)}</td>` : html`
              <td class="num">${fmtKcal(r.barrier_kcal)}</td>
              <td class=${`num ${r.shift_kcal <= -3 ? 'good' : r.shift_kcal >= 3 ? 'bad' : ''}`}>${r.shift_kcal == null ? '—' : (r.shift_kcal > 0 ? '+' : '') + r.shift_kcal.toFixed(1)}</td>
              ${reopt && html`<td class="num muted">${fmtKcal(r.single_point_barrier_kcal)}</td>`}
              <td>${r.t_half ?? '—'}</td><td>${fmtC(r.t_1h_c)}</td><td class="small muted">${fmtC(r.bp_c)}</td>`}
          </tr>`)}
        </tbody></table></div>`}
      ${composite.map((w) => html`<p class="small muted">${w}</p>`)}
    </section>`;
}

function fmtNn(f) { return f == null ? '—' : `${f.toFixed(1)} nN`; }

// A `mepd force` result: which atoms to pull on, per channel; when pulling
// makes a slower channel win; and, if run, the check under force.
export function MechanoPanel({ m }) {
  const [lead, ...notes] = m.warnings || [];
  const channels = m.channels || [];
  return html`
    <section class="conditions">
      ${lead && html`<div class="warn-box small conditions-warn"><p>⚠ ${lead}</p></div>`}
      <ul class="insights">
        ${(m.insights || []).map((i) => html`<li class=${`insight ${i.level === 'selectivity' ? 'accelerates' : i.level}`}>
          <span class="insight-icon" aria-hidden="true">${{ accelerates: '▼', slows: '▲', selectivity: '⇄', check: '✓', caution: '⚠' }[i.level] || 'ℹ'}</span>${i.text}</li>`)}
      </ul>
      <div class="table-scroll"><table class="data">
        <thead><tr><th>Channel</th><th class="num">ΔE‡</th><th>Pull apart to speed up</th><th class="num" title="Barrier change per nN (Bell)">per nN</th>
          <th class="num">at 1 nN</th><th class="num" title="Force for a 1 h half-life">1 h at</th><th>Pull apart to hold back</th><th class="num">per nN</th></tr></thead>
        <tbody>${channels.map((c) => {
          const f = c.favor?.[0], d = c.disfavor?.[0];
          return html`<tr><td>${c.label}${c.kind === 'offtarget' ? html` <span class="small muted">(off-target)</span>` : ''}</td>
            <td class="num">${fmtKcal(c.barrier_kcal)}</td>
            <td class="mono" title=${(c.favor || []).map((x) => `${x.label}: Δq‡ ${x.dq.toFixed(2)} Å`).join('\n')}>${f ? f.label : '—'}</td>
            <td class="num good">${f ? f.per_nN_kcal.toFixed(1) : '—'}</td><td class="num">${f ? fmtKcal(f.barrier_at_1nN) : '—'}</td>
            <td class="num" title=${f?.force_for_1h > (m.max_force ?? 2.5) ? 'Out of reach: beyond the forces scanned, near bond rupture' : ''}>
              ${f?.force_for_1h > (m.max_force ?? 2.5) ? html`<span class="muted">> ${m.max_force ?? 2.5} nN</span>` : fmtNn(f?.force_for_1h)}</td>
            <td class="mono" title=${(c.disfavor || []).map((x) => `${x.label}: Δq‡ ${x.dq.toFixed(2)} Å`).join('\n')}>${d ? d.label : '—'}</td>
            <td class="num bad">${d ? '+' + (-d.per_nN_kcal).toFixed(1) : '—'}</td></tr>`;
        })}</tbody></table></div>
      ${(m.efei || []).length > 0 && html`<div class="table-scroll"><table class="data">
        <thead><tr><th>Check under force</th><th>Pair</th><th class="num">Force</th><th class="num">Bell</th><th class="num">Re-optimized</th><th></th></tr></thead>
        <tbody>${m.efei.map((r) => html`<tr class=${r.status === 'ok' ? '' : 'failed'}><td>${r.label}</td><td class="mono">${r.pair_label}</td>
          <td class="num">${r.force_nN} nN</td><td class="num">${fmtKcal(r.barrier_bell)}</td>
          <td class="num">${r.barrier_efei == null ? '—' : fmtKcal(r.barrier_efei)}</td>
          <td class="small muted">${r.status === 'ok' ? '' : r.status === 'other_minima' ? 'connects other minima' : (r.error || 'failed')}</td></tr>`)}
        </tbody></table></div>`}
      ${notes.map((w) => html`<p class="small muted">${w} (Atom numbers: the viewer's.)</p>`)}
    </section>`;
}

// A `mepd substituents` result: barrier shift (kcal/mol) per site x group,
// for one channel at a time; green lowers the barrier, red raises it.
export function SubstituentPanel({ x }) {
  const channels = x.channels || [];
  const lead = channels.reduce((a, c) => (!a || c.barrier_kcal < a.barrier_kcal ? c : a), null);
  const [chId, setChId] = useState(lead?.id);
  const ch = channels.find((c) => c.id === chId) || lead;
  const cell = {};
  for (const v of x.variants || []) if (v.channel === ch?.id) cell[`${v.site}:${v.group}`] = v;
  const [first, ...rest] = x.warnings || [];
  const shade = (d) => {
    const a = Math.min(1, Math.abs(d) / 10) * 0.55;
    return d < 0 ? `background: rgba(73, 110, 92, ${a})` : `background: rgba(178, 59, 59, ${a})`;
  };
  return html`
    <section class="conditions">
      ${first && html`<div class="warn-box small conditions-warn"><p>⚠ ${first}</p></div>`}
      <ul class="insights">
        ${(x.insights || []).map((i) => html`<li class=${`insight ${i.level === 'selectivity' ? 'accelerates' : i.level}`}>
          <span class="insight-icon" aria-hidden="true">${{ accelerates: '▼', slows: '▲', trend: '↗', selectivity: '⇄' }[i.level] || 'ℹ'}</span>${i.text}</li>`)}
      </ul>
      ${channels.length > 1 && html`<label class="small">Channel <select value=${ch?.id} onChange=${(e) => setChId(e.target.value)}>
        ${channels.map((c) => html`<option value=${c.id}>${c.label} (ΔE‡ ${fmtKcal(c.barrier_kcal)})</option>`)}</select></label>`}
      <div class="table-scroll"><table class="data subst-table">
        <thead><tr><th title="Heavy atom whose hydrogen is replaced (viewer's atom numbers)">Site</th>
          ${(x.groups || []).map((g) => html`<th class="num">${x.group_labels?.[g] || g}</th>`)}
          <th class="num" title="Slope of the shift against Hammett σp, and r²">vs σp</th></tr></thead>
        <tbody>${(x.sites || []).map((s) => {
          const t = (x.trends || []).find((r) => r.site === s.anchor);
          return html`<tr><td class="mono">${s.label}</td>
            ${(x.groups || []).map((g) => {
              const v = cell[`${s.anchor}:${g}`];
              if (!v) return html`<td></td>`;
              if (v.status !== 'ok' || v.clash || v.barrier_kcal < -0.1) {
                const why = v.clash ? 'clashes with the rest' : v.status === 'reacted' ? 'reacted while relaxing'
                  : v.barrier_kcal < -0.1 ? `negative ΔE‡ (${v.barrier_kcal.toFixed(1)})` : (v.error || v.status);
                return html`<td class="num muted" title=${why}>×</td>`;
              }
              return html`<td class="num" style=${shade(v.shift)} title=${`ΔE‡ ${v.barrier_kcal.toFixed(1)} kcal/mol`}>${v.shift > 0 ? '+' : ''}${v.shift.toFixed(1)}</td>`;
            })}
            <td class="num small muted">${t ? `${t.slope > 0 ? '+' : ''}${t.slope.toFixed(0)} (${t.r2.toFixed(2)})` : ''}</td></tr>`;
        })}</tbody></table></div>
      ${rest.map((w) => html`<p class="small muted">${w}</p>`)}
    </section>`;
}

// Follow-ups on a finished TS search or channels run that ask how conditions
// change it (solvent, mechanical force); each keeps its own folder and its
// latest result is shown right here.
// (Mechanical force is off in the web UI for now: not listed here, not offered, not shown.)
const CONDITION_OPS = { solvent: 'conditions', substituents: 'substituents' };
const CONDITION_NAMES = { solvent: 'solvent comparison', substituents: 'substituent scan' };

function LatestCondition({ run, opKey }) {
  const [res, setRes] = useState(null);
  useEffect(() => {
    setRes(null);
    api.get(`/api/jobs/${run.id}/result`).then(setRes).catch(() => {});
  }, [run.id, run.finished]);
  const payload = res?.[CONDITION_OPS[opKey]];
  if (!payload) return null;
  return html`<p class="small muted">Latest ${CONDITION_NAMES[opKey]}
      (<a href="#" onClick=${(e) => { e.preventDefault(); openJob(run.id); }}>full result</a>):</p>
    ${opKey === 'solvent' ? html`<${ConditionsPanel} cond=${payload} />` : html`<${SubstituentPanel} x=${payload} />`}`;
}

const CONDITION_NOTES = {
  substituents: 'Each group replaces one hydrogen, placed along the old bond. Fast relaxes only the group (a first estimate); re-optimizing searches each TS again.',
  solvent: "Solvent energies come from GFN2-xTB's implicit models (added to this job's level unless it is GFN2-xTB itself). Keeping gas-phase geometries takes seconds; re-optimizing searches the TS again in each solvent.",
};

function ConditionsFollowUps({ job }) {
  const operations = useStore((s) => s.operations);
  const runsKey = useStore((s) => Object.values(s.jobs).filter((j) => j.source_job === job.id && CONDITION_OPS[j.op])
    .map((j) => `${j.id}:${j.status}:${j.finished}`).sort().join('|'));
  const runs = useMemo(() => Object.values(state.jobs).filter((j) => j.source_job === job.id && CONDITION_OPS[j.op])
    .sort((a, b) => b.created - a.created), [runsKey]);
  const ops = operations.filter((o) => CONDITION_OPS[o.key] && o.available && (o.source_ops || []).includes(job.op));
  // They all start from a TS with its IRC: nothing to offer without a barrier (a channels run
  // that classified no channel included).
  if (!ops.length || job.status !== 'done' || job.summary?.barrier_kcal == null) return null;
  return html`
    <section class="followups">
      <h3 class="section-title">Reaction conditions</h3>
      ${ops.map((op) => {
        const mine = runs.filter((r) => r.op === op.key);
        const latest = mine.find((r) => r.status === 'done');
        return html`<div key=${op.key}>
          ${latest && html`<${LatestCondition} run=${latest} opKey=${op.key} />`}
          <${FollowUpCard} op=${op} job=${job} runs=${mine} done=${!!latest} note=${CONDITION_NOTES[op.key]} />
        </div>`;
      })}
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
export function JobView({ jobId: openedId }) {
  // Sample more paths runs show on the page of the run they add to: one
  // result, map and tree for all of them; Live, Log and Files per run.
  const jobId = useStore(() => pageJobId(openedId));
  const job = useStore((s) => s.jobs[jobId]);
  const familyKey = useStore((s) => familyOf(s.jobs, jobId)
    .map((j) => `${j.id}:${j.status}:${j.finished || 0}:${j.result_rev || 0}`).join('|'));
  const family = useMemo(() => familyOf(state.jobs, jobId), [familyKey]);
  const askedRun = useStore((s) => s.view.run);
  const askedTab = useStore((s) => s.view.jobTab);
  const [tab, setTab] = useState(null);
  const [runId, setRunId] = useState(null);
  const active = [...family].reverse().find((j) => ['queued', 'running'].includes(j.status));
  const run = family.find((j) => j.id === runId) || active || family[family.length - 1] || job;
  useTick(1000, run?.status === 'running');
  useEffect(() => { setTab(askedTab || null); setRunId(askedRun || null); }, [jobId, askedRun, askedTab]);
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
  // What the family-wide views see: running while any run is.
  const page = family.length > 1 ? { ...job, status: active ? 'running' : job.status,
    finished: Math.max(...family.map((j) => j.finished || 0)),
    result_rev: family.reduce((n, j) => n + (j.result_rev || 0), 0) } : job;
  const hasOutput = job.status !== 'queued';
  const current = tab || (['done'].includes(run.status) || job.external ? 'result' : run.status === 'failed' ? 'log'
    : job.op === 'nanoreactor' ? 'reactor' : 'live');
  const targets = job.targets.structures.map((id) => ({ id, label: structureName(id) }));
  const runLabel = (j, i) => (i === 0 ? (job.op === 'ts' ? 'First search' : 'First run') : `More paths ${i}`);
  return html`
    <div class="job-view">
      <div class="view-head">
        <button class="btn-icon" title="All calculations" onClick=${() => set({ view: { tab: 'jobs', jobId: null } })}>←</button>
        <div class="job-head-main">
          <h2>${job.title}</h2>
          <div class="small muted">
            <span class=${`pill ${page.status}`}>${STATUS_LABEL[page.status]}</span>
            ${job.external ? ' · existing output, read in place' : run.started && html` · ${fmtDuration(jobElapsed(run))}`}
            ${job.profile && html` · profile <b>${job.profile}</b>`}
            ${targets.length > 0 && html` · ${targets.map((t, i) => html`${i ? (job.op === 'ts' || job.op === 'channels' ? ' → ' : ', ') : ''}<a href="#"
              onClick=${(e) => { e.preventDefault(); select({ structures: [t.id] }); set({ view: { tab: 'graph', jobId } }); }}>${t.label}</a>`)}`}
          </div>
        </div>
        <${JobControls} job=${run} />
      </div>
      ${family.length > 1 && html`<div class="segmented run-picker" title="Live, Log, Files and Command show this run; the other tabs cover all of them">
        ${family.map((j, i) => html`<button class=${j.id === run.id ? 'on' : ''} onClick=${() => setRunId(j.id)}>
          ${runLabel(j, i)}${j.status !== 'done' ? html` <span class=${`pill ${j.status}`}>${STATUS_LABEL[j.status]}</span>` : ''}</button>`)}
      </div>`}
      <details class="cmd-line">
        <summary>Command</summary>
        <div class="cmd-body" title="Exactly what ran; paste into a shell to reproduce">
          <code>${run.command}</code>
          <button class="btn-icon" onClick=${() => copy(run.command)} title="Copy">⧉</button>
        </div>
      </details>
      ${run.status === 'failed' && run.error && html`<pre class="error-box">${run.error}</pre>`}
      ${Object.keys(job.live_skipped || {}).length > 0 && html`<p class="level-note small">
        ${Object.keys(job.live_skipped).length} species more than ${job.params?.explore_within ?? job.params?.energy_window} kcal/mol above the seed
        were not added to Explore (the threshold under Explore in this calculation's settings). They are in the results, to add by hand.</p>`}
      ${['cancelled', 'interrupted'].includes(run.status) && html`<p class="warn-box small">${run.error} ${!run.external && html`<button class="btn small" onClick=${() => attempt(() => api.post(`/api/jobs/${run.id}/retry`))}>Resume</button>`}</p>`}
      <div class="tabs">
        ${[['result', 'Results'], job.op === 'nanoreactor' && ['reactor', 'Reactor'], !job.external && job.op !== 'nanoreactor' && ['live', 'Live'], !job.external && family.some((j) => ['channels', 'channels-more'].includes(j.op)) && ['map', 'Path map'],
          ['ts', 'channels', 'channels-more', 'network-splits'].includes(job.op) && ['tree', 'Optimization tree'],
          !job.external && ['log', 'Log'], ['files', 'Files']].filter(Boolean)
          .map(([k, l]) => html`<button class=${current === k ? 'on' : ''} disabled=${!hasOutput && k !== 'live' && k !== 'reactor'} onClick=${() => setTab(k)}>${l}</button>`)}
      </div>
      <div class="tab-body">
        ${current === 'result' && html`<${ResultPanel} job=${page} />`}
        ${current === 'live' && (job.op === 'complex' ? html`<${ComplexLive} key=${run.id} job=${run} />` : html`<${LivePanel} key=${run.id} job=${run} />`)}
        ${current === 'reactor' && html`<${ReactorLive} key=${run.id} job=${run} />`}
        ${current === 'map' && html`<${ChannelsMap} job=${page} />`}
        ${current === 'tree' && html`<${OptTree} job=${page} />`}
        ${current === 'log' && html`<${LogPanel} key=${run.id} job=${run} />`}
        ${current === 'files' && html`<${FilesPanel} key=${run.id} job=${run} />`}
      </div>
    </div>`;
}
