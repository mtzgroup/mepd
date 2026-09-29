// Optimization tree of an MSMEP run: every path search, how splits made
// children, why each branch ended, and each search's optimization step by
// step (all steps' energy profiles overlaid, a step scrubber, barrier and
// path movement per step, the geometry of any image). The data comes from
// mepd/web/opt_tree.py: the finished tree/ folders, or the live record while
// the run goes (or after it died).
import { html, useEffect, useMemo, useRef, useState } from '../lib.js';
import { api } from '../api.js';
import { Viewer3D } from './Viewer3D.js';

const OUTCOME = {
  elementary: ['Elementary', 'ok'],
  split: ['Split', 'split'],
  unresolved: ['Unresolved', 'warn'],
  failed: ['Failed', 'bad'],
  interrupted: ['Interrupted', 'bad'],
  running: ['Running', 'run'],
  skipped: ['Skipped', 'dim'],
  rejected: ['Not run', 'dim'],
};
const outcomeOf = (o) => OUTCOME[o] || [o || '?', 'dim'];
const fmt = (v, d = 1) => (v == null || !isFinite(v) ? '–' : v.toFixed(d));

function useWidth(min = 200) {
  const ref = useRef(null);
  const [w, setW] = useState(480);
  useEffect(() => {
    const ro = new ResizeObserver(([e]) => setW(Math.max(min, Math.round(e.contentRect.width))));
    if (ref.current) ro.observe(ref.current);
    return () => ro.disconnect();
  }, []);
  return [ref, w];
}

// ------------------------------------------------------------ the tree
const NW = 136, NH = 52, GX = 16, GY = 42, PAD = 12;

function layout(nodes) {
  const kids = {};
  const byKey = Object.fromEntries(nodes.map((n) => [n.key, n]));
  const roots = [];
  for (const n of nodes) {
    if (n.parent == null || !byKey[n.parent]) roots.push(n);
    else (kids[n.parent] ||= []).push(n);
  }
  Object.values(kids).forEach((l) => l.sort((a, b) => a.key - b.key));
  const pos = {};
  let leaf = 0, maxDepth = 0;
  const place = (n, depth) => {
    maxDepth = Math.max(maxDepth, depth);
    const ch = kids[n.key] || [];
    if (!ch.length) pos[n.key] = { x: leaf++, depth };
    else {
      ch.forEach((c) => place(c, depth + 1));
      pos[n.key] = { x: (pos[ch[0].key].x + pos[ch[ch.length - 1].key].x) / 2, depth };
    }
  };
  roots.forEach((r) => place(r, 0));
  const px = (p) => PAD + p.x * (NW + GX), py = (p) => PAD + p.depth * (NH + GY);
  return {
    pos: Object.fromEntries(Object.entries(pos).map(([k, p]) => [k, { x: px(p), y: py(p) }])),
    width: PAD * 2 + Math.max(1, leaf) * (NW + GX) - GX,
    height: PAD * 2 + (maxDepth + 1) * (NH + GY) - GY,
  };
}

function TreeDiagram({ nodes, selected, onSelect }) {
  const { pos, width, height } = useMemo(() => layout(nodes), [nodes]);
  return html`<div class="otree-canvas">
    <svg width=${width} height=${height} viewBox="0 0 ${width} ${height}">
      ${nodes.filter((n) => n.parent != null && pos[n.parent]).map((n) => {
        const a = pos[n.parent], b = pos[n.key];
        const x1 = a.x + NW / 2, y1 = a.y + NH, x2 = b.x + NW / 2, y2 = b.y, my = (y1 + y2) / 2;
        return html`<path class=${`otree-edge ${n.outcome === 'skipped' || n.outcome === 'rejected' ? 'dim' : ''}`}
          d=${`M${x1},${y1} C${x1},${my} ${x2},${my} ${x2},${y2}`} />`;
      })}
      ${nodes.map((n) => {
        const p = pos[n.key];
        const [label, tone] = outcomeOf(n.outcome);
        const sub = n.outcome === 'running' ? 'in progress…'
          : n.n_steps ? `${n.n_steps} step${n.n_steps === 1 ? '' : 's'}${n.barrier_kcal != null ? ` · ${fmt(n.barrier_kcal)} kcal` : ''}`
          : n.leaf_status ? n.leaf_status.replaceAll('_', ' ') : 'no search';
        return html`<g class=${`otree-node t-${tone} ${selected === n.key ? 'sel' : ''}`} transform=${`translate(${p.x},${p.y})`}
          onClick=${() => onSelect(n.key)}>
          <title>${n.why || ''}</title>
          <rect class="card" width=${NW} height=${NH} rx="7" />
          <rect class="stripe" width="5" height=${NH} rx="2" />
          <text x="14" y="20" class="k">#${n.key}</text>
          <text x=${NW - 10} y="20" class="o" text-anchor="end">${label}${n.converged === false && n.n_steps ? ' ·⚠' : ''}</text>
          <text x="14" y="39" class="s">${sub}</text>
        </g>`;
      })}
    </svg>
  </div>`;
}

// ------------------------------------------------------------ step plots
function Profiles({ steps, step, frame, onFrame, onStep }) {
  const [box, W] = useWidth();
  const H = 230, m = { l: 46, r: 12, t: 12, b: 28 };
  const all = steps.flatMap((s) => s.e || []);
  if (!all.length) return html`<div ref=${box} class="plot-empty" style=${{ height: 80 }}>No energies recorded for this search</div>`;
  let lo = Math.min(...all), hi = Math.max(...all);
  const pad = (hi - lo || 1) * 0.08; lo -= pad; hi += pad;
  const sx = (x) => m.l + x * (W - m.l - m.r), sy = (y) => m.t + (1 - (y - lo) / (hi - lo)) * (H - m.t - m.b);
  const path = (s) => s.e.map((e, i) => `${i ? 'L' : 'M'}${sx(s.s[i]).toFixed(1)},${sy(e).toFixed(1)}`).join(' ');
  // Up to ~80 background steps, always the first and last.
  const every = Math.max(1, Math.ceil(steps.length / 80));
  const bg = steps.map((s, k) => [s, k]).filter(([s, k]) => s.e && (k % every === 0 || k === steps.length - 1) && k !== step);
  const cur = steps[step];
  const nTicks = 5, tick = (hi - lo) / nTicks, mag = 10 ** Math.floor(Math.log10(tick || 1));
  const st = [1, 2, 5, 10].map((q) => q * mag).find((q) => q >= tick) || tick;
  const ticks = []; for (let v = Math.ceil(lo / st) * st; v <= hi; v += st) ticks.push(+v.toFixed(8));
  const pick = (evt) => {
    const r = evt.currentTarget.getBoundingClientRect(), x = ((evt.clientX - r.left) / r.width) * W;
    let best = 0;
    cur.s.forEach((s, i) => { if (Math.abs(sx(s) - x) < Math.abs(sx(cur.s[best]) - x)) best = i; });
    onFrame(best);
  };
  return html`<div ref=${box} class="otree-prof">
    <svg width="100%" height=${H} viewBox="0 0 ${W} ${H}" onPointerDown=${(e) => cur?.e && pick(e)}
      onPointerMove=${(e) => e.buttons && cur?.e && pick(e)} style="touch-action:none;cursor:crosshair">
      ${ticks.map((t) => html`<g><line class="grid" x1=${m.l} x2=${W - m.r} y1=${sy(t)} y2=${sy(t)} />
        <text class="tick" x=${m.l - 6} y=${sy(t) + 3} text-anchor="end">${t}</text></g>`)}
      <text class="tick" x=${m.l} y=${H - 8}>reactant</text>
      <text class="tick" x=${W - m.r} y=${H - 8} text-anchor="end">product</text>
      <text class="tick" x=${(m.l + W - m.r) / 2} y=${H - 8} text-anchor="middle">path (normalized length) · kcal/mol</text>
      ${bg.map(([s, k]) => html`<path class="bgline" d=${path(s)} style=${`opacity:${(0.08 + 0.4 * (k + 1) / steps.length).toFixed(3)}`}
        onClick=${(e) => { e.stopPropagation(); onStep(k); }} />`)}
      ${cur?.e && html`<path class="curline" d=${path(cur)} />`}
      ${cur?.e && cur.e.map((e, i) => html`<circle class=${`pt ${i === cur.ts ? 'ts' : ''} ${i === frame ? 'on' : ''}`}
        cx=${sx(cur.s[i])} cy=${sy(e)} r=${i === frame ? 5.5 : i === cur.ts ? 4.5 : 3} />`)}
    </svg>
  </div>`;
}

function Spark({ label, ys, step, onStep, unit, log = false, digits = 1 }) {
  const [box, W] = useWidth(160);
  const H = 54, m = { l: 6, r: 6, t: 6, b: 6 };
  const pts = ys.map((y, i) => [i, y]).filter(([, y]) => y != null && isFinite(y) && (!log || y > 0));
  const tf = (y) => (log ? Math.log10(y) : y);
  const vals = pts.map(([, y]) => tf(y));
  const lo = Math.min(...vals), hi = Math.max(...vals);
  const n = Math.max(1, ys.length - 1);
  const sx = (i) => m.l + (i / n) * (W - m.l - m.r), sy = (v) => m.t + (1 - (v - lo) / ((hi - lo) || 1)) * (H - m.t - m.b);
  const pick = (evt) => {
    const r = evt.currentTarget.getBoundingClientRect();
    onStep(Math.max(0, Math.min(ys.length - 1, Math.round(((evt.clientX - r.left) / r.width * W - m.l) / (W - m.l - m.r) * n))));
  };
  return html`<div class="otree-spark" ref=${box}>
    <div class="small"><span class="muted">${label}</span> <b>${ys[step] == null ? '–' : fmt(ys[step], digits)}</b> <span class="muted">${unit}</span></div>
    ${pts.length > 1 ? html`<svg width="100%" height=${H} viewBox="0 0 ${W} ${H}" onPointerDown=${pick}
        onPointerMove=${(e) => e.buttons && pick(e)} style="touch-action:none;cursor:pointer">
      <path class="sparkline" d=${pts.map(([i, y], k) => `${k ? 'L' : 'M'}${sx(i).toFixed(1)},${sy(tf(y)).toFixed(1)}`).join(' ')} />
      <line class="cursor" x1=${sx(step)} x2=${sx(step)} y1="0" y2=${H} />
    </svg>` : html`<div class="plot-empty" style=${{ height: H }}>—</div>`}
  </div>`;
}

// ------------------------------------------------------------ one node
function NodeDetail({ jobId, tree, node, poll }) {
  const [detail, setDetail] = useState(null);
  const [step, setStep] = useState(null);
  const [frame, setFrame] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [err, setErr] = useState(null);
  const token = useRef(0);
  const base = `/api/jobs/${jobId}/trees/${encodeURIComponent(tree.id)}/nodes/${node.key}`;

  const load = (k, keepFrame = false) => {
    const t = ++token.current;
    return api.get(k == null ? base : `${base}?step=${k}`).then((d) => {
      if (t !== token.current) return;
      setDetail(d); setErr(null);
      setStep(d.step);
      if (!keepFrame) setFrame(d.steps[d.step]?.ts ?? 0);
    }).catch((e) => t === token.current && setErr(e.message));
  };
  useEffect(() => { setDetail(null); setStep(null); setPlaying(false); load(null); }, [jobId, tree.id, node.key]);
  // A running node: refresh the live chain; a finished one only if it changed.
  useEffect(() => {
    if (!poll) return undefined;
    const id = setInterval(() => load(playing ? step : null, true), 3000);
    return () => clearInterval(id);
  }, [poll, base, playing, step]);
  useEffect(() => {
    if (!playing || !detail) return undefined;
    const last = detail.steps.length - 1;
    if (step >= last) { setPlaying(false); return undefined; }
    const id = setTimeout(() => load(step + 1), 110);
    return () => clearTimeout(id);
  }, [playing, step, detail]);

  const [label, tone] = outcomeOf(node.outcome);
  const steps = detail?.steps || [];
  const cur = steps[step];
  const live = detail?.live;
  const kindNote = { initial: 'Only the starting path was saved: the search had not finished a step when this record was written.',
    failed: 'The path this branch started from (it failed before saving an optimization).',
    rejected: 'The path that was split but not run.', final: 'Only the final path was saved (no step history).' }[detail?.kind];

  return html`<div class="otree-detail">
    <div class="otree-dhead">
      <span class=${`otag t-${tone}`}>${label}</span>
      <h3>Node #${node.key}${node.depth ? html` <span class="muted small">depth ${node.depth}${node.parent != null ? `, from #${node.parent}` : ''}</span>` : html` <span class="muted small">root</span>`}</h3>
    </div>
    <p class="otree-why">${node.why}</p>
    ${node.converged === false && node.n_steps > 0 && html`<p class="warn-box small">This path search ended before its images met the convergence
      criteria (e.g. an early elementary-step check, or the step limit): the outcome above was decided on that path.</p>`}
    ${node.error && html`<div class="error-box small"><b>${node.error}</b>
      ${node.traceback && html`<details><summary>Traceback</summary><pre>${node.traceback}</pre></details>`}</div>`}
    ${err && html`<p class="error-box small">${err}</p>`}
    ${live && html`<div class="otree-live">
      <div class="small"><span class="otag t-run">live</span> ${live.caption || 'relaxing'}${live.status ? ` · ${live.status}` : ''}</div>
      <p class="small muted">The chain this search is relaxing now. Its step-by-step history appears here when the search finishes.</p>
      <div class="otree-pair">
        <${Profiles} steps=${[{ s: live.s || live.frames.map((_, i) => i / Math.max(1, live.frames.length - 1)), e: live.e, ts: live.ts }]}
          step=${0} frame=${frame} onFrame=${setFrame} onStep=${() => {}} />
        <${Viewer3D} frames=${live.frames} frame=${Math.min(frame, live.frames.length - 1)} height=${250} />
      </div>
    </div>`}
    ${!detail && !err && html`<p class="muted">Loading…</p>`}
    ${detail && !steps.length && !live && html`<p class="muted">No path was saved for this node.</p>`}
    ${steps.length > 0 && html`<div>
      ${kindNote && html`<p class="small muted">${kindNote}</p>`}
      ${steps.length > 1 && html`<div class="otree-scrub">
        <button class="btn small" onClick=${() => { if (step >= steps.length - 1) load(0); setPlaying(!playing); }}>${playing ? '❚❚ Pause' : '▶ Play'}</button>
        <input type="range" min="0" max=${steps.length - 1} value=${step ?? 0} onInput=${(e) => { setPlaying(false); load(+e.target.value); }} />
        <span class="small mono">step ${(step ?? 0) + 1} / ${steps.length}</span>
      </div>`}
      <div class="otree-pair">
        <${Profiles} steps=${steps} step=${step ?? 0} frame=${frame} onFrame=${setFrame} onStep=${(k) => { setPlaying(false); load(k); }} />
        <div>
          ${detail.frames?.length > 0 && html`<${Viewer3D} frames=${detail.frames} frame=${Math.min(frame, detail.frames.length - 1)} height=${230} />`}
          <div class="small muted otree-frame">image ${frame + 1} / ${cur?.n ?? 0}${cur?.e ? ` · ${fmt(cur.e[frame], 2)} kcal/mol` : ''}${frame === cur?.ts ? ' · highest image' : ''}
            ${cur?.grad_rms && html` · |g| ${cur.grad_rms[frame].toExponential(1)} Ha/bohr`}</div>
          ${cur && cur.n > 1 && html`<input class="otree-frame-slider" type="range" min="0" max=${cur.n - 1} value=${frame} onInput=${(e) => setFrame(+e.target.value)} />`}
        </div>
      </div>
      ${steps.length > 1 && html`<div class="otree-sparks">
        <${Spark} label="Barrier" unit="kcal/mol" ys=${steps.map((s) => s.barrier)} step=${step ?? 0} onStep=${(k) => { setPlaying(false); load(k); }} />
        <${Spark} label="Path moved" unit="Å rms" log digits=${4} ys=${steps.map((s) => s.moved)} step=${step ?? 0} onStep=${(k) => { setPlaying(false); load(k); }} />
        ${steps.some((s) => s.ts_grad_rms != null) && html`<${Spark} label="|g| at highest image" unit="Ha/bohr" log digits=${4}
          ys=${steps.map((s) => s.ts_grad_rms)} step=${step ?? 0} onStep=${(k) => { setPlaying(false); load(k); }} />`}
      </div>`}
    </div>`}
  </div>`;
}

// ------------------------------------------------------------ panel
export function OptTree({ job }) {
  const [trees, setTrees] = useState(null);
  const [tid, setTid] = useState(null);
  const [tree, setTree] = useState(null);
  const [sel, setSel] = useState(null);
  const [err, setErr] = useState(null);
  const running = job.status === 'running';

  const loadList = () => api.get(`/api/jobs/${job.id}/trees`).then((l) => {
    setTrees(l);
    setTid((t) => (t && l.some((x) => x.id === t) ? t : l.find((x) => x.running)?.id || l[0]?.id || null));
  }).catch((e) => setErr(e.message));
  const loadTree = (id) => id && api.get(`/api/jobs/${job.id}/trees/${encodeURIComponent(id)}`).then((t) => {
    setTree(t); setErr(null);
    setSel((s) => (s != null && t.nodes.some((n) => n.key === s) ? s
      : (t.nodes.find((n) => n.running) || t.nodes.find((n) => ['failed', 'interrupted'].includes(n.outcome)) || t.nodes[0])?.key ?? null));
  }).catch((e) => setErr(e.message));

  useEffect(() => { setTid(null); setTree(null); setSel(null); loadList(); }, [job.id, job.status]);
  useEffect(() => { setTree(null); loadTree(tid); }, [tid]);
  useEffect(() => {
    if (!running) return undefined;
    const id = setInterval(() => { loadList(); loadTree(tid); }, 3000);
    return () => clearInterval(id);
  }, [running, tid]);

  if (err && !trees) return html`<p class="error-box small">${err}</p>`;
  if (!trees) return html`<p class="muted">Looking for optimization trees…</p>`;
  if (!trees.length) {
    return html`<div class="empty-hint"><p>No optimization tree in this calculation${running ? ' yet' : ''}.</p>
      <p class="small muted">Recursive path searches (MSMEP) record one: each path search, why it split or stopped, and every
      optimization step. ${running ? 'It appears once the first search starts.' : 'Runs made before this view existed only have it once they finished.'}</p></div>`;
  }
  const node = tree?.nodes.find((n) => n.key === sel);
  const counts = (c) => Object.entries(c || {}).map(([o, n]) => html`<span class=${`odot t-${outcomeOf(o)[1]}`} title=${outcomeOf(o)[0]}>${n}</span>`);
  return html`<div class=${`otree ${trees.length > 1 ? 'multi' : ''}`}>
    ${trees.length > 1 && html`<div class="otree-list">
      ${trees.map((t) => html`<button class=${`otree-item ${t.id === tid ? 'on' : ''}`} onClick=${() => setTid(t.id)}>
        <span>${t.label}${t.running ? html` <span class="otag t-run">live</span>` : ''}</span>
        <span class="otree-counts">${t.error ? html`<span class="small muted">unreadable</span>` : counts(t.counts)}</span>
      </button>`)}
    </div>`}
    <div class="otree-main">
      ${!tree && html`<p class="muted">Loading…</p>`}
      ${tree && html`<div>
        <div class="otree-head small muted">
          <b class="otree-title">${tree.label}</b> · ${tree.nodes.length} node${tree.nodes.length === 1 ? '' : 's'}
          ${tree.live ? (tree.running ? ' · recording live' : ' · from the live record (the run did not write its final tree)') : ''}
          <span class="otree-legend">${['elementary', 'split', 'unresolved', 'failed', 'skipped'].map((o) =>
            html`<span><i class=${`sw t-${outcomeOf(o)[1]}`}></i>${outcomeOf(o)[0]}</span>`)}</span>
        </div>
        <${TreeDiagram} nodes=${tree.nodes} selected=${sel} onSelect=${setSel} />
        ${node && html`<${NodeDetail} jobId=${job.id} tree=${tree} node=${node} poll=${running && node.running} />`}
      </div>`}
    </div>
  </div>`;
}
