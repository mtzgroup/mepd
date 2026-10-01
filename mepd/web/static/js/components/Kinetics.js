// Analyze › Kinetics: the network as a microkinetic model (mepd.microkinetics
// via /api/kinetics). Start from a mixture, in a closed flask or with a
// constant feed, and read: how much of the target forms, which barriers
// control it (degree of rate control: lower these), which species are
// traps, where the material flows, and how it all moves with temperature.
import { html, useMemo, useState } from '../lib.js';
import { api, attempt } from '../api.js';
import { set, state, useStore } from '../store.js';
import { depictUrl, isSpecies } from '../util.js';

const fmtC = (x) => (x == null ? '—' : x === 0 ? '0' : Math.abs(x) >= 0.01 && Math.abs(x) < 1000 ? x.toPrecision(3) : x.toExponential(1));
const UNITS = { s: 1, min: 60, h: 3600, d: 86400 };
const RT = (T) => (8.314462618 * T) / 4184;

// ------------------------------------------------------------ charts
// Lines on log-log axes (time vs amount). Series colours: the validated
// categorical slots, in fixed order by the species' rank at the start.
function LineChart({ xs, series, xLabel, yLabel, floor = 1e-12, xlog = true, width = 560, height = 240 }) {
  const [hover, setHover] = useState(null);
  const pad = { l: 56, r: 12, t: 10, b: 34 };
  const W = width - pad.l - pad.r, H = height - pad.t - pad.b;
  const lx = (x) => (xlog ? Math.log10(Math.max(x, 1e-300)) : x);
  const ly = (y) => Math.log10(Math.max(y, floor));
  const xsF = xs.filter((x) => !xlog || x > 0);
  const x0 = Math.min(...xsF.map(lx)), x1 = Math.max(...xsF.map(lx));
  const ys = series.flatMap((s) => s.values).filter((v) => v > 0);
  const y1 = Math.ceil(Math.log10(Math.max(...ys, floor * 10)));
  const y0 = Math.max(Math.floor(Math.log10(Math.min(...ys, 1))), Math.log10(floor));
  const X = (x) => pad.l + ((lx(x) - x0) / Math.max(1e-9, x1 - x0)) * W;
  const Y = (y) => pad.t + (1 - (ly(y) - y0) / Math.max(1e-9, y1 - y0)) * H;
  const ticksY = [];
  const stepY = Math.max(1, Math.ceil((y1 - y0) / 5));
  for (let e = y0; e <= y1; e += stepY) ticksY.push(e);
  const ticksX = [];
  if (xlog) { const sx = Math.max(1, Math.ceil((x1 - x0) / 6)); for (let e = Math.ceil(x0); e <= x1; e += sx) ticksX.push(e); }
  else { for (let k = 0; k <= 5; k += 1) ticksX.push(x0 + ((x1 - x0) * k) / 5); }
  const pts = (vals) => xs.map((x, k) => (xlog && x <= 0 ? null : `${X(x).toFixed(1)},${Y(vals[k]).toFixed(1)}`)).filter(Boolean).join(' ');
  const onMove = (e) => {
    const box = e.currentTarget.getBoundingClientRect();
    const px = ((e.clientX - box.left) / box.width) * width;
    let best = 0, bd = Infinity;
    xs.forEach((x, k) => { if (xlog && x <= 0) return; const d = Math.abs(X(x) - px); if (d < bd) { bd = d; best = k; } });
    setHover(best);
  };
  return html`<div class="kn-chart">
    <svg viewBox=${`0 0 ${width} ${height}`} role="img" aria-label=${`${yLabel} against ${xLabel}`}
      onMouseMove=${onMove} onMouseLeave=${() => setHover(null)}>
      ${ticksY.map((e) => html`<line class="kn-grid" x1=${pad.l} x2=${pad.l + W} y1=${Y(10 ** e)} y2=${Y(10 ** e)} />
        <text class="kn-tick" x=${pad.l - 6} y=${Y(10 ** e) + 4} text-anchor="end">1e${e}</text>`)}
      ${ticksX.map((e) => html`<text class="kn-tick" x=${X(xlog ? 10 ** e : e)} y=${pad.t + H + 16} text-anchor="middle">${xlog ? `1e${e}` : Math.round(e)}</text>`)}
      <text class="kn-axis" x=${pad.l + W / 2} y=${height - 2} text-anchor="middle">${xLabel}</text>
      <text class="kn-axis" transform=${`translate(12 ${pad.t + H / 2}) rotate(-90)`} text-anchor="middle">${yLabel}</text>
      ${series.map((s) => html`<polyline class="kn-line" style=${{ stroke: `var(--series-${s.slot})` }} points=${pts(s.values)} />`)}
      ${hover != null && html`<line class="kn-cross" x1=${X(xs[hover])} x2=${X(xs[hover])} y1=${pad.t} y2=${pad.t + H} />
        ${series.map((s) => html`<circle class="kn-dot" cx=${X(xs[hover])} cy=${Y(s.values[hover])} r="4" style=${{ fill: `var(--series-${s.slot})` }} />`)}`}
    </svg>
    <div class="kn-legend">${series.map((s) => html`<span><i style=${{ background: `var(--series-${s.slot})` }}></i>${s.name}</span>`)}</div>
    ${hover != null && html`<div class="kn-tip">${xLabel.split(' (')[0]} ${fmtC(xs[hover])}:
      ${series.map((s) => html` <span><i style=${{ background: `var(--series-${s.slot})` }}></i>${s.name} ${fmtC(s.values[hover])}</span>`)}</div>`}
  </div>`;
}

// Diverging bars: positive (blue) speeds the target up when that energy is lowered.
function ControlBars({ rows, onPick, T }) {
  const max = Math.max(0.05, ...rows.map((r) => Math.abs(r.x)));
  return html`<div class="kn-bars">${rows.map((r) => {
    const w = (Math.abs(r.x) / max) * 50;
    const factor = Math.exp(r.x / RT(T));
    return html`<button class="kn-bar-row" onClick=${() => onPick && onPick(r)}
      title=${`Lowering it by 1 kcal/mol multiplies the target by ${factor >= 1 ? factor.toFixed(2) : factor.toFixed(3)}`}>
      <span class="kn-bar-label">${r.label}</span>
      <span class="kn-bar-track"><span class="kn-bar-mid"></span>
        <span class=${`kn-bar ${r.x >= 0 ? 'pos' : 'neg'}`} style=${{ width: `${w}%`, [r.x >= 0 ? 'left' : 'right']: '50%' }}></span></span>
      <span class="kn-bar-v mono">${r.x >= 0 ? '+' : ''}${r.x.toFixed(2)}</span>
    </button>`;
  })}</div>`;
}

// ------------------------------------------------------------ setup
function Mixture({ items, onChange, species }) {
  const [q, setQ] = useState('');
  const [open, setOpen] = useState(false);
  const ql = q.trim().toLowerCase();
  const hits = ql && open ? species.filter((s) => `${s.name} ${s.smiles}`.toLowerCase().includes(ql) && !items.some((i) => i.id === s.id)).slice(0, 8) : [];
  const up = (k, patch) => onChange(items.map((it, i) => (i === k ? { ...it, ...patch } : it)));
  return html`<div class="kn-mix">
    ${items.map((it, k) => {
      const rec = state.workspace.structures[it.id];
      return html`<div class="kn-mix-row">
        ${rec?.smiles && html`<img src=${depictUrl(rec.smiles, 60, 45)} alt="" />`}
        <span class="kn-mix-name">${rec?.name || '?'}</span>
        <input type="number" min="0" step="0.1" value=${it.c} title="Starting amount (mol/L)" onInput=${(e) => up(k, { c: +e.target.value })} />
        <span class="small muted">M</span>
        <label class="small" title="Constant feed: held at this amount throughout"><input type="checkbox" checked=${it.held} onChange=${(e) => up(k, { held: e.target.checked })} /> fixed</label>
        <button class="btn-icon" title="Remove" onClick=${() => onChange(items.filter((_, i) => i !== k))}>✕</button>
      </div>`;
    })}
    <div class="cp-search"><input type="search" placeholder="add a species to the start…" value=${q}
      onInput=${(e) => { setQ(e.target.value); setOpen(true); }} onFocus=${() => setOpen(true)} onBlur=${() => setTimeout(() => setOpen(false), 150)} />
      ${hits.length > 0 && html`<ul class="cp-hits">${hits.map((s) => html`<li><button onMouseDown=${(e) => e.preventDefault()}
        onClick=${() => { onChange([...items, { id: s.id, c: 1, held: false }]); setQ(''); setOpen(false); }}>
        ${s.smiles && html`<img src=${depictUrl(s.smiles, 60, 45)} alt="" />`}<span>${s.name}</span></button></li>`)}</ul>`}</div>
  </div>`;
}

function defaultMixture() {
  // The start of the latest nanoreactor run, 1 M each.
  const jobs = Object.values(state.jobs).filter((j) => j.op === 'nanoreactor').sort((a, b) => b.created - a.created);
  const ids = (jobs[0]?.targets?.structures || []).filter((id) => state.workspace.structures[id]);
  return ids.map((id) => ({ id, c: 1, held: false }));
}

// ------------------------------------------------------------ view
export function KineticsView() {
  const structures = useStore((s) => s.workspace.structures);
  const reactions = useStore((s) => s.workspace.reactions || {});
  const species = useMemo(() => {
    const ids = new Set(Object.values(reactions).flatMap((r) => [...r.reactants, ...r.products]));
    return Object.values(structures).filter((s) => isSpecies(s) && (ids.has(s.id) || s.energy != null))
      .sort((a, b) => (ids.has(b.id) - ids.has(a.id)) || a.name.localeCompare(b.name));
  }, [structures, reactions]);
  const saved = state.kinetics || {};
  const [mix, setMix] = useState(saved.mix || defaultMixture());
  const [T, setT] = useState(saved.T ?? 298.15);
  const [tVal, setTVal] = useState(saved.tVal ?? 1);
  const [tUnit, setTUnit] = useState(saved.tUnit ?? 'h');
  const [target, setTarget] = useState(saved.target || '');
  const [what, setWhat] = useState(saved.what || 'amount');
  const [unverified, setUnverified] = useState(saved.unverified ?? true);
  const [sweepOn, setSweepOn] = useState(saved.sweepOn ?? true);
  const [tLo, setTLo] = useState(saved.tLo ?? 250);
  const [tHi, setTHi] = useState(saved.tHi ?? 650);
  const [res, setRes] = useState(saved.res || null);
  const [busy, setBusy] = useState(false);
  const run = async () => {
    setBusy(true);
    const sweep = sweepOn ? Array.from({ length: 9 }, (_, k) => tLo + ((tHi - tLo) * k) / 8) : [];
    const out = await attempt(() => api.post('/api/kinetics', {
      initial: Object.fromEntries(mix.map((m) => [m.id, m.c])), held: mix.filter((m) => m.held).map((m) => m.id),
      temperature: T, time_s: tVal * UNITS[tUnit], target: target || null, include_unverified: unverified,
      control: !!target, control_what: what, sweep }));
    setBusy(false);
    if (out) setRes(out);
    set({ kinetics: { mix, T, tVal, tUnit, target, what, unverified, sweepOn, tLo, tHi, res: out || res } });
  };
  const openReaction = (rid) => set({ analyze: { ...(state.analyze || {}), view: 'reactions', reaction: rid } });

  return html`<div class="kinetics">
    <div class="kn-setup">
      <div class="section-title">Start</div>
      <${Mixture} items=${mix} onChange=${setMix} species=${species} />
      <p class="small muted">"fixed" = a constant feed (held at that amount): the run then heads to a steady state. Without any, it is a closed flask.</p>
      <div class="kn-row"><label>Temperature <input type="number" value=${T} min="1" step="10" onInput=${(e) => setT(+e.target.value)} /> K</label></div>
      <div class="kn-row"><label>Time <input type="number" value=${tVal} min="0" step="1" onInput=${(e) => setTVal(+e.target.value)} /></label>
        <select value=${tUnit} onChange=${(e) => setTUnit(e.target.value)}>${Object.keys(UNITS).map((u) => html`<option value=${u}>${u}</option>`)}</select></div>
      <div class="kn-row"><label>Target <select value=${target} onChange=${(e) => setTarget(e.target.value)}>
        <option value="">— none —</option>
        ${species.map((s) => html`<option value=${s.id}>${s.name}</option>`)}</select></label></div>
      ${target && html`<div class="kn-row segmented">
        <button class=${what === 'amount' ? 'on' : ''} onClick=${() => setWhat('amount')} title="How much of the target there is at the end">amount at the end</button>
        <button class=${what === 'rate' ? 'on' : ''} onClick=${() => setWhat('rate')} title="How fast it forms at the end (with a feed: the steady-state rate)">formation rate</button>
      </div>`}
      <label class="kn-row small"><input type="checkbox" checked=${sweepOn} onChange=${(e) => setSweepOn(e.target.checked)} />
        temperature sweep <input type="number" class="tiny" value=${tLo} onInput=${(e) => setTLo(+e.target.value)} />–<input type="number" class="tiny" value=${tHi} onInput=${(e) => setTHi(+e.target.value)} /> K</label>
      <label class="kn-row small" title="Barriers that are only a path maximum (no TS/IRC confirmed them)"><input type="checkbox" checked=${unverified} onChange=${(e) => setUnverified(e.target.checked)} /> include unverified barriers</label>
      <button class="btn primary" disabled=${busy || !mix.length} onClick=${run}>${busy ? 'Running…' : 'Run kinetics'}</button>
    </div>
    <div class="kn-results">${res ? html`<${Results} res=${res} T=${T} onReaction=${openReaction} />`
      : html`<div class="empty-hint"><p>Pick a starting mixture and (optionally) a target species, then run. The model uses every
          reaction with a barrier: the lower its TS, the faster it goes, forward and back.</p></div>`}</div>
  </div>`;
}

function Results({ res, T, onReaction }) {
  const [showAll, setShowAll] = useState(false);
  if (res.error) return html`<p class="warn-box">${res.error}</p>${res.excluded?.length > 0 && html`<${Excluded} list=${res.excluded} />`}`;
  const byId = Object.fromEntries(res.species.map((s) => [s.id, s]));
  // Series: the species that ever reach a visible amount, largest first; fixed colour by that order.
  const ranked = [...res.species].sort((a, b) => b.max - a.max);
  const shown = ranked.filter((s) => s.max > 1e-12).slice(0, 6);
  const series = shown.map((s, k) => ({ name: s.name, slot: k + 1, values: res.series[s.id] }));
  const tgt = res.target;
  const total = res.species.reduce((a, s) => a + s.c0, 0);
  const allCtl = res.control ? res.steps.map((st, k) => ({ ...st, x: res.control.steps[k] })).filter((r) => Number.isFinite(r.x)) : [];
  const ctl = allCtl.filter((r) => Math.abs(r.x) >= 0.01).sort((a, b) => Math.abs(b.x) - Math.abs(a.x)).slice(0, 8);
  const quiet = allCtl.length - ctl.length;
  // Intermediates and other products only: stabilizing a starting material always slows things (trivially).
  const traps = res.control ? res.species.map((s, k) => ({ label: s.name, x: res.control.species[k], id: s.id, c0: s.c0 }))
    .filter((r) => Number.isFinite(r.x) && Math.abs(r.x) > 0.02 && !(r.c0 > 0) && r.id !== tgt?.id)
    .sort((a, b) => Math.abs(b.x) - Math.abs(a.x)).slice(0, 6) : [];
  const flows = [...res.steps].sort((a, b) => Math.abs(b.extent) - Math.abs(a.extent));
  const timeLabel = res.time_s >= 86400 ? `${+(res.time_s / 86400).toFixed(2)} d` : res.time_s >= 3600 ? `${+(res.time_s / 3600).toFixed(2)} h` : `${+res.time_s.toPrecision(3)} s`;
  return html`<div class="kn-out">
    ${tgt && html`<div class="kn-head">
      <div class="stat"><span class="stat-v">${fmtC(tgt.final)} M</span><span class="stat-l">${tgt.name} after ${timeLabel} at ${Math.round(res.temperature)} K${total ? ` · ${fmtC((100 * tgt.final) / total)}% of the start` : ''}</span></div>
      <div class="stat"><span class="stat-v">${fmtC(tgt.rate)} M/s</span><span class="stat-l">formation rate ${res.mode === 'constant feed' ? (res.steady ? '(steady state)' : '(not yet steady)') : 'at the end'}</span></div>
    </div>`}
    <p class="small muted">${res.mode}${res.mode === 'batch' ? '' : res.steady ? ': steady state reached' : ': not yet steady, run longer'} · ${res.steps.length} reaction steps in the model.</p>

    ${res.control && !ctl.length && html`<section>
      <div class="section-title">What controls ${tgt.name}</div>
      <p class="small">${`No barrier controls it here: by ${timeLabel} at ${Math.round(res.temperature)} K the network has reached ${
        res.mode === 'batch' ? 'equilibrium' : 'a steady state set by the feed'}, so the amount of ${tgt.name} is under thermodynamic control, set by the species' energies. To make more: stabilize ${
        tgt.name} or destabilize what competes with it${traps.length ? ' (below)' : ''}. Use a shorter time or a lower temperature to see which barriers matter before that.`}</p>
      ${traps.length > 0 && html`<${ControlBars} rows=${traps} T=${res.temperature} />`}
    </section>`}
    ${ctl.length > 0 && html`<section>
      <div class="section-title">What controls ${tgt.name} (${res.control.what === 'rate' ? 'its formation rate' : 'its amount'})</div>
      <p class="small muted">Degree of rate control of each TS: + means lowering that barrier gives more ${tgt.name}; − means it diverts material away.
        Hover for the effect of 1 kcal/mol. Click to open the reaction.</p>
      <${ControlBars} rows=${ctl} T=${res.temperature} onPick=${(r) => r.reaction && onReaction(r.reaction)} />
      ${quiet > 0 && html`<p class="small muted">${quiet} other step${quiet > 1 ? 's have' : ' has'} no real control over it (|X| below 0.01).</p>`}
      ${traps.length > 0 && html`<p class="small muted kn-sub">Intermediates and other products: − means stabilizing it lowers ${tgt.name} (a trap or a competing product); + means stabilizing it helps.</p>
        <${ControlBars} rows=${traps} T=${res.temperature} />`}
    </section>`}

    <section>
      <div class="section-title">Amounts over time</div>
      <${LineChart} xs=${res.times} series=${series} xLabel="time (s)" yLabel="amount (M)" />
      <details class="small"><summary>Table</summary>
        <table class="rx-table"><thead><tr><th>Species</th><th>start (M)</th><th>end (M)</th><th>peak (M)</th></tr></thead>
          <tbody>${ranked.slice(0, showAll ? ranked.length : 12).map((s) => html`<tr><td class="rx-eq">${s.name}${s.held ? ' (fixed)' : ''}</td>
            <td class="mono">${fmtC(s.c0)}</td><td class="mono">${fmtC(s.final)}</td><td class="mono">${fmtC(s.max)}</td></tr>`)}</tbody></table>
        ${ranked.length > 12 && html`<button class="btn-link small" onClick=${() => setShowAll(!showAll)}>${showAll ? 'fewer' : `all ${ranked.length}`}</button>`}
      </details>
    </section>

    ${res.sweep && html`<${Sweep} sw=${res.sweep} res=${res} />`}

    <section>
      <div class="section-title">Where the material went</div>
      <table class="rx-table"><thead><tr><th>Reaction</th><th title="Forward barrier from the separated reactants">ΔE‡</th><th title="Net amount converted over the run">converted (M)</th></tr></thead>
        <tbody>${flows.slice(0, 10).map((st) => html`<tr class=${st.reaction ? 'clickable' : ''} onClick=${() => st.reaction && onReaction(st.reaction)}>
          <td class="rx-eq">${st.label}${st.copies > 1 ? html` <span class="small muted">(${st.copies} runs; lowest TS)</span>` : ''}${!st.verified ? html` <span class="small muted" title="path maximum, not TS/IRC-verified">≈</span>` : ''}</td>
          <td class="mono">${st.barrier_fwd.toFixed(1)}</td><td class="mono">${fmtC(st.extent)}</td></tr>`)}</tbody></table>
    </section>
    ${res.excluded?.length > 0 && html`<${Excluded} list=${res.excluded} />`}
    ${res.warnings.map((w) => html`<p class="small muted">⚠ ${w}</p>`)}
  </div>`;
}

function Sweep({ sw, res }) {
  const rows = sw.temperatures.map((t, k) => ({ t, rate: sw.rate[k], ea: sw.apparent_ea_kcal[k] }));
  const ranked = [...res.species].sort((a, b) => b.max - a.max).filter((s) => s.max > 1e-12).slice(0, 6);
  const idx = Object.fromEntries(res.species.map((s, k) => [s.id, k]));
  const series = ranked.map((s, k) => ({ name: s.name, slot: k + 1, values: sw.final.map((f) => f[idx[s.id]]) }));
  const mid = rows[Math.floor(rows.length / 2)];
  return html`<section>
    <div class="section-title">With temperature</div>
    ${res.target && rows.some((r) => r.rate > 0) && html`<p class="small">${res.target.name} forms ${speedup(rows[rows.length - 1].rate / Math.max(rows[0].rate, 1e-300))} faster at ${Math.round(rows[rows.length - 1].t)} K than at ${Math.round(rows[0].t)} K${mid?.ea != null ? ` · apparent activation energy ≈ ${mid.ea.toFixed(1)} kcal/mol around ${Math.round(mid.t)} K` : ''}.</p>`}
    <${LineChart} xs=${sw.temperatures} series=${series} xLabel="temperature (K)" yLabel="amount at the end (M)" xlog=${false} />
  </section>`;
}

function speedup(r) {
  if (!Number.isFinite(r) || r <= 0) return '—';
  return r < 1000 ? `${r.toPrecision(2)}×` : `10^${Math.round(Math.log10(r))} times`;
}

function Excluded({ list }) {
  return html`<details class="small kn-excl"><summary>${list.length} reaction${list.length > 1 ? 's' : ''} not in the model</summary>
    <ul>${list.map((e) => html`<li><span class="rx-eq">${e.label}</span> <span class="muted">— ${e.reason}</span></li>`)}</ul></details>`;
}
