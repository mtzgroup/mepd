// Path map for `mepd channels`: every path being relaxed (one per conformer
// pair / atom mapping, recursive sub-paths too) on one approximate energy
// surface, redrawn as the paths move. See mepd/web/channels_pes.py for the
// coordinates and the surface.
import { html, useEffect, useRef, useState } from '../lib.js';
import { api } from '../api.js';
import { prefs, useStore } from '../store.js';
import { Viewer3D } from './Viewer3D.js';

const PALETTE = [[23, 50, 74], [31, 82, 120], [45, 118, 150], [80, 152, 160], [135, 183, 160], [196, 210, 160], [241, 229, 178], [250, 244, 222]];
const LINE = ['#e8553f', '#3a8fd9', '#f0a24b', '#7a3fa0', '#2fa36b', '#d6457f', '#8c6d1f', '#00a3a3', '#5b5bd6', '#b3452f'];
const W = 660, H = 520, L = 56, R = 16, T = 14, B = 46;

function color(t) {
  t = Math.max(0, Math.min(1, t)) * (PALETTE.length - 1);
  const i = Math.min(PALETTE.length - 2, Math.floor(t)), f = t - i, a = PALETTE[i], b = PALETTE[i + 1];
  return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f];
}
function ticks(a, b, n) {
  const span = b - a || 1, step0 = span / n, mag = Math.pow(10, Math.floor(Math.log10(step0)));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => span / s <= n) || 10 * mag;
  const out = []; for (let t = Math.ceil(a / step) * step; t <= b + 1e-9; t += step) out.push(+t.toFixed(10)); return out;
}
function bilinear(G, xs, ys, x, y) {
  const nx = xs.length, ny = ys.length;
  const fx = (x - xs[0]) / (xs[nx - 1] - xs[0]) * (nx - 1), fy = (y - ys[0]) / (ys[ny - 1] - ys[0]) * (ny - 1);
  const i = Math.max(0, Math.min(nx - 2, Math.floor(fx))), j = Math.max(0, Math.min(ny - 2, Math.floor(fy)));
  const tx = fx - i, ty = fy - j, v = [G[j][i], G[j][i + 1], G[j + 1][i], G[j + 1][i + 1]];
  if (v.some((q) => q === null)) return null;
  return (1 - tx) * (1 - ty) * v[0] + tx * (1 - ty) * v[1] + (1 - tx) * ty * v[2] + tx * ty * v[3];
}
function contours(G, xs, ys, lev) {
  const seg = [];
  for (let j = 0; j < ys.length - 1; j++) for (let i = 0; i < xs.length - 1; i++) {
    const c = [[xs[i], ys[j], G[j][i]], [xs[i + 1], ys[j], G[j][i + 1]], [xs[i + 1], ys[j + 1], G[j + 1][i + 1]], [xs[i], ys[j + 1], G[j + 1][i]]];
    if (c.some((p) => p[2] === null)) continue;
    const pts = [];
    for (let k = 0; k < 4; k++) {
      const a = c[k], b = c[(k + 1) % 4];
      if ((a[2] - lev) * (b[2] - lev) < 0) { const t = (lev - a[2]) / (b[2] - a[2]); pts.push([a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])]); }
    }
    if (pts.length === 2) seg.push(pts); else if (pts.length === 4) { seg.push([pts[0], pts[1]]); seg.push([pts[2], pts[3]]); }
  }
  return seg;
}

function Surface({ map, pairColor, selected, onSelect, hover, setHover }) {
  const canvas = useRef(null);
  const { x: xs, y: ys, E } = map;
  const marks = map.ts || [];
  const ircPts = map.irc?.points || [];
  const pts = map.paths.flatMap((p) => p.points).concat(marks.map((m) => [m.q[0], m.q[1], m.e]), ircPts);
  const allX = pts.map((q) => q[0]).concat(xs), allY = pts.map((q) => q[1]).concat(ys);
  const x0 = Math.min(...allX), x1 = Math.max(...allX), y0 = Math.min(...allY), y1 = Math.max(...allY);
  const energies = (E.flat ? E.flat() : []).filter((v) => v !== null).concat(pts.map((q) => q[2]).filter((v) => v !== null));
  const vmin = Math.min(...energies, 0), vmax = Math.max(Math.min(Math.max(...energies), vmin + 150), vmin + 5);
  const sx = (x) => L + (x - x0) / ((x1 - x0) || 1) * (W - L - R), sy = (y) => T + (1 - (y - y0) / ((y1 - y0) || 1)) * (H - T - B);
  const ix = (px) => x0 + (px - L) / (W - L - R) * (x1 - x0), iy = (py) => y0 + (1 - (py - T) / (H - T - B)) * (y1 - y0);

  useEffect(() => {
    const ctx = canvas.current?.getContext('2d');
    if (!ctx) return;
    ctx.fillStyle = '#fff'; ctx.fillRect(0, 0, W, H);
    if (!xs.length) return;
    const Ec = E.map((row) => row.map((v) => (v === null ? null : Math.min(v, vmax))));
    const img = ctx.createImageData(W - L - R, H - T - B);
    for (let py = 0; py < H - T - B; py++) for (let px = 0; px < W - L - R; px++) {
      const e = bilinear(Ec, xs, ys, ix(px + L), iy(py + T)), o = 4 * (py * (W - L - R) + px);
      if (e === null) { img.data.set(((px + py) % 8) < 2 ? [214, 209, 199, 255] : [244, 242, 236, 255], o); continue; }
      const c = color((e - vmin) / (vmax - vmin)); img.data.set([c[0], c[1], c[2], 255], o);
    }
    ctx.putImageData(img, L, T);
  }, [map]);

  const lev = ticks(vmin, vmax, 12);
  const step = (lev[1] - lev[0]) || 10;
  const segs = xs.length ? lev.flatMap((l) => contours(E, xs, ys, l)) : [];
  const poly = (p) => p.points.map((q) => `${sx(q[0]).toFixed(1)},${sy(q[1]).toFixed(1)}`).join(' ');
  const order = [...map.paths].sort((a, b) => (a.id === selected) - (b.id === selected) || a.active - b.active);
  return html`<div class="cmap-surf" style=${`aspect-ratio:${W}/${H}`}>
    <canvas ref=${canvas} width=${W} height=${H}></canvas>
    <svg viewBox=${`0 0 ${W} ${H}`} onMouseLeave=${() => setHover(null)}>
      <defs><clipPath id="cmapClip"><rect x=${L} y=${T} width=${W - L - R} height=${H - T - B} /></clipPath></defs>
      <g clip-path="url(#cmapClip)">
        ${segs.map(([a, b]) => html`<line x1=${sx(a[0])} y1=${sy(a[1])} x2=${sx(b[0])} y2=${sy(b[1])} stroke="rgba(23,50,74,.28)" stroke-width="0.7" />`)}
        ${order.map((p) => {
          const col = pairColor(p.pair), on = p.id === selected, dim = selected && !on;
          return html`<g style="cursor:pointer" onClick=${() => onSelect(on ? null : p.id)}>
            <polyline points=${poly(p)} fill="none" stroke="rgba(23,50,74,.55)" stroke-width=${on ? 6 : 4} opacity=${dim ? 0.25 : 1} />
            <polyline points=${poly(p)} fill="none" stroke=${col} stroke-width=${on ? 3.6 : 2.2} opacity=${dim ? 0.35 : 1}
              class=${p.active ? 'cmap-active' : ''} stroke-dasharray=${p.monitor !== 'branch-0' && p.monitor !== 'main' ? '5 3' : ''} />
            ${p.points.map((q, k) => html`<circle cx=${sx(q[0])} cy=${sy(q[1])} r=${on ? 3.4 : 2.2} fill=${col} stroke="#fff" stroke-width="0.8"
              opacity=${dim ? 0.3 : 1} onMouseEnter=${() => setHover({ p, k, q })} />`)}
          </g>`;
        })}
        ${ircPts.length > 1 && html`<g style="pointer-events:none">
          <polyline points=${ircPts.map((q) => `${sx(q[0]).toFixed(1)},${sy(q[1]).toFixed(1)}`).join(' ')} fill="none" stroke="rgba(23,50,74,.55)" stroke-width="5" />
          <polyline points=${ircPts.map((q) => `${sx(q[0]).toFixed(1)},${sy(q[1]).toFixed(1)}`).join(' ')} fill="none" stroke="#ffffff" stroke-width="2.8" />
        </g>`}
        ${marks.map((m) => {
          const X = sx(m.q[0]), Y = sy(m.q[1]), col = pairColor(m.pair), r = 7;
          const on = selected && selected.startsWith(`${m.pair}/`);
          const dim = selected && !on;
          const style = m.kind === 'direct' ? '' : m.kind === 'multi-step' ? '3 2' : '1.5 2';
          return html`<g class="cmap-ts" opacity=${dim ? 0.3 : m.kind === 'other' ? 0.75 : 1}
            onMouseEnter=${() => setHover({ ts: m })} style="cursor:pointer">
            <path d=${`M${X - r},${Y - r}L${X + r},${Y + r}M${X - r},${Y + r}L${X + r},${Y - r}`} stroke="#fff" stroke-width="6" stroke-linecap="round" />
            <path d=${`M${X - r},${Y - r}L${X + r},${Y + r}M${X - r},${Y + r}L${X + r},${Y - r}`} stroke=${col} stroke-width="3.2"
              stroke-linecap="round" stroke-dasharray=${style} />
            <circle cx=${X} cy=${Y} r="11" fill="transparent" />
          </g>`;
        })}
      </g>
      ${ticks(x0, x1, 6).map((t) => html`<text x=${sx(t)} y=${H - B + 16} text-anchor="middle">${+t.toFixed(2)}</text>`)}
      ${ticks(y0, y1, 6).map((t) => html`<text x=${L - 6} y=${sy(t) + 4} text-anchor="end">${+t.toFixed(2)}</text>`)}
      <text x=${(L + W - R) / 2} y=${H - 8} text-anchor="middle">${map.axes.x}</text>
      <text x="14" y=${(T + H - B) / 2} transform=${`rotate(-90 14 ${(T + H - B) / 2})`} text-anchor="middle">${map.axes.y}</text>
      ${hover?.ts && html`<g style="pointer-events:none">
        <text x=${Math.min(sx(hover.ts.q[0]) + 12, W - 330)} y=${Math.max(sy(hover.ts.q[1]) - 34, 24)} class="cmap-tip">
          TS ${hover.ts.pair.replace(/_/g, ' ')}: ${hover.ts.e.toFixed(1)} kcal/mol · ${hover.ts.kind === 'direct' ? `direct channel${hover.ts.group ? ` (${hover.ts.group.replace('_', ' ')})` : ''}` : hover.ts.kind === 'multi-step' ? 'step of a multi-step channel' : 'IRC does not connect the ends'}</text>
        ${hover.ts.closest && html`<text x=${Math.min(sx(hover.ts.q[0]) + 12, W - 330)} y=${Math.max(sy(hover.ts.q[1]) - 34, 24) + 15} class="cmap-tip">
          closest image: ${hover.ts.closest.rmsd.toFixed(2)} Å RMSD (image ${hover.ts.closest.image + 1}, ${hover.ts.closest.monitor})</text>`}
      </g>`}
      ${hover && !hover.ts && html`<g style="pointer-events:none">
        <circle cx=${sx(hover.q[0])} cy=${sy(hover.q[1])} r="7" fill="none" stroke="#ffd23f" stroke-width="3" />
        <text x=${Math.min(sx(hover.q[0]) + 10, W - 170)} y=${Math.max(sy(hover.q[1]) - 10, 24)} class="cmap-tip">
          ${hover.p.pair.replace(/_/g, ' ')} · ${hover.p.monitor} · image ${hover.k + 1}: ${hover.q[2] == null ? '—' : `${hover.q[2].toFixed(1)} kcal/mol`}</text>
      </g>`}
    </svg>
    <div class="cmap-legend small">
      <span>Energy</span> <span class="mono">${vmin.toFixed(0)}</span>
      <span class="cmap-bar" style=${`background:linear-gradient(90deg, ${PALETTE.map((c) => `rgb(${c.join(',')})`).join(',')})`}></span>
      <span class="mono">${vmax.toFixed(0)}</span> <span class="muted">kcal/mol vs ${map.reference} · contours every ${+step.toFixed(2)}</span>
    </div>
  </div>`;
}

function PathDetail({ job, path, pairColor, marks = [] }) {
  const [frames, setFrames] = useState(null);
  const [k, setK] = useState(0);
  useEffect(() => {
    let live = true;
    api.get(`/api/jobs/${job.id}/live/${path.pair}`).then((d) => {
      const m = (d.monitors || {})[path.monitor] || d;
      const f = m.geometry?.frames || [];
      if (live) { setFrames(f); setK(Math.max(0, path.points.reduce((best, q, i) => ((q[2] ?? -1e9) > (path.points[best][2] ?? -1e9) ? i : best), 0))); }
    }).catch(() => {});
    return () => { live = false; };
  }, [path.id, path.points.length]);
  const top = Math.max(...path.points.map((q) => q[2] ?? -1e9));
  return html`<div class="cmap-detail">
    <h4><span class="cmap-swatch" style=${`background:${pairColor(path.pair)}`}></span>${path.pair.replace(/_/g, ' ')} · ${path.monitor}${path.active ? ' · relaxing now' : ''}</h4>
    ${path.bonds && html`<p class="small">${path.bonds.kind === 'break-form'
      ? html`x: breaks <b>${path.bonds.x.join(', ') || '—'}</b> · y: forms <b>${path.bonds.y.join(', ') || '—'}</b>`
      : html`x: ${path.bonds.kind} <b>${path.bonds.x.join(', ')}</b> · y: ${path.bonds.kind} <b>${path.bonds.y.join(', ') || '—'}</b>`}</p>`}
    <p class="small muted">Highest image ${top > -1e9 ? `${top.toFixed(1)} kcal/mol` : '—'} on this map's scale.</p>
    ${marks.filter((m) => m.pair === path.pair).map((m) => html`<p class="small">
      ✕ Optimized TS ${m.e.toFixed(1)} kcal/mol (${m.kind === 'direct' ? 'direct channel' : m.kind === 'multi-step' ? 'multi-step channel' : "its IRC doesn't connect the ends"})${m.closest
        ? html` · the chain came within <b>${m.closest.rmsd.toFixed(2)} Å</b> RMSD of it (image ${m.closest.image + 1})` : ''}</p>`)}
    ${frames && frames.length > 0 && html`
      <${Viewer3D} frames=${frames} frame=${Math.min(k, frames.length - 1)} height=${220} />
      <input type="range" min="0" max=${frames.length - 1} value=${Math.min(k, frames.length - 1)} onInput=${(e) => setK(+e.target.value)} />
      <div class="small muted">image ${Math.min(k, frames.length - 1) + 1} of ${frames.length}${path.points[k]?.[2] != null ? ` · ${path.points[k][2].toFixed(1)} kcal/mol` : ''}</div>`}
  </div>`;
}

export function ChannelsMap({ job }) {
  const [mode, setModeState] = useState(() => prefs.get('channelsMapMode', 'bonds'));
  const setMode = (m) => { setModeState(m); prefs.set('channelsMapMode', m); };
  const [map, setMap] = useState(null);
  const [ircTs, setIrcTs] = useState(null);
  const [err, setErr] = useState(null);
  const [selected, setSelected] = useState(null);
  const [hover, setHover] = useState(null);
  // Re-read whenever the run's live data moves on (and every few seconds while it runs).
  const beat = useStore((s) => {
    const st = s.progress[job.id]?.streams || {};
    return Object.values(st).map((v) => v.updated || 0).join(',');
  });
  const [tick, setTick] = useState(0);
  useEffect(() => {
    if (job.status !== 'running') return undefined;
    const t = setInterval(() => setTick((x) => x + 1), 3000);
    return () => clearInterval(t);
  }, [job.status]);
  const busy = useRef(false);
  useEffect(() => {
    if (busy.current) return;
    busy.current = true;
    api.get(`/api/jobs/${job.id}/channels-map?mode=${mode}${mode === 'irc' && ircTs ? `&ts=${encodeURIComponent(ircTs)}` : ''}`).then((m) => { setMap(m); setErr(null); })
      .catch((e) => setErr(e.message)).finally(() => { busy.current = false; });
  }, [mode, ircTs, beat, tick, job.status]);
  const choices = map?.irc_choices || [];

  const pairs = map ? [...new Set(map.paths.map((p) => p.pair))] : [];
  const pairColor = (pair) => LINE[Math.max(0, pairs.indexOf(pair)) % LINE.length];
  const sel = map?.paths.find((p) => p.id === selected);
  return html`<div class="cmap">
    <div class="cmap-head">
      <div class="segmented">
        <button class=${mode === 'bonds' ? 'on' : ''} onClick=${() => setMode('bonds')} title="Each path's own bonds, 0 to 1: broken vs formed, or two groups of bonds formed (More O'Ferrall-Jencks)">Bond progress</button>
        <button class=${mode === 'distance' ? 'on' : ''} onClick=${() => setMode('distance')} title="Distance from the lowest reactant and product conformers, the same for every atom mapping">Distance to reactant vs product</button>
        <button class=${mode === 'irc' ? 'on' : ''} disabled=${!choices.length && mode !== 'irc'} onClick=${() => setMode('irc')}
          title=${choices.length ? 'Post hoc: every chain measured against a computed IRC -- along it, and how far off it' : 'Needs a finished TS optimization with its IRC'}>Along an IRC</button>
      </div>
      ${mode === 'irc' && choices.length > 0 && html`<label class="small">IRC of
        <select value=${map?.irc?.ts || ''} onChange=${(e) => setIrcTs(e.target.value)}>
          ${choices.map((c) => html`<option value=${c.label}>${c.pair.replace(/_/g, ' ')} · ${c.e.toFixed(1)} kcal/mol · ${c.kind === 'direct' ? 'direct channel' : c.kind === 'multi-step' ? 'multi-step' : 'unconnected'}</option>`)}
        </select></label>`}
      <span class="small muted">${map ? `${map.paths.length} path${map.paths.length === 1 ? '' : 's'} · ${pairs.length} pair${pairs.length === 1 ? '' : 's'}` : ''}${job.status === 'running' ? ' · updating live' : ''}</span>
    </div>
    <p class="small muted">${mode === 'irc'
      ? 'Post hoc, against a computed IRC (white; its TS at 0, 0): x is where each image lands along that IRC (Å from the TS, reactant to the left), y how far it is from it (Å, the same numbering-independent distance as the distance view, so every atom mapping is measured against the true path). A chain that found this channel converges onto the white line near x = 0; one that went elsewhere stays off it. '
      : mode === 'bonds'
      ? 'x: how far the bonds that break have stretched; y: how far the bonds that form have closed (0 at the reactant, 1 at the product), each path with its own bonds, so different atom mappings share the square. A concerted path runs along the diagonal; a stepwise one hugs an edge. When a reaction only forms (or only breaks) bonds, e.g. a Diels–Alder, its bonds are split in two groups plotted against each other: synchronous along the diagonal, asynchronous bowed toward an edge. Dashed: sub-paths of a recursive split. '
      : 'x, y: how far each structure is from the lowest reactant and the lowest product conformer (differences of sorted interatomic distances), independent of atom numbering: paths that pass through similar structures run close together.'}
      The colours are a surface fitted to the computed images: their energies and their gradients projected onto these two coordinates. A 2D map keeps only two of a molecule's many directions, so it is exact at the images and approximate between them; hatched: no image nearby.</p>
    ${err && html`<p class="error-box small">${err}</p>`}
    ${(map?.warnings || []).map((w) => html`<p class="level-note small">${w}</p>`)}
    ${map && !map.paths.length && html`<p class="small muted">No paths yet: they appear here as soon as the path searches start.</p>`}
    ${(map?.ts || []).length > 0 && html`<p class="small muted cmap-ts-key">✕ optimized TSs, in their path's colour: solid = a direct channel, dashed = a step of a multi-step channel, dotted = the IRC does not connect the ends. Hover one for how close its chains came.</p>`}
    ${map && map.paths.length > 0 && html`<div class="cmap-body">
      <${Surface} map=${map} pairColor=${pairColor} selected=${selected} onSelect=${setSelected} hover=${hover} setHover=${setHover} />
      <div class="cmap-side">
        <ul class="cmap-pairs small">${map.paths.map((p) => html`<li class=${p.id === selected ? 'on' : ''} onClick=${() => setSelected(p.id === selected ? null : p.id)}>
          <span class="cmap-swatch" style=${`background:${pairColor(p.pair)}`}></span>${p.pair.replace(/_/g, ' ')} <span class="muted">· ${p.monitor}</span>${p.active ? html` <span class="badge">relaxing</span>` : ''}</li>`)}</ul>
        ${sel ? html`<${PathDetail} job=${job} path=${sel} pairColor=${pairColor} marks=${map.ts || []} />` : html`<p class="small muted">Click a path to see its bonds and step through its images.</p>`}
      </div>
    </div>`}
  </div>`;
}
