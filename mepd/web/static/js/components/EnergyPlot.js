// Energy profile as inline SVG: click or drag along it to pick a frame.
import { html, useEffect, useRef, useState } from '../lib.js';

function niceTicks(lo, hi, n = 4) {
  if (!(hi > lo)) return [lo];
  const raw = (hi - lo) / n;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 5, 10].map((m) => m * mag).find((s) => s >= raw) || raw;
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(+v.toFixed(10));
  return out;
}

export function EnergyPlot({ xs, ys, current = null, tsIndex = null, onPick, height = 150, yLabel = 'kcal/mol', compact = false }) {
  const svg = useRef(null);
  const box = useRef(null);
  const [hover, setHover] = useState(null);
  const [W, setW] = useState(480);
  useEffect(() => {
    // Track the real pixel width so the viewBox is 1:1 (no stretched dots).
    const ro = new ResizeObserver(([e]) => setW(Math.max(120, Math.round(e.contentRect.width))));
    if (box.current) ro.observe(box.current);
    return () => ro.disconnect();
  }, []);
  const pts = ys.map((y, i) => ({ i, x: xs?.[i] ?? i, y })).filter((p) => p.y != null && isFinite(p.y));
  if (pts.length < 2) {
    return html`<div ref=${box}><div class="plot-empty" style=${{ height: compact ? 60 : height }}>No energy profile</div></div>`;
  }
  const H = compact ? 70 : height;
  const m = compact ? { l: 6, r: 6, t: 6, b: 6 } : { l: 44, r: 12, t: 12, b: 24 };
  const xmin = Math.min(...pts.map((p) => p.x)), xmax = Math.max(...pts.map((p) => p.x));
  let ymin = Math.min(...pts.map((p) => p.y)), ymax = Math.max(...pts.map((p) => p.y));
  if (ymax - ymin < 1e-6) { ymin -= 1; ymax += 1; }
  const pad = (ymax - ymin) * 0.08;
  ymin -= pad; ymax += pad;
  const sx = (x) => m.l + ((x - xmin) / (xmax - xmin || 1)) * (W - m.l - m.r);
  const sy = (y) => m.t + (1 - (y - ymin) / (ymax - ymin)) * (H - m.t - m.b);
  const d = pts.map((p, k) => `${k ? 'L' : 'M'}${sx(p.x).toFixed(1)},${sy(p.y).toFixed(1)}`).join(' ');

  const nearest = (evt) => {
    const r = svg.current.getBoundingClientRect();
    const px = ((evt.clientX - r.left) / r.width) * W;
    let best = pts[0];
    for (const p of pts) if (Math.abs(sx(p.x) - px) < Math.abs(sx(best.x) - px)) best = p;
    return best;
  };
  const cur = pts.find((p) => p.i === current);
  const ts = pts.find((p) => p.i === tsIndex);

  return html`
    <div ref=${box}><div class="plot" style=${{ height: H }}>
      <svg ref=${svg} viewBox="0 0 ${W} ${H}" width="100%" height=${H}
        onPointerMove=${(e) => { const p = nearest(e); setHover(p); if (e.buttons && onPick) onPick(p.i); }}
        onPointerLeave=${() => setHover(null)}
        onPointerDown=${(e) => onPick && onPick(nearest(e).i)}
        style=${{ cursor: onPick ? 'crosshair' : 'default', touchAction: 'none' }}>
        ${!compact && niceTicks(ymin, ymax).map((t) => html`
          <g>
            <line x1=${m.l} x2=${W - m.r} y1=${sy(t)} y2=${sy(t)} class="grid" />
            <text x=${m.l - 6} y=${sy(t) + 3} class="tick" text-anchor="end">${t}</text>
          </g>`)}
        ${!compact && html`<text x=${10} y=${m.t + (H - m.t - m.b) / 2} class="axis-label"
          transform="rotate(-90 10 ${m.t + (H - m.t - m.b) / 2})" text-anchor="middle">${yLabel}</text>`}
        <path d=${d} class="line" />
        ${!compact && pts.map((p) => html`<circle cx=${sx(p.x)} cy=${sy(p.y)} r="2.2" class="pt" />`)}
        ${ts && html`<circle cx=${sx(ts.x)} cy=${sy(ts.y)} r="4.5" class="ts-pt" />`}
        ${cur && html`<g>
          <line x1=${sx(cur.x)} x2=${sx(cur.x)} y1=${m.t} y2=${H - m.b} class="cursor" />
          <circle cx=${sx(cur.x)} cy=${sy(cur.y)} r="5" class="cur-pt" />
        </g>`}
      </svg>
      ${hover && !compact && html`<div class="plot-tip" style=${{ left: `${(sx(hover.x) / W) * 100}%` }}>
        #${hover.i} · ${hover.y.toFixed(2)} ${yLabel}</div>`}
    </div></div>`;
}
