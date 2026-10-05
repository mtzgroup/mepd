// The nanoreactor, live: the reactor's MD as it runs (the piston wall
// breathing around it), every reaction event lighting up the atoms it
// takes as it happens, a timeline of the events, and a replay of any one
// event with only its molecules and the bonds that form (green) and break
// (red).
import { html, useEffect, useRef, useState } from '../lib.js';
import { api } from '../api.js';
import { ReactionCard, reactionOf } from './Reactions.js';
import { useStore } from '../store.js';

const FPS = 30;   // reactor frames shown per second at 1x (a frame is ~10 fs)
const POLL_S = 2.5;   // seconds between fetches of new frames while the MD runs

// Theme colours, read once per theme: getComputedStyle inside the
// animation loop forces a style/layout pass of the whole page every frame
// (right after the clock was written), which is what made event frames --
// the only ones that read a colour -- stutter.
const _css = {};
let _cssTheme = null;
export function cssVar(name, fallback) {
  const theme = `${document.documentElement.dataset.theme || ''}|${window.matchMedia('(prefers-color-scheme: dark)').matches}`;
  if (theme !== _cssTheme) { for (const k of Object.keys(_css)) delete _css[k]; _cssTheme = theme; }
  if (!(name in _css)) _css[name] = getComputedStyle(document.documentElement).getPropertyValue(name).trim() || fallback;
  return _css[name];
}

export function xyzText(symbols, flat) {
  const lines = [String(symbols.length), ''];
  for (let a = 0; a < symbols.length; a += 1) {
    lines.push(`${symbols[a]} ${flat[3 * a]} ${flat[3 * a + 1]} ${flat[3 * a + 2]}`);
  }
  return `${lines.join('\n')}\n`;
}

// How far from the origin the view must reach: the wide wall, or (a run
// that analyzed an existing trajectory, no wall) the farthest atom.
const _reach = new WeakMap();
function reach(d) {
  if (_reach.has(d)) return _reach.get(d);
  const r = reachOf(d);
  _reach.set(d, r);
  return r;
}

function reachOf(d) {
  const wall = Math.max(0, ...d.radius.filter(Boolean));
  if (wall) return wall + 0.5;
  let r = 0;
  for (let i = 0; i < d.frames.length; i += 10) {
    const f = d.frames[i];
    for (let k = 0; k < f.length; k += 1) r = Math.max(r, Math.abs(f[k]));
  }
  return r + 1;
}

const fmtPs = (fs) => `${(fs / 1000).toFixed(2)} ps`;

// A 3Dmol canvas redrawn per frame: the model is rebuilt each time so bonds
// are perceived anew (they form and break), keeping the camera.
export function useStage(host) {
  const viewer = useRef(null);
  const fitted = useRef(false);
  const el = useRef(null);
  const ro = useRef(null);
  // Created on the first draw: the stage's element may mount after this hook
  // (the page shows a message until the first frames arrive), and may be
  // replaced (after an event replay).
  const ensure = () => {
    if (!window.$3Dmol || !host.current) return null;
    if (viewer.current && el.current === host.current) return viewer.current;
    ro.current?.disconnect();
    viewer.current = window.$3Dmol.createViewer(host.current, { backgroundColor: cssVar('--viewer-bg', '#fff'), antialias: true });
    el.current = host.current;
    fitted.current = false;
    ro.current = new ResizeObserver(() => { viewer.current?.resize(); viewer.current?.render(); });
    ro.current.observe(host.current);
    return viewer.current;
  };
  useEffect(() => () => { ro.current?.disconnect(); viewer.current?.clear(); viewer.current = null; }, []);
  // `extent` (Angstrom around the origin): what the first view must hold,
  // e.g. the wide wall, not just today's atoms.
  const wall = useRef(null);
  // `shapes`: this draw repaints the shapes too (else they stay: the wall).
  const draw = (paint, { refit = false, extent = null, shapes = true } = {}) => {
    const v = ensure();
    if (!v) return;
    const view = v.getView();
    v.removeAllModels();
    if (shapes) { v.removeAllShapes(); v.removeAllLabels(); wall.current = null; }
    if (!fitted.current || refit) {
      if (extent) {   // fit to invisible marker atoms at +-extent, then drop them
        const r = extent;
        const pts = [[r, 0, 0], [-r, 0, 0], [0, r, 0], [0, -r, 0], [0, 0, r], [0, 0, -r]];
        const m = v.addModel(`6\n\n${pts.map((q) => `He ${q.join(' ')}`).join('\n')}\n`, 'xyz');
        v.zoomTo();
        v.removeModel(m);
      } else v.zoomTo();
      paint(v);
      if (!extent) v.zoomTo();
      fitted.current = true;
    } else {
      paint(v);
      v.setView(view);
    }
    v.render();
  };
  // The piston wall: a faint shell plus a wireframe cage (a translucent
  // sphere alone is all but invisible), rebuilt only when it moves.
  const setWall = (r, color) => {
    const v = viewer.current;
    const key = `${r}|${color}`;
    if (!v || wall.current === key) return;
    v.removeAllShapes();
    if (r) {
      const center = { x: 0, y: 0, z: 0 };
      v.addSphere({ center, radius: r, color, opacity: 0.12, resolution: 32 });
      v.addSphere({ center, radius: r, color, opacity: 0.6, wireframe: true, linewidth: 1, resolution: 14 });
    }
    wall.current = key;
    v.render();
  };
  return { draw, setWall, viewer: () => viewer.current, refit: () => { fitted.current = false; wall.current = null; } };
}

export const FULL = { stick: { radius: 0.12 }, sphere: { scale: 0.2 } };
// Bystanders while an event runs: thin and grey, but opaque. (Transparency
// makes 3Dmol depth-sort every frame, and 3D text labels upload a texture
// per frame: either one stalls the animation on a real GPU. The reaction
// is written in an HTML overlay instead.)
let _faded = null, _fadedGrey = null;
const faded = () => {
  const grey = cssVar('--edge-idle', '#9aa7b3');
  if (grey !== _fadedGrey) { _fadedGrey = grey; _faded = { stick: { radius: 0.06, color: grey }, sphere: { scale: 0.12, color: grey } }; }
  return _faded;
};
const short = (text, n = 46) => (text.length > n ? `${text.slice(0, n - 1)}…` : text);

// Play/pause and a frame counter advancing at `speed` x FPS.
function usePlayer(length, { loop = true, onEnd } = {}) {
  const [i, setI] = useState(0);
  const [playing, setPlaying] = useState(true);
  const [speed, setSpeed] = useState(1);
  const lenRef = useRef(length);
  lenRef.current = length;
  useEffect(() => {
    if (!playing) return undefined;
    let raf, last = performance.now(), acc = 0;
    const tick = (now) => {
      acc += ((now - last) / 1000) * FPS * speed;
      last = now;
      if (acc >= 1) {
        const step = Math.floor(acc);
        acc -= step;
        setI((k) => {
          const n = lenRef.current;
          if (!n) return 0;
          if (k + step < n) return k + step;
          if (onEnd) onEnd();
          return loop ? 0 : n - 1;
        });
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [playing, speed]);
  return { i: Math.min(i, Math.max(0, length - 1)), setI, playing, setPlaying, speed, setSpeed };
}

function Transport({ p, children }) {
  return html`<div class="rx-transport">
    <button class="btn small" onClick=${() => p.setPlaying(!p.playing)} title=${p.playing ? 'Pause' : 'Play'}>${p.playing ? '❚❚' : '▶'}</button>
    <div class="segmented">
      ${[0.5, 1, 2, 4].map((s) => html`<button class=${p.speed === s ? 'on' : ''} onClick=${() => p.setSpeed(s)}>${s}×</button>`)}
    </div>
    ${children}
  </div>`;
}

// ------------------------------------------------------------ one event
// An event at full time resolution with only its molecules, looping: bonds
// that form dashed green, bonds that break dashed red.
export function EventPlayer({ jobId, event, height = null, onData }) {
  const host = useRef(null);
  const stage = useStage(host);
  const [data, setData] = useState(null);
  const [err, setErr] = useState(null);
  useEffect(() => {
    setData(null); setErr(null); stage.refit();
    api.get(`/api/jobs/${jobId}/reactor/events/${event}`).then((x) => { setData(x); onData?.(x); })
      .catch((e) => setErr(String(e.message || e)));
  }, [jobId, event]);
  const p = usePlayer(data ? data.frames.length : 0);
  useEffect(() => {
    if (!data) return;
    const flat = data.frames[p.i];
    const ok = cssVar('--ok', '#496e5c'), bad = cssVar('--danger', '#b23b3b');
    let extent = 0;
    for (const f of data.frames) for (let k = 0; k < f.length; k += 1) extent = Math.max(extent, Math.abs(f[k]));
    stage.draw((v) => {
      v.addModel(xyzText(data.symbols, flat), 'xyz');
      v.setStyle({}, FULL);
      for (const c of data.changes) {
        const [a, b] = c.atoms;
        const near = Math.abs(p.i - c.frame) <= 6;
        v.addCylinder({
          start: { x: flat[3 * a], y: flat[3 * a + 1], z: flat[3 * a + 2] },
          end: { x: flat[3 * b], y: flat[3 * b + 1], z: flat[3 * b + 2] },
          radius: near ? 0.07 : 0.035, dashed: true, color: c.formed ? ok : bad, fromCap: 1, toCap: 1,
        });
      }
    }, { extent: extent + 1 });
  }, [data, p.i]);
  const t = data ? (data.first_frame + p.i) * data.dump_fs : 0;
  const phase = !data ? '' : p.i < data.reactant_frame ? 'before' : p.i > data.product_frame ? 'after' : 'reacting';
  return html`<div class="rx-player">
    <div class="rx-stage-wrap">
      <div class="rx-stage" ref=${host} style=${height ? { height: `${height}px` } : null}></div>
      <div class="rx-hud">${data ? html`${fmtPs(t)} · <span class=${`rx-phase ${phase}`}>${phase}</span>` : err || 'loading…'}</div>
      <div class="rx-legend small"><span class="lg-form">- - forms</span> <span class="lg-break">- - breaks</span></div>
    </div>
    ${data && html`<${Transport} p=${p}>
      <input type="range" min="0" max=${data.frames.length - 1} value=${p.i}
        onInput=${(e) => { p.setPlaying(false); p.setI(+e.target.value); }} />
    <//>`}
  </div>`;
}

function EventReplay({ jobId, ev, onClose }) {
  const reaction = useStore((s) => (ev.reaction == null ? null : reactionOf(s.workspace, jobId, ev.reaction)));
  return html`<div class="rx-replay">
    <div class="rx-replay-head">
      <button class="btn small" onClick=${onClose}>← Reactor</button>
      <span class="rx-eq">${ev.label}</span>
    </div>
    ${reaction ? html`<${ReactionCard} r=${reaction} event=${ev.event} />`
      : html`<${EventPlayer} jobId=${jobId} event=${ev.event} />
        <p class="small muted">${ev.reaction == null ? 'An exchange: the same molecules come out as went in.' : 'Its reaction is not in Explore (yet, or deleted).'}</p>`}
  </div>`;
}

// ------------------------------------------------------------ the reactor
// Per frame the work is kept small, so playback stays smooth on any GPU:
//  * no React render per frame: the loop runs on refs, the clock and the
//    playhead are written straight into the DOM, and React only re-renders
//    when the set of events in progress changes (a few times per event);
//  * theme colours are cached (no getComputedStyle per frame);
//  * a late frame never makes playback jump ahead: it just runs slower.

// Before the MD: the packed reactor, then its relaxation inside the wall,
// played through by itself (the latest step is held while it runs; no
// transport: there is nothing to scrub for).
const PREP_TEXT = {
  packed: 'Packed: the molecules placed at random in the wall',
  relaxing: 'Relaxing the packed reactor inside the wall',
  relaxed: 'Relaxed: starting the MD',
};

function PrepView({ prep, running }) {
  const host = useRef(null);
  const stage = useStage(host);
  const p = usePlayer(prep.frames.length, { loop: !running });
  const accent = cssVar('--accent', '#3868b8');
  useEffect(() => {
    const flat = prep.frames[p.i];
    if (!flat) return;
    stage.draw((v) => {
      v.addModel(xyzText(prep.symbols, flat), 'xyz');
      v.setStyle({}, FULL);
    }, { extent: (prep.radius || 8) + 0.5, shapes: false });
    stage.setWall(prep.radius || null, accent);
  }, [prep, p.i]);
  const e = prep.energy_kcal[p.i];
  const step = prep.step[p.i];
  return html`<div class="rx-live">
    <div class="rx-main">
      <div class="rx-stage-wrap">
        <div class="rx-stage" ref=${host}></div>
        ${prep.radius && html`<div class="rx-legend small"><span class="lg-wall">◯ wall</span></div>`}
        <div class="rx-hud">${PREP_TEXT[prep.stage] || ''}${step ? ` · step ${step} of ${prep.steps}` : ''}${e != null && step ? ` · ${e.toFixed(1)} kcal/mol` : ''}${prep.radius ? ` · wall ${prep.radius.toFixed(1)} Å` : ''}${running ? ' · live' : ''}</div>
      </div>
    </div>
    <aside class="rx-events">
      <div class="section-title">Before the MD</div>
      <p class="small muted">The molecules are packed at random into the wall, then relaxed inside it, so that strain from
        the packing does not turn into heat and throw atoms out. The MD and its reaction events show here once it starts.</p>
    </aside>
  </div>`;
}

// The wide radius of the piston (the wall's resting size).
function wallWide(d) {
  if (d._wide === undefined) d._wide = Math.max(0, ...d.radius.filter(Boolean)) || null;
  return d._wide;
}

function activeAt(events, raw) {
  return events.filter((e) => raw >= e.reactant_frame && raw <= e.product_frame);
}

export function ReactorLive({ job }) {
  const host = useRef(null);
  const stage = useStage(host);
  const [d, setD] = useState(null);           // {symbols, frames, radius, stride, dump_fs, events, ...}
  const [open, setOpen] = useState(null);     // an event being replayed
  const [playing, setPlaying] = useState(true);
  const [speed, setSpeed] = useState(1);
  const [activeIds, setActiveIds] = useState('');   // events in progress (React renders on change only)
  const next = useRef(0);
  const mdRate = useRef(0);                   // frames/s the MD delivers (smoothed), while it runs
  const dRef = useRef(null);
  const iRef = useRef(0);
  const hud = useRef(null);
  const headEl = useRef(null);
  const running = ['running', 'queued'].includes(job.status);
  const runningRef = useRef(running);
  runningRef.current = running;
  dRef.current = d;

  useEffect(() => {
    let alive = true, timer;
    next.current = 0;
    mdRate.current = 0;
    setD(null);
    let at = 0;
    const pull = async () => {
      try {
        const r = await api.get(`/api/jobs/${job.id}/reactor?start=${next.current}`);
        if (!alive) return;
        const now = performance.now();
        if (at && r.start > 0) {
          const rate = r.frames.length / ((now - at) / 1000);
          mdRate.current = mdRate.current ? 0.7 * mdRate.current + 0.3 * rate : rate;
        }
        at = now;
        next.current = r.start + r.frames.length * r.stride;
        setD((old) => (old && old.stride === r.stride && r.start > 0
          ? { ...r, frames: old.frames.concat(r.frames), radius: old.radius.concat(r.radius) }
          : r));
      } catch (e) { /* not started yet */ }
      if (alive && ['running', 'queued'].includes(job.status)) timer = setTimeout(pull, POLL_S * 1000);
    };
    pull();
    return () => { alive = false; clearTimeout(timer); };
  }, [job.id, job.status]);

  const accent = cssVar('--accent', '#3868b8');
  const totalFs = d ? Math.max(d.total_ps * 1000, d.n_frames * d.dump_fs) : 1;

  // Draw frame i: the cheap path when nothing but positions changed.
  const drawFrame = (i) => {
    const dd = dRef.current;
    if (!dd || !dd.frames.length) return;
    i = Math.max(0, Math.min(i, dd.frames.length - 1));
    iRef.current = i;
    const flat = dd.frames[i];
    const raw = i * dd.stride;
    const act = activeAt(dd.events, raw);
    const sel = act.map((e) => e.event).join(',');
    // (3Dmol perceives the bonds anew each frame: that is how they form and break.)
    stage.draw((v) => {
      v.addModel(xyzText(dd.symbols, flat), 'xyz');
      v.setStyle({}, act.length ? faded() : FULL);
      if (act.length) v.setStyle({ index: [...new Set(act.flatMap((e) => e.atoms))] }, FULL);
    }, { extent: reach(dd), shapes: false });
    const wide = wallWide(dd);
    const squeezed = dd.radius[i] && wide && dd.radius[i] < wide - 0.05;
    stage.setWall(dd.radius[i] || null, squeezed ? cssVar('--warn', '#b7791f') : accent);
    // The clock and the playhead, without React.
    const tFs = raw * dd.dump_fs;
    if (hud.current) {
      const edge = runningRef.current && i >= dd.frames.length - 1;
      hud.current.textContent = `${fmtPs(tFs)}${dd.radius[i] ? ` · wall ${dd.radius[i].toFixed(1)} Å${squeezed ? ' (squeezing)' : ''}` : ''}${runningRef.current ? (edge ? ' · live · waiting for the MD…' : ' · live') : ''}`;
    }
    if (headEl.current) headEl.current.style.left = `${Math.min(100, (100 * tFs) / Math.max(dd.total_ps * 1000, dd.n_frames * dd.dump_fs))}%`;
    setActiveIds((old) => (old === sel ? old : sel));
  };

  // A fresh stage (first data, back from a replay): the model is gone.
  useEffect(() => { if (d && !open) drawFrame(iRef.current); }, [open, d]);

  useEffect(() => {
    if (!playing || open) return undefined;
    let raf, last = performance.now(), acc = 0;
    const tick = (now) => {
      const n = dRef.current ? dRef.current.frames.length : 0;
      let rate = FPS * speed;
      // Live, near the edge: play at the pace the MD makes frames, a poll
      // or so behind it (faster when further back, slower when closer).
      // At full speed the view caught up in a moment and then stood still
      // until the next poll: start, stop, start.
      if (runningRef.current && mdRate.current > 0) {
        const lag = 1.5 * POLL_S * mdRate.current;
        rate = Math.min(rate, mdRate.current * Math.max(0, n - 1 - iRef.current) / lag);
      }
      acc += ((now - last) / 1000) * rate;
      last = now;
      if (acc >= 1) {
        const step = Math.min(Math.floor(acc), 2 * Math.max(1, Math.round(speed)));
        acc = Math.min(acc - step, 1);   // late frames are dropped, not caught up
        if (n) {
          let k = iRef.current + step;
          if (k >= n) k = runningRef.current ? n - 1 : 0;   // live: wait at the edge; done: loop
          if (k !== iRef.current) drawFrame(k);
        }
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [playing, speed, open]);

  if (open) return html`<${EventReplay} jobId=${job.id} ev=${open} onClose=${() => { setOpen(null); stage.refit(); }} />`;
  if (d && !d.frames.length && d.prep) return html`<${PrepView} prep=${d.prep} running=${running} />`;
  if (!d || !d.frames.length) {
    return html`<div class="empty-hint"><p>${running ? 'The reactor is being packed and relaxed; the MD shows here as it runs.' : 'No reactor trajectory in this output.'}</p></div>`;
  }
  const n = d.frames.length;
  const pct = (fs) => `${Math.min(100, (100 * fs) / totalFs)}%`;
  const seek = (e) => {
    const box = e.currentTarget.getBoundingClientRect();
    const fs = ((e.clientX - box.left) / box.width) * totalFs;
    drawFrame(Math.round(fs / d.dump_fs / d.stride));
  };
  const on = new Set(activeIds ? activeIds.split(',').map(Number) : []);
  const active = d.events.filter((e) => on.has(e.event));
  const shown = [...d.events].sort((a, b) => a.start_fs - b.start_fs);
  return html`<div class="rx-live">
    <div class="rx-main">
      <div class="rx-stage-wrap">
        <div class="rx-stage" ref=${host}></div>
        <div class="rx-hud" ref=${hud}></div>
        ${d.radius.some(Boolean) && html`<div class="rx-legend small" title="The piston: a soft spherical wall that keeps the molecules together, and periodically closes in to push them into each other">
          <span class="lg-wall">◯ wall (wide)</span> <span class="lg-squeeze">◯ squeezing</span></div>`}
        ${active.length > 0 && html`<div class="rx-now">${active.map((e) => html`<div class=${`rx-eq ${e.tentative ? 'tentative' : ''}`} title=${e.label}>
          ${e.tentative ? 'analyzing: ' : ''}${short(e.label, 70)}</div>`)}</div>`}
      </div>
      <div class="rx-transport">
        <button class="btn small" onClick=${() => setPlaying(!playing)} title=${playing ? 'Pause' : 'Play'}>${playing ? '❚❚' : '▶'}</button>
        <div class="segmented">
          ${[0.5, 1, 2, 4].map((sp) => html`<button class=${speed === sp ? 'on' : ''} onClick=${() => setSpeed(sp)}>${sp}×</button>`)}
        </div>
        <div class="rx-timeline" onClick=${seek} title="Click to jump; marks are reaction events">
          <div class="rx-done" style=${{ width: pct(d.n_frames * d.dump_fs) }}></div>
          ${shown.map((e) => html`<i class=${`rx-tick ${on.has(e.event) ? 'on' : ''} ${e.reaction == null ? 'deg' : ''}`}
            style=${{ left: pct(e.start_fs) }} title=${`${fmtPs(e.start_fs)}: ${e.label}`}></i>`)}
          <div class="rx-head" ref=${headEl}></div>
        </div>
      </div>
    </div>
    <aside class="rx-events">
      <div class="section-title">Reaction events · ${shown.length}${d.final ? '' : ' so far'}</div>
      ${!shown.length && html`<p class="small muted">None yet.</p>`}
      <ol>
        ${shown.map((e) => html`<li class=${on.has(e.event) ? 'on' : ''}>
          <button class="rx-ev" onClick=${() => setOpen(e)} title="Open this reaction: its species, energies, complex, MD event and TS">
            <span class="rx-t">${fmtPs(e.start_fs)}</span>
            <span class="rx-eq">${e.label}</span>
            ${e.reaction == null && !e.tentative && html`<span class="small muted">exchange</span>`}
            ${e.tentative && html`<span class="small rx-analyzing" title="Still unfolding at the end of the trajectory: its bonds may change again">analyzing…</span>`}
          </button>
          <button class="btn-link small" title="Jump the reactor to this event"
            onClick=${() => { setPlaying(false); drawFrame(Math.floor(e.reactant_frame / d.stride)); }}>show</button>
        </li>`)}
      </ol>
    </aside>
  </div>`;
}
