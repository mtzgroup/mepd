// A complex being built, live (mepd.web.complex_live): each molecule as the
// program places it -- arriving beside the others, the site found, the pose
// settled -- or the solvent shell growing, or CREST's dynamics exploring
// arrangements. Only while the job runs: no replay afterwards.
import { html, useEffect, useRef, useState } from '../lib.js';
import { api } from '../api.js';
import { FULL, useStage, xyzText } from './ReactorLive.js';

const STEP_S = 1 / 15;     // trajectory frames per second: 15
const TWEEN_S = 0.8;       // a pose settling into the next one
const HOLD_S = 0.7;        // a newcomer shown beside the others before it moves in

const ABOUT = {
  dock: 'xtb docks one molecule at a time: each newcomer waits beside the others while xtb searches for where it binds, then moves into the pose it settled in.',
  nci: 'First docked, then CREST runs dynamics on the complex to explore other arrangements of its molecules, and keeps the best few.',
  qcg: 'CREST adds the solvent molecules around the solute one at a time, then optimizes the cluster.',
};

function extentOf(frames) {
  let r = 0;
  for (const f of frames) for (let k = 0; k < f.xyz.length; k += 1) r = Math.max(r, Math.abs(f.xyz[k]));
  return r + 1;
}

export function ComplexLive({ job }) {
  const running = ['running', 'queued'].includes(job.status);
  const host = useRef(null);
  const stage = useStage(host);
  const [data, setData] = useState(null);
  const framesRef = useRef([]);
  const stageRef = useRef('');
  const hud = useRef(null);
  const reach = useRef(0);
  const restart = useRef(false);

  useEffect(() => {
    if (!running) return undefined;
    let alive = true, timer;
    const pull = async () => {
      try {
        const r = await api.get(`/api/jobs/${job.id}/complex-live`);
        if (!alive) return;
        // A program that starts a file over (CREST's second round): play from there.
        const old = framesRef.current;
        if (r.frames.length < old.length) restart.current = true;
        framesRef.current = r.frames;
        stageRef.current = r.stage;
        setData(r);
      } catch (e) { /* not started yet */ }
      if (alive) timer = setTimeout(pull, 1500);
    };
    pull();
    return () => { alive = false; clearTimeout(timer); };
  }, [job.id, running]);

  const hasFrames = !!(data && data.frames.length);
  useEffect(() => {
    if (!running || !hasFrames) return undefined;
    let raf, last = performance.now();
    let i = 0, t = 0;                     // showing frame i, t seconds into it
    const draw = (flat, frame) => {
      const fs = framesRef.current;
      const r = extentOf(fs);
      const refit = r > reach.current + 1.5;
      if (refit) reach.current = r;
      stage.draw((v) => {
        v.addModel(xyzText(frame.symbols, flat), 'xyz');
        v.setStyle({}, FULL);
      }, { extent: reach.current, refit });
      if (hud.current) hud.current.textContent = `${frame.caption} · live`;
    };
    const tick = (now) => {
      const dt = (now - last) / 1000;
      last = now;
      const fs = framesRef.current;
      if (fs.length) {
        if (restart.current) { restart.current = false; i = Math.max(0, fs.length - 1); t = 0; draw(fs[i].xyz, fs[i]); }
        i = Math.min(i, fs.length - 1);
        const next = fs[i + 1];
        const cur = fs[i];
        const blend = next && next.tween && next.symbols.length === cur.symbols.length;
        if (!next) {                       // caught up: wait here for the program, saying what it does
          t = 0;
          if (hud.current && stageRef.current && stageRef.current !== cur.caption) hud.current.textContent = `${stageRef.current} · live`;
        }
        else {
          const dur = blend ? TWEEN_S : /binds/.test(cur.caption) ? HOLD_S : STEP_S;
          t += dt;
          if (t >= dur) { i += 1; t = 0; draw(fs[i].xyz, fs[i]); }
          else if (blend) {
            const u = t / dur, e = u * u * (3 - 2 * u);
            draw(cur.xyz.map((x, k) => x + (next.xyz[k] - x) * e), next);
          }
        }
      }
      raf = requestAnimationFrame(tick);
    };
    draw(framesRef.current[0].xyz, framesRef.current[0]);
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [running, hasFrames]);

  if (!running) {
    return html`<div class="empty-hint"><p>The build is shown live while the complex is being built. ${job.status === 'done' ? 'See Results for what it found; Explore has the complex.' : ''}</p></div>`;
  }
  if (!hasFrames) {
    return html`<div class="empty-hint"><p>${data?.stage || 'Building the complex…'}
      ${data?.method === 'packed' ? '' : ' The molecules show here as soon as the program places them.'}</p></div>`;
  }
  return html`<div class="rx-live">
    <div class="rx-main">
      <div class="rx-stage-wrap">
        <div class="rx-stage" ref=${host}></div>
        <div class="rx-hud" ref=${hud}></div>
      </div>
      <p class="small muted">${data.stage}</p>
    </div>
    <aside class="rx-events">
      <div class="section-title">Building the complex</div>
      <p class="small muted">${ABOUT[data.method] || ''} Shown as the program writes it; the geometries are then
        minimized at the workspace level.</p>
    </aside>
  </div>`;
}
