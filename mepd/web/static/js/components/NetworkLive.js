// Live view of a reaction network expansion: one stage where the seed turns
// into each proposed product in turn (every animation starts from the seed
// in the same pose), after which the product flies into a growing grid of
// found structures. Click a structure to replay how it formed and see
// whether it is still being minimized or what it became.
import { html, useEffect, useMemo, useRef, useState } from '../lib.js';
import { api } from '../api.js';
import { depictUrl } from '../util.js';
import { Viewer3D } from './Viewer3D.js';

const FRAME_MS = 65, HOLD_MS = 800, FLY_MS = 700, REST_MS = 350;

function statusText(r) {
  return r.outcome || (r.state === 'queued' ? 'queued' : r.state === 'failed' ? 'failed' : 'minimizing');
}

function chipClass(r) {
  return r.outcome ? `outcome-${r.outcome.replace(/\s+/g, '-')}` : r.state === 'queued' ? 'queued' : 'running';
}

function deltaE(r) {
  const [a, b] = r.plot?.y || [];
  return a != null && b != null ? b - a : null;
}

function signed(v) {
  return `${v >= 0 ? '+' : '−'}${Math.abs(v).toFixed(1)}`;
}

const sleep = (ms) => new Promise((res) => setTimeout(res, ms));

export function NetworkLive({ job, reactions }) {
  const byId = useMemo(() => Object.fromEntries(reactions.map((r) => [r.id, r])), [reactions]);
  const ordered = useMemo(() => [...reactions].sort((a, b) => a.id.localeCompare(b.id)), [reactions]);
  const [played, setPlayed] = useState(() => new Set());   // shown in the grid
  const [flying, setFlying] = useState(null);              // on its way to the grid
  const [current, setCurrent] = useState(null);            // { id, frames } on the stage
  const [frame, setFrame] = useState(0);
  const [pinned, setPinned] = useState(null);              // clicked: loops on the stage
  const [filter, setFilter] = useState('all');
  const [seedFrames, setSeedFrames] = useState(null);
  const cache = useRef({});
  const stageRef = useRef(null);
  const tileRefs = useRef({});
  const run = useRef(0);                                  // cancels a running sequence

  useEffect(() => { setPlayed(new Set()); setCurrent(null); setPinned(null); cache.current = {}; }, [job.id]);

  // Every frame of a reaction's animation (finished ones arrive with only
  // their last frame; the rest is fetched once per update).
  const load = async (r) => {
    const key = `${r.id}:${r.updated}`;
    if (cache.current[key]) return cache.current[key];
    let frames = r.geometry?.frames || [];
    if (r.geometry?.truncated || frames.length < 2) {
      try {
        frames = (await api.get(`/api/jobs/${job.id}/live/${encodeURIComponent(r.id)}`)).geometry?.frames || frames;
      } catch { /* keep what we have */ }
    }
    cache.current[key] = frames;
    if (frames.length && !seedFrames) setSeedFrames([frames[0]]);
    return frames;
  };

  const markPlayed = (id) => setPlayed((s) => (s.has(id) ? s : new Set([...s, id])));

  // Play one reaction on the stage: morph, hold on the product, then (auto
  // mode) fly it into the grid or (pinned) loop.
  const play = async (id, loop) => {
    const token = ++run.current;
    const alive = () => run.current === token;
    const r = byId[id];
    if (!r) return;
    const frames = await load(r);
    if (!alive() || !frames.length) return;
    const waiting = ordered.filter((x) => !played.has(x.id)).length;
    const speed = loop ? 1 : waiting > 20 ? 3 : waiting > 8 ? 2 : 1;   // catch up on a long backlog
    do {
      setCurrent({ id, frames });
      for (let i = 0; i < frames.length; i += 1) {
        setFrame(i);
        await sleep(FRAME_MS / speed);
        if (!alive()) return;
      }
      await sleep(HOLD_MS / speed);
      if (!alive()) return;
      if (loop) { setFrame(0); await sleep(REST_MS); }
    } while (loop && alive());
    setFlying(id);
    await sleep(30);                                        // let the target tile mount
    await flyToTile(id, FLY_MS / Math.min(speed, 2));
    if (!alive()) return;
    markPlayed(id);
    setFlying(null);
    setCurrent(null);
    await sleep(REST_MS / speed);
    if (alive()) run.current += 1;                           // idle: the director picks the next
  };

  const flyToTile = async (id, ms) => {
    const r = byId[id];
    const tile = tileRefs.current[id];
    const stage = stageRef.current;
    const smiles = r?.plot?.product_smiles;
    if (!tile || !stage || !smiles) return;
    tile.scrollIntoView?.({ block: 'nearest', behavior: 'smooth' });
    await sleep(60);
    const from = stage.getBoundingClientRect();
    const target = (tile.querySelector('.nl-depict') || tile).getBoundingClientRect();
    const ghost = document.createElement('img');
    ghost.src = depictUrl(smiles, 180, 120);
    ghost.className = 'nl-ghost';
    Object.assign(ghost.style, { left: `${target.left}px`, top: `${target.top}px`, width: `${target.width}px`, height: `${target.height}px` });
    document.body.appendChild(ghost);
    const dx = from.left + from.width / 2 - (target.left + target.width / 2);
    const dy = from.top + from.height / 2 - (target.top + target.height / 2);
    const start = `translate(${dx}px, ${dy}px) scale(2.2)`;
    try {
      await ghost.animate([
        { transform: start, opacity: 0 },
        { transform: start, opacity: 1, offset: 0.18 },
        { transform: 'translate(0, 0) scale(1)', opacity: 1 },
      ], { duration: ms, easing: 'cubic-bezier(.45,.05,.25,1)' }).finished;
    } catch { /* animation cancelled */ }
    ghost.remove();
  };

  // The director: in auto mode, play the next reaction nobody has seen.
  const [tick, setTick] = useState(0);
  useEffect(() => {
    if (pinned || current || flying) return undefined;
    const next = ordered.find((r) => !played.has(r.id));
    if (!next) return undefined;
    const t = setTimeout(() => play(next.id, false).finally(() => setTick((k) => k + 1)), 50);
    return () => clearTimeout(t);
  }, [pinned, current, flying, ordered, played, tick]);

  // A pinned reaction that changed (e.g. finished minimizing) replays with its new frames.
  const pinnedUpdated = pinned ? byId[pinned]?.updated : null;
  useEffect(() => { if (pinned) play(pinned, true); }, [pinned, pinnedUpdated]);

  const pin = (id) => {
    if (current && !pinned) markPlayed(current.id);
    setFlying(null);
    setPinned(id);
  };
  const unpin = () => { run.current += 1; setPinned(null); setCurrent(null); setFlying(null); };
  const skip = () => { run.current += 1; setPlayed(new Set(ordered.map((r) => r.id))); setCurrent(null); setFlying(null); };

  const onStage = current ? byId[current.id] : null;
  const stageFrames = current?.frames || seedFrames;
  const shown = ordered.filter((r) => played.has(r.id) || r.id === flying || r.id === pinned);
  const visible = filter === 'new' ? shown.filter((r) => r.outcome === 'new species' || r.id === flying) : shown;
  const waiting = ordered.length - ordered.filter((r) => played.has(r.id)).length;
  const counts = {
    minimizing: ordered.filter((r) => !r.outcome && r.state !== 'queued').length,
    queued: ordered.filter((r) => r.state === 'queued').length,
    fresh: ordered.filter((r) => r.outcome === 'new species').length,
  };
  const seedSmiles = ordered[0]?.plot?.reactant_smiles;

  return html`
    <div class="netlive">
      <div class="nl-stage card-block" ref=${stageRef}>
        ${stageFrames
          ? html`<${Viewer3D} frames=${stageFrames} frame=${current ? Math.min(frame, stageFrames.length - 1) : 0} height=${420} />`
          : html`<div class="nl-empty muted small">${job.status === 'running'
            ? 'Enumerating products of the seed… each proposed reaction plays here as soon as it is built.'
            : 'No proposed reactions.'}</div>`}
        <div class="nl-caption">
          ${onStage
            ? html`
              <div class="nl-title"><b>#${Number(onStage.id.slice(3))}</b>
                <span class="mono">${onStage.plot.reactant_smiles || '?'}</span> → <span class="mono">${onStage.plot.product_smiles || '?'}</span>
                <span class=${`live-state ${chipClass(onStage)}`}>${statusText(onStage)}</span>
                ${deltaE(onStage) != null && html`<span class="mono muted">${signed(deltaE(onStage))} kcal/mol</span>`}</div>
              <div class="small muted">${onStage.caption || ''}</div>`
            : html`<div class="nl-title">${seedSmiles ? html`Seed <span class="mono">${seedSmiles}</span>` : 'Seed'}</div>
              <div class="small muted">${waiting ? `${waiting} reaction(s) to show` : ordered.length ? 'Every proposed reaction is in the list. Click one to replay it.' : ''}</div>`}
        </div>
        <div class="nl-controls">
          ${pinned && html`<button class="btn small" onClick=${unpin}>Back to live</button>`}
          ${!pinned && waiting > 1 && html`<button class="btn small" onClick=${skip}>Skip ahead (${waiting})</button>`}
        </div>
      </div>
      <div class="nl-list card-block">
        <div class="nl-list-head">
          <h4>Found structures</h4>
          <div class="segmented small">
            <button class=${filter === 'all' ? 'on' : ''} onClick=${() => setFilter('all')}>All ${shown.length}</button>
            <button class=${filter === 'new' ? 'on' : ''} onClick=${() => setFilter('new')}>New species ${shown.filter((r) => r.outcome === 'new species').length}</button>
          </div>
        </div>
        <div class="small muted nl-counts">${ordered.length} proposed · ${counts.minimizing} minimizing · ${counts.queued} queued · ${counts.fresh} new species</div>
        <div class="nl-grid">
          ${visible.map((r) => html`
            <button type="button" key=${r.id} ref=${(el) => { if (el) tileRefs.current[r.id] = el; }}
                class=${`nl-tile ${r.id === flying ? 'arriving' : ''} ${r.id === pinned ? 'pinned' : ''} ${r.outcome && r.outcome !== 'new species' ? 'dim' : ''}`}
                onClick=${() => pin(r.id)} title=${r.caption || ''}>
              ${r.plot?.product_smiles
                ? html`<img class="nl-depict" src=${depictUrl(r.plot.product_smiles, 180, 120)} alt=${r.plot.product_smiles} />`
                : html`<div class="nl-depict"></div>`}
              <div class="nl-tile-foot">
                <span class="small"><b>#${Number(r.id.slice(3))}</b>${deltaE(r) != null ? html` <span class="mono muted">${signed(deltaE(r))}</span>` : ''}</span>
                <span class=${`live-state ${chipClass(r)}`}>${statusText(r)}</span>
              </div>
            </button>`)}
          ${!visible.length && html`<p class="small muted">${ordered.length ? 'Products land here as they are shown.' : 'Nothing yet.'}</p>`}
        </div>
      </div>
    </div>`;
}
