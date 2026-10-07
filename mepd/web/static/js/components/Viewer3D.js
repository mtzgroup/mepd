// 3Dmol.js wrapper.
//
// Two modes: a single `xyz`, or a path as `frames` (xyz strings) + `frame`
// index. A path is parsed into 3Dmol once (addModelsAsFrames); scrubbing
// then only calls setFrame, instead of re-parsing and re-styling the
// molecule on every step. Swapping between molecules with the same atom
// count keeps the camera.
//
// QM/MM: with `system` (a QM/MM system id) or `region` (a region object),
// a structure of that system is drawn by region (see ../qmmm.js), with a
// small bar to show all / only the moving / only the QM atoms, and to colour
// the environment by how far it moved from the first frame.
// `onAtomClick(index)` makes atoms clickable (the region editor);
// `highlight` marks atoms.
import { html, useEffect, useMemo, useRef, useState } from '../lib.js';
import { drawLinks, ENV_MODES, parseXyz, styleRegion, useAutoSystem, useRegion } from '../qmmm.js';

function themeBackground() {
  return getComputedStyle(document.documentElement).getPropertyValue('--viewer-bg').trim() || '#ffffff';
}

const STYLES = {
  stick: { stick: { radius: 0.13 }, sphere: { scale: 0.24 } },
  sphere: { sphere: { scale: 0.32 }, stick: { radius: 0.12 } },
};

function atomCount(xyz) {
  return parseInt(xyz.trimStart().split('\n', 1)[0], 10) || 0;
}

export function Viewer3D({ xyz = null, frames = null, frame = 0, height = 260, labels = false, style = 'stick',
  system = null, region: regionIn = null, onAtomClick = null, highlight = null }) {
  const host = useRef(null);
  const viewer = useRef(null);
  const lastAtoms = useRef(0);
  const loaded = useRef(null);  // the frames array / xyz string currently in the viewer
  const [env, setEnv] = useState('all');
  const [motion, setMotion] = useState(false);
  const source = frames && frames.length ? frames : xyz;
  const first = source ? (Array.isArray(source) ? source[0] : source) : null;
  // No system given: recognize one by its atoms (live paths, trees, maps...).
  const auto = useAutoSystem(regionIn || system || onAtomClick ? null : first);
  const fetched = useRegion(regionIn ? null : (system || auto));
  const region = regionIn || fetched;
  const qmmm = !!(region && first && atomCount(first) === region.natoms);
  // Parsed coordinates, only when colouring by motion needs them.
  const coords = useMemo(() => (qmmm && motion && Array.isArray(source) ? source.map(parseXyz) : null), [qmmm, motion, source]);
  const clickRef = useRef(onAtomClick);
  clickRef.current = onAtomClick;

  useEffect(() => {
    if (!window.$3Dmol || !host.current) return undefined;
    viewer.current = window.$3Dmol.createViewer(host.current, { backgroundColor: themeBackground(), antialias: true });
    const mq = window.matchMedia('(prefers-color-scheme: dark)');
    const onTheme = () => { viewer.current?.setBackgroundColor(themeBackground()); viewer.current?.render(); };
    mq.addEventListener('change', onTheme);
    const ro = new ResizeObserver(() => { viewer.current?.resize(); viewer.current?.render(); });
    ro.observe(host.current);
    return () => { mq.removeEventListener('change', onTheme); ro.disconnect(); viewer.current?.clear(); viewer.current = null; };
  }, []);

  const drawLabels = (v) => {
    v.removeAllLabels();
    if (!labels) return;
    const model = v.getModel();
    if (!model) return;
    model.selectedAtoms({}).forEach((a, i) => v.addLabel(String(i), {
      position: { x: a.x, y: a.y, z: a.z }, fontSize: 10, backgroundOpacity: 0.55, inFront: true,
    }));
  };

  const shown = () => (frames && frames.length ? Math.min(frame, frames.length - 1) : 0);

  const applyStyle = (v) => {
    if (qmmm) {
      styleRegion(v, region, { env, motion, frame: shown(), coords, base: STYLES[style], highlight });
      drawLinks(v, env === 'qm' ? null : region);
    } else {
      v.setStyle({}, STYLES[style] || STYLES.stick);
      v.removeAllShapes();
      if (highlight && highlight.length) v.addStyle({ index: highlight }, { sphere: { scale: 0.42, color: '#3868b8', opacity: 0.55 } });
    }
    if (clickRef.current) {
      v.setClickable({}, true, (atom) => clickRef.current?.(atom.index));
    } else {
      v.setClickable({}, false);
    }
  };

  // (Re)load geometry only when the molecule/path itself changes.
  useEffect(() => {
    const v = viewer.current;
    if (!v) return;
    if (source === loaded.current) return;
    loaded.current = source;
    const view = v.getView();
    v.removeAllModels();
    v.removeAllShapes();
    if (!source) { v.removeAllLabels(); v.render(); return; }
    if (Array.isArray(source)) {
      v.addModelsAsFrames(source.map((s) => (s.endsWith('\n') ? s : `${s}\n`)).join(''), 'xyz');
      v.setFrame(shown());
    } else {
      v.addModel(source, 'xyz');
    }
    applyStyle(v);
    const n = atomCount(first);
    if (n === lastAtoms.current) v.setView(view);
    else if (qmmm) v.zoomTo({ index: region.qm_atoms }); else v.zoomTo();
    lastAtoms.current = n;
    drawLabels(v);
    v.render();
  }, [xyz, frames, style]);

  // Region, display mode or highlight changed: restyle in place.
  useEffect(() => {
    const v = viewer.current;
    if (!v || !loaded.current) return;
    applyStyle(v);
    v.render();
  }, [region, env, motion, coords, highlight, !!onAtomClick]);

  // A region arriving after the geometry: frame the QM atoms once.
  useEffect(() => {
    const v = viewer.current;
    if (v && qmmm && loaded.current) { v.zoomTo({ index: region.qm_atoms }); v.render(); }
  }, [qmmm && region?.id]);

  // Scrubbing: just switch frames.
  useEffect(() => {
    const v = viewer.current;
    if (!v || !frames || !frames.length || loaded.current !== frames) return;
    Promise.resolve(v.setFrame(shown())).then(() => {
      if (qmmm) {
        if (motion) styleRegion(v, region, { env, motion, frame: shown(), coords, base: STYLES[style], highlight });
        drawLinks(v, env === 'qm' ? null : region);
      }
      if (labels) drawLabels(v);
      v.render();
    });
  }, [frame, frames]);

  useEffect(() => {
    const v = viewer.current;
    if (!v) return;
    drawLabels(v);
    v.render();
  }, [labels]);

  if (!window.$3Dmol) {
    return html`<div class="viewer viewer-missing" style=${{ height }}>3D viewer unavailable (3Dmol.js did not load)</div>`;
  }
  return html`<div class="viewer-wrap">
    <div class="viewer" ref=${host} style=${{ height }}></div>
    ${qmmm && html`<div class="qmmm-bar small">
      <span class="qmmm-tag" title=${region.summary}>QM/MM</span>
      <span class="segmented mini">${ENV_MODES.map(([k, label]) => html`<button class=${env === k ? 'on' : ''}
        title=${k === 'all' ? 'QM region, moving and frozen environment' : k === 'active' ? 'Hide the frozen environment' : 'Only the QM atoms'}
        onClick=${() => setEnv(k)}>${label}</button>`)}</span>
      ${Array.isArray(source) && source.length > 1 && html`<label title="Colour the environment by how far it moved from the first frame: grey still, red 1 Å or more; a frozen atom that moved turns magenta">
        <input type="checkbox" checked=${motion} onChange=${(e) => setMotion(e.target.checked)} /> motion</label>`}
      <span class="qmmm-legend"><i class="qm"></i>QM <i class="link"></i>cut bond <i class="act"></i>moving <i class="frz"></i>frozen</span>
    </div>`}
  </div>`;
}
