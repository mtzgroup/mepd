// QM/MM systems in the browser: fetching a system's region (cached per
// region signature) and how a 3D view draws it.
//
// Drawing (Viewer3D's `system` / `region` props):
//   QM atoms           ball and stick, element colours
//   cut QM–MM bonds    a dashed orange bond, and the link H that caps it
//   moving environment thin sticks
//   frozen environment grey lines
// With `motion`, environment atoms are coloured by how far they moved
// from the first frame (grey = still, red = ≥ 1 Å), and a frozen atom that
// moved at all turns magenta: the quickest way to see the environment
// doing something it should not.
import { useEffect, useState } from './lib.js';
import { api } from './api.js';
import { useStore } from './store.js';

const cache = new Map();   // `${sysid}:${sig}` -> Promise<region>

export function fetchRegion(sysid, sig = '') {
  const key = `${sysid}:${sig}`;
  if (!cache.has(key)) cache.set(key, api.get(`/api/qmmm/systems/${sysid}`).catch((e) => { cache.delete(key); throw e; }));
  return cache.get(key);
}

export function useRegion(sysid) {
  const sig = useStore((s) => (sysid ? s.workspace.qmmm_systems?.[sysid]?.sig : null));
  const [region, setRegion] = useState(null);
  useEffect(() => {
    let live = true;
    if (!sysid) { setRegion(null); return undefined; }
    fetchRegion(sysid, sig || '').then((r) => live && setRegion(r)).catch(() => live && setRegion(null));
    return () => { live = false; };
  }, [sysid, sig]);
  return region;
}

export const ENV_MODES = [['all', 'All'], ['active', 'Moving'], ['qm', 'QM only']];

// Atom index sets of a region, built once per region object.
const sets = new WeakMap();
export function regionSets(region) {
  if (!region) return null;
  if (!sets.has(region)) {
    const qm = new Set(region.qm_atoms);
    const frozen = new Set(region.frozen_atoms);
    const all = [...Array(region.natoms).keys()];
    sets.set(region, {
      qm: region.qm_atoms,
      frozen: region.frozen_atoms,
      active: all.filter((i) => !qm.has(i) && !frozen.has(i)),
      mm: all.filter((i) => !qm.has(i)),
      qmSet: qm,
      frozenSet: frozen,
    });
  }
  return sets.get(region);
}

export function parseXyz(text) {
  const lines = text.trim().split('\n');
  const n = parseInt(lines[0], 10) || 0;
  const out = new Float64Array(3 * n);
  for (let i = 0; i < n; i += 1) {
    const p = (lines[2 + i] || '').trim().split(/\s+/);
    out[3 * i] = +p[1]; out[3 * i + 1] = +p[2]; out[3 * i + 2] = +p[3];
  }
  return out;
}

function motionColor(d, frozen) {
  if (frozen) return d > 0.01 ? '#e0119d' : '#9aa3ad';
  const t = Math.max(0, Math.min(1, d / 1.0));
  // grey (still) -> amber -> red (moved 1 Å or more)
  const a = [154, 163, 173], b = [235, 160, 40], c = [210, 40, 40];
  const [p, q, u] = t < 0.5 ? [a, b, t / 0.5] : [b, c, (t - 0.5) / 0.5];
  const m = p.map((x, i) => Math.round(x + (q[i] - x) * u));
  return `rgb(${m[0]},${m[1]},${m[2]})`;
}

// Style the viewer's model for a region. `frame`: the frame shown (for
// motion colours), `coords`: parsed frames (Float64Array each) or null.
export function styleRegion(v, region, { env = 'all', motion = false, frame = 0, coords = null, base = null, highlight = null } = {}) {
  const S = regionSets(region);
  if (!S) return;
  const model = v.getModel();
  if (!model) return;
  const atoms = model.selectedAtoms({});
  if (atoms.length !== region.natoms) { v.setStyle({}, base || { stick: { radius: 0.13 }, sphere: { scale: 0.24 } }); return; }
  v.setStyle({}, {});
  v.setStyle({ index: S.qm }, { stick: { radius: 0.15 }, sphere: { scale: 0.27 } });
  if (env !== 'qm') {
    const moving = S.active;
    let colorOf = null;
    if (motion && coords && coords.length) {
      const c0 = coords[0], ck = coords[Math.min(frame, coords.length - 1)];
      colorOf = (i) => {
        const dx = ck[3 * i] - c0[3 * i], dy = ck[3 * i + 1] - c0[3 * i + 1], dz = ck[3 * i + 2] - c0[3 * i + 2];
        return motionColor(Math.sqrt(dx * dx + dy * dy + dz * dz), S.frozenSet.has(i));
      };
    }
    if (moving.length) {
      v.setStyle({ index: moving }, colorOf
        ? { stick: { radius: 0.08, colorfunc: (a) => colorOf(a.index) } }
        : { stick: { radius: 0.08 } });
    }
    if (env === 'all' && S.frozen.length) {
      v.setStyle({ index: S.frozen }, colorOf
        ? { line: { colorfunc: (a) => colorOf(a.index) } }
        : { line: { color: '#9aa3ad' } });
    }
  }
  if (highlight && highlight.length) {
    v.addStyle({ index: highlight }, { sphere: { scale: 0.42, color: '#3868b8', opacity: 0.55 } });
  }
}

// Dashed cut bonds and link hydrogens at the current frame's positions.
export function drawLinks(v, region) {
  v.removeAllShapes();
  if (!region?.links?.length) return;
  const model = v.getModel();
  if (!model) return;
  const atoms = model.selectedAtoms({});
  if (atoms.length !== region.natoms) return;
  region.links.forEach(([q, m], k) => {
    const A = atoms[q], B = atoms[m];
    if (!A || !B) return;
    const g = region.link_ratios?.[k] ?? 0.72;
    const L = { x: A.x + g * (B.x - A.x), y: A.y + g * (B.y - A.y), z: A.z + g * (B.z - A.z) };
    v.addCylinder({ start: { x: A.x, y: A.y, z: A.z }, end: { x: B.x, y: B.y, z: B.z }, radius: 0.09,
      dashed: true, dashLength: 0.18, gapLength: 0.12, color: '#e8890c', fromCap: 1, toCap: 1 });
    v.addSphere({ center: L, radius: 0.22, color: '#f5b5d2' });
  });
}

// Which QM/MM system a geometry belongs to, worked out from its atoms (the
// same rule as the server's Workspace.qmmm_system_of): same atom count, and
// the same element sequence where the browser can hash it.
const atomsKeyCache = new Map();
async function atomsKey(symbols) {
  const text = symbols.join('|');
  if (atomsKeyCache.has(text)) return atomsKeyCache.get(text);
  let key = null;
  try {
    const buf = await crypto.subtle.digest('SHA-1', new TextEncoder().encode(text));
    key = [...new Uint8Array(buf)].map((b) => b.toString(16).padStart(2, '0')).join('').slice(0, 12);
  } catch { key = null; }   // no WebCrypto (plain http): match on the atom count alone
  atomsKeyCache.set(text, key);
  return key;
}

function symbolsOf(xyz) {
  const lines = xyz.trimStart().split('\n');
  const n = parseInt(lines[0], 10) || 0;
  return lines.slice(2, 2 + n).map((l) => l.trim().split(/\s+/)[0]);
}

export function useAutoSystem(xyz) {
  const systems = useStore((s) => s.workspace.qmmm_systems);
  const [found, setFound] = useState(null);
  useEffect(() => {
    let live = true;
    setFound(null);
    if (!xyz || !systems) return undefined;
    const n = parseInt(xyz.trimStart().split('\n', 1)[0], 10) || 0;
    const cands = Object.values(systems).filter((s) => s.natoms === n);
    if (!cands.length) return undefined;
    atomsKey(symbolsOf(xyz)).then((key) => {
      if (!live) return;
      const hit = key ? cands.find((s) => s.atoms_key === key) : (cands.length === 1 ? cands[0] : null);
      setFound(hit ? hit.id : null);
    });
    return () => { live = false; };
  }, [xyz, systems]);
  return found;
}
