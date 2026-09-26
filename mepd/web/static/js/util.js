import { state } from './store.js';

export const STATUS_LABEL = {
  queued: 'Queued', running: 'Running', done: 'Done', failed: 'Failed',
  cancelled: 'Cancelled', interrupted: 'Interrupted',
};

export function fmtDuration(sec) {
  if (sec == null || !isFinite(sec)) return '';
  sec = Math.max(0, Math.round(sec));
  if (sec < 60) return `${sec}s`;
  const m = Math.floor(sec / 60), s = sec % 60;
  if (m < 60) return `${m}m ${String(s).padStart(2, '0')}s`;
  const h = Math.floor(m / 60);
  return `${h}h ${String(m % 60).padStart(2, '0')}m`;
}

export function jobElapsed(job, now = Date.now() / 1000) {
  if (!job.started) return null;
  return (job.finished || now) - job.started;
}

export function fmtAgo(t) {
  if (!t) return '';
  const d = Date.now() / 1000 - t;
  if (d < 60) return 'just now';
  if (d < 3600) return `${Math.floor(d / 60)} min ago`;
  if (d < 86400) return `${Math.floor(d / 3600)} h ago`;
  return new Date(t * 1000).toLocaleDateString();
}

export function fmtKcal(x, digits = 1) {
  return x == null ? '—' : `${x.toFixed(digits)}`;
}

export function jobsFor({ structure, edge }) {
  return Object.values(state.jobs)
    .filter((j) => (edge && j.targets.edges.includes(edge)) || (structure && j.targets.structures.includes(structure)))
    .sort((a, b) => b.created - a.created);
}

// What an edge "knows": the most urgent job state plus the best barrier any
// finished job (or the result it was imported from) reported for it.
export function edgeStatus(edge, jobs = state.jobs) {
  const related = Object.values(jobs).filter((j) => j.targets.edges.includes(edge.id));
  const has = (st) => related.some((j) => j.status === st);
  const done = related.filter((j) => j.status === 'done' && j.summary?.barrier_kcal != null);
  // A job's barrier only counts for this edge if its IRCs connect the two
  // ends (summary.barrier_verified); otherwise it is shown as unconfirmed.
  const verified = done.filter((j) => j.summary.barrier_verified !== false).map((j) => j.summary.barrier_kcal);
  if (edge.origin?.barrier_kcal != null) verified.push(edge.origin.barrier_kcal);
  const unverified = done.filter((j) => j.summary.barrier_verified === false).map((j) => j.summary.barrier_kcal);
  const barrier = verified.length ? Math.min(...verified) : null;
  const barrierUnverified = !verified.length && unverified.length ? Math.min(...unverified) : null;
  let status = 'idle';
  if (has('running')) status = 'running';
  else if (has('queued')) status = 'queued';
  else if (barrier != null || barrierUnverified != null || has('done') || (edge.origin?.kind === 'job' && !edge.origin.proposed)) status = 'done';
  else if (has('failed')) status = 'failed';
  // Proposed by a network expansion, no path search yet.
  else if (edge.origin?.proposed) status = 'proposed';
  return { status, barrier, barrierUnverified, count: related.length };
}

// Latest status line: live progress if we have it, else what the record had.
export function lastLine(job, progress = state.progress) {
  return progress[job.id]?.last_line ?? job.last_line ?? '';
}

// Everything the graph's edge styling depends on, as one string: the graph
// re-syncs only when this changes, not on every job/progress event.
export function edgeStatusKey(jobs) {
  return Object.values(jobs)
    .filter((j) => j.targets.edges.length)
    .map((j) => `${j.id}:${j.status}:${j.summary?.barrier_kcal ?? ''}:${j.summary?.barrier_verified}:${j.targets.edges.join(',')}`)
    .sort()
    .join('|');
}

// Where a structure stands relative to a level of theory (default: the
// workspace's): {kind: 'ok'|'other'|'none'|'busy'|'failed', text, title}.
export function isTs(rec) {
  return rec.role === 'ts' || (rec.role == null && /\[TS\]$/.test(rec.name || ''));
}

// Playground layout: the part of the canvas the floating controls leave free.
export function playgroundFitMargins() {
  const phone = window.matchMedia('(max-width: 720px), (max-height: 520px)').matches;
  return phone ? { l: 16, t: 140, r: 16, b: 70 } : { l: 40, t: 110, r: 330, b: 36 };
}

export function levelStatus(rec, levelKey = state.levels[state.levelProfile ?? '']?.key) {
  if (isTs(rec)) {
    const at = rec.level ? ` at ${rec.level.label}` : '';
    return { kind: 'ts', text: rec.level ? `TS · ${rec.level.label}` : 'TS', title: `Saddle point${at}; never minimized` };
  }
  if (rec.status === 'optimizing') return { kind: 'busy', text: 'optimizing…', title: 'Being minimized at the workspace level of theory' };
  if (rec.reacted && rec.status === 'ready') return { kind: 'changed', text: 'reacted on minimization', title: rec.status_error || `Reacted while being minimized: ${rec.reacted.from} → ${rec.reacted.to}` };
  if (rec.status === 'not_minimum') return { kind: 'failed', text: 'not a minimum', title: rec.status_error || 'Hessian has an imaginary frequency' };
  if (rec.status === 'opt_failed') return { kind: 'failed', text: 'opt failed', title: rec.status_error || 'Optimization failed' };
  if (!rec.level) {
    const why = rec.origin?.kind === 'smiles' ? 'RDKit/MMFF embedding' : rec.origin?.kind === 'job' ? 'from an imported output (level unknown)' : 'geometry as given';
    return { kind: 'none', text: 'not optimized', title: `Not at any QM level: ${why}` };
  }
  if (rec.level.key !== levelKey) {
    return { kind: 'other', text: rec.level.label ? `${rec.level.label} · other level` : 'other level', title: `Optimized at ${rec.level.profile ?? 'built-in defaults'} (${rec.level.label}), not the workspace level` };
  }
  return { kind: 'ok', text: rec.level.label || 'optimized', title: `Minimum at the workspace level (${rec.level.profile ?? 'built-in defaults'}: ${rec.level.label})` };
}

export function structureName(id) {
  return state.workspace.structures[id]?.name ?? '(deleted)';
}

export function depictUrl(smiles, w = 220, h = 160) {
  return smiles ? `/api/depict?smiles=${encodeURIComponent(smiles)}&w=${w}&h=${h}` : null;
}

export function readFileText(file) {
  return new Promise((resolve, reject) => {
    const r = new FileReader();
    r.onload = () => resolve(r.result);
    r.onerror = () => reject(r.error);
    r.readAsText(file);
  });
}

export function copy(text) {
  try { navigator.clipboard.writeText(text); return true; } catch { return false; }
}

export function downloadText(filename, text, mime = 'text/plain') {
  const url = URL.createObjectURL(new Blob([text], { type: mime }));
  const a = Object.assign(document.createElement('a'), { href: url, download: filename });
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export function safeName(s) {
  return String(s).replace(/[^A-Za-z0-9._-]+/g, '_').replace(/^_+|_+$/g, '').slice(0, 80) || 'mepd';
}

// One xyz frame with a readable comment line (label, frame, energies)
// instead of qcdata's metadata line.
export function xyzWithComment(xyz, comment) {
  const lines = xyz.replace(/\s+$/, '').split('\n');
  lines[1] = comment.replace(/\n/g, ' ');
  return `${lines.join('\n')}\n`;
}

export function entryToXyz(entry, frames = null) {
  const idx = frames ?? entry.frames.map((_, i) => i);
  return idx.map((i) => {
    const f = entry.frames[i];
    const parts = [entry.label, `frame ${i + 1}/${entry.frames.length}`];
    if (i === entry.ts_index) parts.push('TS');
    if (f.energy_hartree != null) parts.push(`E=${f.energy_hartree.toFixed(8)} Eh`);
    if (f.energy_kcal != null) parts.push(`rel=${f.energy_kcal.toFixed(3)} kcal/mol`);
    return xyzWithComment(f.xyz, parts.join(' | '));
  }).join('');
}

// The TS geometry of an entry: its highest point, or its only frame.
export function tsFrameIndex(entry) {
  return entry.ts_index ?? (entry.frames.length === 1 ? 0 : null);
}

// Same breakpoint as the phone layout in style.css.
export const PHONE_QUERY = '(max-width: 720px), (max-height: 520px)';

export function cls(...parts) {
  return parts.filter(Boolean).join(' ');
}

// Conformers of a node, lowest first. ΔE only between conformers whose
// energies come from the same level of theory as the representative's.
export function conformerRows(rec) {
  const confs = rec?.conformers || [];
  const repConf = confs.find((c) => c.id === rec.conformer);
  const key = repConf?.level?.key;
  const same = confs.filter((c) => c.energy != null && c.level?.key === key);
  const floor = same.length ? Math.min(...same.map((c) => c.energy)) : null;
  const rows = confs.map((c, i) => ({
    ...c,
    index: i,
    rep: c.id === rec.conformer,
    dE: floor != null && c.energy != null && c.level?.key === key ? (c.energy - floor) * 627.509474 : null,
    source: c.origin?.kind === 'job' ? c.origin.label : c.origin?.kind === 'smiles' ? `SMILES ${c.origin.input}` : (c.origin?.kind || ''),
  }));
  rows.sort((a, b) => (a.dE ?? 1e9) - (b.dE ?? 1e9) || a.index - b.index);
  return rows;
}

export function conformerLabel(r) {
  const e = r.dE != null ? `${r.dE >= 0 ? '+' : ''}${r.dE.toFixed(1)} kcal/mol` : (r.level ? r.level.label : 'no energy');
  return `#${r.index + 1} · ${e}${r.rep ? ' · lowest' : ''}`;
}

