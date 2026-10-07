// QM/MM in the UI: a system's region (what is QM, what moves, what is
// frozen) and editing it, and the checks shown next to every QM/MM result
// so that a path through an environment can be seen to make sense.
import { html, useEffect, useMemo, useState } from '../lib.js';
import { api, attempt } from '../api.js';
import { useStore } from '../store.js';
import { useRegion } from '../qmmm.js';
import { Viewer3D } from './Viewer3D.js';

// "0-3 7 9" <-> [0,1,2,3,7,9]
export function parseIdx(text) {
  const out = new Set();
  for (const tok of String(text || '').replace(/,/g, ' ').split(/\s+/)) {
    if (!tok) continue;
    const m = tok.match(/^(\d+)-(\d+)$/);
    if (m) for (let i = +m[1]; i <= +m[2]; i += 1) out.add(i);
    else if (/^\d+$/.test(tok)) out.add(+tok);
  }
  return [...out].sort((a, b) => a - b);
}

export function fmtIdx(list) {
  const idx = [...new Set(list)].sort((a, b) => a - b);
  const parts = [];
  for (let k = 0; k < idx.length;) {
    let j = k;
    while (j + 1 < idx.length && idx[j + 1] === idx[j] + 1) j += 1;
    parts.push(j === k ? `${idx[k]}` : `${idx[k]}-${idx[j]}`);
    k = j + 1;
  }
  return parts.join(' ');
}

const MM_LABELS = { gfnff: 'GFN-FF', tip3p: 'TIP3P water (electrostatic)', gfn2: 'GFN2-xTB', gfn1: 'GFN1-xTB',
  amber: 'AMBER (OpenMM)', terachem: 'TeraChem QM/MM' };

// Click atoms (or type indices) to choose the QM region; see the cut bonds,
// moving shell and problems before saving.
function RegionEditor({ rec, xyz, region, onDone }) {
  const [qm, setQm] = useState(region ? region.qm_text : '');
  const [qmCharge, setQmCharge] = useState(region ? region.qm_charge : rec.charge);
  const [mult, setMult] = useState(region ? region.qm_multiplicity : rec.multiplicity);
  const [radius, setRadius] = useState(region?.active_radius ?? 6);
  const [mm, setMm] = useState(region?.mm || 'gfnff');
  const [embedding, setEmbedding] = useState(region?.embedding || 'mechanical');
  const [preview, setPreview] = useState(null);
  const [busy, setBusy] = useState(false);
  const sel = useMemo(() => parseIdx(qm), [qm]);
  useEffect(() => {
    if (!sel.length) { setPreview(null); return undefined; }
    let live = true;
    const t = setTimeout(() => {
      api.post('/api/qmmm/preview', { structure: rec.id, qm_atoms: fmtIdx(sel), qm_charge: +qmCharge,
        qm_multiplicity: +mult, active_radius: radius === '' ? null : +radius, mm, embedding })
        .then((p) => live && setPreview(p)).catch((e) => live && setPreview({ error: e.message }));
    }, 250);
    return () => { live = false; clearTimeout(t); };
  }, [fmtIdx(sel), qmCharge, mult, radius, mm, embedding]);
  const toggle = (i) => {
    const s = new Set(sel);
    if (s.has(i)) s.delete(i); else s.add(i);
    setQm(fmtIdx([...s]));
  };
  const save = async () => {
    setBusy(true);
    const body = { qm_atoms: fmtIdx(sel), qm_charge: +qmCharge, qm_multiplicity: +mult,
      active_radius: radius === '' ? null : +radius, mm, embedding };
    const out = rec.qmmm
      ? await attempt(() => api.put(`/api/qmmm/systems/${rec.qmmm}`, body), 'QM/MM region updated')
      : await attempt(() => api.post('/api/qmmm/systems', { ...body, structure: rec.id, optimize: true }),
        'QM/MM system created; minimizing it embedded');
    setBusy(false);
    if (out) onDone();
  };
  const shownRegion = preview && !preview.error ? preview : null;
  return html`<div class="qmmm-panel qmmm-edit">
    <h4>${rec.qmmm ? 'Edit the QM region' : 'Define a QM region (QM/MM)'}</h4>
    <p class="small muted">Click atoms to add or remove them from the QM region (shown as balls).
      The rest is the environment: within the moving shell it relaxes, beyond it it is frozen.</p>
    <${Viewer3D} xyz=${xyz} region=${shownRegion} onAtomClick=${toggle} highlight=${sel} height=${280} />
    <label class="field"><span class="field-label">QM atoms (indices from 0)</span>
      <input type="text" value=${qm} onInput=${(e) => setQm(e.target.value)} placeholder="e.g. 0-11 15" /></label>
    <div class="row small">
      <label>QM charge <input class="tiny" type="number" value=${qmCharge} onInput=${(e) => setQmCharge(e.target.value)} /></label>
      <label>multiplicity <input class="tiny" type="number" min="1" value=${mult} onInput=${(e) => setMult(e.target.value)} /></label>
      <label title="Environment atoms within this distance of the QM region move; the rest is frozen. Empty = nothing frozen">
        moving shell (Å) <input class="tiny" type="number" min="0" step="0.5" value=${radius} onInput=${(e) => setRadius(e.target.value)} /></label>
      <label>environment <select value=${mm} onChange=${(e) => setMm(e.target.value)}>
        ${['gfnff', 'tip3p', 'gfn2', 'gfn1'].concat(region?.mm === 'amber' ? ['amber'] : []).map((k) => html`<option value=${k}>${MM_LABELS[k]}</option>`)}
      </select></label>
      ${mm === 'amber' && html`<label title="Electrostatic: the AMBER charges enter the QM calculation (needs a QM level that takes point charges, e.g. Psi4)">
        <input type="checkbox" checked=${embedding === 'electrostatic'} onChange=${(e) => setEmbedding(e.target.checked ? 'electrostatic' : 'mechanical')} /> electrostatic</label>`}
    </div>
    ${(mm === 'tip3p' || embedding === 'electrostatic') && html`<p class="small muted">Electrostatic embedding: the water's charges polarize the QM region (the
      energy of forming ions is felt). It needs a QM level that takes point charges: choose xTB (GFN2) or Psi4 in Settings.</p>`}
    ${preview?.error && html`<p class="warn-box small">${preview.error}</p>`}
    ${shownRegion && html`<p class="small">${shownRegion.summary}</p>`}
    ${shownRegion?.problems?.map((p) => html`<p class="warn-box small">⚠ ${p}</p>`)}
    ${rec.qmmm && html`<p class="small muted">Energies computed with the current region stay as they are; they are not compared with ones from the new region.</p>`}
    <div class="row-actions">
      <button class="btn primary" disabled=${busy || !shownRegion} onClick=${save}>${rec.qmmm ? 'Save region' : 'Make QM/MM system'}</button>
      <button class="btn-link small" onClick=${onDone}>Cancel</button>
    </div>
  </div>`;
}

function Check({ ok, warn, value, label, title }) {
  return html`<span class=${`qmmm-check ${ok ? '' : warn ? 'warn' : 'bad'}`} title=${title}><b>${value}</b><span>${label}</span></span>`;
}

// Badges for the worst frame of a diagnose() report.
function CheckBadges({ report }) {
  const w = report.worst || {};
  return html`<div class="qmmm-checks">
    <${Check} ok=${w.frozen <= 1e-3} value=${`${(w.frozen ?? 0).toFixed(3)} Å`} label="frozen atoms moved"
      title="Largest move of a frozen environment atom (should be 0)" />
    <${Check} ok=${!w.mm_changes} value=${w.mm_changes || 0} label="MM bonds changed"
      title="Bonds made or broken among environment atoms: the force field cannot describe that, so it should be 0" />
    <${Check} ok=${(w.boundary ?? 0) <= 1.3} warn=${(w.boundary ?? 0) <= 1.6} value=${w.boundary ? `${w.boundary.toFixed(2)}×` : '—'}
      label="cut-bond stretch" title="Longest QM–MM boundary bond relative to its covalent length (should stay near 1)" />
    <${Check} ok=${w.contact == null || w.contact >= 1.5} warn=${w.contact >= 1.2} value=${w.contact != null ? `${w.contact.toFixed(2)} Å` : '—'}
      label="closest QM–MM contact" title="Shortest non-bonded distance between a QM and an environment atom" />
  </div>`;
}

// Several series against frame index, each scaled to its own range.
function Spark({ series, current, onPick }) {
  const n = Math.max(0, ...series.map((s) => s.ys.length));
  if (n < 2) return null;
  const W = 300, H = 64, pad = 4;
  const x = (i) => pad + (i / (n - 1)) * (W - 2 * pad);
  const lines = series.map((s) => {
    const ys = s.ys.filter((v) => v != null && Number.isFinite(v));
    const lo = s.min ?? Math.min(...ys), hi = s.max ?? Math.max(...ys);
    const span = hi - lo || 1;
    const pts = s.ys.map((v, i) => (v == null ? null : `${x(i).toFixed(1)},${(H - pad - ((v - lo) / span) * (H - 2 * pad)).toFixed(1)}`)).filter(Boolean);
    return html`<polyline fill="none" stroke=${s.color} stroke-width="1.6" points=${pts.join(' ')} />`;
  });
  const pick = (e) => {
    if (!onPick) return;
    const r = e.currentTarget.getBoundingClientRect();
    onPick(Math.max(0, Math.min(n - 1, Math.round(((e.clientX - r.left) / r.width * W - pad) / (W - 2 * pad) * (n - 1)))));
  };
  return html`<div>
    <svg class="qmmm-spark" viewBox=${`0 0 ${W} ${H}`} preserveAspectRatio="none" onClick=${pick}>
      ${lines}
      ${current != null && html`<line class="cur" x1=${x(current)} x2=${x(current)} y1="0" y2=${H} />`}
    </svg>
    <div class="qmmm-series">${series.map((s) => html`<span title=${s.title || ''}><i style=${{ background: s.color }}></i>${s.label}</span>`)}</div>
  </div>`;
}

// Next to a QM/MM result entry: is the path sensible? Frame by frame.
export function QmmmCheck({ job, entry, frame, onFrame }) {
  const [report, setReport] = useState(null);
  const [error, setError] = useState(null);
  const splitJob = useStore((s) => Object.values(s.jobs).filter((j) => j.op === 'qmmm-inspect' && j.source_job === job.id
    && j.params?.entry === entry.id).map((j) => `${j.id}:${j.status}`).join(','));
  useEffect(() => {
    let live = true;
    setReport(null); setError(null);
    api.get(`/api/jobs/${job.id}/qmmm-check?entry=${encodeURIComponent(entry.id)}`)
      .then((r) => live && setReport(r)).catch((e) => live && setError(e.message));
    return () => { live = false; };
  }, [job.id, entry.id, job.result_rev, splitJob]);
  if (error) return html`<p class="small muted">QM/MM checks unavailable: ${error}</p>`;
  if (!report) return html`<p class="small muted">Checking the QM/MM path…</p>`;
  const fr = report.frames || [];
  const many = fr.length > 1;
  const split = report.split;
  const runSplit = () => attempt(() => api.post('/api/jobs', { op: 'qmmm-inspect', params: { entry: entry.id }, source_job: job.id }),
    'Splitting energies into QM and environment');
  const cur = fr[Math.min(frame, fr.length - 1)];
  return html`<div class="qmmm-panel">
    <h4>QM/MM checks</h4>
    <${CheckBadges} report=${report} />
    ${report.warnings.map((w) => html`<p class="warn-box small">⚠ ${w}</p>`)}
    ${many && html`<${Spark} current=${frame} onPick=${onFrame} series=${[
      { label: 'QM atoms moved (RMSD)', color: '#3868b8', ys: fr.map((f) => f.qm_rmsd), min: 0 },
      { label: 'moving environment (RMSD)', color: '#e8890c', ys: fr.map((f) => f.mm_rmsd), min: 0 },
      { label: 'closest QM–MM contact', color: '#7a4fb5', ys: fr.map((f) => f.closest_contact) },
    ]} />`}
    ${cur && html`<p class="small muted">Frame ${frame + 1}: QM RMSD ${cur.qm_rmsd.toFixed(2)} Å · environment RMSD ${cur.mm_rmsd.toFixed(2)} Å
      (largest move ${cur.mm_max_move.toFixed(2)} Å)${cur.closest_contact != null ? ` · closest contact ${cur.closest_contact.toFixed(2)} Å (${cur.closest_pair.join('–')})` : ''}
      ${cur.qm_formed.length || cur.qm_broken.length ? html` · QM bonds ${cur.qm_formed.map((b) => `+${b.join('-')}`).concat(cur.qm_broken.map((b) => `−${b.join('-')}`)).join(' ')}` : ''}</p>`}
    ${many && (split?.qm_kcal
      ? html`<div class="small"><b>Energy split</b> (kcal/mol from frame 1)</div>
        <${Spark} current=${frame} onPick=${onFrame} series=${[
          { label: 'total', color: '#202a33', ys: split.total_kcal },
          { label: 'QM region', color: '#3868b8', ys: split.qm_kcal },
          { label: 'environment', color: '#e8890c', ys: split.environment_kcal },
        ].map((s) => ({ ...s, min: Math.min(...split.total_kcal, ...split.qm_kcal, ...split.environment_kcal),
          max: Math.max(...split.total_kcal, ...split.qm_kcal, ...split.environment_kcal) }))} />
        ${split.qm_kcal[frame] != null && html`<p class="small muted">Frame ${frame + 1}: QM ${split.qm_kcal[frame].toFixed(1)},
          environment ${split.environment_kcal[frame].toFixed(1)}, total ${split.total_kcal[frame].toFixed(1)} kcal/mol</p>`}`
      : split?.status && split.status !== 'done' && split.status !== 'failed'
        ? html`<p class="small muted">Splitting energies… (${split.status})</p>`
        : html`<button class="btn small" title="Recompute each frame as QM region + environment: does the barrier come from the reaction or from the solvent/protein around it?"
            onClick=${runSplit}>Split energies: QM vs environment</button>`)}
  </div>`;
}

// Inspector: a QM/MM structure (its region, checks, editing).
export function QmmmStructure({ rec, xyz, labels, shownConformer }) {
  const region = useRegion(rec.qmmm);
  const sys = useStore((s) => s.workspace.qmmm_systems?.[rec.qmmm]);
  const [editing, setEditing] = useState(false);
  const [check, setCheck] = useState(null);
  useEffect(() => {
    let live = true;
    setCheck(null);
    const q = shownConformer ? `&conformer=${encodeURIComponent(shownConformer)}` : '';
    api.get(`/api/qmmm/check?structure=${rec.id}${q}`).then((r) => live && setCheck(r)).catch(() => {});
    return () => { live = false; };
  }, [rec.id, shownConformer, rec.conformer, sys?.sig]);
  if (editing) return html`<${RegionEditor} rec=${rec} xyz=${xyz} region=${region} onDone=${() => setEditing(false)} />`;
  const last = check?.frames?.[1];
  return html`
    <${Viewer3D} xyz=${xyz} labels=${labels} system=${rec.qmmm} height=${260} />
    <div class="qmmm-panel">
      <h4>QM/MM system · ${sys?.name || ''}</h4>
      ${region ? html`<p class="small">${region.summary}</p>` : html`<p class="small muted">Loading the region…</p>`}
      ${(sys?.problems || []).map((p) => html`<p class="warn-box small">⚠ ${p}</p>`)}
      ${last && html`<p class="small muted">Against the system as set up: frozen atoms moved ${check.worst.frozen.toFixed(3)} Å,
        environment RMSD ${last.mm_rmsd.toFixed(2)} Å, ${last.mm_bond_changes.length ? html`<b class="warn-text">${last.mm_bond_changes.length} MM bonds changed</b>` : 'no MM bonds changed'}.</p>`}
      <p class="small muted">${`Every calculation on this structure runs embedded: the profile's level for the QM atoms, ${MM_LABELS[sys?.mm] || sys?.mm || 'the force field'} for the rest.`}</p>
      <button class="btn small" onClick=${() => setEditing(true)}>Edit the QM region</button>
    </div>`;
}

// Inspector: make a plain structure (e.g. an uploaded active site) QM/MM.
export function DefineQmmm({ rec, xyz }) {
  const [open, setOpen] = useState(false);
  if (!open) {
    return html`<a href="#" class="small" title="Pick a QM region in this structure; the rest becomes its force-field environment"
      onClick=${(e) => { e.preventDefault(); setOpen(true); }}>Define a QM region (QM/MM) →</a>`;
  }
  return html`<${RegionEditor} rec=${rec} xyz=${xyz} region=${null} onDone=${() => setOpen(false)} />`;
}

// Add panel: a whole system (protein, solvated cluster) as a QM/MM system,
// from a file, or a TeraChem QM/MM input on this machine.
export function QmmmAdd({ onDone }) {
  const [mode, setMode] = useState('file');
  const [file, setFile] = useState(null);
  const [qm, setQm] = useState('');
  const [charge, setCharge] = useState('0');
  const [qmCharge, setQmCharge] = useState('');
  const [mult, setMult] = useState('1');
  const [radius, setRadius] = useState('6');
  const [mm, setMm] = useState('gfnff');
  const [path, setPath] = useState('');
  const [busy, setBusy] = useState(false);
  const add = async () => {
    setBusy(true);
    let out;
    if (mode === 'file') {
      const fd = new FormData();
      fd.append('file', file);
      fd.append('qm_atoms', qm);
      fd.append('charge', charge || '0');
      fd.append('multiplicity', mult || '1');
      if (qmCharge !== '') fd.append('qm_charge', qmCharge);
      if (radius !== '') fd.append('active_radius', radius);
      fd.append('mm', mm);
      out = await attempt(() => api.post('/api/qmmm/upload', fd), 'QM/MM system added; minimizing it embedded');
    } else {
      out = await attempt(() => api.post('/api/qmmm/systems', { terachem: path, mm, optimize: true,
        active_radius: radius === '' ? null : +radius }), 'TeraChem system converted; minimizing it embedded');
    }
    setBusy(false);
    if (out) onDone?.(out.structure);
  };
  return html`<div class="qmmm-edit small">
    <div class="segmented mini">
      <button class=${mode === 'file' ? 'on' : ''} onClick=${() => setMode('file')}>XYZ / PDB file</button>
      <button class=${mode === 'tc' ? 'on' : ''} onClick=${() => setMode('tc')}>TeraChem input</button>
    </div>
    ${mode === 'file' ? html`
      <input type="file" accept=".xyz,.pdb" onChange=${(e) => setFile(e.target.files[0] || null)} />
      <label class="field"><span class="field-label">QM atoms (indices from 0; refine later by clicking atoms)</span>
        <input type="text" value=${qm} onInput=${(e) => setQm(e.target.value)} placeholder="e.g. 0-23" /></label>
      <div class="row">
        <label>system charge <input class="tiny" type="number" value=${charge} onInput=${(e) => setCharge(e.target.value)} /></label>
        <label>QM charge <input class="tiny" type="number" value=${qmCharge} placeholder=${charge} onInput=${(e) => setQmCharge(e.target.value)} /></label>
        <label>multiplicity <input class="tiny" type="number" min="1" value=${mult} onInput=${(e) => setMult(e.target.value)} /></label>
      </div>` : html`
      <label class="field"><span class="field-label">Path to tc.in on the server (its prmtop, rst7 and qmindices next to it)</span>
        <input type="text" value=${path} onInput=${(e) => setPath(e.target.value)} placeholder="/path/to/tc.in" /></label>
      <p class="muted">Runs now without TeraChem: the AMBER environment through OpenMM, the QM region at the workspace level.</p>`}
    <div class="row">
      <label title="Environment within this distance of the QM region moves; the rest is frozen">moving shell (Å)
        <input class="tiny" type="number" min="0" step="0.5" value=${radius} onInput=${(e) => setRadius(e.target.value)} /></label>
      ${mode === 'file' && html`<label>environment <select value=${mm} onChange=${(e) => setMm(e.target.value)}>
        ${['gfnff', 'gfn2', 'gfn1'].map((k) => html`<option value=${k}>${MM_LABELS[k]}</option>`)}</select></label>`}
    </div>
    <button class="btn primary" disabled=${busy || (mode === 'file' ? !file || !parseIdx(qm).length : !path.trim())} onClick=${add}>
      ${busy ? 'Adding…' : 'Add QM/MM system'}</button>
  </div>`;
}
