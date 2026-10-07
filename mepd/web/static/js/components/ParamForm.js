// Renders any operation's parameter form from its pydantic JSON schema.
// Field extras from mepd.web.operations: group, advanced, cli, cli_kind.
import { html, useEffect, useState } from '../lib.js';
import { prefs } from '../store.js';

// A number field that lets you type: it keeps its own text while you edit
// ("0.", "-", "1e-" are fine half-way) and reports a number only when the text
// parses as one. A number input re-rendered from the parsed value on every
// keystroke loses the "." (Chrome reports "0." as empty, which reset the field).
export function NumberInput({ value, onChange, integer = false, min, max, placeholder = '', id, className = '', title }) {
  const shown = (v) => (v == null || Number.isNaN(v) ? '' : String(v));
  const [text, setText] = useState(shown(value));
  const [editing, setEditing] = useState(false);
  useEffect(() => { if (!editing) setText(shown(value)); }, [value, editing]);
  const pattern = integer ? /^-?\d+$/ : /^-?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?$/;
  const bad = text.trim() !== '' && (!pattern.test(text.trim())
    || (min != null && +text < min) || (max != null && +text > max));
  return html`<input id=${id} type="text" inputmode=${integer ? 'numeric' : 'decimal'} class=${`${className} ${bad ? 'invalid' : ''}`}
    value=${text} placeholder=${placeholder} title=${bad ? `Not a valid number${min != null ? ` (at least ${min})` : ''}${max != null ? ` (at most ${max})` : ''}` : title}
    onFocus=${() => setEditing(true)}
    onBlur=${() => { setEditing(false); setText(shown(value)); }}
    onInput=${(e) => {
      const raw = e.target.value;
      setText(raw);
      const t = raw.trim();
      if (t === '') { onChange(null); return; }
      if (pattern.test(t)) {
        const v = integer ? parseInt(t, 10) : parseFloat(t);
        if ((min == null || v >= min) && (max == null || v <= max)) onChange(v);
      }
    }} />`;
}

function fieldType(prop) {
  const variants = prop.anyOf ? prop.anyOf.filter((v) => v.type !== 'null') : [prop];
  const base = variants[0] || {};
  const nullable = !!prop.anyOf?.some((v) => v.type === 'null');
  if (base.enum || prop.enum) return { kind: 'enum', options: base.enum || prop.enum, nullable };
  if (base.type === 'array' && base.items?.enum) return { kind: 'multi', options: base.items.enum };
  return { kind: base.type || 'string', nullable, min: base.minimum ?? base.exclusiveMinimum, max: base.maximum };
}

function Field({ name, prop, value, onChange }) {
  const t = fieldType(prop);
  const hint = [prop.description, prop.cli ? `CLI: ${prop.cli}` : ''].filter(Boolean).join('\n');
  const id = `f-${name}`;
  let control;
  if (t.kind === 'boolean') {
    return html`
      <label class="field field-bool" title=${hint}>
        <input type="checkbox" class="switch" checked=${!!value} onChange=${(e) => onChange(e.target.checked)} />
        <span class="field-label">${prop.title || name}</span>
        ${prop.description && html`<span class="field-help">${prop.description}</span>`}
      </label>`;
  }
  // Field extras may name what each value is shown as (`labels`).
  const shown = (o) => prop.labels?.[o] ?? o;
  if (t.kind === 'multi') {
    const on = new Set(value || []);
    control = html`<div class="chips" role="group">
      ${t.options.map((o) => html`<button type="button" aria-pressed=${on.has(o)} class=${`chip ${on.has(o) ? 'on' : ''}`}
        title=${shown(o)} onClick=${() => onChange(on.has(o) ? (value || []).filter((v) => v !== o) : [...(value || []), o])}>
        ${String(shown(o)).replace(/\s*\(.*\)$/, '')}</button>`)}
    </div>`;
  } else if (t.kind === 'enum') {
    const options = t.options;
    control = options.length <= 3 && options.every((o) => String(shown(o)).length <= 24)
      ? html`<div class="segmented" role="radiogroup">
          ${options.map((o) => html`<button type="button" role="radio" aria-checked=${value === o}
            class=${value === o ? 'on' : ''} onClick=${() => onChange(o)}>${shown(o)}</button>`)}
        </div>`
      : html`<select id=${id} value=${value ?? ''} onChange=${(e) => onChange(e.target.value)}>
          ${options.map((o) => html`<option value=${o}>${shown(o)}</option>`)}
        </select>`;
  } else if (t.kind === 'integer' || t.kind === 'number') {
    control = html`<${NumberInput} id=${id} integer=${t.kind === 'integer'} min=${t.min} max=${t.max} value=${value}
      placeholder=${t.nullable ? 'default' : String(prop.default ?? '')}
      onChange=${(v) => onChange(v == null ? (t.nullable ? null : prop.default) : v)} />`;
  } else {
    control = html`<input id=${id} type="text" value=${value ?? ''} placeholder=${t.nullable ? 'default' : ''}
      onInput=${(e) => onChange(e.target.value === '' && t.nullable ? null : e.target.value)} />`;
  }
  return html`
    <div class="field" title=${hint}>
      <label class="field-label" for=${id}>${prop.title || name}</label>
      ${control}
      ${prop.description && html`<span class="field-help">${prop.description}</span>`}
    </div>`;
}

// Pull remembered values back inside the schema's limits (e.g. demo caps
// that were added after the value was saved in this browser).
export function clampToSchema(schema, values) {
  const out = { ...values };
  for (const [k, p] of Object.entries(schema?.properties || {})) {
    const max = p.maximum ?? p.anyOf?.find((v) => v.maximum != null)?.maximum;
    if (max != null && typeof out[k] === 'number' && out[k] > max) out[k] = max;
  }
  return out;
}

// A calculation form remembers only the settings you changed from its
// defaults, so a default that changes later still reaches every setting you
// never touched. (Version 2 of the stored form: version 1 kept every value,
// defaults included, so one changed default stayed stale forever; those are
// not read, which starts every form once from the current defaults.)
const REMEMBERED = (key) => `params2:${key}`;
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);

export function rememberedValues(op, extra = {}) {
  const defaults = defaultsFor(op.schema);
  return clampToSchema(op.schema, { ...defaults, ...prefs.get(REMEMBERED(op.key), {}), ...extra });
}

export function rememberValues(op, values) {
  const defaults = defaultsFor(op.schema);
  const changed = Object.fromEntries(Object.entries(values).filter(([k, v]) => k in defaults && !same(v, defaults[k])));
  prefs.set(REMEMBERED(op.key), changed);
}

export function defaultsFor(schema) {
  const out = {};
  for (const [k, p] of Object.entries(schema?.properties || {})) out[k] = p.default ?? null;
  return out;
}

// Mirrors operations.requirement_met: "other" or "other=a|b".
function requirementMet(values, requires) {
  if (!requires) return true;
  const [name, allowed] = requires.split('=');
  return allowed === undefined ? !!values[name] : allowed.split('|').includes(String(values[name]));
}

export function ParamForm({ schema, values, onChange }) {
  if (!schema?.properties || !Object.keys(schema.properties).length) {
    return html`<p class="muted small">No parameters.</p>`;
  }
  // group "Hidden": set by the page itself (e.g. Sandbox › Stop & analyze), never a form field.
  const entries = Object.entries(schema.properties).filter(([, p]) => p.group !== 'Hidden' && requirementMet(values, p.requires));
  const basic = entries.filter(([, p]) => !p.advanced);
  const advanced = entries.filter(([, p]) => p.advanced);
  const groups = {};
  for (const [k, p] of advanced) (groups[p.group || 'Advanced'] ||= []).push([k, p]);
  const changed = (k) => values[k] !== (schema.properties[k].default ?? null);
  const nChanged = advanced.filter(([k]) => changed(k)).length;
  const set = (k) => (v) => onChange({ ...values, [k]: v });
  return html`
    <div class="param-form">
      ${basic.map(([k, p]) => html`<${Field} key=${k} name=${k} prop=${p} value=${values[k]} onChange=${set(k)} />`)}
      ${advanced.length > 0 && html`
        <details class="advanced">
          <summary>Advanced${nChanged ? html` <span class="badge">${nChanged} changed</span>` : ''}</summary>
          ${Object.entries(groups).map(([g, fields]) => html`
            <fieldset>
              <legend>${g}</legend>
              ${fields.map(([k, p]) => html`<${Field} key=${k} name=${k} prop=${p} value=${values[k]} onChange=${set(k)} />`)}
            </fieldset>`)}
          <button type="button" class="btn-link small" onClick=${() => onChange(defaultsFor(schema))}>Reset all to defaults</button>
        </details>`}
    </div>`;
}
