// Renders any operation's parameter form from its pydantic JSON schema.
// Field extras from mepd.web.operations: group, advanced, cli, cli_kind.
import { html } from '../lib.js';

function fieldType(prop) {
  const variants = prop.anyOf ? prop.anyOf.filter((v) => v.type !== 'null') : [prop];
  const base = variants[0] || {};
  const nullable = !!prop.anyOf?.some((v) => v.type === 'null');
  if (base.enum || prop.enum) return { kind: 'enum', options: base.enum || prop.enum, nullable };
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
  if (t.kind === 'enum') {
    const options = t.options;
    control = options.length <= 3 && options.every((o) => String(o).length <= 16)
      ? html`<div class="segmented" role="radiogroup">
          ${options.map((o) => html`<button type="button" role="radio" aria-checked=${value === o}
            class=${value === o ? 'on' : ''} onClick=${() => onChange(o)}>${o}</button>`)}
        </div>`
      : html`<select id=${id} value=${value ?? ''} onChange=${(e) => onChange(e.target.value)}>
          ${options.map((o) => html`<option value=${o}>${o}</option>`)}
        </select>`;
  } else if (t.kind === 'integer' || t.kind === 'number') {
    control = html`<input id=${id} type="number" step=${t.kind === 'integer' ? 1 : 'any'} min=${t.min} max=${t.max}
      value=${value ?? ''} placeholder=${t.nullable ? 'default' : ''}
      onInput=${(e) => {
        const raw = e.target.value;
        if (raw === '') onChange(t.nullable ? null : prop.default);
        else onChange(t.kind === 'integer' ? parseInt(raw, 10) : parseFloat(raw));
      }} />`;
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
  const entries = Object.entries(schema.properties).filter(([, p]) => requirementMet(values, p.requires));
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
