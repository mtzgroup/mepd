// Single place that pins the UI's third-party modules (vendored under
// static/vendor so the page works offline; no build step).
export {
  html, render, Component,
  useState, useEffect, useRef, useMemo, useCallback, useLayoutEffect,
} from '../vendor/preact-htm.module.js';
