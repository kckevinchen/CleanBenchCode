/* Shared utilities used by every page in the labelling workbench.
 *
 * These were previously redefined verbatim in runs.js, compare.js,
 * external_compare.js, column_board.js, and label.js. Centralising
 * them removes ~80 LOC of duplication and lets us evolve fetch
 * behaviour (auth headers, retries, error formatting) in one place.
 *
 * Loaded as an ES module - consumers `import { api, $, $$, escapeHtml }
 * from '../shared/utils.js'` (or `./shared/utils.js`).
 */

/** querySelector. */
export const $ = (sel, root = document) => root.querySelector(sel);

/** querySelectorAll, returned as a real Array (for .map / .filter). */
export const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

/** Escape a value for safe HTML interpolation (text + attribute contexts).
 *
 * Coerces null/undefined to "". Covers the OWASP-recommended five
 * characters so the same helper is safe inside element text and
 * inside single- or double-quoted attribute values.
 */
export function escapeHtml(s) {
  return String(s ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

/** JSON fetch wrapper.
 *
 * Sets ``content-type: application/json`` by default, propagates any
 * caller-supplied init fields, and throws on non-2xx responses with a
 * message that includes the (possibly JSON-encoded) error body. The
 * 400-char body cap keeps stack traces readable when the backend
 * returns a large HTML error page.
 */
export async function api(path, options = {}) {
  const res = await fetch(path, {
    headers: { "content-type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    let detail = body.slice(0, 400);
    try {
      const parsed = JSON.parse(body);
      if (parsed && typeof parsed === "object" && parsed.detail) {
        detail = typeof parsed.detail === "string"
          ? parsed.detail
          : JSON.stringify(parsed.detail);
      }
    } catch (_) {
      // body wasn't JSON; fall back to the truncated text we already have.
    }
    throw new Error(`${res.status} ${res.statusText}: ${detail}`);
  }
  return res.json();
}
