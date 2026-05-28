/* Viewport auto-zoom for the data-cleaning workbench.
 *
 * The three-pane labelling layout (320px sidebar + 1fr centre + 380px
 * detail) and the runs dashboard were tuned against a ~1600-1920px
 * wide design viewport. On 1366x768 / 1280x720 laptops everything
 * still LOOKS the same to the layout engine, so the centre CSV viewer
 * gets squeezed and operators end up horizontally scrolling inside
 * cells. The existing piecemeal ``@media (max-width: ...)`` shrinks
 * help, but they only adjust a handful of paddings + font-sizes -
 * the grid template, table column widths, and absolute pixel paddings
 * deeper in the stylesheet still feel oversized.
 *
 * This script sets ``html { zoom: N }`` based on viewport size so the
 * whole UI scales uniformly: DOWN on smaller screens (so the centre
 * CSV viewer doesn't squeeze), and UP on larger screens (so 13px
 * type doesn't disappear on 1440p / 4K monitors). ``zoom`` is the
 * simplest way to scale a whole page in one go because the browser
 * reflows content at the new logical size - unlike ``transform:
 * scale()``, which doesn't update layout bounds. ``zoom`` is
 * supported in every current browser engine (Chromium, WebKit, and
 * Firefox 126+).
 *
 * The script runs synchronously from ``<head>`` BEFORE stylesheets
 * parse, so the very first paint is at the right scale (no FOUC).
 *
 * Power-user escape hatches (set from devtools, persist across
 * reloads via localStorage):
 *
 *     // Disable auto-zoom (force 1.0):
 *     window.__workbenchZoom.setOverride('off');
 *
 *     // Pin a specific zoom factor:
 *     window.__workbenchZoom.setOverride(0.85);
 *
 *     // Return to automatic:
 *     window.__workbenchZoom.setOverride(null);
 */
(function () {
  'use strict';

  // The reference viewport every CSS pixel value in styles.css /
  // runs.css / mt_label.css was tuned against. Picked at the lower
  // end of "comfortable" (1600x900) so the auto-zoom scales DOWN
  // on smaller screens and scales UP on larger ones (up to the
  // ``MAX_ZOOM`` ceiling below) - matching what operators actually
  // expect on their physical screens.
  var DESIGN_WIDTH = 1600;
  var DESIGN_HEIGHT = 900;

  // Hard floor: at 0.65 the 13px base font is ~8.5px, which is the
  // smallest readable size in our typeface. Anything below that and
  // the operator should rotate their monitor or use real browser zoom.
  var MIN_ZOOM = 0.65;
  // Hard ceiling: cap the up-scale on very large screens so the UI
  // doesn't balloon out of proportion on operator-pinned 4K kiosk
  // setups. 1.5x at the design height yields ~19.5px base text at
  // 1440p / ~26px at 2160p which sits comfortably above the
  // operator's reading distance. Power users on 5K+ can still pin a
  // higher factor via ``__workbenchZoom.setOverride(1.75)``.
  var MAX_ZOOM = 1.5;

  var STORAGE_KEY = 'workbench:zoom-override';

  function pickZoom() {
    var w = window.innerWidth || document.documentElement.clientWidth;
    var h = window.innerHeight || document.documentElement.clientHeight;
    if (!w || !h) {
      return 1;
    }
    // Fit by the MOST-constrained axis so neither width nor height
    // gets clipped (portrait monitors / windowed sessions / devtools
    // docked-right halve the width but keep height; docked-bottom
    // does the opposite). ``Math.min`` makes both cases just work.
    var fit = Math.min(w / DESIGN_WIDTH, h / DESIGN_HEIGHT);
    if (fit > MAX_ZOOM) return MAX_ZOOM;
    if (fit < MIN_ZOOM) return MIN_ZOOM;
    return fit;
  }

  function readOverride() {
    try {
      return window.localStorage.getItem(STORAGE_KEY);
    } catch (e) {
      // Cross-origin sandboxes / private-mode quirks can throw on
      // localStorage access. Treat as "no override".
      return null;
    }
  }

  function writeOverride(value) {
    try {
      if (value === null || value === undefined) {
        window.localStorage.removeItem(STORAGE_KEY);
      } else {
        window.localStorage.setItem(STORAGE_KEY, String(value));
      }
    } catch (e) {
      // Same swallow as readOverride - storage being unavailable
      // shouldn't crash the page.
    }
  }

  function resolveZoom() {
    var override = readOverride();
    if (override === 'off' || override === '1' || override === '1.0') {
      return 1;
    }
    if (override) {
      var n = parseFloat(override);
      if (!isNaN(n) && isFinite(n)) {
        // Allow override outside the auto-zoom clamp (0.4-2.0) so
        // power users can pin a tight zoom for kiosk displays etc.
        if (n < 0.4) n = 0.4;
        if (n > 2.0) n = 2.0;
        return n;
      }
    }
    return pickZoom();
  }

  function apply() {
    var z = resolveZoom();
    var root = document.documentElement;
    if (!root) return;
    // Setting ``zoom`` rather than ``style.zoom`` directly on the
    // CSSStyleDeclaration keeps the inline-style cascade clean -
    // any operator-supplied stylesheet can still override it via
    // ``html { zoom: 1 !important }`` if they really want to.
    root.style.zoom = String(z);
    // Publish the zoom + its inverse as CSS custom properties so
    // viewport-unit declarations can compensate. The wart we're
    // working around: CSS ``zoom`` PAINTS descendants at scale, but
    // ``100vh`` inside the zoomed subtree still resolves against the
    // un-zoomed viewport. So ``body { height: 100vh; zoom: 0.85 }``
    // paints the body at 85% of the window, leaving 15% blank at the
    // bottom. With ``--vh-scale = 1/z``, full-viewport rules become
    // ``height: calc(100vh * var(--vh-scale, 1))`` which inflates
    // the logical height by 1/z, so the painted height fills the
    // viewport exactly. At z=1 the multiplication is a no-op.
    var inv = 1 / z;
    root.style.setProperty('--workbench-zoom', String(z));
    root.style.setProperty('--vh-scale', String(inv));
    root.style.setProperty('--vw-scale', String(inv));
    // Expose the resolved factor on the root element so CSS / JS
    // that wants to opt-in to extra adjustments can read it via
    // ``getComputedStyle(document.documentElement).getPropertyValue(...)``
    // or ``document.documentElement.dataset.viewportZoom``.
    root.dataset.viewportZoom = z.toFixed(3);
  }

  // First paint: must apply before stylesheets parse so there is no
  // un-zoomed flash. The script tag should sit in <head> with no
  // ``defer``/``async`` attribute.
  apply();

  // Recompute on viewport changes. ``resize`` fires for window
  // resizes, fullscreen toggles, tab moves between displays of
  // different DPI, and devtools open/close. ``requestAnimationFrame``
  // throttles so we restyle at most once per frame.
  var pending = 0;
  window.addEventListener('resize', function () {
    if (pending) return;
    pending = window.requestAnimationFrame(function () {
      pending = 0;
      apply();
    });
  });

  // Tiny public API for devtools tinkering / future settings UI.
  window.__workbenchZoom = {
    apply: apply,
    pickZoom: pickZoom,
    setOverride: function (value) {
      writeOverride(value);
      apply();
    },
    getOverride: readOverride,
    getCurrent: function () {
      return parseFloat(
        (document.documentElement && document.documentElement.dataset.viewportZoom) || '1',
      );
    },
  };
})();
