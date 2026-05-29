// Behaviour for the header "Export" dropdown (a native <details>).
//
// Native <details> toggles on summary click but never closes on its own;
// this adds the expected dropdown affordances:
//   * click anywhere outside  -> close
//   * Escape                  -> close (and refocus the summary)
//   * pick an export action    -> close
//
// Kept as a tiny standalone module (loaded with type="module", so it runs
// after the DOM is parsed and is CSP-safe under script-src 'self') rather
// than bloating label.js. Touches only #export-menu, so it can't interfere
// with the labelling logic.

const menu = document.getElementById("export-menu");

if (menu) {
  const summary = menu.querySelector("summary");

  // Outside-click closes the menu. Clicks on the <select> or its options
  // are inside the menu, so picking an error type keeps it open.
  document.addEventListener("click", (e) => {
    if (menu.open && !menu.contains(e.target)) menu.open = false;
  });

  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && menu.open) {
      menu.open = false;
      summary?.focus();
    }
  });

  // Picking an export action closes the menu (the click still propagates
  // to label.js's own handlers / the conflict-guard first).
  menu.querySelectorAll(".export-menu-body a[id]").forEach((a) => {
    a.addEventListener("click", () => { menu.open = false; });
  });
}
