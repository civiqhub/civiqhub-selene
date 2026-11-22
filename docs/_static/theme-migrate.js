// Copyright 2026 CiviqHub SAS
// SPDX-License-Identifier: Apache-2.0

// The light-only site (PR #10) wrote localStorage.theme = "light" for every
// visitor. That was not a choice. Clear it once so Furo falls back to "auto".
// Must run in the <head>, before Furo reads the key in base.html.
(() => {
  try {
    if (
      globalThis.localStorage &&
      localStorage.getItem("theme") === "light" &&
      localStorage.getItem("civiqhub-theme-migrated") !== "1"
    ) {
      localStorage.removeItem("theme");
      localStorage.setItem("civiqhub-theme-migrated", "1");
    }
  } catch {
    // Storage can be unavailable; Furo then keeps "auto".
  }
})();
