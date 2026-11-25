// Copyright 2026 CiviqHub SAS
// SPDX-License-Identifier: Apache-2.0

// Render the formulas that Sphinx wrote into the page (docs/_ext/katex_local.py) with
// the copy of KaTeX under _static/katex/. Each formula element goes to auto-render on its
// own, so that text outside the formulas is never read as TeX. The macros of conf.py
// arrive as JSON in the page.
(() => {
  const formulaSelector = "span.math.notranslate, div.math.notranslate";

  function readMacros() {
    const element = document.getElementById("selene-katex-macros");
    return element ? JSON.parse(element.textContent || "{}") : {};
  }

  function renderFormulas() {
    if (typeof window.renderMathInElement !== "function") {
      return;
    }
    const macros = readMacros();
    document.querySelectorAll(formulaSelector).forEach((element) => {
      window.renderMathInElement(element, {
        delimiters: [
          { left: "\\[", right: "\\]", display: true },
          { left: "\\(", right: "\\)", display: false },
        ],
        // A copy per formula: a definition made in one formula does not reach the next.
        macros: { ...macros },
        // A formula KaTeX cannot read shows as an error in place; the rest still renders.
        throwOnError: false,
        // No command may link to or load a resource (\href, \url, \includegraphics).
        trust: false,
        strict: "warn",
      });
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", renderFormulas);
  } else {
    renderFormulas();
  }
})();
