// Copyright 2026 CiviqHub SAS
// SPDX-License-Identifier: Apache-2.0

// Nested module names in the sidebar are one token in <span class="pre">.
// Insert a break after each dot so they wrap between segments, never mid-word.
(() => {
  const wrapSidebarNames = () => {
    document.querySelectorAll(".sidebar-tree .pre").forEach((node) => {
      if (!(node instanceof HTMLElement) || node.dataset.civiqhubWrap === "1") {
        return;
      }
      node.dataset.civiqhubWrap = "1";
      const text = node.textContent ?? "";
      if (!text.includes(".")) {
        return;
      }
      node.replaceChildren();
      const parts = text.split(".");
      parts.forEach((part, index) => {
        node.append(part);
        if (index < parts.length - 1) {
          node.append(".");
          node.append(document.createElement("wbr"));
        }
      });
    });
  };

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", wrapSidebarNames);
  } else {
    wrapSidebarNames();
  }
})();
