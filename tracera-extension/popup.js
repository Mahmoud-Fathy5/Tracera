/**
 * Tracera Extension — Popup Script
 * ==================================
 * Just checks the API health status on popup open.
 * No user configuration needed — API URL is hardcoded in background.js
 */

(function () {
  "use strict";

  const statusDot  = document.getElementById("status-dot");
  const statusText = document.getElementById("status-text");

  // Ask the background service worker to check API health
  chrome.runtime.sendMessage({ action: "tracera-health-check" }, (response) => {
    if (chrome.runtime.lastError) {
      setStatus("error", "Extension error");
      return;
    }

    if (response && response.ok) {
      const model = response.data?.model || "Online";
      setStatus("ok", `Connected — ${model}`);
    } else {
      setStatus("error", response?.error || "Cannot reach server");
    }
  });

  function setStatus(state, text) {
    statusDot.className = "status__dot";
    if (state === "ok") statusDot.classList.add("status__dot--ok");
    else if (state === "error") statusDot.classList.add("status__dot--error");
    else statusDot.classList.add("status__dot--loading");

    statusText.textContent = text;
  }
})();
