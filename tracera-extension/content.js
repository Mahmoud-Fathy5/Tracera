/**
 * Tracera Extension — Content Script
 * ====================================
 * Injects overlay UI into the page to display:
 *   - Loading spinner when analyzing
 *   - Results panel with verdict, confidence, and attribution
 */

(function () {
  "use strict";

  let overlayEl = null;

  // ---------------------------------------------------------------
  // 1. Listen for Messages from Background
  // ---------------------------------------------------------------
  chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (message.action === "tracera-show-loading") {
      showLoading(message.imageUrl);
    }

    if (message.action === "tracera-show-result") {
      showResult(message.result);
    }
  });

  // ---------------------------------------------------------------
  // 2. Show Loading Overlay
  // ---------------------------------------------------------------
  function showLoading(imageUrl) {
    removeOverlay();

    overlayEl = document.createElement("div");
    overlayEl.id = "tracera-overlay";
    overlayEl.innerHTML = `
      <div class="tracera-panel">
        <div class="tracera-panel__header">
          <div class="tracera-panel__brand">
            <svg class="tracera-panel__logo" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
              <path d="M12 2L3 7v10l9 5 9-5V7l-9-5z" stroke="currentColor" stroke-width="1.5" fill="none"/>
              <path d="M12 8a4 4 0 100 8 4 4 0 000-8z" stroke="currentColor" stroke-width="1.5"/>
              <line x1="4" y1="12" x2="20" y2="12" stroke="currentColor" stroke-width="1" opacity="0.5"/>
            </svg>
            <span>Tracera</span>
          </div>
          <button class="tracera-panel__close" id="tracera-close">&times;</button>
        </div>
        <div class="tracera-panel__body tracera-panel__body--loading">
          <div class="tracera-spinner">
            <div class="tracera-spinner__ring"></div>
            <div class="tracera-spinner__ring tracera-spinner__ring--2"></div>
          </div>
          <p class="tracera-loading-text">Analyzing image…</p>
          <p class="tracera-loading-sub">Running spectral forensic analysis</p>
        </div>
      </div>
    `;

    document.body.appendChild(overlayEl);

    // Close button
    overlayEl.querySelector("#tracera-close").addEventListener("click", removeOverlay);
    // Click outside to close
    overlayEl.addEventListener("click", (e) => {
      if (e.target === overlayEl) removeOverlay();
    });

    // Animate in
    requestAnimationFrame(() => {
      if (overlayEl) overlayEl.classList.add("tracera-overlay--visible");
    });
  }

  // ---------------------------------------------------------------
  // 3. Show Result Overlay
  // ---------------------------------------------------------------
  function showResult(result) {
    removeOverlay();

    const isError = !!result.error;
    const isFake = !isError && result.verdict === "Fake";
    const isReal = !isError && result.verdict === "Real";

    let bodyHTML = "";

    if (isError) {
      bodyHTML = `
        <div class="tracera-result tracera-result--error">
          <div class="tracera-result__icon">
            <svg viewBox="0 0 24 24" fill="none"><circle cx="12" cy="12" r="10" stroke="currentColor" stroke-width="2"/><line x1="8" y1="8" x2="16" y2="16" stroke="currentColor" stroke-width="2"/><line x1="16" y1="8" x2="8" y2="16" stroke="currentColor" stroke-width="2"/></svg>
          </div>
          <p class="tracera-result__title">Analysis Failed</p>
          <p class="tracera-result__message">${escapeHTML(result.error)}</p>
        </div>
      `;
    } else {
      const confidence = Math.round(result.confidence * 100);
      const verdictClass = isFake ? "tracera-result--fake" : "tracera-result--real";
      const verdictIcon = isFake
        ? `<svg viewBox="0 0 24 24" fill="none"><path d="M12 2L1 21h22L12 2z" stroke="currentColor" stroke-width="2"/><line x1="12" y1="9" x2="12" y2="15" stroke="currentColor" stroke-width="2"/><circle cx="12" cy="18" r="1" fill="currentColor"/></svg>`
        : `<svg viewBox="0 0 24 24" fill="none"><path d="M12 2a10 10 0 1010 10A10 10 0 0012 2z" stroke="currentColor" stroke-width="2"/><path d="M8 12l3 3 5-5" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>`;

      let attributionHTML = "";
      if (isFake && result.attribution) {
        const attrConf = result.attribution_confidence
          ? `${(result.attribution_confidence * 100).toFixed(1)}%`
          : "";
        const attrIcon = result.attribution === "GAN"
          ? `<svg viewBox="0 0 24 24" fill="none"><rect x="3" y="3" width="18" height="18" rx="3" stroke="currentColor" stroke-width="1.5"/><path d="M8 12h8M12 8v8" stroke="currentColor" stroke-width="1.5"/></svg>`
          : `<svg viewBox="0 0 24 24" fill="none"><circle cx="12" cy="12" r="9" stroke="currentColor" stroke-width="1.5"/><path d="M12 3c-5 4-5 14 0 18M12 3c5 4 5 14 0 18" stroke="currentColor" stroke-width="1"/></svg>`;

        attributionHTML = `
          <div class="tracera-attr">
            <div class="tracera-attr__icon">${attrIcon}</div>
            <div class="tracera-attr__info">
              <span class="tracera-attr__label">Generator Identified</span>
              <span class="tracera-attr__value">${escapeHTML(result.attribution)}</span>
              ${attrConf ? `<span class="tracera-attr__conf">${attrConf} confidence</span>` : ""}
            </div>
          </div>
        `;
      }

      bodyHTML = `
        <div class="tracera-result ${verdictClass}">
          <div class="tracera-result__icon">${verdictIcon}</div>
          <p class="tracera-result__verdict">${result.verdict.toUpperCase()}</p>
          <div class="tracera-gauge">
            <svg viewBox="0 0 100 100">
              <g transform="rotate(-90 50 50)">
                <circle cx="50" cy="50" r="42" class="tracera-gauge__bg"/>
                <circle cx="50" cy="50" r="42" class="tracera-gauge__fill ${isFake ? 'tracera-gauge__fill--fake' : 'tracera-gauge__fill--real'}"
                        stroke-dasharray="${2 * Math.PI * 42}"
                        stroke-dashoffset="${2 * Math.PI * 42 * (1 - result.confidence)}"
                        />
              </g>
              <text x="50" y="45" class="tracera-gauge__value">${confidence}%</text>
              <text x="50" y="62" class="tracera-gauge__label">Confidence</text>
            </svg>
          </div>
          ${attributionHTML}
        </div>
      `;
    }

    overlayEl = document.createElement("div");
    overlayEl.id = "tracera-overlay";
    overlayEl.innerHTML = `
      <div class="tracera-panel">
        <div class="tracera-panel__header">
          <div class="tracera-panel__brand">
            <svg class="tracera-panel__logo" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
              <path d="M12 2L3 7v10l9 5 9-5V7l-9-5z" stroke="currentColor" stroke-width="1.5" fill="none"/>
              <path d="M12 8a4 4 0 100 8 4 4 0 000-8z" stroke="currentColor" stroke-width="1.5"/>
              <line x1="4" y1="12" x2="20" y2="12" stroke="currentColor" stroke-width="1" opacity="0.5"/>
            </svg>
            <span>Tracera</span>
          </div>
          <button class="tracera-panel__close" id="tracera-close">&times;</button>
        </div>
        <div class="tracera-panel__body">
          ${bodyHTML}
        </div>
        <div class="tracera-panel__footer">
          <span>Powered by GramNet v3 Spectral Analysis</span>
        </div>
      </div>
    `;

    document.body.appendChild(overlayEl);

    overlayEl.querySelector("#tracera-close").addEventListener("click", removeOverlay);
    overlayEl.addEventListener("click", (e) => {
      if (e.target === overlayEl) removeOverlay();
    });

    requestAnimationFrame(() => {
      if (overlayEl) overlayEl.classList.add("tracera-overlay--visible");
    });
  }

  // ---------------------------------------------------------------
  // 4. Remove Overlay
  // ---------------------------------------------------------------
  function removeOverlay() {
    if (overlayEl) {
      var el = overlayEl;    // capture the current element
      overlayEl = null;       // clear immediately so new overlays aren't affected
      el.classList.remove("tracera-overlay--visible");
      setTimeout(function () {
        if (el && el.parentNode) {
          el.parentNode.removeChild(el);
        }
      }, 300);
    }
  }

  // ---------------------------------------------------------------
  // 5. Utility
  // ---------------------------------------------------------------
  function escapeHTML(str) {
    const div = document.createElement("div");
    div.textContent = str;
    return div.innerHTML;
  }
})();
