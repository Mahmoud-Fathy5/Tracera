/**
 * Tracera Extension — Background Service Worker
 * ===============================================
 * Handles:
 *   - Context menu creation ("Analyze with Tracera")
 *   - Fetching images from URLs
 *   - Sending images to the Tracera API
 *   - Communicating results back to the content script
 */

// =================================================================
// ⬇️  SET YOUR API URL HERE AFTER DEPLOYING TO AZURE  ⬇️
// =================================================================
// Replace this with your actual deployed URL, for example:
//   "https://tracera-app.azurewebsites.net"
//   "https://your-custom-domain.com"
//   "http://localhost:5000"  (for local testing)
//
const API_BASE_URL = "https://ma7moud05-tracera.hf.space/";
// =================================================================

// ---------------------------------------------------------------
// 1. Create Context Menu on Install
// ---------------------------------------------------------------
chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({
    id: "tracera-analyze",
    title: "🔍 Analyze with Tracera (AI vs Real)",
    contexts: ["image"],
  });
});

// ---------------------------------------------------------------
// 2. Handle Context Menu Click
// ---------------------------------------------------------------
chrome.contextMenus.onClicked.addListener(async (info, tab) => {
  if (info.menuItemId !== "tracera-analyze") return;

  const imageUrl = info.srcUrl;
  if (!imageUrl) return;

  // Notify content script to show loading overlay
  try {
    await chrome.tabs.sendMessage(tab.id, {
      action: "tracera-show-loading",
      imageUrl: imageUrl,
    });
  } catch (e) {
    console.warn("Could not send message to content script:", e);
  }

  try {
    const endpoint = `${API_BASE_URL}/api/predict`;

    // Fetch the image as a blob
    const imageBlob = await fetchImageAsBlob(imageUrl);

    if (!imageBlob) {
      await sendResult(tab.id, {
        error: "Could not fetch the image. It may be protected or cross-origin restricted.",
      });
      return;
    }

    // Validate file type
    const validTypes = ["image/jpeg", "image/png", "image/webp"];
    if (!validTypes.includes(imageBlob.type)) {
      await sendResult(tab.id, {
        error: `Unsupported image type: ${imageBlob.type}. Tracera supports JPEG, PNG, and WebP.`,
      });
      return;
    }

    // Validate file size (10 MB limit)
    if (imageBlob.size > 10 * 1024 * 1024) {
      await sendResult(tab.id, {
        error: "Image is too large (max 10 MB).",
      });
      return;
    }

    // Create form data and send to API
    const formData = new FormData();
    formData.append("image", imageBlob, "image.jpg");

    const response = await fetch(endpoint, {
      method: "POST",
      body: formData,
    });

    const data = await response.json();

    if (!response.ok) {
      await sendResult(tab.id, {
        error: data.error || `Server error (${response.status})`,
      });
      return;
    }

    // Send successful result
    await sendResult(tab.id, {
      success: true,
      verdict: data.verdict,
      confidence: data.confidence,
      attribution: data.attribution,
      attribution_confidence: data.attribution_confidence,
      imageUrl: imageUrl,
    });

  } catch (err) {
    console.error("Tracera analysis error:", err);
    await sendResult(tab.id, {
      error: `Connection error: ${err.message}. Make sure the Tracera API is running.`,
    });
  }
});

// ---------------------------------------------------------------
// 3. Fetch Image as Blob
// ---------------------------------------------------------------
async function fetchImageAsBlob(url) {
  try {
    // Handle data URLs directly
    if (url.startsWith("data:")) {
      const response = await fetch(url);
      return await response.blob();
    }

    // Handle blob URLs
    if (url.startsWith("blob:")) {
      const response = await fetch(url);
      return await response.blob();
    }

    // Regular URL — try CORS first, then no-cors fallback
    try {
      const response = await fetch(url, { mode: "cors" });
      if (response.ok) {
        return await response.blob();
      }
    } catch (corsError) {
      const response = await fetch(url, { mode: "no-cors" });
      return await response.blob();
    }

    return null;
  } catch (e) {
    console.error("Failed to fetch image:", e);
    return null;
  }
}

// ---------------------------------------------------------------
// 4. Send Result to Content Script
// ---------------------------------------------------------------
async function sendResult(tabId, result) {
  try {
    await chrome.tabs.sendMessage(tabId, {
      action: "tracera-show-result",
      result: result,
    });
  } catch (e) {
    console.warn("Could not send result to content script:", e);
  }
}

// ---------------------------------------------------------------
// 5. Respond to popup health check requests
// ---------------------------------------------------------------
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.action === "tracera-health-check") {
    fetch(`${API_BASE_URL}/api/health`)
      .then((res) => res.json())
      .then((data) => sendResponse({ ok: true, data }))
      .catch((err) => sendResponse({ ok: false, error: err.message }));
    return true; // keep channel open for async response
  }
});
