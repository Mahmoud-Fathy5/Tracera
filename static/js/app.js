/**
 * Tracera — Frontend Application Logic
 * =====================================
 * Handles:
 *   - Drag-and-drop file upload with validation
 *   - API communication with /api/predict
 *   - Result rendering with animated gauge
 *   - Scroll-triggered animations
 *   - Stat counter animations
 */

(function () {
    "use strict";

    // ---------------------------------------------------------------
    // DOM Elements
    // ---------------------------------------------------------------
    const nav           = document.getElementById("main-nav");
    const uploadZone    = document.getElementById("upload-zone");
    const uploadContent = document.getElementById("upload-content");
    const previewArea   = document.getElementById("preview-area");
    const previewImage  = document.getElementById("preview-image");
    const removeBtn     = document.getElementById("remove-btn");
    const fileInput     = document.getElementById("file-input");
    const analyzeBtn    = document.getElementById("analyze-btn");
    const analyzeBtnTxt = document.getElementById("analyze-btn-text");
    const loadingState  = document.getElementById("loading-state");
    const errorState    = document.getElementById("error-state");
    const errorMessage  = document.getElementById("error-message");
    const errorDismiss  = document.getElementById("error-dismiss");
    const resultsArea   = document.getElementById("results-area");
    const verdictBadge  = document.getElementById("verdict-badge");
    const verdictText   = document.getElementById("verdict-text");
    const gaugeFill     = document.getElementById("gauge-fill");
    const gaugeValue    = document.getElementById("gauge-value");
    const attrArea      = document.getElementById("attribution-area");
    const attrText      = document.getElementById("attribution-text");
    const attrConf      = document.getElementById("attribution-conf");
    const tryAgainBtn   = document.getElementById("try-again-btn");
    const hamburgerBtn  = document.getElementById("hamburger-btn");
    const navLinks      = document.getElementById("nav-links");

    // ---------------------------------------------------------------
    // Constants
    // ---------------------------------------------------------------
    const MAX_FILE_SIZE  = 10 * 1024 * 1024; // 10 MB
    const ALLOWED_TYPES  = new Set(["image/jpeg", "image/png", "image/webp"]);
    const GAUGE_CIRCUMFERENCE = 2 * Math.PI * 52; // ~326.73

    let selectedFile = null;

    // ---------------------------------------------------------------
    // 1. Navigation — Scroll Effect
    // ---------------------------------------------------------------
    let lastScrollY = 0;
    window.addEventListener("scroll", function () {
        const scrollY = window.scrollY;
        if (scrollY > 40) {
            nav.classList.add("is-scrolled");
        } else {
            nav.classList.remove("is-scrolled");
        }
        lastScrollY = scrollY;
    }, { passive: true });

    // ---------------------------------------------------------------
    // 1b. Hamburger Menu
    // ---------------------------------------------------------------
    if (hamburgerBtn && navLinks) {
        hamburgerBtn.addEventListener("click", function () {
            hamburgerBtn.classList.toggle("is-active");
            navLinks.classList.toggle("is-open");
        });
        // Close menu when a link is clicked
        navLinks.querySelectorAll(".nav__link").forEach(function (link) {
            link.addEventListener("click", function () {
                hamburgerBtn.classList.remove("is-active");
                navLinks.classList.remove("is-open");
            });
        });
    }

    // ---------------------------------------------------------------
    // 1c. Hero Particles
    // ---------------------------------------------------------------
    var particlesContainer = document.getElementById("hero-particles");
    if (particlesContainer) {
        for (var i = 0; i < 20; i++) {
            var dot = document.createElement("div");
            dot.className = "hero__particle";
            dot.style.left = Math.random() * 100 + "%";
            dot.style.top = (60 + Math.random() * 40) + "%";
            dot.style.animationDelay = (Math.random() * 8) + "s";
            dot.style.animationDuration = (6 + Math.random() * 6) + "s";
            dot.style.width = (2 + Math.random() * 4) + "px";
            dot.style.height = dot.style.width;
            particlesContainer.appendChild(dot);
        }
    }

    // ---------------------------------------------------------------
    // 2. Scroll-Triggered Animations (Intersection Observer)
    // ---------------------------------------------------------------
    const animatedElements = document.querySelectorAll("[data-animate]");

    const observer = new IntersectionObserver(
        function (entries) {
            entries.forEach(function (entry) {
                if (entry.isIntersecting) {
                    const delay = parseInt(entry.target.dataset.delay || "0", 10);
                    setTimeout(function () {
                        entry.target.classList.add("is-visible");
                    }, delay);
                    observer.unobserve(entry.target);
                }
            });
        },
        { threshold: 0.15, rootMargin: "0px 0px -50px 0px" }
    );

    animatedElements.forEach(function (el) { observer.observe(el); });

    // ---------------------------------------------------------------
    // 3. Stat Counter Animation
    // ---------------------------------------------------------------
    const statValues = document.querySelectorAll(".stats__value[data-count]");
    let statsCounted = false;

    const statsObserver = new IntersectionObserver(
        function (entries) {
            entries.forEach(function (entry) {
                if (entry.isIntersecting && !statsCounted) {
                    statsCounted = true;
                    animateCounters();
                    statsObserver.disconnect();
                }
            });
        },
        { threshold: 0.5 }
    );

    const statsSection = document.querySelector(".stats");
    if (statsSection) statsObserver.observe(statsSection);

    function animateCounters() {
        statValues.forEach(function (el) {
            const target = parseFloat(el.dataset.count);
            const isDecimal = target % 1 !== 0;
            const duration = 1500;
            const startTime = performance.now();

            function updateCounter(now) {
                const elapsed = now - startTime;
                const progress = Math.min(elapsed / duration, 1);
                // Ease-out cubic
                const ease = 1 - Math.pow(1 - progress, 3);
                const current = target * ease;

                el.textContent = isDecimal ? current.toFixed(1) : Math.round(current);

                if (progress < 1) {
                    requestAnimationFrame(updateCounter);
                }
            }

            requestAnimationFrame(updateCounter);
        });
    }

    // ---------------------------------------------------------------
    // 4. File Upload — Drag & Drop + Click
    // ---------------------------------------------------------------
    // Click to open file dialog
    uploadZone.addEventListener("click", function (e) {
        if (e.target === removeBtn || removeBtn.contains(e.target)) return;
        fileInput.click();
    });

    fileInput.addEventListener("change", function () {
        if (fileInput.files.length > 0) {
            handleFile(fileInput.files[0]);
        }
    });

    // Drag events
    uploadZone.addEventListener("dragover", function (e) {
        e.preventDefault();
        uploadZone.classList.add("is-dragover");
    });

    uploadZone.addEventListener("dragleave", function (e) {
        e.preventDefault();
        uploadZone.classList.remove("is-dragover");
    });

    uploadZone.addEventListener("drop", function (e) {
        e.preventDefault();
        uploadZone.classList.remove("is-dragover");
        if (e.dataTransfer.files.length > 0) {
            handleFile(e.dataTransfer.files[0]);
        }
    });

    // Remove image
    removeBtn.addEventListener("click", function (e) {
        e.stopPropagation();
        resetUpload();
    });

    function handleFile(file) {
        // Validate type
        if (!ALLOWED_TYPES.has(file.type)) {
            showError("Invalid file type. Please upload a JPEG, PNG, or WebP image.");
            return;
        }

        // Validate size
        if (file.size > MAX_FILE_SIZE) {
            showError("File too large. Maximum size is 10 MB.");
            return;
        }

        selectedFile = file;

        // Show preview
        const reader = new FileReader();
        reader.onload = function (e) {
            previewImage.src = e.target.result;
            uploadContent.style.display = "none";
            previewArea.style.display = "block";
            analyzeBtn.disabled = false;
            hideError();
            hideResults();
        };
        reader.readAsDataURL(file);
    }

    function resetUpload() {
        selectedFile = null;
        fileInput.value = "";
        previewImage.src = "";
        uploadContent.style.display = "";
        previewArea.style.display = "none";
        analyzeBtn.disabled = true;
        hideError();
        hideResults();
    }

    // ---------------------------------------------------------------
    // 5. Analyze — Send to API
    // ---------------------------------------------------------------
    analyzeBtn.addEventListener("click", function () {
        if (!selectedFile || analyzeBtn.disabled) return;
        runAnalysis();
    });

    async function runAnalysis() {
        // UI state: loading
        analyzeBtn.disabled = true;
        analyzeBtnTxt.textContent = "Analyzing…";
        loadingState.style.display = "block";
        hideError();
        hideResults();

        try {
            const formData = new FormData();
            formData.append("image", selectedFile);

            const response = await fetch("/api/predict", {
                method: "POST",
                body: formData,
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || "Server error. Please try again.");
            }

            showResults(data);
        } catch (err) {
            showError(err.message || "Connection error. Please check your network.");
        } finally {
            loadingState.style.display = "none";
            analyzeBtn.disabled = false;
            analyzeBtnTxt.textContent = "Analyze Image";
        }
    }

    // ---------------------------------------------------------------
    // 6. Results Rendering
    // ---------------------------------------------------------------
    function showResults(data) {
        resultsArea.style.display = "block";

        // Verdict
        const isFake = data.verdict === "Fake";
        verdictText.textContent = data.verdict.toUpperCase();
        verdictBadge.className = "try__verdict " + (isFake ? "try__verdict--fake" : "try__verdict--real");

        // Gauge
        const confidence = data.confidence;
        const displayConf = Math.round(confidence * 100);
        const offset = GAUGE_CIRCUMFERENCE * (1 - confidence);

        // Reset gauge for animation
        gaugeFill.style.transition = "none";
        gaugeFill.setAttribute("stroke-dashoffset", GAUGE_CIRCUMFERENCE);

        requestAnimationFrame(function () {
            requestAnimationFrame(function () {
                gaugeFill.style.transition = "stroke-dashoffset 1.2s cubic-bezier(0.65, 0, 0.35, 1), stroke 0.3s ease";
                gaugeFill.setAttribute("stroke-dashoffset", offset);
            });
        });

        // Animate gauge value
        animateValue(gaugeValue, 0, displayConf, 1200, "%");

        // Gauge color
        if (isFake) {
            gaugeFill.classList.add("try__gauge-fill--danger");
        } else {
            gaugeFill.classList.remove("try__gauge-fill--danger");
        }

        // Attribution
        if (isFake && data.attribution) {
            attrArea.style.display = "block";
            attrText.textContent = data.attribution;
            if (data.attribution_confidence != null) {
                attrConf.textContent = (data.attribution_confidence * 100).toFixed(1) + "% confidence";
            } else {
                attrConf.textContent = "";
            }
        } else {
            attrArea.style.display = "none";
        }

        // Smooth scroll to results
        resultsArea.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }

    function animateValue(element, start, end, duration, suffix) {
        const startTime = performance.now();

        function update(now) {
            const elapsed = now - startTime;
            const progress = Math.min(elapsed / duration, 1);
            const ease = 1 - Math.pow(1 - progress, 3);
            const current = Math.round(start + (end - start) * ease);
            element.textContent = current + suffix;
            if (progress < 1) requestAnimationFrame(update);
        }

        requestAnimationFrame(update);
    }

    function hideResults() {
        resultsArea.style.display = "none";
    }

    // ---------------------------------------------------------------
    // 7. Error Handling
    // ---------------------------------------------------------------
    function showError(message) {
        errorMessage.textContent = message;
        errorState.style.display = "flex";
    }

    function hideError() {
        errorState.style.display = "none";
    }

    errorDismiss.addEventListener("click", hideError);

    // ---------------------------------------------------------------
    // 8. Try Again
    // ---------------------------------------------------------------
    tryAgainBtn.addEventListener("click", function () {
        resetUpload();
        uploadZone.scrollIntoView({ behavior: "smooth", block: "center" });
    });

    // ---------------------------------------------------------------
    // 9. Smooth Scroll for Nav Links
    // ---------------------------------------------------------------
    document.querySelectorAll('a[href^="#"]').forEach(function (link) {
        link.addEventListener("click", function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute("href"));
            if (target) {
                target.scrollIntoView({ behavior: "smooth", block: "start" });
            }
        });
    });

})();
