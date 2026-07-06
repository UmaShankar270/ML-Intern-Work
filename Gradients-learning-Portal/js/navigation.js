// Ensure LMS namespace exists
window.LMS = window.LMS || {};

/**
 * Path Offset Calculator
 * Determines the relative prefix (e.g., '../../' or './') dynamically
 * based on the nesting depth of the current page.
 */
LMS.getPathOffset = function() {
  const path = window.location.pathname.replace(/\\/g, '/');
  
  // Count depth by searching for track subdirectories
  if (path.includes('/track-1/') || path.includes('/track-2/') || path.includes('/content/')) {
    // If inside track-1/01-What_is_ML/
    return "../../";
  }
  return "./";
};

LMS.navigation = {
  injectHeader() {
    const headerEl = document.getElementById("header-container");
    if (!headerEl) return;

    const offset = LMS.getPathOffset();
    const activePage = window.location.pathname.split("/").pop() || "index.html";

    headerEl.innerHTML = `
      <div class="header-container">
        <a href="${offset}index.html" class="logo-area">
          <span class="logo-icon">▲</span>
          <span class="logo-text">GRADIENTS</span>
          <span class="badge">LMS</span>
        </a>
        
        <nav class="main-nav">
          <a href="${offset}index.html" class="nav-link ${activePage === 'index.html' ? 'active' : ''}">Dashboard</a>
          <a href="${offset}course-overview.html?track=fundamentals" class="nav-link ${activePage === 'course-overview.html' ? 'active' : ''}">Track 1</a>
          <a href="${offset}course-overview.html?track=dim-reduction" class="nav-link ${activePage === 'course-overview.html' ? 'active' : ''}">Track 2</a>
          <a href="${offset}curriculum.html" class="nav-link ${activePage === 'curriculum.html' ? 'active' : ''}">Curriculum</a>
        </nav>

        <div class="header-actions">
          <button class="theme-toggle-btn" onclick="LMS.theme.toggle()" title="Toggle Dark/Light Mode">
            &#9790;
          </button>
          
          <div class="global-progress-pill" title="Overall Course Progress">
            <span class="progress-label">Progress</span>
            <div class="progress-bar-mini-bg">
              <div id="header-progress-fill" class="progress-bar-mini-fill" style="width: 0%"></div>
            </div>
            <span id="header-progress-pct" class="progress-percent">0%</span>
          </div>
        </div>
      </div>
    `;
  },

  injectFooter() {
    const footerEl = document.getElementById("footer-container");
    if (!footerEl) return;

    const offset = LMS.getPathOffset();

    footerEl.innerHTML = `
      <div class="container footer-grid">
        <div>
          <h4>GRADIENTS</h4>
          <p>Machine Learning Internship Learning Portal. Structured educational path from beginner to advanced concepts.</p>
        </div>
        <div>
          <h5>Tracks</h5>
          <ul>
            <li><a href="${offset}curriculum.html">Track 1: ML Fundamentals</a></li>
            <li><a href="${offset}curriculum.html">Track 2: Dimensionality Reduction</a></li>
          </ul>
        </div>
      </div>
      <div class="footer-bottom">
        <p>&copy; ${new Date().getFullYear()} Gradients LMS. Designed for interactive ML mastery.</p>
      </div>
    `;
  }
};

document.addEventListener("DOMContentLoaded", () => {
  LMS.navigation.injectHeader();
  LMS.navigation.injectFooter();
});
