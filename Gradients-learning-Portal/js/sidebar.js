// Ensure LMS namespace exists
window.LMS = window.LMS || {};

LMS.sidebar = {
  render() {
    const sidebarEl = document.getElementById("sidebar-container");
    if (!sidebarEl) return;

    const offset = LMS.getPathOffset();
    const activeInfo = this.getActiveState();
    
    const activeTrack = activeInfo.track || LMS.curriculum.tracks[0];
    const activeModule = activeInfo.module;
    const activeLesson = activeInfo.lesson;

    const stats = LMS.progress.getTrackStats(activeTrack.id);

    let html = `
      <div class="sidebar-header">
        <a href="${offset}course-overview.html?track=${activeTrack.id}" class="back-link">&larr; Track Overview</a>
        <h3>${activeTrack.title}</h3>
        
        <div class="track-progress-container">
          <div class="progress-info">
            <span>Overall Track Progress</span>
            <span class="pct">${stats.percent}%</span>
          </div>
          <div class="progress-bar-bg">
            <div class="progress-bar-fill" style="width: ${stats.percent}%"></div>
          </div>
          <div class="progress-fraction">${stats.completed}/${stats.total} Lessons Complete</div>
        </div>
      </div>
      
      <div class="sidebar-search-box">
        <input type="text" id="sidebar-search-input" class="search-input" placeholder="Search lessons..." oninput="LMS.search.handleSearch(this.value)">
      </div>

      <div class="sidebar-scrollable">
        <nav class="modules-accordion">
    `;

    activeTrack.modules.forEach((mod, modIdx) => {
      const isCurrentModule = activeModule && mod.id === activeModule.id;
      const trackFolder = activeTrack.id === "fundamentals" ? "track-1" : "track-2";
      const displayModuleIdx = activeTrack.id === "dim-reduction" ? (modIdx + 6) : (modIdx + 1);

      html += `
        <div class="accordion-item ${isCurrentModule ? 'active' : ''}">
          <button class="accordion-header" onclick="this.closest('.accordion-item').classList.toggle('active')">
            <div class="module-title-group">
              <span class="module-index">Module ${displayModuleIdx}</span>
              <span class="module-name">${mod.title}</span>
            </div>
            <span class="accordion-icon">&darr;</span>
          </button>
          
          <div class="accordion-content">
            <ul class="lesson-list">
      `;

      mod.lessons.forEach(les => {
        const isCurrentLesson = activeLesson && les.id === activeLesson.id;
        const isCompleted = LMS.progress.isCompleted(les.id);
        
        html += `
          <li class="lesson-item ${isCurrentLesson ? 'current' : ''}" data-lesson-id="${les.id}" data-lesson-title="${les.title.toLowerCase()}">
            <span class="completion-checkbox ${isCompleted ? 'checked' : ''}" 
                  onclick="LMS.progress.toggle('${les.id}')" 
                  title="Toggle Completion Status">
              ${isCompleted ? '&#10003;' : ''}
            </span>
            <a href="${offset}${trackFolder}/${mod.folder}/${les.file}" class="lesson-link">
              <span class="lesson-number">${les.id}</span>
              <span class="lesson-title">${les.title}</span>
            </a>
          </li>
          ${isCurrentLesson && les.subtopics && les.subtopics.length ? this.renderSubtopics(les.subtopics) : ''}
        `;
      });

      html += `
            </ul>
          </div>
        </div>
      `;
    });

    html += `
        </nav>
      </div>
    `;

    sidebarEl.innerHTML = html;
  },

  getActiveState() {
    const path = window.location.pathname.replace(/\\/g, '/');
    let activeTrack = null;
    let activeModule = null;
    let activeLesson = null;

    for (const track of LMS.curriculum.tracks) {
      const trackFolder = track.id === "fundamentals" ? "track-1" : "track-2";
      for (const mod of track.modules) {
        for (const les of mod.lessons) {
          const suffix = `${trackFolder}/${mod.folder}/${les.file}`;
          if (path.endsWith(suffix) || path.includes(suffix)) {
            activeTrack = track;
            activeModule = mod;
            activeLesson = les;
            return { track: activeTrack, module: activeModule, lesson: activeLesson };
          }
        }
      }
    }
    return { track: activeTrack, module: activeModule, lesson: activeLesson };
  },

  renderSubtopics(subtopics) {
    let html = `<ul class="subtopic-list">`;
    subtopics.forEach(st => {
      html += `<li class="subtopic-item">${st}</li>`;
    });
    html += `</ul>`;
    return html;
  },

  injectUtilityPanel() {
    const grid = document.querySelector(".lesson-grid-layout");
    if (!grid) return;
    if (document.getElementById("utility-container")) return;

    const activeInfo = this.getActiveState();
    const lesson = activeInfo.lesson;
    if (!lesson) return;

    const subtopics = lesson.subtopics || [];

    let html = `
      <aside id="utility-container" class="lesson-utility-panel">
        <div class="utility-section">
          <div class="utility-section-title">📑 Lesson Outline</div>
          <ul class="utility-outline-list">
            ${subtopics.length ? subtopics.map(st => `<li class="utility-outline-item">${st}</li>`).join('') : '<li style="font-size:0.78rem;color:var(--text-muted);">No outline available</li>'}
          </ul>
        </div>
        <div class="utility-section">
          <div class="utility-section-title">📖 Reading Progress</div>
          <div class="utility-reading-progress">
            <div class="urp-bar"><div class="urp-fill" id="urp-fill"></div></div>
            <span class="urp-label" id="urp-label">0% read</span>
          </div>
          <button class="utility-bookmark-btn" onclick="LMS.sidebar.toggleBookmark()">
            ${this.isBookmarked() ? '🔖 Bookmarked' : '🔖 Bookmark this lesson'}
          </button>
        </div>
      </aside>
    `;

    grid.insertAdjacentHTML("beforeend", html);
    grid.classList.add("has-utility-panel");

    // Track visit
    if (activeInfo.track && activeInfo.module && activeInfo.lesson) {
      LMS.progress.trackLessonVisit(
        activeInfo.lesson.id,
        activeInfo.lesson.title,
        activeInfo.module.title,
        activeInfo.track.id,
        activeInfo.lesson.file
      );
    }
  },

  setupReadingProgress() {
    const fill = document.getElementById("urp-fill");
    const label = document.getElementById("urp-label");
    if (!fill || !label) return;

    const content = document.querySelector(".lesson-player-body") || document.querySelector(".lesson-body-scrollable");
    if (!content) return;

    content.addEventListener("scroll", () => {
      const scrollTop = content.scrollTop;
      const scrollHeight = content.scrollHeight - content.clientHeight;
      const pct = scrollHeight > 0 ? Math.min(Math.round((scrollTop / scrollHeight) * 100), 100) : 0;
      fill.style.width = `${pct}%`;
      label.textContent = `${pct}% read`;
    });
  },

  isBookmarked() {
    const activeInfo = this.getActiveState();
    if (!activeInfo.lesson) return false;
    const bookmarks = JSON.parse(localStorage.getItem("gradients_lms_bookmarks") || "[]");
    return bookmarks.includes(activeInfo.lesson.id);
  },

  toggleBookmark() {
    const activeInfo = this.getActiveState();
    if (!activeInfo.lesson) return;
    let bookmarks = JSON.parse(localStorage.getItem("gradients_lms_bookmarks") || "[]");
    const idx = bookmarks.indexOf(activeInfo.lesson.id);
    if (idx > -1) {
      bookmarks.splice(idx, 1);
    } else {
      bookmarks.push(activeInfo.lesson.id);
    }
    localStorage.setItem("gradients_lms_bookmarks", JSON.stringify(bookmarks));
    // Re-inject utility panel to refresh button
    const old = document.getElementById("utility-container");
    if (old) old.remove();
    const grid = document.querySelector(".lesson-grid-layout");
    if (grid) grid.classList.remove("has-utility-panel");
    this.injectUtilityPanel();
  },

  setupLayout() {
    const grid = document.querySelector(".lesson-grid-layout");
    if (!grid) return;

    // Remove container constraints from parent <main>
    const main = document.querySelector("main.container");
    if (main) {
      main.style.maxWidth = "100%";
      main.style.padding = "0";
      main.style.paddingBottom = "0";
    }

    // Add player-layout class for 3-column flex
    grid.classList.add("lesson-player-layout");

    // Convert content area to player-content (scroll container)
    const contentArea = grid.querySelector(".lesson-content-area");
    if (contentArea) {
      contentArea.classList.add("lesson-player-content");
      const body = contentArea.querySelector(".lesson-body-scrollable");
      if (body) {
        body.classList.add("lesson-player-body");
      }
    }

    // Move breadcrumb inside content area if it's outside the grid
    const breadcrumb = document.querySelector(".breadcrumb-container");
    if (breadcrumb && contentArea && !grid.contains(breadcrumb)) {
      contentArea.insertBefore(breadcrumb, contentArea.firstChild);
    }

    // Inject right utility panel
    this.injectUtilityPanel();

    // Set up scroll-based reading progress
    this.setupReadingProgress();

    // Re-render math in sidebar/utility dynamic content
    if (typeof LMS.renderMath === 'function') {
      LMS.renderMath();
    }
  }
};

document.addEventListener("DOMContentLoaded", () => {
  LMS.sidebar.render();

  // Set up 3-column layout for lesson pages
  if (document.querySelector(".lesson-grid-layout")) {
    LMS.sidebar.setupLayout();
  }
});
