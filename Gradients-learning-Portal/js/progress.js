// Ensure LMS namespace exists
window.LMS = window.LMS || {};

(function() {
  const PROGRESS_KEY = "gradients_lms_completed_lessons";
  const ACTIVITY_KEY = "gradients_lms_recent_activity";
  let completed = new Set(JSON.parse(localStorage.getItem(PROGRESS_KEY)) || []);

  window.LMS.progress = {
    isCompleted(lessonId) {
      return completed.has(lessonId);
    },

    toggle(lessonId) {
      if (completed.has(lessonId)) {
        completed.delete(lessonId);
      } else {
        completed.add(lessonId);
      }
      localStorage.setItem(PROGRESS_KEY, JSON.stringify(Array.from(completed)));
      this.updateAllUI();
    },

    reset() {
      completed.clear();
      localStorage.removeItem(PROGRESS_KEY);
      this.updateAllUI();
    },

    getTrackStats(trackId) {
      const track = LMS.curriculum.tracks.find(t => t.id === trackId);
      if (!track) return { percent: 0, completed: 0, total: 0 };

      let total = 0;
      let completedCount = 0;

      track.modules.forEach(mod => {
        mod.lessons.forEach(les => {
          total++;
          if (completed.has(les.id)) {
            completedCount++;
          }
        });
      });

      const percent = total > 0 ? Math.round((completedCount / total) * 100) : 0;
      return { percent, completed: completedCount, total };
    },

    getModuleStats(trackId, moduleId) {
      const track = LMS.curriculum.tracks.find(t => t.id === trackId);
      if (!track) return { percent: 0, completed: 0, total: 0 };

      const mod = track.modules.find(m => m.id === moduleId);
      if (!mod) return { percent: 0, completed: 0, total: 0 };

      let total = 0;
      let completedCount = 0;

      mod.lessons.forEach(les => {
        total++;
        if (completed.has(les.id)) {
          completedCount++;
        }
      });

      const percent = total > 0 ? Math.round((completedCount / total) * 100) : 0;
      return { percent, completed: completedCount, total };
    },

    getOverallStats() {
      const s1 = this.getTrackStats("fundamentals");
      const s2 = this.getTrackStats("dim-reduction");
      const completed = s1.completed + s2.completed;
      const total = s1.total + s2.total;
      const percent = total > 0 ? Math.round((completed / total) * 100) : 0;
      return { percent, completed, total, remaining: total - completed };
    },

    getCurrentLessonInfo() {
      for (const track of LMS.curriculum.tracks) {
        for (const mod of track.modules) {
          for (const les of mod.lessons) {
            if (!completed.has(les.id)) {
              return { track, module: mod, lesson: les };
            }
          }
        }
      }
      const lastTrack = LMS.curriculum.tracks[LMS.curriculum.tracks.length - 1];
      const lastMod = lastTrack.modules[lastTrack.modules.length - 1];
      const lastLes = lastMod.lessons[lastMod.lessons.length - 1];
      return { track: lastTrack, module: lastMod, lesson: lastLes };
    },

    getLessonState(track, mod, lessonId) {
      let found = false;
      for (const les of mod.lessons) {
        if (les.id === lessonId) { found = true; break; }
        if (!completed.has(les.id) && les.id !== lessonId) {
          return "locked";
        }
      }
      return "current";
    },

    trackLessonVisit(lessonId, lessonTitle, moduleTitle, trackId, file) {
      let visits = JSON.parse(localStorage.getItem(ACTIVITY_KEY)) || [];
      visits = visits.filter(v => v.lessonId !== lessonId);
      visits.unshift({ lessonId, lessonTitle, moduleTitle, trackId, file, timestamp: Date.now() });
      if (visits.length > 10) visits = visits.slice(0, 10);
      localStorage.setItem(ACTIVITY_KEY, JSON.stringify(visits));
    },

    getRecentActivity() {
      return JSON.parse(localStorage.getItem(ACTIVITY_KEY)) || [];
    },

    getAchievements() {
      const totalCompleted = this.getOverallStats().completed;
      const t1s = this.getTrackStats("fundamentals");
      const t2s = this.getTrackStats("dim-reduction");
      return [
        { id: "first", icon: "🎯", title: "First Lesson", desc: "Complete your first lesson", unlocked: totalCompleted >= 1 },
        { id: "starter", icon: "🚀", title: "Getting Started", desc: "Complete 5 lessons", unlocked: totalCompleted >= 5 },
        { id: "dedicated", icon: "💪", title: "Dedicated Learner", desc: "Complete 10 lessons", unlocked: totalCompleted >= 10 },
        { id: "halfway", icon: "⚡", title: "Half Way There", desc: "Complete 25 lessons", unlocked: totalCompleted >= 25 },
        { id: "advanced", icon: "🧠", title: "Advanced Learner", desc: "Complete 40 lessons", unlocked: totalCompleted >= 40 },
        { id: "fundamentals", icon: "🏆", title: "Master of ML Fundamentals", desc: "Complete all Track 1 lessons", unlocked: t1s.completed >= t1s.total && t1s.total > 0 },
        { id: "dr", icon: "🔬", title: "Dimensionality Reduction Expert", desc: "Complete all Track 2 lessons", unlocked: t2s.completed >= t2s.total && t2s.total > 0 },
        { id: "grandmaster", icon: "👑", title: "Grandmaster", desc: "Complete all 84 lessons", unlocked: totalCompleted >= 84 }
      ];
    },

    renderDashboard() {
      const el = document.getElementById("dashboard-container");
      if (!el) return;

      const overall = this.getOverallStats();
      const currentInfo = this.getCurrentLessonInfo();
      const recentActivity = this.getRecentActivity();
      const achievements = this.getAchievements();

      const trackFolder1 = "track-1";
      const trackFolder2 = "track-2";
      const s1 = this.getTrackStats("fundamentals");
      const s2 = this.getTrackStats("dim-reduction");
      const track1 = LMS.curriculum.tracks[0];
      const track2 = LMS.curriculum.tracks[1];

      const continueTrack1 = this.getFirstIncompleteLesson(track1);
      const continueTrack2 = this.getFirstIncompleteLesson(track2);

      function getTrackLabelClass(trackId) {
        return trackId === "fundamentals" ? "cyan" : "purple";
      }
      function getGradientClass(trackId) {
        return trackId === "fundamentals" ? "cyan" : "purple";
      }

      let html = `
        <div class="dashboard-page">
          <div class="section-title-row">
            <h2>My Learning Tracks</h2>
            <a href="course-overview.html" class="section-link">View All &rarr;</a>
          </div>
          <div class="dashboard-tracks-grid">
            ${this.renderTrackCard(track1, s1, trackFolder1, continueTrack1, "cyan", "purple")}
            ${this.renderTrackCard(track2, s2, trackFolder2, continueTrack2, "purple", "cyan")}
          </div>

          <div class="section-title-row" style="margin-top: 2.5rem;">
            <h2>Learning Statistics</h2>
          </div>
          <div class="dashboard-stats-grid">
            <div class="stat-card glass">
              <div class="stat-card-icon blue">📚</div>
              <div class="stat-card-info">
                <h4>${overall.total}</h4>
                <p>Total Lessons</p>
              </div>
            </div>
            <div class="stat-card glass">
              <div class="stat-card-icon green">✅</div>
              <div class="stat-card-info">
                <h4>${overall.completed}</h4>
                <p>Completed</p>
              </div>
            </div>
            <div class="stat-card glass">
              <div class="stat-card-icon cyan">📖</div>
              <div class="stat-card-info">
                <h4>${overall.remaining}</h4>
                <p>Remaining</p>
              </div>
            </div>
            <div class="stat-card glass">
              <div class="stat-card-icon purple">⏱️</div>
              <div class="stat-card-info">
                <h4>${Math.round(overall.completed * 0.25)}h</h4>
                <p>Study Time</p>
              </div>
            </div>
          </div>

          <div class="dashboard-progress-section">
            <div class="section-title-row">
              <h2>Learning Progress</h2>
            </div>
            <div class="progress-main-card glass">
              <div class="progress-main-top">
                <h3>Overall Course Progress</h3>
                <span class="progress-big-number gradient-text">${overall.percent}%</span>
              </div>
              <div class="progress-main-bar">
                <div class="progress-main-fill" style="width: ${overall.percent}%"></div>
              </div>
              <div class="progress-main-details">
                <div class="progress-detail-item">
                  <span class="label">Track 1</span>
                  <span class="value">${s1.completed}/${s1.total} (${s1.percent}%)</span>
                </div>
                <div class="progress-detail-item">
                  <span class="label">Track 2</span>
                  <span class="value">${s2.completed}/${s2.total} (${s2.percent}%)</span>
                </div>
                <div class="progress-detail-item">
                  <span class="label">Current Module</span>
                  <span class="value current">${currentInfo.module ? currentInfo.module.title : "All Complete!"}</span>
                </div>
                <div class="progress-detail-item">
                  <span class="label">Current Lesson</span>
                  <span class="value current">${currentInfo.lesson ? currentInfo.lesson.title : "—"}</span>
                </div>
              </div>
            </div>
          </div>

          <div class="dashboard-activity-section">
            <div class="section-title-row">
              <h2>Recent Activity</h2>
            </div>
            <div class="activity-list">
              ${recentActivity.length === 0 ? '<p style="color: var(--text-muted); padding: 1rem 0;">No recent activity yet. Start a lesson to track your progress!</p>' : ''}
              ${recentActivity.map(a => {
                const trackF = a.trackId === "fundamentals" ? "track-1" : "track-2";
                const mod = LMS.curriculum.tracks.flatMap(t => t.modules).find(m => m.title === a.moduleTitle);
                const les = mod ? mod.lessons.find(l => l.id === a.lessonId) : null;
                const href = les ? `${trackF}/${mod.folder}/${les.file}` : "#";
                const timeAgo = this.getTimeAgo(a.timestamp);
                return `
                  <div class="activity-item glass">
                    <div class="activity-icon">📖</div>
                    <div class="activity-info">
                      <div class="activity-title">${a.lessonTitle}</div>
                      <div class="activity-meta">${a.moduleTitle} &middot; ${timeAgo}</div>
                    </div>
                    <a href="${href}" class="activity-link">Continue &rarr;</a>
                  </div>
                `;
              }).join('')}
            </div>
          </div>

          <div class="dashboard-achievements-section">
            <div class="section-title-row">
              <h2>Achievements</h2>
            </div>
            <div class="achievements-grid">
              ${achievements.map(a => `
                <div class="achievement-badge glass ${a.unlocked ? '' : 'locked'}">
                  <div class="badge-icon">${a.unlocked ? a.icon : '🔒'}</div>
                  <h4>${a.title}</h4>
                  <p>${a.desc}</p>
                </div>
              `).join('')}
            </div>
          </div>
        </div>
      `;
      el.innerHTML = html;
    },

    renderTrackCard(track, stats, trackFolder, continueLesson, labelClass, progressClass) {
      const modulesCount = track.modules.length;
      const lessonsCount = stats.total;
      const totalMins = lessonsCount * 10;
      const href = continueLesson ? `${trackFolder}/${continueLesson.module.folder}/${continueLesson.lesson.file}` : "#";
      const label = track.id === "fundamentals" ? "Track 1" : "Track 2";
      const icon = track.id === "fundamentals" ? "🧠" : "🗜️";
      return `
        <div class="dashboard-track-card glass">
          <div class="dt-card-top">
            <span class="dt-track-label ${labelClass}">${label}</span>
            <span class="dt-track-icon">${icon}</span>
          </div>
          <h3>${track.title}</h3>
          <p>${track.description}</p>
          <div class="dt-meta-row">
            <span class="dt-meta-item">📦 <strong>${modulesCount}</strong> Modules</span>
            <span class="dt-meta-item">📊 <strong>${lessonsCount}</strong> Lessons</span>
            <span class="dt-meta-item">⏱️ <strong>${totalMins}</strong> min</span>
          </div>
          <div class="dt-progress-bar-bg">
            <div class="dt-progress-bar-fill ${progressClass}" style="width: ${stats.percent}%"></div>
          </div>
          <div class="dt-progress-label">${stats.completed}/${lessonsCount} lessons completed &middot; ${stats.percent}%</div>
          <div class="dt-card-footer">
            <a href="${href}" class="btn btn-primary">Continue &rarr;</a>
            <a href="course-overview.html?track=${track.id}" class="btn btn-secondary">View Details</a>
          </div>
        </div>
      `;
    },

    getFirstIncompleteLesson(track) {
      for (const mod of track.modules) {
        for (const les of mod.lessons) {
          if (!completed.has(les.id)) {
            return { module: mod, lesson: les };
          }
        }
      }
      const lastMod = track.modules[track.modules.length - 1];
      const lastLes = lastMod.lessons[lastMod.lessons.length - 1];
      return { module: lastMod, lesson: lastLes };
    },

    getLessonStateInModule(mod) {
      let foundIncomplete = false;
      return mod.lessons.map(les => {
        if (completed.has(les.id)) return "completed";
        if (!foundIncomplete) { foundIncomplete = true; return "current"; }
        return "locked";
      });
    },

    getTimeAgo(timestamp) {
      const mins = Math.floor((Date.now() - timestamp) / 60000);
      if (mins < 1) return "Just now";
      if (mins < 60) return `${mins}m ago`;
      const hours = Math.floor(mins / 60);
      if (hours < 24) return `${hours}h ago`;
      const days = Math.floor(hours / 24);
      if (days < 7) return `${days}d ago`;
      return `${Math.floor(days / 7)}w ago`;
    },

    renderTrackPage(trackId) {
      const el = document.getElementById("track-container");
      if (!el) return;

      const track = LMS.curriculum.tracks.find(t => t.id === trackId);
      if (!track) {
        el.innerHTML = `<p>Track not found.</p>`;
        return;
      }

      const stats = this.getTrackStats(track.id);
      const trackFolder = track.id === "fundamentals" ? "track-1" : "track-2";
      const continueInfo = this.getFirstIncompleteLesson(track);
      const continueHref = continueInfo ? `${trackFolder}/${continueInfo.module.folder}/${continueInfo.lesson.file}` : "#";
      const totalMins = stats.total * 10;
      const isTrack1 = track.id === "fundamentals";
      const accentClass = isTrack1 ? "cyan" : "purple";

      const outcomes = isTrack1 ? [
        "Understand the shift from rule-based code to data-driven ML.",
        "Diagnose bias-variance tradeoffs and overfitting/underfitting.",
        "Apply L1, L2, and Elastic Net regularization effectively.",
        "Engineer features: encoding, scaling, polynomial expansion.",
        "Implement cross-validation strategies without data leakage."
      ] : [
        "Understand when and why to apply dimensionality reduction.",
        "Project labeled data using Fisher's Linear Discriminant.",
        "Decompose variance using PCA and interpret scree plots.",
        "Visualize high-dimensional manifolds with t-SNE and UMAP.",
        "Build autoencoders and variational autoencoders from scratch."
      ];

      const difficulty = isTrack1 ? "Beginner to Intermediate" : "Intermediate to Advanced";

      let html = `
        <div class="track-page">
          <div class="track-header-section">
            <div class="track-header-grid">
              <div class="track-title-group">
                <span class="dt-track-label ${accentClass}" style="margin-bottom: 0.5rem; display: inline-block;">${isTrack1 ? "Track 1" : "Track 2"}</span>
                <h1>${track.title}</h1>
                <p class="track-subtitle">${track.description}</p>
                <div class="track-meta-pills">
                  <span class="track-meta-pill"><span class="pill-icon">📦</span> ${track.modules.length} Modules</span>
                  <span class="track-meta-pill"><span class="pill-icon">📊</span> ${stats.total} Lessons</span>
                  <span class="track-meta-pill"><span class="pill-icon">⏱️</span> ${totalMins} min</span>
                  <span class="track-meta-pill"><span class="pill-icon">🎯</span> ${difficulty}</span>
                </div>
              </div>
              <div class="track-progress-side-card glass">
                <span class="tp-label">Your Progress</span>
                <div class="tp-big-progress">
                  <span class="tp-pct gradient-text">${stats.percent}%</span>
                  <span class="tp-frac">${stats.completed}/${stats.total}</span>
                </div>
                <div class="tp-bar-bg">
                  <div class="tp-bar-fill" style="width: ${stats.percent}%"></div>
                </div>
                <a href="${continueHref}" class="btn btn-primary">Continue Learning &rarr;</a>
              </div>
            </div>
          </div>

          <div class="track-outcomes-section">
            <div class="section-title-block">
              <h2>Learning Outcomes</h2>
              <p>What you will accomplish after completing this track.</p>
            </div>
            <div class="outcomes-grid">
              ${outcomes.map(o => `
                <div class="outcome-item glass">
                  <span class="outcome-check">✓</span>
                  <span class="outcome-text">${o}</span>
                </div>
              `).join('')}
            </div>
          </div>

          <div class="track-desc-section">
            <div class="section-title-block">
              <h2>About This Track</h2>
            </div>
            <div class="track-desc-tabs" id="track-desc-tabs">
              <span class="track-desc-tab active" data-tab="what">What You'll Learn</span>
              <span class="track-desc-tab" data-tab="who">Who This Is For</span>
              <span class="track-desc-tab" data-tab="prereqs">Prerequisites</span>
              <span class="track-desc-tab" data-tab="outcome">Outcome</span>
            </div>
            <div class="track-desc-panel active" data-panel="what">
              ${isTrack1 ? `
                <p>This foundational track builds your understanding of Machine Learning from first principles. You will explore the core ML pipeline, understand cost functions, and learn to diagnose model performance through bias-variance analysis. Every concept is motivated intuitively, formalized mathematically, and implemented in Python.</p>
                <ul>
                  <li>Understand what makes ML different from traditional programming</li>
                  <li>Diagnose underfitting vs overfitting systematically</li>
                  <li>Apply L1, L2, and Elastic Net regularization with confidence</li>
                  <li>Master feature engineering: encoding, scaling, interactions, selection</li>
                  <li>Implement K-Fold, Stratified, and LOOCV strategies</li>
                </ul>
              ` : `
                <p>This advanced track dives deep into dimensionality reduction techniques. You will progress from classical linear methods (LDA, PCA) to modern non-linear embeddings (t-SNE, UMAP), and finally to deep learning-based approaches (Autoencoders, VAEs). Each method is motivated by a specific structural problem and evaluated on real datasets.</p>
                <ul>
                  <li>Understand the mathematical foundations behind each DR technique</li>
                  <li>Implement PCA via eigendecomposition and SVD</li>
                  <li>Visualize complex high-dimensional manifolds with t-SNE and UMAP</li>
                  <li>Build undercomplete and variational autoencoders in Python</li>
                  <li>Compare trade-offs: interpretability, speed, global vs local preservation</li>
                </ul>
              `}
            </div>
            <div class="track-desc-panel" data-panel="who">
              ${isTrack1 ? `
                <p>This track is designed for a wide range of learners:</p>
                <ul>
                  <li><strong>Students</strong> with basic math and programming who want to build a structured ML foundation</li>
                  <li><strong>Data enthusiasts</strong> looking to formalize their understanding of ML algorithms</li>
                  <li><strong>Interns & Juniors</strong> preparing for technical ML interviews</li>
                </ul>
              ` : `
                <p>This track is best suited for:</p>
                <ul>
                  <li><strong>Learners</strong> who have completed Track 1 or have equivalent foundational knowledge</li>
                  <li><strong>Data Scientists</strong> working with high-dimensional datasets</li>
                  <li><strong>Researchers</strong> exploring visualization and compression techniques</li>
                </ul>
              `}
            </div>
            <div class="track-desc-panel" data-panel="prereqs">
              ${isTrack1 ? `
                <ul>
                  <li>Basic programming (Python recommended)</li>
                  <li>High school level mathematics (algebra, probability)</li>
                  <li>Curiosity and logical thinking</li>
                </ul>
              ` : `
                <ul>
                  <li>Completion of Track 1 (ML Fundamentals) or equivalent knowledge</li>
                  <li>Familiarity with linear algebra, calculus, and probability</li>
                  <li>Basic Python and NumPy experience</li>
                </ul>
              `}
            </div>
            <div class="track-desc-panel" data-panel="outcome">
              ${isTrack1 ? `
                <p>By the end of this track, you will be able to independently build, evaluate, and debug ML models. You will understand the structural intuition behind every model component, diagnose training issues using learning curves, and apply proper validation strategies.</p>
              ` : `
                <p>By the end of this track, you will be able to critically evaluate when dimensionality reduction is appropriate, choose the right technique for your data structure, and implement both classical and deep learning compression methods with a solid grasp of their mathematical foundations.</p>
              `}
            </div>
          </div>

          <div class="track-modules-section">
            <div class="section-title-block">
              <h2>Modules</h2>
              <p>${track.modules.length} modules &middot; ${stats.total} lessons</p>
            </div>
            ${track.modules.map((mod, modIdx) => {
              const modStats = this.getModuleStats(track.id, mod.id);
              const lessonStates = this.getLessonStateInModule(mod);
              const modNum = track.id === "dim-reduction" ? modIdx + 6 : modIdx + 1;
              return `
                <div class="module-card glass" data-module-id="${mod.id}">
                  <div class="module-card-header" onclick="this.closest('.module-card').classList.toggle('expanded')">
                    <div class="module-card-header-left">
                      <div class="module-card-number">${String(modNum).padStart(2, '0')}</div>
                      <div class="module-card-info">
                        <h4>${mod.title}</h4>
                        <p>${modStats.total} lessons</p>
                      </div>
                    </div>
                    <div class="module-card-right">
                      <div class="module-card-stats">
                        <span>📊 ${modStats.completed}/${modStats.total}</span>
                        <span>⏱️ ${modStats.total * 10}m</span>
                      </div>
                      <span class="module-card-arrow">▾</span>
                    </div>
                  </div>
                  <div class="module-card-progress">
                    <div class="mcp-bar"><div class="mcp-fill" style="width: ${modStats.percent}%"></div></div>
                    <span class="mcp-label">${modStats.percent}%</span>
                  </div>
                  <div class="module-card-body">
                    <ul class="module-lesson-list">
                      ${mod.lessons.map((les, lIdx) => {
                        const state = lessonStates[lIdx];
                        const stateIcon = state === "completed" ? "✓" : state === "current" ? "●" : "🔒";
                        const href = state !== "locked" ? `${trackFolder}/${mod.folder}/${les.file}` : "#";
                        return `
                          <li class="module-lesson-item ${state}">
                            <span class="lesson-state-icon">${stateIcon}</span>
                            <span class="lesson-state-title">${les.id} &middot; ${les.title}</span>
                            ${state !== "locked" ? `<a href="${href}" class="lesson-state-link">${state === "completed" ? "Review" : "Start"} &rarr;</a>` : '<span class="lesson-state-link" style="color: var(--text-muted);">Locked</span>'}
                          </li>
                        `;
                      }).join('')}
                    </ul>
                    <a href="${trackFolder}/${mod.folder}/${mod.lessons[0].file}" class="btn btn-primary" style="margin-top: 0.75rem; width: 100%;">Start Module &rarr;</a>
                  </div>
                </div>
              `;
            }).join('')}
          </div>
        </div>
      `;
      el.innerHTML = html;

      // Bind description tabs
      const tabs = document.querySelectorAll("#track-desc-tabs .track-desc-tab");
      tabs.forEach(tab => {
        tab.addEventListener("click", () => {
          tabs.forEach(t => t.classList.remove("active"));
          tab.classList.add("active");
          document.querySelectorAll(".track-desc-panel").forEach(p => p.classList.remove("active"));
          const panel = document.querySelector(`.track-desc-panel[data-panel="${tab.dataset.tab}"]`);
          if (panel) panel.classList.add("active");
        });
      });
    },

    updateAllUI() {
      const stats1 = this.getTrackStats("fundamentals");
      const stats2 = this.getTrackStats("dim-reduction");
      const totalCompleted = stats1.completed + stats2.completed;
      const totalLessons = stats1.total + stats2.total;
      const overallPercent = totalLessons > 0 ? Math.round((totalCompleted / totalLessons) * 100) : 0;

      const fill = document.getElementById("header-progress-fill");
      const pctLabel = document.getElementById("header-progress-pct");
      if (fill) fill.style.width = `${overallPercent}%`;
      if (pctLabel) pctLabel.textContent = `${overallPercent}%`;

      if (LMS.sidebar && typeof LMS.sidebar.render === "function") {
        LMS.sidebar.render();
      }

      const activeInfo = LMS.sidebar ? LMS.sidebar.getActiveState() : {};
      const lessonBtn = document.getElementById("toggle-complete-btn");
      if (lessonBtn && activeInfo.lesson) {
        const isCompleted = this.isCompleted(activeInfo.lesson.id);
        if (isCompleted) {
          lessonBtn.className = "btn btn-secondary btn-completed";
          lessonBtn.innerHTML = "&#10003; Completed";
        } else {
          lessonBtn.className = "btn btn-secondary";
          lessonBtn.innerHTML = "Mark as Complete";
        }
      }

      this.renderCurriculumPage();

      const track1Pct = document.getElementById("track-1-pct");
      const track1Frac = document.getElementById("track-1-frac");
      const track2Pct = document.getElementById("track-2-pct");
      const track2Frac = document.getElementById("track-2-frac");
      
      if (track1Pct) track1Pct.textContent = `Progress: ${stats1.percent}%`;
      if (track1Frac) track1Frac.textContent = `${stats1.completed}/${stats1.total} Lessons`;
      if (track2Pct) track2Pct.textContent = `Progress: ${stats2.percent}%`;
      if (track2Frac) track2Frac.textContent = `${stats2.completed}/${stats2.total} Lessons`;

      // Re-render dashboard if it exists
      this.renderDashboard();
      // Re-render track page if it exists
      const trackContainer = document.getElementById("track-container");
      if (trackContainer && trackContainer.dataset.trackId) {
        this.renderTrackPage(trackContainer.dataset.trackId);
      }
    },

    renderCurriculumPage() {
      const pageEl = document.getElementById("curriculum-page-container");
      if (!pageEl) return;

      const offset = LMS.getPathOffset();

      let html = `
        <div class="curriculum-hero">
          <div class="container">
            <h1>Course Curriculum</h1>
            <p class="lead-text">Study and navigate the entire structured curriculum below. Your progress updates dynamically.</p>
            <button id="reset-progress-btn" class="btn btn-secondary btn-reset">
              &#8634; Reset All Learning Progress
            </button>
          </div>
        </div>

        <div class="container curriculum-layout">
      `;

      LMS.curriculum.tracks.forEach(track => {
        const trackStats = this.getTrackStats(track.id);

        html += `
          <section class="track-detail-section">
            <div class="track-detail-header">
              <div class="track-title-block">
                <h2>${track.title}</h2>
                <p>${track.description}</p>
              </div>
              <div class="track-pct-pill">
                <span class="pct-num">${trackStats.percent}%</span>
                <span class="pct-lbl">Completed</span>
              </div>
            </div>

            <div class="track-modules-list">
        `;

        track.modules.forEach((mod, modIdx) => {
          const modStats = this.getModuleStats(track.id, mod.id);
          const trackFolder = track.id === "fundamentals" ? "track-1" : "track-2";

          html += `
            <div class="module-block glass">
              <div class="module-block-header">
                <div>
                  <span class="module-number-label">Module ${modIdx + 1}</span>
                  <h3>${mod.title}</h3>
                </div>
                <span class="module-progress-text">${modStats.completed}/${modStats.total} Complete</span>
              </div>

              <div class="module-lessons-grid">
          `;

          mod.lessons.forEach(les => {
            const isCompleted = this.isCompleted(les.id);

            html += `
              <div class="lesson-card ${isCompleted ? 'completed' : ''}">
                <div class="lesson-card-state">
                  <span class="card-checkbox ${isCompleted ? 'checked' : ''}" 
                        onclick="LMS.progress.toggle('${les.id}')">
                    ${isCompleted ? '&#10003;' : ''}
                  </span>
                  <span class="lesson-id-tag">${les.id}</span>
                </div>
                <h4 class="lesson-card-title">${les.title}</h4>
                <a href="${offset}${trackFolder}/${mod.folder}/${les.file}" class="lesson-card-link">
                  Study Lesson &rarr;
                </a>
              </div>
            `;
          });

          html += `
              </div>
            </div>
          `;
        });

        html += `
            </div>
          </section>
        `;
      });

      html += `
        </div>
      `;

      pageEl.innerHTML = html;

      const resetBtn = document.getElementById("reset-progress-btn");
      if (resetBtn) {
        resetBtn.addEventListener("click", () => {
          if (confirm("Are you sure you want to reset all learning progress? This cannot be undone.")) {
            this.reset();
          }
        });
      }
    }
  };

  document.addEventListener("DOMContentLoaded", () => {
    LMS.progress.updateAllUI();

    // Render dashboard if container exists
    if (document.getElementById("dashboard-container")) {
      LMS.progress.renderDashboard();
    }

    // Render track page from URL param
    const trackContainer = document.getElementById("track-container");
    if (trackContainer) {
      const params = new URLSearchParams(window.location.search);
      const trackId = params.get("track") || "fundamentals";
      trackContainer.dataset.trackId = trackId;
      LMS.progress.renderTrackPage(trackId);
    }

    const lessonBtn = document.getElementById("toggle-complete-btn");
    if (lessonBtn) {
      lessonBtn.addEventListener("click", () => {
        const activeInfo = LMS.sidebar ? LMS.sidebar.getActiveState() : {};
        if (activeInfo.lesson) {
          LMS.progress.toggle(activeInfo.lesson.id);
        }
      });
    }
  });
})();
