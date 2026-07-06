(function () {
  'use strict';

  var DELIMITERS = [
    { left: '$$', right: '$$', display: true },
    { left: '$', right: '$', display: false },
    { left: '\\(', right: '\\)', display: false },
    { left: '\\[', right: '\\]', display: true }
  ];

  function renderAllMath() {
    if (typeof renderMathInElement === 'undefined') return;
    try {
      renderMathInElement(document.body, {
        delimiters: DELIMITERS,
        throwOnError: false
      });
    } catch (e) { /* silent */ }

    renderMathEqElements();
  }

  function renderMathEqElements() {
    if (typeof katex === 'undefined') return;
    document.querySelectorAll('.math-eq').forEach(function (el) {
      if (el.querySelector('.katex')) return;
      var text = el.textContent.trim();
      if (!text) return;
      text = text.replace(/^\\\[|\\\]$/g, '').replace(/^\\\(|\\\)$/g, '').trim();
      if (!text) return;
      try {
        katex.render(text, el, {
          displayMode: !!el.closest('.math-card'),
          throwOnError: false
        });
      } catch (e) {
        el.textContent = text;
      }
    });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', renderAllMath);
  } else {
    renderAllMath();
  }

  window.LMS = window.LMS || {};
  window.LMS.renderMath = renderAllMath;
})();
