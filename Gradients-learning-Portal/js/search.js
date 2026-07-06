// Ensure LMS namespace exists
window.LMS = window.LMS || {};

LMS.search = {
  handleSearch(query) {
    const cleanQuery = query.trim().toLowerCase();
    const lessonItems = document.querySelectorAll(".lesson-item");
    const accordionItems = document.querySelectorAll(".accordion-item");

    if (!cleanQuery) {
      // Restore all lesson listings and default module folds
      lessonItems.forEach(item => {
        item.style.display = "flex";
      });
      // Show modules
      accordionItems.forEach(item => {
        item.style.display = "block";
      });
      return;
    }

    // Process matching states
    accordionItems.forEach(acc => {
      let hasMatch = false;
      const items = acc.querySelectorAll(".lesson-item");
      
      items.forEach(item => {
        const title = item.getAttribute("data-lesson-title") || "";
        const id = item.getAttribute("data-lesson-id") || "";
        
        if (title.includes(cleanQuery) || id.includes(cleanQuery)) {
          item.style.display = "flex";
          hasMatch = true;
        } else {
          item.style.display = "none";
        }
      });

      // Show/Hide the accordion block entirely based on child matches
      if (hasMatch) {
        acc.style.display = "block";
        acc.classList.add("active"); // auto-expand matching modules
      } else {
        acc.style.display = "none";
      }
    });
  }
};
