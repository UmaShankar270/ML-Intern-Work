// Initialize global LMS namespace
window.LMS = window.LMS || {};

/**
 * Master Curriculum Config — Single Source of Truth
 * Defines Course → Track → Module → Lesson → Subtopic hierarchy
 * with subtopics for sidebar ToC and curriculum page previews.
 */
window.LMS.curriculum = {
  tracks: [
    // ──────────────────────────────────────────────
    // TRACK 1: Machine Learning Fundamentals
    // ──────────────────────────────────────────────
    {
      id: "fundamentals",
      title: "Machine Learning Fundamentals",
      description: "Master the foundations of Machine Learning from first principles. Understand the core workflow, diagnostics, bias-variance tradeoff, regularization, and evaluation strategies.",
      modules: [
        // ── Module 01 ──
        {
          id: "01-What_is_ML",
          title: "Introduction to Machine Learning",
          folder: "01-What_is_ML",
          lessons: [
            { id: "1.1", title: "The Paradigm Shift", file: "1.1-introduction.html", subtopics: ["The Problem with Rule-Based Systems", "What is Machine Learning?", "Formal Definition (Tom Mitchell)", "Traditional Programming vs ML", "When to Use ML"] },
            { id: "1.2", title: "Why Machine Learning?", file: "1.2-why-ml.html", subtopics: ["Limits of Hand-Coded Rules", "The Learning Paradigm Shift", "Why ML Works in Practice", "Problem Scale and Adaptability"] },
            { id: "1.3", title: "Traditional Programming vs ML", file: "1.3-traditional-vs-ml.html", subtopics: ["Rule-Based vs Data-Driven", "Visual Comparison", "Core Mechanics", "Decision Guide"] },
            { id: "1.4", title: "Core Learning Paradigms", file: "1.4-ai-vs-ml.html", subtopics: ["Supervised Learning", "Unsupervised Learning", "Reinforcement Learning", "Paradigm Decision Tree"] },
            { id: "1.5", title: "Types of ML Tasks", file: "1.5-types.html", subtopics: ["Classification", "Regression", "Clustering", "Dimensionality Reduction", "Task Comparison"] },
            { id: "1.6", title: "Empirical Risk Minimization", file: "1.6-applications.html", subtopics: ["The Mathematical Goal", "ERM Formulation", "Variable Descriptions", "Why ERM is the Foundation"] },
            { id: "1.7", title: "Loss Functions & the ML Pipeline", file: "1.7-workflow.html", subtopics: ["Mean Squared Error (MSE)", "Binary Cross-Entropy (BCE)", "Differentiability Requirement", "ML Pipeline Steps", "Data Leakage", "Accuracy Paradox", "Overfitting Prevention"] },
            { id: "1.8", title: "Summary & Interview Preparation", file: "1.8-summary.html", subtopics: ["Key Interview Questions", "Further Reading"] },
            { id: "1.9", title: "Practice Notebook", file: "1.9-notebook.html", subtopics: ["Concept Recap", "Python Code", "Output & Visualizations", "Exercises"] }
          ]
        },
        // ── Module 02 ──
        {
          id: "02-Bias",
          title: "Bias-Variance Dilemma",
          folder: "02-Bias",
          lessons: [
            { id: "2.1", title: "Introduction to Generalization", file: "2.1-introduction.html", subtopics: ["Generalization Concept", "What is Bias?", "What is Variance?", "Learning Curves Overview"] },
            { id: "2.2", title: "What is Bias?", file: "2.2-bias.html", subtopics: ["Intuition of Bias", "The Pre-judging Teacher Analogy", "Underfitting Symptom", "Real-world Examples"] },
            { id: "2.3", title: "What is Variance?", file: "2.3-variance.html", subtopics: ["Intuition of Variance", "Overfitting Symptom", "Fluctuation Across Datasets", "Bias vs Variance Contrast"] },
            { id: "2.4", title: "Underfitting Diagnostics", file: "2.4-underfitting.html", subtopics: ["Diagnostic Indicators", "Learning Curve Analysis", "When More Data Won't Help", "How to Reduce Bias"] },
            { id: "2.5", title: "Overfitting Diagnostics", file: "2.5-overfitting.html", subtopics: ["Train-Test Gap", "Validation Curve Analysis", "Early Stopping Signals", "How to Reduce Variance"] },
            { id: "2.6", title: "Bias-Variance Tradeoff Mathematics", file: "2.6-tradeoff.html", subtopics: ["Expected Error Decomposition", "Bias-Variance Formula", "Mathematical Derivation", "Tradeoff Visualization"] },
            { id: "2.7", title: "Real-world Diagnostic Examples", file: "2.7-examples.html", subtopics: ["Case Studies", "Decision Guide", "Remediation Strategies", "When to Accept Bias"] },
            { id: "2.8", title: "Summary & Interview Preparation", file: "2.8-summary.html", subtopics: ["Key Interview Questions", "Further Reading"] },
            { id: "2.9", title: "Notebook Practice", file: "2.9-notebook.html", subtopics: ["Concept Recap", "Python Code", "Output & Visualizations", "Exercises"] }
          ]
        },
        // ── Module 03 ──
        {
          id: "03-Regularization",
          title: "Regularization",
          folder: "03-Regularization",
          lessons: [
            { id: "3.1", title: "The Overfitting Problem", file: "3.1-introduction.html", subtopics: ["Memorization vs Generalization", "The Memorizing Student Analogy", "What is Regularization?", "Spice Tax Analogy"] },
            { id: "3.2", title: "Why Regularization is Needed", file: "3.2-why-regularization.html", subtopics: ["Cost Function Overview", "Penalty Term Intuition", "Bias-Variance Tradeoff via Regularization"] },
            { id: "3.3", title: "L1 Regularization (Lasso)", file: "3.3-l1-lasso.html", subtopics: ["L1 Penalty Formulation", "Feature Selection Property", "Sparsity Inducement", "Geometric Constraint Space"] },
            { id: "3.4", title: "L2 Regularization (Ridge)", file: "3.4-l2-ridge.html", subtopics: ["L2 Penalty Formulation", "Weight Shrinkage Effect", "When to Use Ridge", "L1 vs L2 Geometric Comparison"] },
            { id: "3.5", title: "Elastic Net", file: "3.5-elastic-net.html", subtopics: ["Combined L1+L2 Penalty", "Correlated Feature Handling", "Lasso vs Ridge vs Elastic Net"] },
            { id: "3.6", title: "Choosing Regularization Strength", file: "3.6-choosing.html", subtopics: ["Lambda Hyperparameter", "Cross-Validation for Lambda", "C Parameter in Scikit-learn", "Feature Scaling Requirement"] },
            { id: "3.7", title: "Summary & Interview Preparation", file: "3.7-summary.html", subtopics: ["Key Interview Questions", "Further Reading"] },
            { id: "3.8", title: "Notebook Practice", file: "3.8-notebook.html", subtopics: ["Concept Recap", "Python Code", "Output & Visualizations", "Exercises"] }
          ]
        },
        // ── Module 04 ──
        {
          id: "04-Feature_Engineering",
          title: "Feature Engineering",
          folder: "04-Feature_Engineering",
          lessons: [
            { id: "4.1", title: "Introduction to Feature Engineering", file: "4.1-introduction.html", subtopics: ["What is Feature Engineering?", "Diamond Cutting Analogy", "Feature Engineering vs Feature Selection", "The Refining Process"] },
            { id: "4.2", title: "Encoding Categorical Variables", file: "4.2-encoding.html", subtopics: ["One-Hot Encoding", "Label Encoding", "Target Encoding", "When to Use Each"] },
            { id: "4.3", title: "Feature Scaling Techniques", file: "4.3-scaling.html", subtopics: ["Why Scaling is Mandatory", "Z-Score Standardization", "Min-Max Normalization", "Worked Numerical Example"] },
            { id: "4.4", title: "Polynomial & Interaction Features", file: "4.4-polynomial.html", subtopics: ["Interaction Features", "Polynomial Features", "Feature Count Formula", "When to Use Polynomial Expansion"] },
            { id: "4.5", title: "Feature Selection Methods", file: "4.5-selection.html", subtopics: ["Filter Methods", "Wrapper Methods", "Embedded Methods", "Visual Cluster Separation"] },
            { id: "4.6", title: "Pipelines & Best Practices", file: "4.6-pipelines.html", subtopics: ["Data Leakage Risk", "Scikit-learn Pipeline", "Chaining Transformers", "Best Practices"] },
            { id: "4.7", title: "Summary & Interview Preparation", file: "4.7-summary.html", subtopics: ["Key Interview Questions", "Further Reading"] },
            { id: "4.8", title: "Notebook Practice", file: "4.8-notebook.html", subtopics: ["Concept Recap", "Python Code", "Output & Visualizations", "Exercises"] }
          ]
        },
        // ── Module 05 ──
        {
          id: "05-Cross_validation_Strategies",
          title: "Cross-Validation Strategies",
          folder: "05-Cross_validation_Strategies",
          lessons: [
            { id: "5.1", title: "Limits of Single Split Evaluation", file: "5.1-introduction.html", subtopics: ["Validation Variance Problem", "The One-Exam Analogy", "What is Cross-Validation?", "Why Average Over Folds"] },
            { id: "5.2", title: "K-Fold Cross-Validation", file: "5.2-kfold.html", subtopics: ["K-Fold Mechanics", "Mathematical Formulation", "Worked Numerical Example", "When to Use K-Fold"] },
            { id: "5.3", title: "Stratified K-Fold", file: "5.3-stratified.html", subtopics: ["Class Proportion Preservation", "Imbalanced Datasets", "Stratified vs Standard K-Fold"] },
            { id: "5.4", title: "LOOCV & Shuffle Split Variations", file: "5.4-variations.html", subtopics: ["Leave-One-Out CV", "Shuffle Split", "Time Series Cross-Validation", "Strategy Comparisons"] },
            { id: "5.5", title: "Data Leakage Pitfalls & Solutions", file: "5.5-leakage.html", subtopics: ["Preprocessing Inside the Loop", "Common Leakage Sources", "CV Workflow Diagram", "Strategy Comparisons"] },
            { id: "5.6", title: "Summary & Interview Preparation", file: "5.6-summary.html", subtopics: ["Key Interview Questions", "Further Reading"] },
            { id: "5.7", title: "Notebook Practice", file: "5.7-notebook.html", subtopics: ["Concept Recap", "Python Code", "Output & Visualizations", "Exercises"] }
          ]
        }
      ]
    },

    // ──────────────────────────────────────────────
    // TRACK 2: Dimensionality Reduction
    // ──────────────────────────────────────────────
    {
      id: "dim-reduction",
      title: "Dimensionality Reduction",
      description: "Dive deep into methods to compress representations, visualize high-dimensional data, and build probabilistic generative models.",
      modules: [
        // ── Module 06 ──
        {
          id: "06-LDA",
          title: "Linear Discriminant Analysis (LDA)",
          folder: "06-LDA",
          lessons: [
            { id: "6.1", title: "Supervised vs Unsupervised DR", file: "6.1-supervised-vs-unsupervised.html", subtopics: ["What is LDA?", "PCA vs LDA Comparison", "Why Labels Matter", "Cancer Screening Example"] },
            { id: "6.2", title: "Fisher's Linear Discriminant", file: "6.2-fishers-criterion.html", subtopics: ["Class Means", "Global Mean", "Separability Objective", "Spot the Difference Analogy"] },
            { id: "6.3", title: "Scatter Matrices (Within & Between)", file: "6.3-scatter-matrices.html", subtopics: ["Within-Class Scatter (Sw)", "Between-Class Scatter (Sb)", "Optimization Objective", "Practical Significance"] },
            { id: "6.4", title: "Mathematical Derivation of LDA", file: "6.4-derivation.html", subtopics: ["Eigenproblem Solution", "Projection Equation", "C-1 Component Limit", "Binary Classification Case"] },
            { id: "6.5", title: "Multi-class LDA & Projection Limits", file: "6.5-multiclass.html", subtopics: ["Multi-class Extension", "Projection Matrix", "Iris Dataset Example", "Visualization"] },
            { id: "6.6", title: "Assumptions & Limitations", file: "6.6-assumptions.html", subtopics: ["Normal Distribution", "Equal Covariance", "Independent Observations", "Linear Separability", "Singularity Problem"] },
            { id: "6.7", title: "Summary & Interview Preparation", file: "6.7-summary.html", subtopics: ["Key Interview Questions", "Further Reading"] },
            { id: "6.8", title: "Notebook Practice", file: "6.8-notebook.html", subtopics: ["Concept Recap", "Python Code", "Output & Visualizations", "Exercises"] }
          ]
        },
        // ── Module 07 ──
        {
          id: "07-PCA",
          title: "Principal Component Analysis (PCA)",
          folder: "07-PCA",
          lessons: [
            { id: "7.1", title: "Intuition of Dimensionality Reduction", file: "7.1-introduction.html", subtopics: ["What is Dimensionality Reduction?", "Why It Is Needed", "PCA Core Concept", "Textbook Summarization Analogy"] },
            { id: "7.2", title: "Variance Maximization Formulation", file: "7.2-variance-vs-reconstruction.html", subtopics: ["Mean Centering", "Covariance Matrix", "Eigen Decomposition", "Explained Variance Ratio"] },
            { id: "7.3", title: "Covariance Matrix & Eigen Decomposition", file: "7.3-covariance-matrix.html", subtopics: ["Mathematical Blueprint", "Worked Numerical Example", "Component Projection", "Symbol Descriptions"] },
            { id: "7.4", title: "Singular Value Decomposition (SVD) Link", file: "7.4-svd-link.html", subtopics: ["SVD Relationship", "Computational Efficiency", "Numerical Stability", "When to Use SVD"] },
            { id: "7.5", title: "Scree Plots & Explained Variance", file: "7.5-scree-plots.html", subtopics: ["Explained Variance Ratio", "Choosing k Components", "Scree Plot Elbow Method", "Cumulative Variance Threshold"] },
            { id: "7.6", title: "Assumptions & Limitations", file: "7.6-limitations.html", subtopics: ["Linearity Assumption", "Variance Equals Information", "Orthogonality Constraint", "Interpretability Trade-off"] },
            { id: "7.7", title: "Summary & Interview Preparation", file: "7.7-summary.html", subtopics: ["Key Interview Questions", "Further Reading"] },
            { id: "7.8", title: "Notebook Practice", file: "7.8-notebook.html", subtopics: ["Concept Recap", "Python Code", "Output & Visualizations", "Exercises"] }
          ]
        },
        // ── Module 08 ──
        {
          id: "08-t-SNE",
          title: "t-Distributed Stochastic Neighbor Embedding (t-SNE)",
          folder: "08-t-SNE",
          lessons: [
            { id: "8.1", title: "Neighbor Embedding Intuition", file: "8.1-introduction.html", subtopics: ["What is t-SNE?", "Why Neighbor Embeddings?", "Classroom Seating Analogy", "Visualization Only Warning"] },
            { id: "8.2", title: "The Crowding Problem", file: "8.2-crowding-problem.html", subtopics: ["Volume Scaling in High-D", "The Studio Apartment Analogy", "Why Gaussian Fails in 2D", "Student t-Distribution Solution"] },
            { id: "8.3", title: "Similarity Distributions", file: "8.3-similarity-distributions.html", subtopics: ["High-D Gaussian Similarity", "Symmetrized Joint Probability", "Low-D Cauchy Similarity", "KL Divergence Cost Function"] },
            { id: "8.4", title: "KL Divergence & Numerical Example", file: "8.4-kl-divergence.html", subtopics: ["KL Divergence Minimization", "Worked Numerical Example", "Local vs Global Preservation", "The Stretching Effect"] },
            { id: "8.5", title: "Hyperparameter Tuning", file: "8.5-hyperparameters.html", subtopics: ["Perplexity Tuning", "Learning Rate", "Early Exaggeration", "Max Iterations", "PCA Pre-processing"] },
            { id: "8.6", title: "Limitations & Trade-offs", file: "8.6-limitations.html", subtopics: ["Stochastic Nature", "Cluster Size Interpretation", "Out-of-Sample Limitation", "When Not to Use t-SNE"] },
            { id: "8.7", title: "Summary & Interview Preparation", file: "8.7-summary.html", subtopics: ["Key Interview Questions", "Further Reading"] },
            { id: "8.8", title: "Notebook Practice", file: "8.8-notebook.html", subtopics: ["Concept Recap", "Python Code", "Output & Visualizations", "Exercises"] }
          ]
        },
        // ── Module 09 ──
        {
          id: "09-UMAP",
          title: "Uniform Manifold Approximation & Projection (UMAP)",
          folder: "09-UMAP",
          lessons: [
            { id: "9.1", title: "Manifold Learning Intuition", file: "9.1-manifolds.html", subtopics: ["What is UMAP?", "The Sphere Analogy", "Why Another DR Technique?", "The Vine Map Analogy"] },
            { id: "9.2", title: "Fuzzy Similarity in High-D", file: "9.2-fuzzy-similarities.html", subtopics: ["Asymmetric Fuzzy Similarity", "Local Offset Rho", "Adaptive Bandwidth Sigma", "NN-Descent Speedup"] },
            { id: "9.3", title: "Symmetrization & Low-D Similarity", file: "9.3-symmetrization.html", subtopics: ["Fuzzy Set Union", "Low-D Student-t Approximation", "Cross-Entropy Attraction-Repulsion"] },
            { id: "9.4", title: "Cross-Entropy Loss", file: "9.4-cross-entropy.html", subtopics: ["Attraction vs Repulsion Forces", "Why Cross-Entropy Beats KL", "Worked Numerical Example", "Interpretation of Results"] },
            { id: "9.5", title: "Layout Optimization & SGD", file: "9.5-layout-optimization.html", subtopics: ["Spectral Initialization", "Force-Directed Layout", "Negative Sampling", "Algorithm Workflow"] },
            { id: "9.6", title: "Hyperparameters: Neighbors & Min-dist", file: "9.6-hyperparameters.html", subtopics: ["n_neighbors Tuning", "min_dist Tuning", "Local vs Global Balance", "Practical Guidelines"] },
            { id: "9.7", title: "PCA vs t-SNE vs UMAP", file: "9.7-comparisons.html", subtopics: ["Speed & Scalability", "Local vs Global Preservation", "When to Use Each", "Comparison Matrix"] },
            { id: "9.8", title: "Summary & Interview Preparation", file: "9.8-summary.html", subtopics: ["Key Interview Questions", "Further Reading"] },
            { id: "9.9", title: "Notebook Practice", file: "9.9-notebook.html", subtopics: ["Concept Recap", "Python Code", "Output & Visualizations", "Exercises"] }
          ]
        },
        // ── Module 10 ──
        {
          id: "10-Autoencoders-VAE",
          title: "Autoencoders & VAEs",
          folder: "10-Autoencoders-VAE",
          lessons: [
            { id: "10.1", title: "Undercomplete Autoencoders", file: "10.1-undercomplete.html", subtopics: ["Why Dimensionality Reduction in DL?", "Encoder-Bottleneck-Decoder Architecture", "Linear vs Non-Linear Compression", "Story Summarization Analogy"] },
            { id: "10.2", title: "PCA Relationship", file: "10.2-pca-relationship.html", subtopics: ["Linear Autoencoder = PCA", "Weight Matrix as PCA Directions", "Reconstruction Perspective"] },
            { id: "10.3", title: "Autoencoder Math & Loss Functions", file: "10.3-math-loss.html", subtopics: ["Encoder Equation", "Decoder Equation", "Reconstruction Loss (MSE)", "Worked Forward Pass"] },
            { id: "10.4", title: "Limitations of Traditional Autoencoders", file: "10.4-limitations.html", subtopics: ["Fragmented Latent Space", "No Generative Capability", "Disorganized Embeddings", "Posterior Collapse Risk"] },
            { id: "10.5", title: "Variational Autoencoders (VAEs)", file: "10.5-probabilistic-latents.html", subtopics: ["Why VAEs Were Introduced", "Probabilistic Encoding", "Latent Space Continuity", "Facial Features Analogy"] },
            { id: "10.6", title: "ELBO Loss & KL Divergence", file: "10.6-elbo-kl-loss.html", subtopics: ["VAE Loss Function", "Reconstruction + KL Divergence", "Worked Numerical Example", "Beta Hyperparameter"] },
            { id: "10.7", title: "The Reparameterization Trick", file: "10.7-reparameterization.html", subtopics: ["Stochastic Sampling Problem", "Reparameterization Solution", "Backpropagation Through Noise", "Graph Flow Diagram"] },
            { id: "10.8", title: "Autoencoders vs VAE vs PCA", file: "10.8-comparisons.html", subtopics: ["Comparison Matrix", "When to Use Each", "Anomaly Detection Application", "Practical Trade-offs"] },
            { id: "10.9", title: "Summary & Interview Preparation", file: "10.9-summary.html", subtopics: ["Key Interview Questions", "Further Reading"] },
            { id: "10.10", title: "Notebook Practice", file: "10.10-notebook.html", subtopics: ["Concept Recap", "Python Code", "Output & Visualizations", "Exercises"] }
          ]
        }
      ]
    }
  ]
};

// ──────────────────────────────────────────────
// Helper: Get flat lesson list by module ID
// ──────────────────────────────────────────────
LMS.getLessonByModuleAndId = function(moduleId, lessonId) {
  for (const track of LMS.curriculum.tracks) {
    for (const mod of track.modules) {
      if (mod.id === moduleId) {
        return mod.lessons.find(l => l.id === lessonId);
      }
    }
  }
  return null;
};

// ──────────────────────────────────────────────
// Helper: Resolve full path to lesson file
// ──────────────────────────────────────────────
LMS.resolveLessonPath = function(moduleId, lessonFile) {
  const folder = LMS.getModuleFolder(moduleId);
  return `track-${folder}/` + folder + "/" + lessonFile;
};

// ──────────────────────────────────────────────
// Helper: Get module folder name
// ──────────────────────────────────────────────
LMS.getModuleFolder = function(moduleId) {
  for (const track of LMS.curriculum.tracks) {
    for (const mod of track.modules) {
      if (mod.id === moduleId) return mod.folder;
    }
  }
  return null;
};

// Theme Management
const THEME_KEY = "gradients_lms_theme";

LMS.theme = {
  current: localStorage.getItem(THEME_KEY) || "dark",

  init() {
    this.apply(this.current);
  },

  apply(theme) {
    document.documentElement.setAttribute("data-theme", theme);
    localStorage.setItem(THEME_KEY, theme);
    this.current = theme;

    const toggles = document.querySelectorAll(".theme-toggle-btn");
    toggles.forEach(btn => {
      btn.innerHTML = theme === "dark" ? "&#9728;" : "&#9790;";
    });
  },

  toggle() {
    const nextTheme = this.current === "dark" ? "light" : "dark";
    this.apply(nextTheme);
  }
};

// Document Load triggers
document.addEventListener("DOMContentLoaded", () => {
  LMS.theme.init();
});
