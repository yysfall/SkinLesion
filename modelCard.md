# Skin Lesion Classification with Explainability

## Model Overview
- **CNN:** MobileNetV3 trained to classify skin lesion images as **benign** or **malignant**.  
- **Explainability:** Grad-CAM visualizations highlight areas of the lesion influencing the prediction.  
- **NLP Component:** TF-IDF + Logistic Regression analyzes lesion descriptions to provide additional risk context.  
- **RL Component:** Q-learning agent dynamically adjusts classification thresholds to optimize sensitivity while controlling false positives.  
- **Integration:** CNN and NLP outputs are combined in a fusion layer; RL tunes the threshold for final prediction.

---

## Intended Use
- **Primary Use:** Educational and research purposes in AI-assisted dermatology studies.  
- **Users:** Students, researchers, and developers exploring AI pipelines and explainability in medical imaging.  
- **Important Note:** **Not for clinical diagnosis or treatment decisions.** All outputs are informational only.  

---

## Metrics
- **ROC-AUC:** Measures overall discriminative ability of the CNN and baseline models.  
- **Accuracy:** Percentage of correct predictions (secondary reference).  
- **Sensitivity at fixed specificity:** Evaluates performance for high-risk (malignant) detection.  
- **RL Reward:** Combines sensitivity and false positive rate for threshold tuning evaluation.  

**Week 2 placeholder results:**  
WALAPA

**Learning Curves / Grad-CAM Visuals:**  
WALAPA

---

## Limitations
- **Dataset size & diversity:** Limited images may not cover all skin types, ages, or rare lesion types.  
- **Bias:** Underperformance possible for underrepresented skin tones due to dataset imbalance.  
- **Explainability:** Grad-CAM provides qualitative insight but does not guarantee clinically interpretable reasoning.  
- **NLP Support:** Simple TF-IDF may not capture nuanced clinical language in descriptions.  
- **RL Thresholding:** Q-learning agent is a prototype and may not generalize to real-world deployment.  

