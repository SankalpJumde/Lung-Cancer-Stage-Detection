
# 🫁 Lung Cancer Stage Detection 🔍🧠  
**"Detect. Classify. Save Lives."**

---

## 🧭 Introduction

Lung cancer remains one of the deadliest cancers worldwide — but **early, accurate detection can change everything**. This project is a deep learning-powered approach to **classifying lung cancer stages using CT scan images**, making diagnosis faster, smarter, and potentially life-saving.

Whether for research, diagnostics, or real-world deployment, this system is designed with **impact** in mind.

---

## 💡 What This Project Does

- 📤 **Uploads CT scan images** of the patient
- 🧠 **Processes and analyzes the image** via a trained CNN model
- 🏷️ **Predicts the cancer stage**, classified into:
  - **Stage I** – Normal case  
  - **Stage II** – Benign case  
  - **Stage III** – Indolent case  
  - **Stage IV** – Malignant case
- 👤 **Displays patient name, image, and result** on a dynamic result tab

---

## 🧬 Tech Stack

| Layer        | Tech Used                |
|--------------|---------------------------|
| ML Model     | TensorFlow / Keras (CNN) |
| Image Handling | OpenCV, NumPy           |
| Web App      | Flask (Python)           |
| Frontend     | HTML/CSS + Bootstrap     |

---

## 🗂️ Dataset

Used a categorized dataset of lung CT scans with labeled stages.

📦 *[Click Here](https://www.kaggle.com/datasets/adityamahimkar/iqothnccd-lung-cancer-dataset)*

---

## 📷 Sample UI Output

![image](https://github.com/user-attachments/assets/30727aff-a9b8-45e0-a253-6b1e53ab01bc)

---

## 🚀 Getting Started

### 1️⃣ Clone this repo:
```bash
git clone https://github.com/yourusername/lung-cancer-stage-detection.git
cd lung-cancer-stage-detection
```

### 2️⃣ Install the dependencies:
```bash
pip install -r requirements.txt
```

### 3️⃣ Launch the web app:
```bash
python app.py
```

### 4️⃣ Use the interface:
- Upload a CT scan image  
- Enter the patient’s name  
- Click "Submit" to see results

---

## 🌍 Real-World Use Cases

- 🏥 **Hospitals & Clinics**: Decision support for radiologists  
- 💻 **Remote Diagnosis**: For patients in low-resource areas  
- 📊 **Research Labs**: Exploring diagnostic model performance  
- 📱 **Future App Integrations**: Instant mobile analysis for healthcare pros  

---

## 🔮 Future Scope & Vision

### 🔗 Web & Portal Enhancements
- Multi-user login (Doctors, Labs, Patients)  
- Downloadable PDF reports with signatures  

### 📱 Mobile App Integration
- Scan image directly via camera  
- Instant feedback & stage prediction

### 🔁 Model Expansion
- Multi-type cancer classification  
- Support for 3D CT slices (DICOM)

### 📈 Smart Insights & Dashboards
- Live stats on detection patterns  
- Alerts for high-risk uploads  
- Case review analytics for health institutions

---

## 📚 References

1. [Artificial Intelligence in lung cancer screening: Detection, classification, prediction, and prognosis](https://www.researchgate.net/publication/379642910_Artificial_intelligence_in_lung_cancer_screening_Detection_classification_prediction_and_prognosis)  
2. [Deep learning for lungs cancer detection: a review](https://link.springer.com/article/10.1007/s10462-024-10807-1#:~:text=2020)
3. [Comparing CNN-based and transformer-based models for identifying lung cancer: which is more effective?](https://www.researchgate.net/publication/376681092_Comparing_CNN-based_and_transformer-based_models_for_identifying_lung_cancer_which_is_more_effective)  
4. [Advancements in Early Detection of Lung Cancer in Public Health: A Comprehensive Study Utilizing Machine Learning Algorithms and Predictive Models](https://www.researchgate.net/publication/377559452_Advancements_in_Early_Detection_of_Lung_Cancer_in_Public_Health_A_Comprehensive_Study_Utilizing_Machine_Learning_Algorithms_and_Predictive_Models)
---

## 🤝 Acknowledgements

To the warriors fighting cancer, the researchers building tools, and the developers writing code with purpose — this one’s for you.  
Special thanks to open-source communities, medical researchers, and everyone helping tech heal lives.

---

## 🧑‍💻 Developed By

**Sankalp Jumde**  
🎓 B.Tech AI, Class of 2026  
🔗 [[LikendIn](https://www.linkedin.com/in/sankalp-jumde/)] | [[GitHub](https://github.com/SankalpJumde)] | [[Mail](sankalpkrishna1103@gmail.com)]

---

## 📄 License

This project is licensed under the **MIT License** — because knowledge should be open, especially when lives are at stake.
