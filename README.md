# 👋 Hi there, I'm Debopam Chowdhury

## 🚀 About Me

Machine Learning & Software Engineer with expertise in ML/DL, MLOps, Mathematics, RAG, and Flutter. I'm passionate about building deployable AI-based solutions and have a strong foundation in Machine Learning, Computer Science, and Mathematics.

- 🌱 Currently working on LLM Pre-Training from scratch using PyTorch & Math
- 💼 Recently completed a contract as a Generative AI Engineer
- 🔭 Solved 100+ problems on LeetCode
- 🎓 BE in Information Science from Acharya Institute of Technology (2021-2025)
- 📫 Reach me at: debopamwork@gmail.com

## 🛠️ Skills

| Category | Technologies & Skills |
|----------|------------------------|
| **Programming Languages & Tools** | Python, Java, Dart, C++, HTML5, CSS3, JavaScript, Git |
| **Machine Learning & Deep Learning** | TensorFlow, scikit-learn, Keras, PyTorch, Hugging Face, Neural Networks, LSTMs, CNNs, Transformers, LLM Finetuning, Hyperparameter Optimization, LoRA/QLoRA, Clustering Algorithms, Decision Trees, Bagging & Boosting, Anomaly Detection |
| **Data Handling & Analysis** | Pandas, NumPy, SQL, NoSQL, Data Manipulation, Data Preparation |
| **Cloud & DevOps** | Docker, AWS, Hugging Face Spaces, FastAPI, CI/CD Pipeline |
| **Mathematics** | Linear Algebra, Probability, Statistics, Boosting Methods |
| **Generative AI & RAG** | Vector Embeddings, Indexing, Chunking, RAG Pipelines, LlamaIndex, LangChain, Colpali, Byaldi, Vector Databases, LangGraph, CrewAI |
| **Mobile & Web Development** | Flutter, Firebase, Riverpod, FastAPI, Google OAuth |

## 💼 Experience

### 1. GENERATIVE AI ENGINEER (Contract - Remote)
> **Private Client, Sydney, Australia** | October-November 2024

- Developed secure, on-premise solutions for complex PDF with images and charts Q&A and knowledge retrieval
- Built data ingestion pipeline for 1000's of documents with automatic task scheduler
- Implemented multimodal RAG pipelines (Byaldi, Colqwen2, Pixtral 12B) optimized for diverse document types (70% accuracy improvement)
- Containerized the application with Docker for deployment flexibility

**Technologies:** NLP, Vision Embeddings, Local Multimodal RAG, LangChain, Pixtral 12B, Col-Qwen2, Byaldi

> [Small open-source contribution - Byaldi - 575 ✰](https://github.com/AnswerDotAI/byaldi/pull/50)

### 2. AI GYM BUDDY (Langchain | Flutter | Riverpod | Gemini)
> Solo Creator: Design - Code - Deploy - Marketing ---- Deployed✅

Personalized AI-driven workout app with smart equipment detection and progress tracking.

- 650+ registered users
- AI instrument detection (camera or gallery)
- Personalized workout routines based on available equipment
- Dynamic video tutorial finder
- Google OAuth integration

**Technologies:** Dart, Flutter, Firebase, Gemini 2.0, Riverpod, LangChain, FastAPI, Google OAuth, Deep Learning

📱 [Google Play Store](https://play.google.com/store/apps/details?id=com.aigymbuddy.me&hl=en) | 🌍 [Website](http://www.aigymbuddy.in/) | 🎥 [1-Min Demo Video](https://youtube.com/shorts/0ZR0IWiZJQE?si=IsL4hi5JQe9FYAgk)

### 3. FLUTTER DEVELOPER (Contract)
> **Focus-flow, Remote** | Nov-Dec 2024 

### 4. FLUTTER DEVELOPER INTERN
> **UNFILTR, INC, Bengaluru, India** | Jan-June 2023

# 🔥 Projects

## Deep Learning

### 1. NSFW Classifier - Image Content Moderation System
*High-accuracy NSFW content detection system for social media platforms* ___Deployed✅

- 96% accuracy & 0.92 F1 score
- Incrementally trained on 130,000 sample images
- Two-phase training with EfficientNetV2-M

**Technologies:** TensorFlow, Incremental Training, Transfer Learning, ReduceLRonPlateau

[🌍 Live Webapp + Architecture + Training Code + Training Data](https://nsfw-content-detector.streamlit.app/)

---

#### 2. IBM EMPLOYEE ATTRITION PREDICTOR
*End-to-end ML application predicting employee attrition with 85% AUC* ____Deployed✅

- Hyperparameter optimized models (MLP, XGBoost, Logistic Regression)
- FastAPI backend with Pydantic schema validation
- Containerized with Docker and deployed on AWS EC2
- CI/CD pipeline with GitHub Actions

**Technologies:** TensorFlow, AWS, Docker, FastAPI, CI/CD Pipeline, Multi-Layer Perceptron Neural Network, XGBoost, Logistic Regression, Hyperparameter Tuned Models, GitHub Actions, Pydantic, Flutter Web, Reverse-Proxy-Server: Caddy

[🎥Explanation Video](https://www.linkedin.com/posts/debopam-chowdhury-param-600619229_machinelearning-deeplearning-aws-activity-7244476917884608512-DfbD?utm_source=share&utm_medium=member_desktop) | [🌍Live Webapp + Architecture + Training Code + Training Data](https://www.debopamchowdhury.works/)

#### 3. Scalable Deep Learning Based Recommendation System
*Scalable recommendation system capable of handling 25M+ candidates* ____Deployed✅

- Hybrid architecture with candidate generation and re-ranking
- Custom 4-tower deep learning model trained from scratch using Tensorflow, using Nvidia's 2xT4 GPUs
- Resistant to cold-start problem

**Technologies:** TensorFlow, Faiss, Vector DB, Distributed GPU Training, Langchain, BGE, Streamlit

[🌍 Live Webapp + Architecture + Training Code + Training Data](https://debopam-movie-recommendation-system.streamlit.app/)

---

### 4. Non-Sequential Breast Cancer Classification System
*Multi-output deep learning model for breast cancer detection* ____Deployed✅

- Published in IRJET
- Processes both mammogram images and tabular clinical data
- Fine-tuned EfficientNetV2B3 for feature extraction
- Distributed training with TensorFlow's MirroredStrategy in Nvidia 2xT4-GPUs

**Technologies:** TensorFlow, Transfer Learning, EfficientNetV2, FusedMB-CNN

[📃IRJET Published Paper](https://www.irjet.net/archives/V12/i2/IRJET-V12I211.pdf) | [🌍 Live Webapp + Architecture + Training Code + Evaluation Metrics](https://debopamparam-bcd-inference-3vqotx.streamlit.app/)

---

## NLP & LLM

### 5. LLM-Finetuning (Local SQL Agent by Finetuning SLM)
*SQL agent created by finetuning Qwen2.5-3B-Coder-Instruct model* ____Deployed✅

- Supervised finetuning with QLora, with High Quality SQL synthetic data, generated from ChatGpt-4o.
- Quantized from BF16 to int4(q4_k_m) for lightweight inference
- Integrated with Ollama and LlamaCpp

**Technologies:** Supervised Finetuning, Unsloth, LlamaCPP, Docker, DuckDB, Langchain, Huggingface Spaces

[🌍 Live Webapp + Architecture + Training Code + Training Data](https://huggingface.co/spaces/DebopamC/Natual_Language-to-SQL-Qwen2.5-3B-FineTuned) | [🖥️ Run Locally Via Ollama](https://ollama.com/debopam/Text-to-SQL__Qwen2.5-Coder-3B-FineTuned)

---

### 6. LLM - Continued Pre-Training (CPT + SFT)
> *Ongoing (80% Done)*

Teaching Qwen2.5-0.5B to learn Bengali language through CPT and adapting it to English-to-Bengali translation using SFT, all under 400MB memory for edge device translation tasks.

**Technologies:** Unsloth, CPT, SFT, Edge Deployment, Language Model Optimization

---

### 7. LLM-Pre-Training from Scratch - using Pytorch & Math
> *Upcoming* -- Currently studying all Mathematical Concepts -- Plan to finish this project within 7 April, 2025.

Building a small LLama-style foundational model from scratch using PyTorch, Flash-Attention, and mathematics for less than $50 using Runpod-GPUs.

**Technologies:** PyTorch, Flash-Attention, LLM Architecture, Runpod-GPUs

---

## Synthetic Data Generation

### 8. TurboML knowledge distillation Synthetic Data Generation
*1.3K Synthetic SFT dataset made without using any 3rd party Library* ____Deployed✅

- Q&A dataset about TurboML with 1,343 technical questions and detailed answers
- Covers implementation, troubleshooting, architecture design, and performance optimization

[🌍 Hugging Face Datasets](https://huggingface.co/datasets/DebopamC/TurboML_Synthetic_QnA_Dataset)

---

## Generative AI

### 9. TurboML Chatbot with Grounding
*Made TurboML Chat Agent by scraping all the Docs + Pypi package with Grounding Links* ____Deployed✅

[🌍 Check it out live](https://turboml-chat.streamlit.app/)

---

### Mini Projects

1. Efficient Parallel Implementation of Forward Prop, K-means using Math & Numpy Broadcasting
2. Machine translation using Encoder-Decoder Architecture using Bi-directional LSTMs and Bahdanau Attention
3. Math behind Gating mechanism of a LSTM & GRU cell
4. Learning LangGraph & CrewAi
5. Real-time Data-Center Anomaly Detection in Streaming Data with TurboML (HST + AdaBoost)

### Hackathon Project

#### Image Entity Extraction with Qwen2 VL: Large-Scale Inference
*Amazon ML Challenge Hackathon Competition - Ranked 172 out of ~75,000 participants*

- Developed large-scale image-to-text inference pipeline using Qwen2 VL: 2B
- Incorporated image preprocessing, Regex, and parallel processing
- F1-Score of 0.47

[Click Here to see the code](https://colab.research.google.com/drive/1V5F1XMlYNHzv-hA9xmIJ-Jx5vuD0fKAR?usp=sharing)

## 🎓 Education

| Degree/Certificate | Institution | Year | Result |
|--------------------|-------------|------|--------|
| **BE in Information Science** | Acharya Institute of Technology, Bangalore | 2021-2025 | CGPA: 8.32 |
| **Higher Secondary Education** | Kalyani Public School, Barasat, Kolkata | 2019-2021 | 90% |
| **Secondary Education** | Sacred Heart Day High School, Kolkata | 2019 | 77% |

## 🏆 Achievements

- Ranked 172 out of 75,000 participants in the Amazon ML Challenge Hackathon 2024
- 2nd place out of 60 in the TechnioD Hackathon
- Finalist in IIT Bombay's Mood Indigo Bengaluru Event
- Open-source contribution to Byaldi - 575 ✰: [Fix langchain integration not present in pypi tar & whl-- pyproject.toml](https://github.com/AnswerDotAI/byaldi/pull/50)

## 📺 YouTube

- [Understand the Importance of Math in ML with Numpy Broadcasting](https://youtu.be/QnnZyEeUdhE?si=6GuAa8hj5e7YvfaG)
- [Let's Visualize Optimal Floyd's Cycle Detection Algorithm with a String](https://youtu.be/C55wl2wcb9A?si=pV8Wlhbtf5W_j3un)
- [Python Compilerrrr???](https://youtu.be/vvkp3ak7uCI?si=xsRcTANstVBM6HcY)

## 📫 Contact Me

- Email: debopamwork@gmail.com
- LinkedIn: [Connect with me](https://www.linkedin.com/in/debopam-chowdhury-param-600619229/)

