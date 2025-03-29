// config.js
window.portfolioConfig = {
  profileImage: "images/profile.png",
  videos: [
    {
      title: "Let’s Visualize Vectorization like never before - Understand the Importance of Math in ML with Numpy Broadcasting",
      youtubeId: "QnnZyEeUdhE"
    },
    {
      title: "Let’s Visualize Optimal Floyd’s Cycle Detection Algorithm with a String",
      youtubeId: "C55wl2wcb9A"
    },
    {
      title: "Python Compilerrrr???",
      youtubeId: "vvkp3ak7uCI"
    }
  ],
  projects: [
      {
          title: "NSFW Classifier - Image Content Moderation System",
          category: "Deep Learning",
          
          description: "### Goal 🎯\nDetect and remove explicit image content on platforms like Instagram/Facebook.\n\n### Performance 📊\nAchieved **96% Accuracy & 0.92 F1 Score**.\n\n### Training Strategy 🧠\nImplemented **Incremental (2-Phase) Training** from scratch on **130,000 images** using 16GB Nvidia P100 GPUs.\n- **Phase 1:** 15 epochs with a frozen EfficientNetV2-M extractor (LR 10⁻⁴).\n- **Phase 2:** 15 epochs unfreezing Block-7 (frozen BatchNorm), lower LR (2x10⁻⁵ -> 10⁻⁶ via ReduceLROnPlateau).\n\n### Techniques ✨\nUtilized data augmentation, EarlyStopping, ModelCheckpoint, and ReduceLROnPlateau for robust training.",
          image: "images/nsfw-classifier.png",
          tags: ["TensorFlow", "Incremental Training", "Transfer Learning", "ReduceLRonPlateau", "EfficientNetV2-M", "Distributed GPU Training", "Computer Vision", "Streamlit", "FusedMB-CNN"],
          live: "https://nsfw-content-detector.streamlit.app/",
      },
      {
        title: "AI-Powered Financial Dispute Resolution System",
        category: "Scalable API Development",
        // Description already updated correctly
        description: "### Backend Robustness 🔒\n- Created a **robust, scalable, and modular** FastAPI backend, ensuring **maintainability & scalability** with proper **routes, services, pydantic schema handling, database interactions,** and **unit tests**.\n\n### AI-Powered Analysis 🧠\n- When a new dispute is inserted via API, an **AI Model (LLM)** automatically analyzes it and provides structured insights, including:\n  - **priority_level, priority_reason, insights, follow-up_questions, probable_solutions, possible_reasons, risk_score, risk_factors**\n- These insights help prioritize disputes, ensuring **higher-priority cases** are reviewed first while assisting tech support with **AI-generated suggestions & follow-ups**.\n\n### Frontend Capabilities 🖥️\n- Developed an **interactive Streamlit front-end**(70% Complete) to **demonstrate workflow with GUI**.\n\n### Deployment 🚀\n- **Dockerized** the project using a **single container** that runs both the **front-end** (port 5670) and **backend** (port 8000) via **sub-processes**.\n- This enables the same **Docker container** to be **deployed anywhere** for showcasing the **full prototype**.",
        image: "images/financial-dispute.png",
        tags: ["FastAPI", "Docker", "Streamlit", "LLM", "Synthetic Data Generation", "Unit Testing", "Pydantic", "PostgreSQL"],
        live:"https://huggingface.co/spaces/DebopamC/AI_Financial_Dispute_Automation",
        github: "https://github.com/DebopamParam/AI-Powered_Dispute_Resolution",
      },
      {
          title: "IBM EMPLOYEE ATTRITION PREDICTOR",
          category: "Deep Learning",
          
          description: "### Objective 🎯\nPredict employee attrition with **85% AUC** to enhance retention.\n\n### Modeling 🛠️\nDeveloped & **Hyperparameter Optimized** Multi-Layer Perceptron (MLP), XGBoost, and Logistic Regression models, including both training and inference pipelines.\n\n### Backend ⚙️\nBuilt a **FastAPI backend** for real-time predictions, using **Pydantic** for robust schema validation.\n\n### Deployment ☁️\n**Containerized with Docker**, deployed on **AWS EC2**, and managed via **AWS ECR**.\n\n### CI/CD 🔄\nImplemented an automated **CI/CD pipeline** using **GitHub Actions** for seamless updates.\n\n### Frontend 💻\nCreated a user-friendly **Flutter Web** interface for interaction.\n\n### Security 🛡️\nSecured HTTPS traffic using **Caddy** as a reverse proxy server.",
          image: "images/employee-attrition.png",
          tags: ["TensorFlow", "AWS", "EC2", "ECR", "Docker", "FastAPI", "CI/CD Pipeline", "Multi-Layer Perceptron", "XGBoost", "Logistic Regression", "Hyperparameter-Tuning", "GitHub Actions", "Pydantic", "Flutter Web", "Reverse-Proxy-Server: Caddy"],
          live: "https://www.debopamchowdhury.works/",
          video: "https://www.linkedin.com/posts/debopam-chowdhury-param-600619229_machinelearning-deeplearning-aws-activity-7244476917884608512-DfbD"
      },
      {
          title: "Rapid Prototype Banking API",
          category: "Scalable API Development",
          description: "### Core Infrastructure 🏗️\n- Implemented a **Docker-based** development environment using **PostgreSQL, Redis, Kafka,** and **Zookeeper**.\n- Designed an **efficient database schema** and set up **Redis locking** for concurrent transactions.\n- Configured **Kafka event streaming** for transaction processing and notifications.\n\n### API Implementation ⚙️\n- Built **RESTful endpoints** for account creation, balance inquiry, and transaction processing using **FastAPI**.\n- Implemented **comprehensive validation**, **atomic transaction handling**, and **credit/debit operations** with balance checks.\n\n### Deployment Strategy 🚀\n- Utilized **Docker Compose** for container orchestration, enabling easy setup anywhere.",
          image: "images/banking-api.png",
          tags: ["FastAPI", "PostgreSQL", "Redis", "Kafka", "Docker", "Docker-Compose", "RESTful API", "Event Streaming", "Concurrency Control", "Backend Development"],
          github: "https://github.com/DebopamParam/Scalable-Banking-API_Level-1",
      },
      {
          title: "Scalable Deep Learning Based Recommendation System",
          category: "Deep Learning",
          
          description: "### Scalability 📈\nDesigned to handle **25 Million+** candidates efficiently.\n\n### Architecture 🏗️\nImplemented a **Hybrid Model**:\n- **Candidate Generation:** Filters top 100 candidates from millions using efficient vector space search (Faiss).\n- **Re-ranking Model:** Custom **4-Tower Deep Learning Model** (trained from scratch) ranks the filtered candidates to display the top 30, identifying user patterns, choices, ratings, and genres.\n\n### Cold-Start Resistance 🧊\nArchitecture designed to mitigate the cold-start problem.\n\n### Training 🔥\nUtilized **Distributed GPU Training** (e.g., 2x Nvidia T4) with TensorFlow.\n\n### Features ✨\nIncludes accessible data visualizations and statistics.",
          image: "images/recommendation-system.png",
          tags: ["TensorFlow", "Distributed GPU Training", "Faiss", "Vector DB", "Recommendation System", "Deep Learning", "Langchain", "Streamlit"],
          live: "https://debopam-movie-recommendation-system.streamlit.app/",
      },
      {
          title: "Non-Sequential Breast Cancer Classification System",
          category: "Deep Learning",
          
          description: "### Multi-Modal & Multi-Output 🧬\nDeveloped a novel deep learning model predicting cancer presence, invasiveness, and difficult-negative status.\n\n### Architecture 🏗️\nLeverages a **non-sequential architecture** to process both **mammogram images** and **tabular clinical data**.\n\n### Feature Extraction ✨\nUtilized a pre-trained **EfficientNetV2B3**, **fine-tuning** layers from Block 6 onwards for task-specific adaptation.\n\n### Training 🔥\nAccelerated training using **TensorFlow's MirroredStrategy** for distributed training on 2xT4 GPUs (9 hours on Kaggle).\n\n### Publication 📜\nWork published in IRJET (International Research Journal of Engineering and Technology).",
          image: "images/cancer-classification.png",
          tags: ["TensorFlow", "Transfer Learning", "Fine-Tuning", "EfficientNetV2", "FusedMB-CNN", "Distributed Training", "Computer Vision", "Streamlit"],
          live: "https://debopamparam-bcd-inference-3vqotx.streamlit.app/",
          paper: "https://www.irjet.net/archives/V12/i2/IRJET-V12I211.pdf"
      },
      {
          title: "LLM-Finetuning (Local SQL Agent)",
          category: "NLP & LLM",
          
          description: "### Model 🧠\n**Supervised Finetuned** Qwen2.5-3B-Coder-Instruct on high-quality SQL data using **Unsloth** and **QLoRA** (Quantized Low-Rank Adapters).\n\n### Efficiency ⚡\n**Quantized** the model from BF16 to **int4 (q4_k_m)** for reduced size and lightweight inference.\n\n### Inference 💡\nCreated local inference options using both **Ollama** and **LlamaCpp**.\n\n### Agentic Workflow 🤖\nDeveloped a **Langchain SQL agent** that decides whether to execute the generated SQL query or simply answer the user's question based on context.\n\n### Features ✨\nIntegrated group CSV upload, schema extractor, and an Automatic/Manual SQL executor using **DuckDB**.\n\n### Deployment 🚀\nDeployed serverlessly on **Hugging Face Spaces** using **Docker**.",
          image: "images/llm-finetuning.png",
          tags: ["Supervised Finetuning", "QLoRA", "Unsloth", "Quantization", "LlamaCPP", "Ollama", "Docker", "DuckDB", "Langchain", "HuggingFace Spaces", "Agentic Workflow"],
          live: "https://huggingface.co/spaces/DebopamC/Natual_Language-to-SQL-Qwen2.5-3B-FineTuned",
          ollama: "https://ollama.com/debopam/Text-to-SQL__Qwen2.5-Coder-3B-FineTuned"
      },
      {
          title: "TurboML Chatbot with Grounding",
          category: "Generative AI",
          
          description: "### Functionality 💬\nCreated a **Chat Agent** specifically for the **TurboML library**.\n\n### Grounding 🔗\nAchieved reliable and accurate responses by **scraping TurboML's official documentation** and **PyPI package information**.\n\n### Technique 🛠️\nImplemented a **Retrieval-Augmented Generation (RAG)** pipeline using **LangChain** to ground the chatbot's answers in the scraped context.",
          image: "images/turboml-chat.png",
          tags: ["RAG", "LangChain", "Vector DB", "Embeddings", "Streamlit", "Web Scraping", "Chatbot"],
          live: "https://turboml-chat.streamlit.app/",
      },

  ],
  socialMedia: {
    linkedin: "https://www.linkedin.com/in/debopam-chowdhury-param-600619229/",
    github: "https://github.com/DebopamParam",
    youtube: "https://www.youtube.com/@DCparam/videos",
    email: "debopamwork@gmail.com"
  }
};