// config.js
window.portfolioConfig = {
    profileImage: "images/profile.png",
    videos: [
      {
        title: "Let’s Visualize Vectorization like never before",
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
            description: "High-accuracy NSFW content detection system for social media platforms with 96% accuracy & 0.92 F1 score. Incrementally trained on 130,000 sample images with two-phase training using EfficientNetV2-M.",
            image: "images/nsfw-classifier.png",
            tags: ["TensorFlow", "Distributed GPU Training", "FusedMB-CNN", "Incremental Training", "Transfer Learning", "ReduceLRonPlateau", "Streamlit"],
            live: "https://nsfw-content-detector.streamlit.app/",
        },
        {
            title: "IBM EMPLOYEE ATTRITION PREDICTOR",
            category: "Deep Learning",
            description: "End-to-end ML application predicting employee attrition with 85% AUC. Hyperparameter optimized models (MLP, XGBoost, Logistic Regression). FastAPI backend with Pydantic schema validation. Containerized with Docker and deployed on AWS EC2.",
            image: "images/employee-attrition.png",
            tags: ["TensorFlow", "AWS", "Docker", "FastAPI", "CI/CD Pipeline", "ANN", "XGBoost", "Hyperparameter-Tuning", "GitHub Actions", "Pydantic", "Flutter Web", "Reverse-Proxy-Server: Caddy"],
            live: "https://www.debopamchowdhury.works/",
            video: "https://www.linkedin.com/posts/debopam-chowdhury-param-600619229_machinelearning-deeplearning-aws-activity-7244476917884608512-DfbD"
        },
        {
            title: "Scalable Deep Learning Based Recommendation System",
            category: "Deep Learning",
            description: "Scalable recommendation system capable of handling 25M+ candidates. Hybrid architecture with candidate generation and re-ranking. Custom 4-tower deep learning model trained from scratch using Tensorflow with Nvidia's 2xT4 GPUs.",
            image: "images/recommendation-system.png",
            tags: ["TensorFlow", "Distributed GPU Training", "MovieLens", "Faiss", "Numpy-Pandas", "Langchain", "Streamlit"],
            live: "https://debopam-movie-recommendation-system.streamlit.app/",
        },
        {
            title: "Non-Sequential Breast Cancer Classification System",
            category: "Deep Learning",
            description: "Multi-output deep learning model for breast cancer detection published in IRJET. Processes both mammogram images and tabular clinical data. Fine-tuned EfficientNetV2B3 for feature extraction.",
            image: "images/cancer-classification.png",
            tags: ["TensorFlow", "Transfer Learning", "EfficientNetV2", "FusedMB-CNN", "Streamlit"],
            live: "https://debopamparam-bcd-inference-3vqotx.streamlit.app/",
            paper: "https://www.irjet.net/archives/V12/i2/IRJET-V12I211.pdf"
        },
        {
            title: "LLM-Finetuning (Local SQL Agent)",
            category: "NLP & LLM",
            description: "SQL agent created by finetuning Qwen2.5-3B-Coder-Instruct model with supervised finetuning using QLora. Quantized from BF16 to int4(q4_k_m) for lightweight inference.",
            image: "images/llm-finetuning.png",
            tags: ["Supervised Finetuning", "Unsloth", "LlamaCPP", "Docker", "Langchain", "HuggingFace-Spaces"],
            live: "https://huggingface.co/spaces/DebopamC/Natual_Language-to-SQL-Qwen2.5-3B-FineTuned",
            ollama: "https://ollama.com/debopam/Text-to-SQL__Qwen2.5-Coder-3B-FineTuned"
        },
        {
            title: "TurboML Chatbot with Grounding",
            category: "Generative AI",
            description: "Made TurboML Chat Agent by scraping all the Docs + Pypi package with Grounding Links.",
            image: "images/turboml-chat.png",
            tags: ["RAG", "LangChain", "Vector DB", "Embeddings", "Streamlit"],
            live: "https://turboml-chat.streamlit.app/",
        }
    ],
    socialMedia: {
      linkedin: "https://www.linkedin.com/in/debopam-chowdhury-param-600619229/",
      github: "https://github.com/DebopamParam",
      youtube: "https://www.youtube.com/@DCparam/videos",
      email: "debopamwork@gmail.com"
    }
  };