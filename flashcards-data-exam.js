const flashcardsData = [
    // Domain 1: Fundamentals of AI and ML (20% - 15 cards)
    {
        id: 1,
        category: 'domain1',
        categoryName: 'Domain 1: AI/ML Fundamentals (20%)',
        question: 'What are the three main types of Machine Learning?',
        answer: 'Supervised Learning (labeled data), Unsupervised Learning (unlabeled data), and Reinforcement Learning (reward-based learning through trial and error).'
    },
    {
        id: 2,
        category: 'domain1',
        categoryName: 'Domain 1: AI/ML Fundamentals (20%)',
        question: 'What is the difference between Classification and Regression?',
        answer: 'Classification predicts discrete categories (e.g., spam/not spam), while Regression predicts continuous numerical values (e.g., house prices, temperature).'
    },
    {
        id: 3,
        category: 'domain1',
        categoryName: 'Domain 1: AI/ML Fundamentals (20%)',
        question: 'What is the ML model lifecycle?',
        answer: 'Data Collection → Data Preparation → Model Training → Model Evaluation → Model Deployment → Monitoring & Retraining. This cycle ensures models stay accurate over time.'
    },
    {
        id: 4,
        category: 'domain1',
        categoryName: 'Domain 1: AI/ML Fundamentals (20%)',
        question: 'What is Overfitting vs Underfitting?',
        answer: 'Overfitting: Model memorizes training data, performs poorly on new data. Underfitting: Model is too simple, performs poorly on both training and new data.'
    },
    {
        id: 5,
        category: 'domain1',
        categoryName: 'Domain 1: AI/ML Fundamentals (20%)',
        question: 'What is Training Data vs Testing Data vs Validation Data?',
        answer: 'Training: Used to teach the model. Validation: Used to tune hyperparameters during training. Testing: Used to evaluate final model performance on unseen data.'
    },
    {
        id: 6,
        category: 'domain1',
        categoryName: 'Domain 1: AI/ML Fundamentals (20%)',
        question: 'What is Feature Engineering?',
        answer: 'The process of selecting, transforming, and creating input variables (features) from raw data to improve model performance. Includes normalization, encoding, and feature selection.'
    },
    {
        id: 7,
        category: 'domain1',
        categoryName: 'Domain 1: AI/ML Fundamentals (20%)',
        question: 'What is Model Inference?',
        answer: 'The process of using a trained ML model to make predictions on new, unseen data in production. This is when the model is actively deployed and serving predictions.'
    },
    {
        id: 8,
        category: 'domain1',
        categoryName: 'Domain 1: AI/ML Fundamentals (20%)',
        question: 'What are common AI/ML use cases in business?',
        answer: 'Fraud detection, customer churn prediction, recommendation systems, demand forecasting, sentiment analysis, image/video analysis, chatbots, and predictive maintenance.'
    },
    {
        id: 9,
        category: 'domain1',
        categoryName: 'Domain 1: AI/ML Fundamentals (20%)',
        question: 'What is Deep Learning and when is it used?',
        answer: 'ML using neural networks with multiple layers. Best for complex patterns in large datasets: image recognition, NLP, speech recognition, and autonomous vehicles.'
    },
    {
        id: 10,
        category: 'domain1',
        categoryName: 'Domain 1: AI/ML Fundamentals (20%)',
        question: 'What is Transfer Learning?',
        answer: 'Using a pre-trained model as a starting point for a new task. Saves time and resources by leveraging knowledge from similar problems. Common in computer vision and NLP.'
    },
    {
        id: 11,
        category: 'domain1',
        categoryName: 'Domain 1: AI/ML Fundamentals (20%)',
        question: 'What is Model Accuracy vs Precision vs Recall?',
        answer: 'Accuracy: Overall correctness. Precision: Of predicted positives, how many are correct. Recall: Of actual positives, how many were found. Choose based on business needs.'
    },
    {
        id: 12,
        category: 'domain1',
        categoryName: 'Domain 1: AI/ML Fundamentals (20%)',
        question: 'What is Clustering and when is it used?',
        answer: 'Unsupervised learning that groups similar data points. Use cases: customer segmentation, anomaly detection, document organization, and market basket analysis.'
    },
    {
        id: 13,
        category: 'domain1',
        categoryName: 'Domain 1: AI/ML Fundamentals (20%)',
        question: 'What is Computer Vision?',
        answer: 'AI field enabling computers to interpret visual information. Applications: facial recognition, object detection, medical imaging, autonomous vehicles, and quality inspection.'
    },
    {
        id: 14,
        category: 'domain1',
        categoryName: 'Domain 1: AI/ML Fundamentals (20%)',
        question: 'What is Natural Language Processing (NLP)?',
        answer: 'AI for understanding and generating human language. Applications: chatbots, translation, sentiment analysis, text summarization, and named entity recognition.'
    },
    {
        id: 15,
        category: 'domain1',
        categoryName: 'Domain 1: AI/ML Fundamentals (20%)',
        question: 'What is Anomaly Detection?',
        answer: 'Identifying unusual patterns that don\'t conform to expected behavior. Use cases: fraud detection, network intrusion, equipment failure prediction, and quality control.'
    },

    // Domain 2: Fundamentals of GenAI (24% - 18 cards)
    {
        id: 16,
        category: 'domain2',
        categoryName: 'Domain 2: GenAI Fundamentals (24%)',
        question: 'What is Generative AI?',
        answer: 'AI that creates new content (text, images, code, audio, video) based on patterns learned from training data. Examples: ChatGPT, DALL-E, GitHub Copilot, Midjourney.'
    },
    {
        id: 17,
        category: 'domain2',
        categoryName: 'Domain 2: GenAI Fundamentals (24%)',
        question: 'What is a Foundation Model (FM)?',
        answer: 'Large AI model trained on vast amounts of data that can be adapted for many tasks. Also called Large Language Models (LLMs) for text. Examples: GPT-4, Claude, Llama.'
    },
    {
        id: 18,
        category: 'domain2',
        categoryName: 'Domain 2: GenAI Fundamentals (24%)',
        question: 'What is Amazon Bedrock?',
        answer: 'AWS managed service providing access to foundation models from AI21, Anthropic, Cohere, Meta, Stability AI, and Amazon through a single API. No infrastructure management needed.'
    },
    {
        id: 19,
        category: 'domain2',
        categoryName: 'Domain 2: GenAI Fundamentals (24%)',
        question: 'What is Prompt Engineering?',
        answer: 'Crafting effective inputs to get desired outputs from AI models. Techniques: clear instructions, examples (few-shot), context, role assignment, and iterative refinement.'
    },
    {
        id: 20,
        category: 'domain2',
        categoryName: 'Domain 2: GenAI Fundamentals (24%)',
        question: 'What is Zero-shot vs Few-shot vs Fine-tuning?',
        answer: 'Zero-shot: Task with no examples. Few-shot: Task with 1-10 examples in prompt. Fine-tuning: Retraining model on custom dataset for specific domain.'
    },
    {
        id: 21,
        category: 'domain2',
        categoryName: 'Domain 2: GenAI Fundamentals (24%)',
        question: 'What is a Token in GenAI?',
        answer: 'Unit of text processed by models (~4 characters or 0.75 words). Models have token limits (e.g., 4K, 8K, 100K tokens). Affects cost and context window size.'
    },
    {
        id: 22,
        category: 'domain2',
        categoryName: 'Domain 2: GenAI Fundamentals (24%)',
        question: 'What is Model Hallucination?',
        answer: 'When AI generates false or nonsensical information that sounds plausible. Caused by training data gaps or model limitations. Mitigate with RAG, fact-checking, and grounding.'
    },
    {
        id: 23,
        category: 'domain2',
        categoryName: 'Domain 2: GenAI Fundamentals (24%)',
        question: 'What is Temperature in GenAI models?',
        answer: 'Parameter controlling randomness (0-1). Low (0-0.3): Focused, deterministic. Medium (0.4-0.7): Balanced. High (0.8-1): Creative, diverse. Adjust based on use case.'
    },
    {
        id: 24,
        category: 'domain2',
        categoryName: 'Domain 2: GenAI Fundamentals (24%)',
        question: 'What is the Context Window?',
        answer: 'Maximum amount of text (in tokens) a model can process at once. Includes prompt + response. Larger windows allow more context but cost more.'
    },
    {
        id: 25,
        category: 'domain2',
        categoryName: 'Domain 2: GenAI Fundamentals (24%)',
        question: 'What are Embeddings?',
        answer: 'Numerical representations of text that capture semantic meaning. Used for similarity search, clustering, and RAG. Generated by embedding models like Amazon Titan Embeddings.'
    },
    {
        id: 26,
        category: 'domain2',
        categoryName: 'Domain 2: GenAI Fundamentals (24%)',
        question: 'What is Model Customization in Bedrock?',
        answer: 'Three approaches: Prompt Engineering (no training), Fine-tuning (custom training data), and Continued Pre-training (domain-specific knowledge). Choose based on needs and resources.'
    },
    {
        id: 27,
        category: 'domain2',
        categoryName: 'Domain 2: GenAI Fundamentals (24%)',
        question: 'What is Amazon Titan?',
        answer: 'AWS family of foundation models. Includes Titan Text (text generation), Titan Embeddings (semantic search), and Titan Image Generator (image creation).'
    },
    {
        id: 28,
        category: 'domain2',
        categoryName: 'Domain 2: GenAI Fundamentals (24%)',
        question: 'What is Claude (Anthropic)?',
        answer: 'Foundation model available in Bedrock. Known for long context windows (100K+ tokens), strong reasoning, and Constitutional AI for safety. Good for analysis and coding.'
    },
    {
        id: 29,
        category: 'domain2',
        categoryName: 'Domain 2: GenAI Fundamentals (24%)',
        question: 'What is Stable Diffusion?',
        answer: 'Open-source image generation model available in Bedrock. Creates images from text descriptions. Used for art, design, marketing, and content creation.'
    },
    {
        id: 30,
        category: 'domain2',
        categoryName: 'Domain 2: GenAI Fundamentals (24%)',
        question: 'What are GenAI use cases in business?',
        answer: 'Content creation, code generation, customer service chatbots, document summarization, data analysis, personalization, creative design, and knowledge management.'
    },
    {
        id: 31,
        category: 'domain2',
        categoryName: 'Domain 2: GenAI Fundamentals (24%)',
        question: 'What is Bedrock Agents?',
        answer: 'Fully managed capability to build AI agents that can execute multi-step tasks, call APIs, query databases, and use tools to complete complex workflows autonomously.'
    },
    {
        id: 32,
        category: 'domain2',
        categoryName: 'Domain 2: GenAI Fundamentals (24%)',
        question: 'What is Bedrock Knowledge Bases?',
        answer: 'Managed RAG solution that connects foundation models to your data sources. Automatically handles chunking, embedding, and retrieval for accurate, grounded responses.'
    },
    {
        id: 33,
        category: 'domain2',
        categoryName: 'Domain 2: GenAI Fundamentals (24%)',
        question: 'What is Model Evaluation in GenAI?',
        answer: 'Assessing quality using metrics like accuracy, relevance, coherence, toxicity, and bias. Use human evaluation, automated metrics, and A/B testing for production models.'
    },

    // Domain 3: Applications of Foundation Models (28% - 21 cards)
    {
        id: 34,
        category: 'domain3',
        categoryName: 'Domain 3: FM Applications (28%)',
        question: 'What is RAG (Retrieval Augmented Generation)?',
        answer: 'Technique that retrieves relevant information from external sources before generating responses. Reduces hallucinations, provides current data, and grounds answers in facts.'
    },
    {
        id: 35,
        category: 'domain3',
        categoryName: 'Domain 3: FM Applications (28%)',
        question: 'How does RAG work?',
        answer: '1) User query → 2) Retrieve relevant docs from knowledge base → 3) Combine query + docs in prompt → 4) FM generates grounded response. Uses vector databases for similarity search.'
    },
    {
        id: 36,
        category: 'domain3',
        categoryName: 'Domain 3: FM Applications (28%)',
        question: 'What is Amazon SageMaker?',
        answer: 'Fully managed ML service for building, training, and deploying models at scale. Includes notebooks, training jobs, endpoints, pipelines, and MLOps tools.'
    },
    {
        id: 37,
        category: 'domain3',
        categoryName: 'Domain 3: FM Applications (28%)',
        question: 'What is SageMaker JumpStart?',
        answer: 'Pre-built ML solutions and foundation models you can deploy with one click. Includes 100+ models for text, vision, and more. Accelerates development.'
    },
    {
        id: 38,
        category: 'domain3',
        categoryName: 'Domain 3: FM Applications (28%)',
        question: 'What is Amazon Lex?',
        answer: 'Service for building conversational interfaces (chatbots) using voice and text. Powers Alexa. Includes NLU, automatic speech recognition, and integration with Lambda.'
    },
    {
        id: 39,
        category: 'domain3',
        categoryName: 'Domain 3: FM Applications (28%)',
        question: 'What is Amazon Kendra?',
        answer: 'Intelligent search service powered by ML. Understands natural language queries, ranks results by relevance, and provides direct answers from documents.'
    },
    {
        id: 40,
        category: 'domain3',
        categoryName: 'Domain 3: FM Applications (28%)',
        question: 'What is Amazon Comprehend?',
        answer: 'NLP service for extracting insights from text. Features: sentiment analysis, entity recognition, key phrase extraction, language detection, and topic modeling.'
    },
    {
        id: 41,
        category: 'domain3',
        categoryName: 'Domain 3: FM Applications (28%)',
        question: 'What is Amazon Rekognition?',
        answer: 'Computer vision service for image and video analysis. Detects objects, faces, text, scenes, activities, and inappropriate content. No ML expertise required.'
    },
    {
        id: 42,
        category: 'domain3',
        categoryName: 'Domain 3: FM Applications (28%)',
        question: 'What is Amazon Textract?',
        answer: 'Extracts text, handwriting, and data from scanned documents. Goes beyond OCR by understanding forms, tables, and document structure. Used for document processing automation.'
    },
    {
        id: 43,
        category: 'domain3',
        categoryName: 'Domain 3: FM Applications (28%)',
        question: 'What is Amazon Transcribe?',
        answer: 'Speech-to-text service that converts audio to text. Features: real-time and batch, speaker identification, custom vocabularies, and automatic punctuation.'
    },
    {
        id: 44,
        category: 'domain3',
        categoryName: 'Domain 3: FM Applications (28%)',
        question: 'What is Amazon Polly?',
        answer: 'Text-to-speech service that converts text into lifelike speech. Supports 60+ voices in 30+ languages. Features: SSML, neural voices, and custom lexicons.'
    },
    {
        id: 45,
        category: 'domain3',
        categoryName: 'Domain 3: FM Applications (28%)',
        question: 'What is Amazon Translate?',
        answer: 'Neural machine translation service for fast, high-quality language translation. Supports 75+ languages, real-time and batch translation, and custom terminology.'
    },
    {
        id: 46,
        category: 'domain3',
        categoryName: 'Domain 3: FM Applications (28%)',
        question: 'What is Amazon Personalize?',
        answer: 'Service for building personalized recommendation systems. Uses ML to deliver customized product, content, and search recommendations based on user behavior.'
    },
    {
        id: 47,
        category: 'domain3',
        categoryName: 'Domain 3: FM Applications (28%)',
        question: 'What is Amazon Forecast?',
        answer: 'Time-series forecasting service using ML. Predicts future values based on historical data. Use cases: demand planning, inventory management, and financial forecasting.'
    },
    {
        id: 48,
        category: 'domain3',
        categoryName: 'Domain 3: FM Applications (28%)',
        question: 'What is Amazon Fraud Detector?',
        answer: 'Managed service for detecting fraud using ML. Identifies suspicious online activities like fake accounts, payment fraud, and account takeovers. No ML expertise needed.'
    },
    {
        id: 49,
        category: 'domain3',
        categoryName: 'Domain 3: FM Applications (28%)',
        question: 'What is Amazon CodeWhisperer?',
        answer: 'AI coding companion that generates code suggestions in real-time. Supports 15+ languages, security scanning, and reference tracking. Integrated with IDEs.'
    },
    {
        id: 50,
        category: 'domain3',
        categoryName: 'Domain 3: FM Applications (28%)',
        question: 'What is Amazon Q?',
        answer: 'Generative AI assistant for business. Helps with tasks, answers questions, generates content, and provides insights. Integrates with AWS services and enterprise data.'
    },
    {
        id: 51,
        category: 'domain3',
        categoryName: 'Domain 3: FM Applications (28%)',
        question: 'When to use Bedrock vs SageMaker?',
        answer: 'Bedrock: Quick deployment of pre-trained FMs, no infrastructure management, pay-per-use. SageMaker: Custom model development, full control, training from scratch, MLOps.'
    },
    {
        id: 52,
        category: 'domain3',
        categoryName: 'Domain 3: FM Applications (28%)',
        question: 'What is Model Deployment in SageMaker?',
        answer: 'Deploy models to real-time endpoints (low latency), batch transform (large datasets), or serverless inference (variable traffic). Choose based on latency and cost needs.'
    },
    {
        id: 53,
        category: 'domain3',
        categoryName: 'Domain 3: FM Applications (28%)',
        question: 'What is SageMaker Model Monitor?',
        answer: 'Continuously monitors ML models in production for data drift, model quality degradation, and bias. Sends alerts when performance degrades so you can retrain.'
    },
    {
        id: 54,
        category: 'domain3',
        categoryName: 'Domain 3: FM Applications (28%)',
        question: 'What is Vector Database for RAG?',
        answer: 'Stores embeddings for semantic search. Popular options: Amazon OpenSearch Service, pgvector (RDS), Pinecone, or FAISS. Essential for RAG implementations.'
    },

    // Domain 4: Guidelines for Responsible AI (14% - 11 cards)
    {
        id: 55,
        category: 'domain4',
        categoryName: 'Domain 4: Responsible AI (14%)',
        question: 'What is Responsible AI?',
        answer: 'Developing and deploying AI systems that are fair, transparent, accountable, secure, and respect privacy. Ensures AI benefits society while minimizing harm.'
    },
    {
        id: 56,
        category: 'domain4',
        categoryName: 'Domain 4: Responsible AI (14%)',
        question: 'What is AI Bias and how to mitigate it?',
        answer: 'Unfair outcomes due to biased training data or algorithms. Mitigate: diverse training data, bias detection tools, regular audits, and inclusive development teams.'
    },
    {
        id: 57,
        category: 'domain4',
        categoryName: 'Domain 4: Responsible AI (14%)',
        question: 'What is AI Fairness?',
        answer: 'Ensuring AI systems treat all groups equitably without discrimination based on race, gender, age, etc. Measure using fairness metrics and conduct bias testing.'
    },
    {
        id: 58,
        category: 'domain4',
        categoryName: 'Domain 4: Responsible AI (14%)',
        question: 'What is AI Transparency and Explainability?',
        answer: 'Transparency: Understanding how AI systems work. Explainability: Understanding specific decisions. Use tools like SageMaker Clarify for model interpretability.'
    },
    {
        id: 59,
        category: 'domain4',
        categoryName: 'Domain 4: Responsible AI (14%)',
        question: 'What is Amazon SageMaker Clarify?',
        answer: 'Tool for detecting bias in data and models, and explaining model predictions. Provides bias metrics, feature importance, and SHAP values for interpretability.'
    },
    {
        id: 60,
        category: 'domain4',
        categoryName: 'Domain 4: Responsible AI (14%)',
        question: 'What is AI Privacy and Data Protection?',
        answer: 'Protecting personal data used in AI systems. Practices: data minimization, anonymization, encryption, consent management, and compliance with GDPR/CCPA.'
    },
    {
        id: 61,
        category: 'domain4',
        categoryName: 'Domain 4: Responsible AI (14%)',
        question: 'What is Human-in-the-Loop (HITL)?',
        answer: 'Keeping humans involved in AI decision-making, especially for high-stakes decisions. Humans review, validate, or override AI outputs. Use Amazon SageMaker Ground Truth.'
    },
    {
        id: 62,
        category: 'domain4',
        categoryName: 'Domain 4: Responsible AI (14%)',
        question: 'What is AI Accountability?',
        answer: 'Clear responsibility for AI system outcomes. Includes documentation, audit trails, version control, and defined escalation paths for issues.'
    },
    {
        id: 63,
        category: 'domain4',
        categoryName: 'Domain 4: Responsible AI (14%)',
        question: 'What is Content Filtering in GenAI?',
        answer: 'Blocking harmful, toxic, or inappropriate content. Bedrock provides Guardrails to filter inputs/outputs based on policies. Essential for safe AI applications.'
    },
    {
        id: 64,
        category: 'domain4',
        categoryName: 'Domain 4: Responsible AI (14%)',
        question: 'What are Bedrock Guardrails?',
        answer: 'Safety controls for GenAI applications. Filter harmful content, block sensitive topics, redact PII, and enforce brand guidelines. Applied to both prompts and responses.'
    },
    {
        id: 65,
        category: 'domain4',
        categoryName: 'Domain 4: Responsible AI (14%)',
        question: 'What is Model Governance?',
        answer: 'Policies and processes for managing AI models throughout their lifecycle. Includes approval workflows, documentation, monitoring, and compliance tracking.'
    },

    // Domain 5: Security, Compliance, and Governance (14% - 11 cards)
    {
        id: 66,
        category: 'domain5',
        categoryName: 'Domain 5: Security & Governance (14%)',
        question: 'What is the AWS Shared Responsibility Model for AI?',
        answer: 'AWS: Security OF the cloud (infrastructure, services). Customer: Security IN the cloud (data, access control, encryption, compliance). Both share responsibility.'
    },
    {
        id: 67,
        category: 'domain5',
        categoryName: 'Domain 5: Security & Governance (14%)',
        question: 'How does IAM work with AI services?',
        answer: 'Use IAM policies to control who can access AI services and what actions they can perform. Follow least privilege principle. Use roles for service-to-service access.'
    },
    {
        id: 68,
        category: 'domain5',
        categoryName: 'Domain 5: Security & Governance (14%)',
        question: 'What is Data Encryption for AI workloads?',
        answer: 'Encrypt data at rest (S3, EBS) using KMS and in transit (TLS/SSL). Bedrock and SageMaker support encryption by default. Use customer-managed keys for control.'
    },
    {
        id: 69,
        category: 'domain5',
        categoryName: 'Domain 5: Security & Governance (14%)',
        question: 'What is VPC for AI services?',
        answer: 'Deploy AI services in VPC for network isolation. Use VPC endpoints for private connectivity. SageMaker supports VPC mode for training and inference.'
    },
    {
        id: 70,
        category: 'domain5',
        categoryName: 'Domain 5: Security & Governance (14%)',
        question: 'What is AWS CloudTrail for AI?',
        answer: 'Logs all API calls to AI services for auditing and compliance. Track who accessed what, when, and from where. Essential for security investigations.'
    },
    {
        id: 71,
        category: 'domain5',
        categoryName: 'Domain 5: Security & Governance (14%)',
        question: 'What is Amazon Macie?',
        answer: 'Data security service that uses ML to discover, classify, and protect sensitive data in S3. Identifies PII, financial data, and credentials. Important for AI data governance.'
    },
    {
        id: 72,
        category: 'domain5',
        categoryName: 'Domain 5: Security & Governance (14%)',
        question: 'What is Data Residency and Sovereignty?',
        answer: 'Keeping data in specific geographic regions for compliance. Choose AWS regions carefully. Bedrock and SageMaker data stays in your selected region.'
    },
    {
        id: 73,
        category: 'domain5',
        categoryName: 'Domain 5: Security & Governance (14%)',
        question: 'What is AWS Compliance for AI?',
        answer: 'AWS AI services comply with standards like SOC, ISO, HIPAA, GDPR, and FedRAMP. Check AWS Artifact for compliance reports. Customer responsible for their use case compliance.'
    },
    {
        id: 74,
        category: 'domain5',
        categoryName: 'Domain 5: Security & Governance (14%)',
        question: 'What is Model Versioning and Lineage?',
        answer: 'Track model versions, training data, and parameters. SageMaker Model Registry stores models with metadata. Essential for reproducibility and rollback.'
    },
    {
        id: 75,
        category: 'domain5',
        categoryName: 'Domain 5: Security & Governance (14%)',
        question: 'What is Cost Optimization for AI workloads?',
        answer: 'Use Spot Instances for training, right-size instances, use serverless inference for variable traffic, monitor with Cost Explorer, and set budgets/alerts.'
    },
    {
        id: 76,
        category: 'domain5',
        categoryName: 'Domain 5: Security & Governance (14%)',
        question: 'What is AWS Well-Architected Framework for AI/ML?',
        answer: 'Best practices across 6 pillars: Operational Excellence, Security, Reliability, Performance Efficiency, Cost Optimization, and Sustainability. Apply to AI workloads.'
    }
];
