const flashcardsData = [
    {
        id: 1,
        category: 'domain1',
        categoryName: 'Domain 1: AI/ML Fundamentals (20%)',
        question: 'What is Artificial Intelligence (AI)?',
        answer: 'Technology that makes computers act smart, like humans - understanding language, recognizing images, making decisions.'
    },
    {
        id: 2,
        category: 'domain1',
        categoryName: 'Domain 1: AI/ML Fundamentals (20%)',
        question: 'What is Machine Learning (ML)?',
        answer: 'Teaching computers to learn from examples instead of programming every rule; like showing a child pictures of cats until they recognize cats on their own.'
    },
    {
        id: 3,
        category: 'domain1',
        categoryName: 'Domain 1: AI/ML Fundamentals (20%)',
        question: 'What is Deep Learning?',
        answer: 'A more advanced type of ML that uses layers of learning (like peeling an onion) to understand complex patterns; powers things like voice assistants.'
    },
    {
        id: 4,
        category: 'domain1',
        categoryName: 'Domain 1: AI/ML Fundamentals (20%)',
        question: 'What is the difference between Training and Inferencing?',
        answer: 'Training: Teaching an AI model by showing it lots of examples until it learns patterns. Inferencing: When a trained AI model makes predictions or decisions on new data it has not seen before.'
    },
    {
        id: 5,
        category: 'domain1',
        categoryName: 'Domain 1: AI/ML Fundamentals (20%)',
        question: 'What is Bias in AI?',
        answer: 'When AI makes unfair decisions because the data it learned from was unbalanced or reflected human prejudices.'
    },
    {
        id: 6,
        category: 'domain1',
        categoryName: 'Domain 1: AI/ML Fundamentals (20%)',
        question: 'What is Overfitting vs Underfitting?',
        answer: 'Overfitting: Model memorized too much and cannot handle new situations. Underfitting: Model did not learn enough from the training data.'
    },
    {
        id: 7,
        category: 'domain1',
        categoryName: 'Domain 1: AI/ML Fundamentals (20%)',
        question: 'What are the three types of Machine Learning?',
        answer: 'Supervised Learning (teaching with labeled examples), Unsupervised Learning (AI finds patterns without labels), Reinforcement Learning (AI learns by trial and error with rewards).'
    },
    {
        id: 8,
        category: 'domain1',
        categoryName: 'Domain 1: AI/ML Fundamentals (20%)',
        question: 'What is Labeled vs Unlabeled Data?',
        answer: 'Labeled Data: Data tagged with the correct answer (like photos marked cat or dog). Unlabeled Data: Raw data without any tags or answers provided.'
    },
    {
        id: 9,
        category: 'domain1',
        categoryName: 'Domain 1: AI/ML Fundamentals (20%)',
        question: 'What are the 4 types of Inference?',
        answer: 'Real-time (under 60s, instant responses), Serverless (occasional traffic, auto-scales), Asynchronous (up to 1GB, up to 1 hour), Batch Transform (gigabytes, can take days).'
    },
    {
        id: 10,
        category: 'domain1',
        categoryName: 'Domain 1: AI/ML Fundamentals (20%)',
        question: 'What is SVD (Singular Value Decomposition)?',
        answer: 'Breaks large datasets into simpler components while preserving important patterns. Used for recommendations and matrix factorization.'
    },
    {
        id: 11,
        category: 'domain2',
        categoryName: 'Domain 2: GenAI Fundamentals (24%)',
        question: 'What is Generative AI?',
        answer: 'AI that creates new content - writes text, generates images, composes music. Examples: ChatGPT, image generators.'
    },
    {
        id: 12,
        category: 'domain2',
        categoryName: 'Domain 2: GenAI Fundamentals (24%)',
        question: 'What is a Large Language Model (LLM)?',
        answer: 'A type of AI trained on massive amounts of text that can understand and generate human-like responses; the brain behind chatbots.'
    },
    {
        id: 13,
        category: 'domain2',
        categoryName: 'Domain 2: GenAI Fundamentals (24%)',
        question: 'What are Foundation Models?',
        answer: 'Pre-trained, large AI models (like GPT, Claude) that can be customized for specific tasks without starting from zero.'
    },
    {
        id: 14,
        category: 'domain2',
        categoryName: 'Domain 2: GenAI Fundamentals (24%)',
        question: 'What is the difference between Continued Pre-training and Fine-tuning?',
        answer: 'Continued pre-training uses unlabeled data to pre-train a model, whereas fine-tuning uses labeled data to train a model.'
    },
    {
        id: 15,
        category: 'domain2',
        categoryName: 'Domain 2: GenAI Fundamentals (24%)',
        question: 'What are Tokens in GenAI?',
        answer: 'Small chunks of text (like words or parts of words) that AI models process; you pay based on how many tokens you use.'
    },
    {
        id: 16,
        category: 'domain2',
        categoryName: 'Domain 2: GenAI Fundamentals (24%)',
        question: 'What are Embeddings?',
        answer: 'Converting words or images into numbers so computers can understand and compare them.'
    },
    {
        id: 17,
        category: 'domain2',
        categoryName: 'Domain 2: GenAI Fundamentals (24%)',
        question: 'What are Hallucinations in AI?',
        answer: 'AI generates responses that sound authoritative but are inaccurate or false.'
    },
    {
        id: 18,
        category: 'domain2',
        categoryName: 'Domain 2: GenAI Fundamentals (24%)',
        question: 'What is Temperature in inference parameters?',
        answer: 'Randomness of responses; low (0.1-0.3) = predictable and safe; high (0.7-1.0) = creative and varied.'
    },
    {
        id: 19,
        category: 'domain2',
        categoryName: 'Domain 2: GenAI Fundamentals (24%)',
        question: 'What is Zero-shot Prompting?',
        answer: 'Asking the model to perform a task without any examples.'
    },
    {
        id: 20,
        category: 'domain2',
        categoryName: 'Domain 2: GenAI Fundamentals (24%)',
        question: 'What is Few-shot Prompting?',
        answer: 'Providing a few examples to help the model understand what you want.'
    },
    {
        id: 21,
        category: 'domain3',
        categoryName: 'Domain 3: FM Applications (28%)',
        question: 'What is Amazon Bedrock?',
        answer: 'Access to ready-made AI models from top companies (Claude, Llama, etc.) without building from scratch; includes Guardrails for safety. Two pricing: On-Demand/Batch and Provisioned Throughput.'
    },
    {
        id: 22,
        category: 'domain3',
        categoryName: 'Domain 3: FM Applications (28%)',
        question: 'What is Amazon SageMaker?',
        answer: 'Complete toolkit to build, train, and launch your own custom ML models; like a workshop with all the tools you need.'
    },
    {
        id: 23,
        category: 'domain3',
        categoryName: 'Domain 3: FM Applications (28%)',
        question: 'What is SageMaker Canvas?',
        answer: 'No-code, visual ML model building for business users.'
    },
    {
        id: 24,
        category: 'domain3',
        categoryName: 'Domain 3: FM Applications (28%)',
        question: 'What is SageMaker Clarify?',
        answer: 'Detects bias and explains model predictions.'
    },
    {
        id: 25,
        category: 'domain3',
        categoryName: 'Domain 3: FM Applications (28%)',
        question: 'What is Amazon Rekognition?',
        answer: 'Analyzes photos and videos to find faces, objects, text, or inappropriate content.'
    },
    {
        id: 26,
        category: 'domain3',
        categoryName: 'Domain 3: FM Applications (28%)',
        question: 'What is Amazon Polly?',
        answer: 'Converts written text into spoken words in natural-sounding voices and multiple languages.'
    },
    {
        id: 27,
        category: 'domain3',
        categoryName: 'Domain 3: FM Applications (28%)',
        question: 'What is Amazon Transcribe?',
        answer: 'Converts speech/audio into written text; automatic note-taker for meetings or calls.'
    },
    {
        id: 28,
        category: 'domain3',
        categoryName: 'Domain 3: FM Applications (28%)',
        question: 'What is Amazon Translate?',
        answer: 'Translates text from one language to another automatically.'
    },
    {
        id: 29,
        category: 'domain3',
        categoryName: 'Domain 3: FM Applications (28%)',
        question: 'What is Amazon Comprehend?',
        answer: 'Reads text and figures out sentiment (positive/negative), finds key topics, extracts important information.'
    },
    {
        id: 30,
        category: 'domain3',
        categoryName: 'Domain 3: FM Applications (28%)',
        question: 'What is Amazon Lex?',
        answer: 'Builds conversational chatbots that understand questions and respond appropriately.'
    },
    {
        id: 31,
        category: 'domain3',
        categoryName: 'Domain 3: FM Applications (28%)',
        question: 'What is Amazon Textract?',
        answer: 'Extracts text and data from scanned documents, forms, and tables.'
    },
    {
        id: 32,
        category: 'domain3',
        categoryName: 'Domain 3: FM Applications (28%)',
        question: 'What is Amazon Kendra?',
        answer: 'Intelligent search engine that understands questions and finds answers in company documents.'
    },
    {
        id: 33,
        category: 'domain3',
        categoryName: 'Domain 3: FM Applications (28%)',
        question: 'What is RAG (Retrieval Augmented Generation)?',
        answer: 'Make AI search your documents for facts before answering; prevents hallucinations.'
    },
    {
        id: 34,
        category: 'domain3',
        categoryName: 'Domain 3: FM Applications (28%)',
        question: 'What is BLEU?',
        answer: 'Measures quality of machine translation.'
    },
    {
        id: 35,
        category: 'domain4',
        categoryName: 'Domain 4: Responsible AI (14%)',
        question: 'What are the Six Core Dimensions of Responsible AI?',
        answer: 'Fairness, Explainability, Privacy and Security, Safety, Controllability, Veracity and Robustness.'
    },
    {
        id: 36,
        category: 'domain4',
        categoryName: 'Domain 4: Responsible AI (14%)',
        question: 'What is Fairness in AI?',
        answer: 'AI treats all individuals and groups equitably without discrimination.'
    },
    {
        id: 37,
        category: 'domain4',
        categoryName: 'Domain 4: Responsible AI (14%)',
        question: 'What is Explainability in AI?',
        answer: 'Ability to understand how the AI system arrived at its results.'
    },
    {
        id: 38,
        category: 'domain4',
        categoryName: 'Domain 4: Responsible AI (14%)',
        question: 'What is Safety in AI?',
        answer: 'Ensuring AI does not produce harmful, toxic, or dangerous outputs.'
    },
    {
        id: 39,
        category: 'domain4',
        categoryName: 'Domain 4: Responsible AI (14%)',
        question: 'What are Generative AI Risks?',
        answer: 'Hallucinations (false info), Data privacy (sensitive info exposed), Quality control (not meeting expectations), Toxicity and safety (hateful content), Intellectual property (reproducing copyrighted content).'
    },
    {
        id: 40,
        category: 'domain5',
        categoryName: 'Domain 5: Security & Governance (14%)',
        question: 'What is Amazon Bedrock Guardrails?',
        answer: 'Sets safety rules to prevent harmful, biased, or inappropriate AI content.'
    },
    {
        id: 41,
        category: 'domain5',
        categoryName: 'Domain 5: Security & Governance (14%)',
        question: 'What is AWS IAM?',
        answer: 'Controls who can access AWS services by creating users, groups, roles, and permissions.'
    },
    {
        id: 42,
        category: 'domain5',
        categoryName: 'Domain 5: Security & Governance (14%)',
        question: 'What is AWS KMS?',
        answer: 'Creates and manages encryption keys for securing data.'
    },
    {
        id: 43,
        category: 'domain5',
        categoryName: 'Domain 5: Security & Governance (14%)',
        question: 'What is AWS CloudWatch?',
        answer: 'Monitors AWS resources and tracks performance metrics.'
    },
    {
        id: 44,
        category: 'domain5',
        categoryName: 'Domain 5: Security & Governance (14%)',
        question: 'What is AWS CloudTrail?',
        answer: 'Records all API calls and user activity for auditing.'
    },
    {
        id: 45,
        category: 'domain5',
        categoryName: 'Domain 5: Security & Governance (14%)',
        question: 'What is Amazon Macie?',
        answer: 'Discovers and protects sensitive data using machine learning.'
    },
    {
        id: 46,
        category: 'domain5',
        categoryName: 'Domain 5: Security & Governance (14%)',
        question: 'What is Bedrock + Guardrails used for?',
        answer: 'Bedrock provides foundation models; Guardrails sets safety rules to prevent harmful/biased content (Ensuring AI responses are safe and compliant).'
    },
    {
        id: 47,
        category: 'domain5',
        categoryName: 'Domain 5: Security & Governance (14%)',
        question: 'What is SageMaker + Clarify used for?',
        answer: 'SageMaker trains models; Clarify detects bias in training data and predictions (Ensuring fairness in ML models).'
    },
    {
        id: 48,
        category: 'domain5',
        categoryName: 'Domain 5: Security & Governance (14%)',
        question: 'What is Bedrock + RAG used for?',
        answer: 'Bedrock provides foundation models; RAG connects them to your documents (Preventing hallucinations by grounding responses in facts).'
    },
    {
        id: 49,
        category: 'scenarios',
        categoryName: 'Exam Scenarios',
        question: 'Scenario: Analyze customer reviews for sentiment',
        answer: 'Amazon Comprehend'
    },
    {
        id: 50,
        category: 'scenarios',
        categoryName: 'Exam Scenarios',
        question: 'Scenario: Convert call recordings to text',
        answer: 'Amazon Transcribe'
    },
    {
        id: 51,
        category: 'scenarios',
        categoryName: 'Exam Scenarios',
        question: 'Scenario: Build a website chatbot',
        answer: 'Amazon Lex'
    },
    {
        id: 52,
        category: 'scenarios',
        categoryName: 'Exam Scenarios',
        question: 'Scenario: Translate product descriptions',
        answer: 'Amazon Translate'
    },
    {
        id: 53,
        category: 'scenarios',
        categoryName: 'Exam Scenarios',
        question: 'Scenario: Extract data from invoices',
        answer: 'Amazon Textract'
    },
    {
        id: 54,
        category: 'scenarios',
        categoryName: 'Exam Scenarios',
        question: 'Scenario: Detect inappropriate images',
        answer: 'Amazon Rekognition'
    },
    {
        id: 55,
        category: 'scenarios',
        categoryName: 'Exam Scenarios',
        question: 'Scenario: Search company documents',
        answer: 'Amazon Kendra'
    },
    {
        id: 56,
        category: 'scenarios',
        categoryName: 'Exam Scenarios',
        question: 'Scenario: Use pre-built AI without coding',
        answer: 'Amazon Bedrock'
    },
    {
        id: 57,
        category: 'scenarios',
        categoryName: 'Exam Scenarios',
        question: 'Scenario: Prevent AI from making things up',
        answer: 'RAG with Amazon Bedrock'
    },
    {
        id: 58,
        category: 'scenarios',
        categoryName: 'Exam Scenarios',
        question: 'Scenario: Ensure AI responses are safe',
        answer: 'Amazon Bedrock Guardrails'
    },
];

// Total: 58 flashcards from AI Practitioner document
