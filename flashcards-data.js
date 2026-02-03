const flashcardsData = [
    // Core Concepts (1-10)
    {
        id: 1,
        category: 'core',
        categoryName: 'Core Concepts',
        question: 'What is Artificial Intelligence (AI)?',
        answer: 'Technology that enables machines to perform tasks that typically require human intelligence, such as visual perception, speech recognition, decision-making, and language translation.'
    },
    {
        id: 2,
        category: 'core',
        categoryName: 'Core Concepts',
        question: 'What is Machine Learning (ML)?',
        answer: 'A subset of AI where systems learn from data and improve their performance over time without being explicitly programmed. The system identifies patterns and makes decisions based on data.'
    },
    {
        id: 3,
        category: 'core',
        categoryName: 'Core Concepts',
        question: 'What is Deep Learning (DL)?',
        answer: 'A subset of ML that uses neural networks with multiple layers (deep neural networks) to learn from large amounts of data. Particularly effective for image recognition, NLP, and complex pattern recognition.'
    },
    {
        id: 4,
        category: 'core',
        categoryName: 'Core Concepts',
        question: 'What is the difference between AI, ML, and DL?',
        answer: 'AI is the broadest concept (machines mimicking human intelligence). ML is a subset of AI (learning from data). DL is a subset of ML (using deep neural networks). Think: AI ⊃ ML ⊃ DL'
    },
    {
        id: 5,
        category: 'core',
        categoryName: 'Core Concepts',
        question: 'What is Supervised Learning?',
        answer: 'A type of ML where the model learns from labeled training data. Each example has input features and a known output (label). The model learns to map inputs to outputs. Examples: spam detection, price prediction.'
    },
    {
        id: 6,
        category: 'core',
        categoryName: 'Core Concepts',
        question: 'What is Unsupervised Learning?',
        answer: 'A type of ML where the model learns from unlabeled data. The system finds hidden patterns or structures without predefined labels. Examples: customer segmentation, anomaly detection.'
    },
    {
        id: 7,
        category: 'core',
        categoryName: 'Core Concepts',
        question: 'What is Reinforcement Learning?',
        answer: 'A type of ML where an agent learns by interacting with an environment, receiving rewards for good actions and penalties for bad ones. The agent learns optimal behavior through trial and error.'
    },
    {
        id: 8,
        category: 'core',
        categoryName: 'Core Concepts',
        question: 'What is a Neural Network?',
        answer: 'A computing system inspired by biological neural networks in the brain. It consists of interconnected nodes (neurons) organized in layers that process and transmit information to learn patterns from data.'
    },
    {
        id: 9,
        category: 'core',
        categoryName: 'Core Concepts',
        question: 'What is Natural Language Processing (NLP)?',
        answer: 'A branch of AI that enables computers to understand, interpret, and generate human language. Used in chatbots, translation, sentiment analysis, and text summarization.'
    },
    {
        id: 10,
        category: 'core',
        categoryName: 'Core Concepts',
        question: 'What is Computer Vision?',
        answer: 'A field of AI that enables computers to interpret and understand visual information from images and videos. Used in facial recognition, object detection, and medical image analysis.'
    },
    
    // ML Components (11-20)
    {
        id: 11,
        category: 'components',
        categoryName: 'ML Components',
        question: 'What are Features?',
        answer: 'Input variables or attributes used to make predictions. For example, in house price prediction, features might include square footage, number of bedrooms, location, and age of the house.'
    },
    {
        id: 12,
        category: 'components',
        categoryName: 'ML Components',
        question: 'What is a Label (Target)?',
        answer: 'The output variable you\'re trying to predict in supervised learning. For example, in spam detection, the label is "spam" or "not spam."'
    },
    {
        id: 13,
        category: 'components',
        categoryName: 'ML Components',
        question: 'What is Training Data?',
        answer: 'The dataset used to teach a machine learning model. It contains examples with features and (in supervised learning) their corresponding labels. The model learns patterns from this data.'
    },
    {
        id: 14,
        category: 'components',
        categoryName: 'ML Components',
        question: 'What is Testing Data?',
        answer: 'A separate dataset used to evaluate how well the trained model performs on new, unseen data. It helps assess if the model can generalize beyond the training data.'
    },
    {
        id: 15,
        category: 'components',
        categoryName: 'ML Components',
        question: 'What is a Model?',
        answer: 'The mathematical representation or algorithm that has learned patterns from training data and can make predictions on new data. It\'s the output of the training process.'
    },
    {
        id: 16,
        category: 'components',
        categoryName: 'ML Components',
        question: 'What is Training?',
        answer: 'The process of feeding data to a machine learning algorithm so it can learn patterns and relationships. The model adjusts its parameters to minimize prediction errors.'
    },
    {
        id: 17,
        category: 'components',
        categoryName: 'ML Components',
        question: 'What is Inference?',
        answer: 'The process of using a trained model to make predictions on new, unseen data. This is when the model is deployed and actively being used.'
    },
    {
        id: 18,
        category: 'components',
        categoryName: 'ML Components',
        question: 'What is Overfitting?',
        answer: 'When a model learns the training data too well, including noise and outliers, and performs poorly on new data. The model memorizes rather than generalizes.'
    },
    {
        id: 19,
        category: 'components',
        categoryName: 'ML Components',
        question: 'What is Underfitting?',
        answer: 'When a model is too simple to capture the underlying patterns in the data. It performs poorly on both training and testing data.'
    },
    {
        id: 20,
        category: 'components',
        categoryName: 'ML Components',
        question: 'What is a Dataset?',
        answer: 'A collection of data used for training, testing, or validating machine learning models. It typically consists of multiple examples with features and (optionally) labels.'
    },
    
    // ML Algorithms (21-30)
    {
        id: 21,
        category: 'algorithms',
        categoryName: 'ML Algorithms',
        question: 'What is Classification?',
        answer: 'A supervised learning task where the model predicts discrete categories or classes. Examples: email spam/not spam, disease diagnosis (positive/negative), image recognition (cat/dog/bird).'
    },
    {
        id: 22,
        category: 'algorithms',
        categoryName: 'ML Algorithms',
        question: 'What is Regression?',
        answer: 'A supervised learning task where the model predicts continuous numerical values. Examples: house prices, temperature forecasting, stock prices, sales predictions.'
    },
    {
        id: 23,
        category: 'algorithms',
        categoryName: 'ML Algorithms',
        question: 'What is Clustering?',
        answer: 'An unsupervised learning technique that groups similar data points together based on their characteristics. Examples: customer segmentation, document organization, image compression.'
    },
    {
        id: 24,
        category: 'algorithms',
        categoryName: 'ML Algorithms',
        question: 'What is a Decision Tree?',
        answer: 'A supervised learning algorithm that makes decisions by splitting data based on feature values, creating a tree-like structure. Easy to interpret and visualize.'
    },
    {
        id: 25,
        category: 'algorithms',
        categoryName: 'ML Algorithms',
        question: 'What is a Random Forest?',
        answer: 'An ensemble learning method that combines multiple decision trees to improve accuracy and reduce overfitting. Each tree votes on the final prediction.'
    },
    {
        id: 26,
        category: 'algorithms',
        categoryName: 'ML Algorithms',
        question: 'What is Linear Regression?',
        answer: 'A regression algorithm that models the relationship between features and a continuous target variable using a straight line (or hyperplane in multiple dimensions).'
    },
    {
        id: 27,
        category: 'algorithms',
        categoryName: 'ML Algorithms',
        question: 'What is Logistic Regression?',
        answer: 'Despite its name, it\'s a classification algorithm that predicts the probability of an instance belonging to a particular class. Commonly used for binary classification.'
    },
    {
        id: 28,
        category: 'algorithms',
        categoryName: 'ML Algorithms',
        question: 'What is K-Means Clustering?',
        answer: 'An unsupervised learning algorithm that partitions data into K clusters by minimizing the distance between data points and their cluster centers.'
    },
    {
        id: 29,
        category: 'algorithms',
        categoryName: 'ML Algorithms',
        question: 'What is a Recommendation System?',
        answer: 'An algorithm that suggests items to users based on their preferences, behavior, or similarities to other users. Examples: Netflix movie recommendations, Amazon product suggestions.'
    },
    {
        id: 30,
        category: 'algorithms',
        categoryName: 'ML Algorithms',
        question: 'What is Anomaly Detection?',
        answer: 'The process of identifying unusual patterns or outliers in data that don\'t conform to expected behavior. Used in fraud detection, network security, and quality control.'
    },
    
    // Generative AI (31-40)
    {
        id: 31,
        category: 'genai',
        categoryName: 'Generative AI',
        question: 'What is Generative AI?',
        answer: 'AI systems that can create new content (text, images, code, audio) based on patterns learned from training data. Examples: ChatGPT, DALL-E, GitHub Copilot.'
    },
    {
        id: 32,
        category: 'genai',
        categoryName: 'Generative AI',
        question: 'What is a Foundation Model?',
        answer: 'A large AI model trained on vast amounts of data that can be adapted for many different tasks. Examples: GPT-4, BERT, Claude. Also called Large Language Models (LLMs) for text.'
    },
    {
        id: 33,
        category: 'genai',
        categoryName: 'Generative AI',
        question: 'What is Amazon Bedrock?',
        answer: 'AWS service that provides access to foundation models from various AI companies through a single API. Allows you to build generative AI applications without managing infrastructure.'
    },
    {
        id: 34,
        category: 'genai',
        categoryName: 'Generative AI',
        question: 'What is Prompt Engineering?',
        answer: 'The practice of crafting effective input prompts to get desired outputs from AI models. Involves writing clear instructions, providing examples, and structuring queries effectively.'
    },
    {
        id: 35,
        category: 'genai',
        categoryName: 'Generative AI',
        question: 'What is RAG (Retrieval Augmented Generation)?',
        answer: 'A technique that enhances AI responses by retrieving relevant information from external knowledge sources before generating an answer. Improves accuracy and reduces hallucinations.'
    },
    {
        id: 36,
        category: 'genai',
        categoryName: 'Generative AI',
        question: 'What is Fine-tuning?',
        answer: 'The process of taking a pre-trained model and training it further on specific data to adapt it for a particular task or domain.'
    },
    {
        id: 37,
        category: 'genai',
        categoryName: 'Generative AI',
        question: 'What is a Token?',
        answer: 'A unit of text that AI models process. Can be a word, part of a word, or punctuation. Models have token limits (e.g., 4,000 tokens = ~3,000 words).'
    },
    {
        id: 38,
        category: 'genai',
        categoryName: 'Generative AI',
        question: 'What is Model Hallucination?',
        answer: 'When an AI model generates false or nonsensical information that sounds plausible. The model "makes up" facts that weren\'t in its training data.'
    },
    {
        id: 39,
        category: 'genai',
        categoryName: 'Generative AI',
        question: 'What is Transfer Learning?',
        answer: 'Using knowledge gained from training on one task to improve performance on a different but related task. Saves time and computational resources.'
    },
    {
        id: 40,
        category: 'genai',
        categoryName: 'Generative AI',
        question: 'What is Zero-shot Learning?',
        answer: 'When a model can perform a task it wasn\'t explicitly trained for, using only its general knowledge and a prompt description of the task.'
    },
    
    // AWS Services (41-50)
    {
        id: 41,
        category: 'aws',
        categoryName: 'AWS AI Services',
        question: 'What is Amazon SageMaker?',
        answer: 'AWS\'s fully managed service for building, training, and deploying machine learning models at scale. Provides tools for the entire ML lifecycle.'
    },
    {
        id: 42,
        category: 'aws',
        categoryName: 'AWS AI Services',
        question: 'What is Amazon Rekognition?',
        answer: 'AWS service for image and video analysis. Can detect objects, faces, text, scenes, and activities. Used for content moderation, facial recognition, and celebrity detection.'
    },
    {
        id: 43,
        category: 'aws',
        categoryName: 'AWS AI Services',
        question: 'What is Amazon Comprehend?',
        answer: 'AWS natural language processing (NLP) service that extracts insights from text. Can perform sentiment analysis, entity recognition, language detection, and topic modeling.'
    },
    {
        id: 44,
        category: 'aws',
        categoryName: 'AWS AI Services',
        question: 'What is Amazon Lex?',
        answer: 'AWS service for building conversational interfaces (chatbots) using voice and text. Powers Amazon Alexa and can be integrated into applications.'
    },
    {
        id: 45,
        category: 'aws',
        categoryName: 'AWS AI Services',
        question: 'What is Amazon Polly?',
        answer: 'AWS text-to-speech service that converts text into lifelike speech. Supports multiple languages and voices for creating voice-enabled applications.'
    },
    {
        id: 46,
        category: 'aws',
        categoryName: 'AWS AI Services',
        question: 'What is Amazon Transcribe?',
        answer: 'AWS speech-to-text service that automatically converts audio to text. Supports real-time and batch transcription with speaker identification and custom vocabularies.'
    },
    {
        id: 47,
        category: 'aws',
        categoryName: 'AWS AI Services',
        question: 'What is Amazon Translate?',
        answer: 'AWS neural machine translation service that provides fast, high-quality language translation. Supports dozens of languages for real-time and batch translation.'
    },
    {
        id: 48,
        category: 'aws',
        categoryName: 'AWS AI Services',
        question: 'What is Amazon Forecast?',
        answer: 'AWS time-series forecasting service that uses ML to predict future values based on historical data. Used for demand planning, resource allocation, and financial forecasting.'
    },
    {
        id: 49,
        category: 'aws',
        categoryName: 'AWS AI Services',
        question: 'What is Amazon Personalize?',
        answer: 'AWS service for building personalized recommendation systems. Uses ML to deliver customized product, content, and search recommendations.'
    },
    {
        id: 50,
        category: 'aws',
        categoryName: 'AWS AI Services',
        question: 'What is Amazon Textract?',
        answer: 'AWS service that automatically extracts text, handwriting, and data from scanned documents. Goes beyond OCR by understanding forms and tables.'
    }
];
