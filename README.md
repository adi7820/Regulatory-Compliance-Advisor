# Regulatory-Compliance-Advisor
This project is dedicated to the development of an AI-powered advisory system that assists businesses in navigating the complex and ever-evolving landscape of industry regulations. The system is designed to offer actionable compliance guidance by leveraging a Retrieval-Augmented Generation (RAG) framework, ensuring that users stay informed about the latest regulatory changes.

## Key Features
### 1. Retrieval-Augmented Generation (RAG) Setup
Our system employs a RAG architecture to maintain an up-to-date understanding of regulatory changes. The system provides precise and relevant compliance guidance, helping businesses stay compliant across multiple jurisdictions.

### 2. Cost-Effectiveness and Proprietary Models
While proprietary models like those from OpenAI offer high performance, they can be cost-prohibitive, especially for continuous large-scale monitoring. Our solution addresses this by balancing performance with cost-effectiveness, making it viable for sustained operations in large-scale regulatory environments.

### 3. Data Privacy and Security
Our approach mitigates the risks associated with using proprietary models, ensuring that sensitive data is handled with the highest standards of security.

### 4. Domain-Specific Fine-Tuning
Generic responses can undermine the effectiveness of AI in specialized domains. To combat this, our system incorporates domain-specific fine-tuning, ensuring that the generated advice is both relevant and accurate for specific sectors like Finance, Healthcare, and Data Privacy.

### 5. Diverse Data Sources
Our system ingests information from a variety of sources, including:

- PDF Documents
- Wikipedia Pages
- Websites
This diverse input ensures a comprehensive understanding of regulatory landscapes across regions such as the US, European Union, and India.

## Technical Framework
### RAG Architecture
- Data Preparation:
  - Embeddings: OpenAI Embedding Model
  - VectorDB: PineCone
- Generation: GPT-4o-Mini
- Evaluation: RAGAS
- Deployment: Docker
- User Interface: Gradio
### Vector Database Configuration
- Chunk Size: 500
- Indices: 3
- Retrieval Search Algorithm: Similarity Search
- Retrieved Chunks: 20
- Generation Output Tokens: 512
- Temperature: 0

## Open-Source Pipeline
We also developed a secondary architecture based entirely on open-source technologies:
- Embedding Model: Stella
- VectorDB: Qdrant
- Generation Model: Llama 3.1 8B (finetuned and non-finetuned variants)
- Challenges: The finetuned Llama model produced suboptimal outputs, leading us to revert to the non-finetuned Llama 3.1 8B Instruct model for generation tasks.

### Prompt Data Format
- Context Chunks: 20 (each of 512 tokens)
- Relevant Chunks: 4 (random order)
- Components: User Query, Generated Answer

### Finetuning Process
- Model: Llama 3.1 8B
- Techniques: LoRA Finetuning, LoRA Adaptor merging
- Final Model: Merged using SLEPR with the Llama 3.1 8B Instruct model
###
## Run Using Docker Setup
  ### 1. Pull the Docker image:
  ```bash
  docker pull adi1710/rag-gradio-app:v2
  ```
  ### 2. Run the Docker Container
  ```bash
  docker run --gpus all -p 7860:7860 -e OPENAI_API_KEY=your_openai_api_key adi1710/rag-gradio-app:v2
  ```
  ### 3. Access the Application:
  Once the container is running, you can access the application through your web browser at `http://localhost:7860`.

        

