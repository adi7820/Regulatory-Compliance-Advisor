# Regulatory-Compliance-Advisor
GenAI based project

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

        

