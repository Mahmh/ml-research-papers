# Chain-of-Thought Prompting Elicits Reasoning in Large Language Models
<img src="./thumbnail.png" alt="Before vs. After Using CoT">

This project explores the use of Chain-of-Thought (CoT) prompting to enhance reasoning capabilities in large language models. By leveraging CoT, models can break down complex problems into intermediate steps, improving performance on tasks requiring multi-step reasoning.

Paper details:
- Published: 2022-01
- Link: https://arxiv.org/abs/2201.11903

## Quick Start
Follow these steps to set up and run this project:

1. **Clone the repository**

    ```bash
    git clone https://github.com/Mahmh/ml-research-papers
    ```

2. **Create and activate a virtual environment**

    - **Unix / macOS**  
      ```bash
      python3 -m venv venv
      source venv/bin/activate
      pip install -e .
      cd cot-reasoning
      ```
    - **Windows (PowerShell)**  
      ```powershell
      python -m venv venv
      .\venv\Scripts\Activate.ps1
      pip install -e .
      cd cot-reasoning
      ```

3. **Install dependencies**
    ```bash
    pip install --upgrade pip
    pip install -r requirements.txt
    ```

4. **Launch the notebook**
    ```bash
    jupyter lab
    ```