# Master Thesis

This project contains a Python script `survey.py` that is used to conduct surveys and collect responses using LLMs. The script reads survey questions and scenarios from specified directories and can run in both dry run mode and with actual API calls to OpenAI.

## Purpose

The `survey.py` script is designed to create and manage LLM-surveys. It allows users to define questions, collect responses, and store the results for analysis. The script can calculate the required sample size for the survey based on the provided parameters.

## How to Run

To run the `survey.py` script, follow these steps:

1. Ensure you have Python installed on your system.
2. Install the required dependencies using:

```sh
pip install -r requirements.txt
```

3. Open a terminal or command prompt.
4. Navigate to the directory containing `survey.py`.
5. Run the script using the following command:

```sh
python survey.py
```

6. Follow the prompts to provide the necessary inputs:
    * Enter the directory containing the survey files (e.g., `survey1`, `survey2`, `survey3`).
    * Enter your OpenAI API key (or leave empty for dry run mode).
    * Enter the name of the output file to save the survey responses.

