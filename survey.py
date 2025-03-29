#!python3

import openai
import random
import json
import os
import pandas as pd
import math
from statsmodels.stats.power import TTestIndPower


def read_lines_from_text_file(file_name):
    """
    Read lines from a text file and return a list of non-empty lines.

    Args:
        file_name (str): The name of the text file to read.

    Returns:
        list: A list of non-empty lines from the text file.
    """
    # Make file name relative to the script's location
    file_name = os.path.join(os.path.dirname(__file__), file_name)

    with open(file_name, "r") as open_file:
        stripped_lines = [line.strip() for line in open_file]
        return [line for line in stripped_lines if line]


def read_text_from_file(file_name):
    """
    Read the contents of a text file and return them as a single string.

    Args:
        file_name (str): The name of the text file to read.

    Returns:
        str: The contents of the text file as a single string.
    """
    lines = read_lines_from_text_file(file_name)
    return " ".join(lines)


def read_dictionary_from_file(file_name):
    """
    Reads a dictionary from a text file.

    Args:
        file_name (str): The name of the text file.

    Returns:
        dict: The dictionary read from the file.

    File Format:
        The text file should have one key-value pair per line, separated by a colon (:).
        Example:
        key1:value1
        key2:value2
        ...
    """
    def make_pair(line):
        pair = [el.strip() for el in line.split(":", 2)]
        return ["", line] if len(pair) < 2 else pair

    lines = read_lines_from_text_file(file_name)
    list_of_pairs = [make_pair(line) for line in lines]

    # Combine list of pairs into a dictionry of keys and multiple values
    dictionary = {}
    for key, value in list_of_pairs:
        if key not in dictionary:
            dictionary[key] = [value]
        else:
            dictionary[key].append(value)
    return dictionary


# A LLM session keeping track of the state of the conversation
class LLMSession:

    def __init__(self, pre_prompt, config):
        self.pre_prompt = pre_prompt
        self.config = config or {}
        self.client = openai.OpenAI(
            api_key = self.config.get("api_key"),
            base_url = self.config.get("base_url")
        )
        self.messages = [
            {"role": "system", "content": pre_prompt}
        ]

    def query(self, prompt):
        """
        Query the OpenAI API with the given prompt and return the response.
        """

        # Add prompt to the conversation state
        self.messages.append({"role": "user", "content": prompt})

        # Query the api
        completion = self.client.chat.completions.create(
            model=self.config.get("model"),
            messages=self.messages
        )

        # Query the api
        reply = completion.choices[0].message.content

        # Keep track of the conversation state
        self.messages.append({"role": "assistent", "content": reply})

        return reply
    
    def add_user_message(self, message):
        self.messages.append({"role": "user", "content": message})


# A debug session for testing the conversation
class DebugSession:

    prefix = "DEBUG_SESSION - "

    def __init__(self, pre_prompt):
        print(f"{DebugSession.prefix}Running new debug session")
        print(f"{DebugSession.prefix}* System: {pre_prompt}")
        pass

    def query(self, prompt):
        print(f"{DebugSession.prefix}* User: {prompt}")
        print(f"{DebugSession.prefix}* Assistant: [DRY_RUN]")
        return "[DRY_RUN]"
    
    def add_user_message(self, message):
        print(f"{DebugSession.prefix}* User: {message}")



def find_sample_size(effect_size, alpha, power, num_groups):
    """
    Calculate the sample size needed for a given effect size, alpha, power, and number of groups.

    Parameters:
    effect_size (float): The effect size.
    alpha (float): The significance level.
    power (float): The desired power.
    num_groups (int): The number of groups.

    Returns:
    int: The sample size needed for each group (rounded up).
    """
    power_analysis = TTestIndPower()
    sample_size = power_analysis.solve_power(effect_size=effect_size, alpha=alpha, power=power, nobs1=None, ratio=1.0, alternative="two-sided")

    return math.ceil(sample_size)


if __name__ == "__main__":
    # Random seed for reproducibility
    random.seed(42)

    # Query directory from user
    directory = input("Enter the directory containing the survey files: ")

    # Query whether to use causal inference or not
    is_causal_inference = input("Do you want to use causal inference? (y/N): ").lower() == "y"

    # Read general instruction pre-prompt for LLM
    if is_causal_inference:
        pre_prompt_template = read_text_from_file(f"{directory}/prompt_causal_inference.txt")

        # Load causal inference data
        causal_inference_data = pd.read_csv("causal_inference_consumer_data.csv")
    else:
        pre_prompt = read_text_from_file(f"{directory}/prompt.txt")

    # Read the three experimental scenarios
    scenarios = read_dictionary_from_file(f"{directory}/scenarios.txt")

    # Read survey questions (formatted for LLM)
    survey_questions = read_dictionary_from_file(f"{directory}/questions.txt")
    survey_question_categories = sorted(list(survey_questions.keys()))

    # Query if dry running
    dry_run = input("Do you want to do a dry run? (y,N): ").lower() == "y"

    # Query OpenAI API config from user
    llm_config = {}

    if dry_run:
        print("Running in dry run mode. No API key provided")
    else:
        api_key = input("Enter your OpenAI API key (or empty): ")

        if api_key:
            llm_config["api_key"] = api_key

        base_url = input("Enter your OpenAI base URL (or empty): ")
    
        if base_url:
            llm_config["base_url"] = base_url
        
        model = input("Enter your model: ")

        if model:
            llm_config["model"] = model
        

    # Query the number of trials from the user
    num_trials = int(input("Enter the number of trials: "))
                     
    # Ask if the user wants to do a "within-subject" or "between-subject" survey
    within_subject = input("Do you want to do a within-subject survey? (y/N): ").lower() == "y"

    # Query output file name from user
    output_file = input("Enter the name of the output csv file (e.g. out.csv): ")

    # Add csv extension if not provided
    if not output_file.endswith(".csv"):
        output_file += ".csv"

    # Add results directory if not provided
    if not output_file.startswith("results/") and not output_file.startswith("/") and not output_file.startswith("."):
        output_file = f"results/{output_file}"

    print(f"Sample size calculation: {num_trials} trials needed for the survey.")

    # Create empty array for dataset
    rows = []

    # Run trials (adjust as needed)
    for trial_id in range(num_trials):

        # Randomly select a scenario
        
        if within_subject:
            # Shuffle randomly the scenarios
            selected_scenario = random.sample(list(scenarios.keys()), len(scenarios.keys()))
        else:
            selected_scenario = [random.choice(list(scenarios.keys()))]
        
        scenario_texts = [scenarios[scenario][0] for scenario in selected_scenario]

        print(f"{trial_id + 1}/{num_trials} Running LLM survey for: {selected_scenario}")

        question_id = 1

        if is_causal_inference:
            # Choose randomly one row from causal inference data
            row = causal_inference_data.sample(n=1, random_state=trial_id)

            # Inject causal inference data into the pre-prompt
            # In the prompt, we use the column names as placeholders
            pre_prompt = pre_prompt_template.format(
                **row.to_dict(orient="records")[0]
            )

        if dry_run:
            session = DebugSession(pre_prompt)
        else:
            session = LLMSession(pre_prompt, llm_config)

        # Run the survey for each scenario
        for scenario, scenario_text in zip(selected_scenario, scenario_texts):
            session.add_user_message("Now we switch to the following scenario: " + scenario_text)

            # Query the LLM for each survey question
            for category in survey_question_categories:
                for question in survey_questions[category]:

                    # If no API key provided, skip API call
                    llm_response = session.query(question)

                    # Question id as a 3 digit number with the question
                    question_column = f"{question_id:03d} {question}"
                    question_id += 1

                    # Store the survey response in the DataFrame
                    rows.append({
                        "TrialId": trial_id,
                        "Scenario": scenario,
                        "Category": category,
                        "Question": question_column,
                        "Response": llm_response
                    })

    # Convert the list of rows to a DataFrame
    df = pd.DataFrame(rows, columns=["TrialId", "Scenario", "Category", "Question", "Response"])

    # Put questions as columns
    df = df.pivot_table(index=["TrialId", "Scenario"], columns=["Question", "Category"], values="Response", aggfunc="first").reset_index()

    # Save the survey responses to a CSV file with columns as trials and rows as questions
    df.to_csv(output_file, index=False)

    print(f"LLM survey responses saved to '{output_file}'.")
