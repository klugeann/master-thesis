#!python3survey3

import openai
import random
import json
import os
import pandas as pd
import math
from statsmodels.stats.power import TTestIndPower


skippable_categories = ["00 Demographics", 
                        "11 Daily Internet Usage", 
                        "12 Frequency of Online Purchases",
                        "13 Money Spent for Online Purchases",
                        "14 Social Media Usage"]


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

def read_pairs_from_file(file_name):
    """
    Reads pairs of key-value from a text file.

    Args:
        file_name (str): The name of the text file.
    
    Returns:
        list: A list of pairs read from the file.
    """
    def make_pair(line):
        pair = [el.strip() for el in line.split(":", 2)]
        return ["", line] if len(pair) < 2 else pair

    lines = read_lines_from_text_file(file_name)
    list_of_pairs = [make_pair(line) for line in lines]

    return list_of_pairs


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
    list_of_pairs = read_pairs_from_file(file_name)

    # Combine list of pairs into a dictionry of keys and multiple values
    dictionary = {}
    for key, value in list_of_pairs:
        if key not in dictionary:
            dictionary[key] = [value]
        else:
            dictionary[key].append(value)
    return dictionary


def deduplicate_list(input_list):
    """
    Remove duplicates from a list while preserving the order of elements.

    Args:
        input_list (list): The list from which to remove duplicates.

    Returns:
        list: A new list with duplicates removed.
    """
    return list(dict.fromkeys(input_list))


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
            messages=self.messages,
            timeout=60
        )

        # Query the api
        reply = completion.choices[0].message.content

        # Keep track of the conversation state
        self.messages.append({"role": "assistant", "content": reply})

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
    # Query directory from user or use environment variable
    directory = os.getenv("SURVEY_DIRECTORY") or input("Enter the directory containing the survey files: ")

    # Query whether to use causal inference or not
    is_causal_inference_env = os.getenv("USE_CAUSAL_INFERENCE", "")
    is_causal_inference = is_causal_inference_env.lower() == "y" if is_causal_inference_env else input("Do you want to use causal inference? (y/N): ").lower() == "y"

    # Read general instruction pre-prompt for LLM
    if is_causal_inference:
        pre_prompt_template = read_text_from_file(f"{directory}/prompt_causal_inference.txt")

        # Load causal inference data
        causal_inference_data = pd.read_csv(os.getenv("CAUSAL_INFERENCE_DATA", "causal_inference_consumer_data.csv"))
    else:
        pre_prompt = read_text_from_file(f"{directory}/prompt.txt")

    # Read the three experimental scenarios
    scenarios = read_dictionary_from_file(f"{directory}/scenarios.txt")

    # Read survey questions (formatted for LLM)
    survey_questions = read_pairs_from_file(f"{directory}/questions.txt")
    survey_question_categories = deduplicate_list(map(lambda x: x[0], survey_questions))

    # Query if dry running
    dry_run_env = os.getenv("DRY_RUN", "")
    dry_run = dry_run_env.lower() == "y" if dry_run_env else input("Do you want to do a dry run? (y,N): ").lower() == "y"

    # Query OpenAI API config from user or use environment variables
    llm_config = {}

    if dry_run:
        print("Running in dry run mode. No API key provided")
    else:
        api_key = os.getenv("OPENAI_API_KEY") or input("Enter your OpenAI API key (or empty): ")

        if api_key:
            llm_config["api_key"] = api_key

        base_url = os.getenv("OPENAI_BASE_URL") or input("Enter your OpenAI base URL (or empty): ")
    
        if base_url:
            llm_config["base_url"] = base_url
        
        model = os.getenv("OPENAI_MODEL") or input("Enter your model: ")

        if model:
            llm_config["model"] = model
        

    # Query the number of trials from the user or use environment variable
    num_trials = int(os.getenv("NUM_TRIALS") or input("Enter the number of trials: "))
                     
    # Ask if the user wants to do a "within-subject" or "between-subject" survey
    within_subject_env = os.getenv("WITHIN_SUBJECT", "")
    within_subject = within_subject_env.lower() == "y" if within_subject_env else input("Do you want to do a within-subject survey? (y/N): ").lower() == "y"

    # Query output file name from user or use environment variable
    output_file = os.getenv("OUTPUT_FILE") or input("Enter the name of the output csv file (e.g. out.csv): ")

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
        
        scenario_id = 0

        # Run the survey for each scenario
        for scenario, scenario_text in zip(selected_scenario, scenario_texts):
            question_id = 1
            scenario_id += 1

            scenario_prompt = "Please answer the following questions based on the following new scenario: \n\n"    

            new_scenario_prompt = "Now we switch to another scenario." + \
                "You are still the same person as before with same identity, age, employment status, education and so on," + \
                f"but you are now in a different situation. {scenario_prompt}\n\n"

            if within_subject and scenario_id > 1:
                session.add_user_message(new_scenario_prompt + scenario_text)
            else:
                session.add_user_message(scenario_prompt + scenario_text)
            
            # Query the LLM for each survey question
            for category in survey_question_categories:
                if within_subject and scenario_id > 1 and category in skippable_categories:
                    # Skip demographics questions for within-subject design
                    continue

                for question in map(lambda x: x[1], filter(lambda x: x[0] == category, survey_questions)):
                    llm_response = session.query(question)

                    # Question id as a 3 digit number with the question
                    question_column = f"{question_id:03d} {question}"
                    question_id += 1

                    # Store the survey response in the DataFrame
                    rows.append({
                        "TrialId": (f"{trial_id}-{scenario_id:02d}" if within_subject else f"{trial_id}"),
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
