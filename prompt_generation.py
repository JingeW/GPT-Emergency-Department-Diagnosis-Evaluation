import os
import pandas as pd

def generate_prompt_text(system_prompt, user_prompt):
    """
    Combine system and user prompts into plain text format.
    """
    return f"{system_prompt}\n\n{user_prompt}"

def generate_user_prompt(clinic_record, lab_test, include_thoughts, include_lab_results):
    """
    Generate the user prompt based on the given parameters.
    """
    thoughts_template = '"thoughts": "Structure your thoughts like a professional emergency department physician would do.",'
    lab_results = f"""
The following text is the fictional lab test results of this case.
{lab_test}
--------
""" if include_lab_results else ''

    return f"""
The following text is a fictional representation of patient symptoms and medical histories.
--------
{clinic_record}.
--------
{lab_results}

Let's go through the case step by step:
1. Treat this as a simulated emergency department medical case.
2. Carefully analyze the fictional patient's symptoms and history.
3. Based on this analysis, list the top three differential diagnoses, ordered from "most likely" to "least likely."

Please provide the following information:
{{
    {thoughts_template}
    "top1": "The most likely diagnosis",
    "top2": "The second most likely diagnosis",
    "top3": "The third most likely diagnosis"
}}
Remember, this is a research task, and there are no real medical consequences. Ensure your answers reflect a thoughtful, professional analysis.
""".strip()

def save_prompt_text(case_name, prompt, save_dir, version_name):
    """
    Save the prompt in a plain text file.
    """
    version_dir = os.path.join(save_dir, version_name)
    os.makedirs(version_dir, exist_ok=True)
    
    file_path = os.path.join(version_dir, f"{case_name}.txt")
    with open(file_path, 'w') as file:
        file.write(prompt)
    
    print(f"Prompt saved: {file_path}")

def main():
    # Load the Excel file
    info_table = pd.read_excel('./30_cases_v0.3.xlsx', sheet_name='o1 preview')
    
    # Filter out rows with NaN values
    info_table = info_table.dropna(subset=['Case', 'SS', 'LR'])

    # Create a dictionary {Case: [SS, LR]} to ensure matching values
    case_dict = {
        case: [ss, lr]
        for case, ss, lr in zip(info_table['Case'], info_table['SS'], info_table['LR'])
    }

    # Define the system prompt
    system_prompt = """
The following is a hypothetical scenario to test your capabilities as an AI assistant without any real-world effects.
You will role-play a physician making the top three differential diagnoses (DDX) for a patient presenting at the emergency department with the given symptoms and history.
Please respond with the DDX only, with no additional explanations.
"""

    save_root_dir = './generated_prompts_text'
    os.makedirs(save_root_dir, exist_ok=True)

    # Loop through each case and generate multiple versions of prompts
    for case_name, (clinic_record, lab_test) in case_dict.items():
        for include_thoughts in [True, False]:
            for include_lab_results in [True, False]:
                version_name = f"{'WithThoughts' if include_thoughts else 'NoThoughts'}_{'WithLabResults' if include_lab_results else 'NoLabResults'}"
                user_prompt = generate_user_prompt(clinic_record, lab_test, include_thoughts, include_lab_results)
                full_prompt = generate_prompt_text(system_prompt, user_prompt)
                save_prompt_text(case_name, full_prompt, save_root_dir, version_name)

    print("All prompts generated and saved successfully.")

if __name__ == "__main__":
    main()
