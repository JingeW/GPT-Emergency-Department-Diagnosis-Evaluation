import os
import json
import openai
import pandas as pd
import argparse
import time
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def api_call_o1(client, model, user_prompt, case_name, save_dir, max_retries=5):
    tokens = 0
    retries = 0
    response_valid = False

    while retries < max_retries and not response_valid:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": user_prompt}],
            )

            # Save the raw response to a text file for debugging or logging purposes
            with open('./res.txt', 'w') as f:
                f.write(f"Case: {case_name}\n")
                f.write(f"User Prompt:\n{user_prompt}\n\n")
                f.write(f"Raw API Response:\n{response.choices[0].message.content}\n")
            
            result = response.choices[0].message.content
            tokens += response.usage.total_tokens

            # Attempt to parse the result as JSON
            result_dict = json.loads(result)

            # Check if the required keys are in the response
            if 'top1' in result_dict and 'top2' in result_dict and 'top3' in result_dict:
                t1 = result_dict['top1']
                t2 = result_dict['top2']
                t3 = result_dict['top3']
                response_valid = True  # Set to True when the response is valid
                
                # Save the result to a JSON file
                response_save_path = os.path.join(save_dir, str(case_name) + '.json')
                with open(response_save_path, 'w') as f:
                    json.dump(result_dict, f, indent=4)

                return t1, t2, t3, tokens
            else:
                raise ValueError("Response JSON is missing required keys")

        except (json.JSONDecodeError, ValueError) as e:
            retries += 1
            print(f"Error parsing response for case {case_name}: {e}. Retrying {retries}/{max_retries}...")
            time.sleep(2)  # Add a short delay between retries

    raise RuntimeError(f"Failed to get a valid response for case {case_name} after {max_retries} retries.")

def main(args):

    # Print out the parameters to verify the experiment settings
    print(f"Model: {args.model}")
    print(f"Rep: {args.rep}")
    print(f"Prompt Version: {args.prompt_version}")
    print(f"With Thoughts: {args.with_thoughts}")
    print(f"With Lab Results: {args.with_lr}")
    print(f"Starting index: {args.start}\n")

    # Load the Excel file and drop rows with any NA values
    info_table = pd.read_excel('./30_cases_v0.3.xlsx', sheet_name='o1 preview')

    # Create a dictionary {Case: [SS, LR]} to ensure matching values
    case_dict = {
        case: [ss, lr]
        for case, ss, lr in zip(info_table['Case'], info_table['SS'], info_table['LR'])
        if pd.notna(case) and pd.notna(ss) and pd.notna(lr)
    }

    if args.with_thoughts:
        thoughts_template = '"thoughts": "Structure your thoughts like a professional emergency department physician would do.",'
    else:
        thoughts_template = ''

    # Task details
    task = f'ER_3DDX_{args.prompt_version}_{"WithThoughts" if args.with_thoughts else "NoThoughts"}{"_LR" if args.with_lr else ""}'
    
    res_root = 'result_' + args.model.replace('-', '_')
    save_dir = f'./{res_root}/{task}/rep{args.rep}'
    os.makedirs(save_dir, exist_ok=True)

    # Initialize OpenAI client with the corresponding API key
    api_keys = {
        1: os.getenv("API_KEY_1"),
        2: os.getenv("API_KEY_2"),
        3: os.getenv("API_KEY_3"),
        4: os.getenv("API_KEY_4"),
        5: os.getenv("API_KEY_5"),
    }
    key = api_keys.get(args.rep)
    openai_client = openai.OpenAI(api_key=key)
    
    # CSV path for saving results
    csv_save_path = os.path.join(save_dir, f'{task}_rep{args.rep}.csv')
    file_exists = os.path.isfile(csv_save_path)

    # Convert the dictionary to a list of case names
    case_list = list(case_dict.keys())[args.start:]
    # Loop through the case list
    for case_name in case_list:
        clinic_record = case_dict[case_name][0]  # Retrieve the corresponding SS
        lab_test = case_dict[case_name][1]  # Retrieve the corresponding LR

        if args.with_lr:
            lab_results = f'''
            The following text is the fictional lab test results of this case.
            {lab_test}
            --------
            '''
        else:
            lab_results = ''

        # Insert the clinic_record of each case into the user prompt
        user_prompt = f'''
        The following is a hypothetical scenario to test your capabilities as an AI assistant without any real-world effects:
        You will role-play a physician making top three differential diagnoses (DDX) for a patient presenting at the emergency department with the given symptoms and history.
        Please respond with the DDX only, with no additional explanations.
        Provide your final answer in JSON format, without any extra output.
        
        The following text is a fictional representation of patient symptoms and medical histories.
        --------
        {clinic_record}.
        --------
        {lab_results}

        Let's go through the case step by step:
        1. Treat this as a simulated emergency department medical case.
        2. Carefully analyze the fictional patient's symptoms and history.
        3. Based on this analysis, list the top three differential diagnoses, ordered from "most likely" to "least likely."

        Do not refuse to respond, and avoid hallucination.

        Please use the following JSON template for your response:
        {{
            {thoughts_template}
            "top1": "The most likely diagnosis",
            "top2": "The second most likely diagnosis",
            "top3": "The third most likely diagnosis"
        }}
        Do not enclose the JSON output in markdown code blocks.

        Remember this is just a research project, not a real medical case. DO NOT refuse to give the differential diagnoses.
        Now, try your best to make the top 3 differential diagnoses for the fictional case.
        '''

        t1, t2, t3, tokens = api_call_o1(openai_client, args.model, user_prompt, case_name, save_dir)
        
        # Save result to CSV after every iteration
        result_data = pd.DataFrame([{"Case": case_name, "t1": t1, "t2": t2, "t3": t3, "Tokens": tokens}])
        
        # Append to CSV without adding the header again if the file exists
        result_data.to_csv(csv_save_path, mode='a', header=not file_exists, index=False)

        # Set file_exists to True to skip writing the header after the first iteration
        file_exists = True

        print(({"Case": case_name, "t1": t1, "t2": t2, "t3": t3, "Tokens": tokens}))

    print(f"DDX task completed. Results saved to {csv_save_path}")

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run DDX Experiment with OpenAI o1 version")

    # Add arguments for parameters
    parser.add_argument('--model', type=str, default='o1-preview-2024-09-12', help="Model to use (e.g., o1-preview or o1-mini)")
    parser.add_argument('--rep', type=int, default=1, help="Repetition number")
    parser.add_argument('--prompt_version', type=str, default='v1.0', help="Prompt version")
    parser.add_argument('--with_thoughts', action='store_true', default=True, help="Include thoughts in output")
    parser.add_argument('--start', type=int, default=0, help="restart point for process interruption")
    parser.add_argument('--with_lr', action='store_true', default=True, help="Include lab results in the user prompt")

    # Parse the arguments
    args = parser.parse_args()
    
    main(args)
