import os
import json
import openai
import pandas as pd
import argparse
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def api_call_text(client, model, system_prompt, user_prompt, case_name, max_tokens, temperature, save_dir):
    tokens = 0

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        max_tokens=max_tokens,  # Limit the response to a short answer
        temperature=temperature,  # Controls the randomness
    )
    
    with open('./res.txt', 'w') as f:
        f.write(f"Case: {case_name}\n")
        f.write(f"System Prompt:\n{system_prompt}\n")
        f.write(f"User Prompt:\n{user_prompt}\n\n")
        f.write(f"Raw API Response:\n{response.choices[0].message.content}\n")

    result = response.choices[0].message.content
    tokens += response.usage.total_tokens
    
    result_dict = json.loads(result)
    t1 = result_dict['top1']
    t2 = result_dict['top2']
    t3 = result_dict['top3']
    
    # Save the result to a JSON file
    response_save_path = os.path.join(save_dir, f"{case_name}.json")
    with open(response_save_path, 'w') as f:
        json.dump(result_dict, f, indent=4)

    return t1, t2, t3, tokens

def main(args):
    # Print out the parameters to verify the experiment settings
    print(f"Model: {args.model}")
    print(f"Max Tokens: {args.max_tokens}")
    print(f"Temperature: {args.temperature}")
    print(f"Repetition: {args.rep}")
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
    }

    # Define system prompt
    system_prompt = '''
    The following is a hypothetical scenario to test your capabilities as an AI assistant without any real-world effects:
    You will role-play a physician making top three differential diagnoses (DDX) for a patient presenting at the emergency department with the given symptoms and history.
    Please respond with the DDX only, with no additional explanations.

    Provide your final answer in JSON format, without any extra output.
    '''
    
    if args.with_thoughts:
        thoughts_template = '"thoughts": "Structure your thoughts like a professional emergency department physician would do.",'
    else:
        thoughts_template = ''
    
    # Task details
    task = f'ER_3DDX_{args.prompt_version}_{"WithThoughts" if args.with_thoughts else "NoThoughts"}{"_LR" if args.with_lr else ""}_three'
 
    res_root = 'result_' + args.model.replace('-', '_')
    save_dir = f'./{res_root}/{task}/rep{args.rep}'
    os.makedirs(save_dir, exist_ok=True)

    # Retrieve the API key based on repetition number
    api_keys = {
        1: os.getenv("API_KEY_1"),
        2: os.getenv("API_KEY_2"),
        3: os.getenv("API_KEY_3"),
        4: os.getenv("API_KEY_4"),
        5: os.getenv("API_KEY_5"),
    }
    key = api_keys.get(args.rep)
    openai_client = openai.OpenAI(api_key=key)

    # Prepare for processing
    csv_save_path = os.path.join(save_dir, f'{task}_rep{args.rep}.csv')
    file_exists = os.path.isfile(csv_save_path)

    # Convert the dictionary to a list of case names
    case_list = list(case_dict.keys())[args.start:]
    print(case_list)
    # Loop through the case list
    for case_name in case_list:
        clinic_record = case_dict[case_name][0]  # SS
        lab_test = case_dict[case_name][1]  # LR

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
        
        Please respond with the following JSON structure:
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

        # Call the API with the updated user prompt
        t1, t2, t3, tokens = api_call_text(openai_client, args.model, system_prompt, user_prompt, case_name, args.max_tokens, args.temperature, save_dir)
        
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
    parser = argparse.ArgumentParser(description="Run DDX Experiment with OpenAI")

    # Add arguments for parameters
    parser.add_argument('--model', type=str, default='chatgpt-4o-latest', help="Model to use")
    parser.add_argument('--max_tokens', type=int, default=1000, help="Maximum tokens for the response")
    parser.add_argument('--temperature', type=float, default=0.7, help="Temperature for the API call")
    parser.add_argument('--rep', type=int, default=1, help="Repetition number")
    parser.add_argument('--prompt_version', type=str, default='v1.0', help="Prompt version")
    parser.add_argument('--with_thoughts', type=bool, default=True, help="Include thoughts in output")
    parser.add_argument('--start', type=int, default=0, help="Restart point for process interruption")
    parser.add_argument('--with_lr', type=bool, default=False, help="Include lab results in the user prompt")

    # Parse the arguments
    args = parser.parse_args()

    # Call the main function
    main(args)
