"""This file contains the code with which the LLM takes the test and its uncertainty is measured."""

# Generic libraries
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import pandas as pd
import itertools
from tqdm import tqdm
import argparse
import numpy as np
import os
import warnings
import logging
import random
import string

# Importing Transformers
# Set HF_home to the folder that has a lot of space. This has to be done before importing transformers.
os.environ['HF_HOME'] = '/scratch/' + \
    str(open('../tokens/HPC_ACCOUNT_ID.txt', 'r').read())
print("HF_HOME set to:", os.environ['HF_HOME'])

MODEL_NAME_DICT = {
    # Quantised models:
    # Small models:  [1-10 billion parameters]
    'phi3_5-chat': 'unsloth/Phi-3.5-mini-instruct-bnb-4bit',              # 2B
    'Llama3_2-3b-chat': 'unsloth/Llama-3.2-3B-Instruct-bnb-4bit',         # 3B
    'Qwen2_5-3b-chat': 'unsloth/Qwen2.5-3B-Instruct-bnb-4bit',            # 3B
    'Llama3_1-8b-chat': 'unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit',   # 8B

    # Medium-sized models: [10-60 billion parameters]
    'Qwen2_5-14b-chat': 'unsloth/Qwen2.5-14B-Instruct-bnb-4bit',          # 14B
    'Qwen2_5-32b-chat': 'unsloth/Qwen2.5-32B-Instruct-bnb-4bit',          # 32B
    'Yi-34b-chat': 'unsloth/yi-34b-chat-bnb-4bit',                       # 34B

    # Large models: [more than 60 billion parameters]
    'Llama3_1-70b-chat': 'unsloth/Meta-Llama-3.1-70B-Instruct-bnb-4bit',  # 70B
    'Qwen2_5-72b-chat': 'unsloth/Qwen2.5-72B-Instruct-bnb-4bit',          # 72B

    # Default-Precision models:
    # Small models:  [1-10 billion parameters]
    'phi3_5-chat_fp': 'microsoft/Phi-3.5-mini-instruct',                  # 2B
    'Llama3_2-3b-chat_fp': 'meta-llama/Llama-3.2-3B-Instruct',            # 3B
    'Qwen2_5-3b-chat_fp': 'Qwen/Qwen2.5-3B-Instruct',                     # 3B
    'Llama3_1-8b-chat_fp': 'meta-llama/Llama-3.1-8B-Instruct',           # 8B

    # Medium-sized models: [10-60 billion parameters]
    'Qwen2_5-14b-chat_fp': 'Qwen/Qwen2.5-14B-Instruct',                   # 14B
    'Qwen2_5-32b-chat_fp': 'Qwen/Qwen2.5-32B-Instruct',                   # 32B
    'Yi-34b-chat_fp': '01-ai/Yi-34B-Chat',                                # 34B

    # Large models: [more than 60 billion parameters]
    'Llama3_1-70b-chat_fp': 'meta-llama/Llama-3.1-70B-Instruct',          # 70B
    'Qwen2_5-72b-chat_fp': 'Qwen/Qwen2.5-72B-Instruct',                   # 72B
}

ROOT_DIR = "../"
TEST_MODE = False
DATASET = ""
MODEL_NAME = ""
PROMPT_STYLE = ""
WARNINGS = False
KEEP_OUTPUT_TEXT = True
OUTPUT_FILE_NAME = ""
HF_TOKEN = ""
NUM_CHOICE_PERMUTATIONS = 0


def parse_args():
    parser = argparse.ArgumentParser(
        description='Measure uncertainty of a model on the usmle_2024, Biopsychology or cmcqrd item difficulty dataset')
    parser.add_argument('-t', '--test_mode', action="store_true",
                        help='Test mode only uses 10 questions', default=False)
    parser.add_argument('-d', '--dataset', type=str,
                        help='Dataset to use ("usmle", "bio" or "cmcqrd")', required=True)
    parser.add_argument('-m', '--model', type=str,
                        help='Model Name', required=True)
    parser.add_argument('-p', '--prompt', type=int,
                        help='Which prompt formulation to use, 1 or 2', default=1)
    parser.add_argument('-w', '--warnings', action="store_true",
                        help='If enabled, certain warnings are shown', default=False)
    parser.add_argument('-kot', '--keep_output_text', action="store_true",
                        help='If enabled, the output text is kept in the output file', default=False)
    parser.add_argument('-np', '--number_permutations', type=int,
                        help='Number of choice permutations, defaults to 10. Use all combinations if there are fewer than the passed value', default=10)

    return dict(vars(parser.parse_args()))


def set_globals(args):
    output_file_suffix = ""
    if args['test_mode']:
        output_file_suffix += "_TEST"
    if args['prompt'] == 2:
        output_file_suffix += "_prompt_2"
    globals()["TEST_MODE"] = args['test_mode']
    globals()["DATASET"] = args['dataset']
    globals()["MODEL_NAME"] = MODEL_NAME_DICT[args['model']]
    globals()["PROMPT_STYLE"] = args['prompt']
    globals()["WARNINGS"] = args['warnings']
    globals()["KEEP_OUTPUT_TEXT"] = args['keep_output_text']
    globals()["NUM_CHOICE_PERMUTATIONS"] = args['number_permutations']
    globals()["OUTPUT_FILE_NAME"] = ROOT_DIR + "data/" + DATASET + \
        "/with_uncertainty/" + args['model'] + output_file_suffix + ".csv"
    globals()["HF_TOKEN"] = load_token(ROOT_DIR + "tokens/HF_TOKEN.txt")

    print("Running with specs:")
    print("Test mode: ", TEST_MODE)
    print("Dataset: ", DATASET)
    print("Model: ", MODEL_NAME)
    print("Prompt style: ", PROMPT_STYLE)
    print("Show Warnings: ", WARNINGS)
    print("Keep output text: ", KEEP_OUTPUT_TEXT)
    print("Output file: ", OUTPUT_FILE_NAME)
    print("Number of choice permutations: ", NUM_CHOICE_PERMUTATIONS)
    print("HF Token: ", "Token read!" if HF_TOKEN else "No token.")


def set_warnings():
    if not WARNINGS:
        warnings.filterwarnings("ignore", category=UserWarning)
        # Only show errors. Default is to show warnings and errors.
        os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
        # For tensorflow: Hides INFO messages
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        os.environ['POSSIBLE_USER_WARNINGS'] = 'off'
        logging.getLogger("lightning.pytorch.utilities.rank_zero").setLevel(
            logging.ERROR)
        logging.getLogger("lightning.pytorch.accelerators.cuda").setLevel(
            logging.ERROR)


def load_token(file_path):
    try:
        with open(file_path, 'r') as file:
            content = file.read()
        return str(content).strip()
    except FileNotFoundError:
        print(
            f"Warning: Trying to load token, but the file '{file_path}' does not exist.")
        return None


class Generator():
    def __init__(self):
        # The quantization config is ignored for models that are pre-quantized. A warning will pop up, but it can be ignored.
        if 'fp' in MODEL_NAME:
            self.model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME, device_map="auto", token=HF_TOKEN)
        else:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
            self.model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME, device_map="auto", token=HF_TOKEN, quantization_config=quantization_config)

        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME, padding_side="left", token=HF_TOKEN)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Figure out the token ids of the answer choices 'a' to 'j'
        answer_choices = 'abcdefghij'
        self.answer_choices_indices_dic = {}
        for choice in answer_choices:
            tokens = [choice.upper(), f' {choice.upper()}', f'{choice.upper()} ', f' {choice.upper()} ',
                      choice, f' {choice}', f'{choice} ', f' {choice} ']
            self.answer_choices_indices_dic[choice] = [
                x for x in self.tokenizer.convert_tokens_to_ids(tokens) if x is not None
            ]

    def calc_probabilities_for_prompt(self, question, num_of_choices):
        '''
        Given a question and a list of answer choices, returns the probabilities for each answer choice and the
        generated output text, if KEEP_OUTPUT_TEXT is True.

        Args:
            question (str): The question to be asked.
            num_of_choices (int): The number of answer choices.

        Returns:
            scaled_probs (list): The probabilities for each answer choice, scaled to sum to 1.
            output_text (str): The generated output text, if KEEP_OUTPUT_TEXT is True.        
        '''

        prompt = None
        # 1st, prepare the prompt
        if 'chat' in MODEL_NAME:  # Chat model
            system_message = "As a student taking an exam, you are asked a multiple-choice question."

            chat = [{"role": "system", "content": system_message},
                    {"role": "user", "content": question}]

            prompt = self.tokenizer.apply_chat_template(
                chat, tokenize=False, add_generation_prompt=True, return_tensors="pt")
        else:  # Non-chat model
            prompt = question

        # 2nd, prepare the model inputs
        model_inputs = self.tokenizer(
            prompt, return_tensors="pt", padding=True).to("cuda")

        # 3rd, generate the output
        output = self.model.generate(**model_inputs, max_new_tokens=10, do_sample=False,
                                     output_scores=True, num_return_sequences=1, return_dict_in_generate=True,
                                     renormalize_logits=False, pad_token_id=self.tokenizer.eos_token_id)

        # 4th, get the probabilities for the next token
        next_token_candidates_tensor = output.scores[0][0]
        # next_token_candidates_tensor holds the probability of each vocabulary token being the next token
        # convert the logits to probabilities
        probabilities = torch.nn.functional.softmax(
            next_token_candidates_tensor, dim=0)

        # 5th, get the probabilities for the answer choices, and scale them to 0-1
        choices_letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
        unscaled_probs = [0] * num_of_choices
        for choice_idx in range(num_of_choices):
            # gather the probabilities for all relevant tokens for this choice, e.g., "A", 'a', ' A'
            probs_for_choice = [probabilities[idx].item(
            ) for idx in self.answer_choices_indices_dic[choices_letters[choice_idx]]]
            # pick the max probability for that choice
            unscaled_probs[choice_idx] = max(probs_for_choice)

        # Scale the probabilities to sum to 1
        total = sum(unscaled_probs)
        scaled_probs = [p / total for p in unscaled_probs]

        # 6th, decode the output and store that as well
        output_text = ""
        if KEEP_OUTPUT_TEXT:
            output_text = self.tokenizer.decode(
                output.sequences[0], skip_special_tokens=True)

        return scaled_probs, output_text


def format_question_string(question, choices):
    """Given a question and a list of answer choices, returns a string with the question and the answer choices"""
    prompt = ""
    question_string = question
    for i in range(len(choices)):
        question_string += f" \n{string.ascii_uppercase[i]}: {choices[i]}"

    if PROMPT_STYLE == 1:
        prompt = f"""Below is a multiple-choice question. Choose the letter which best answers the question. Keep your response as brief as possible; just state the letter corresponding to your answer with no explanation.

        Question:

        {question_string}

        Response:\n
        """
    elif PROMPT_STYLE == 2:
        prompt = f"""You will be presented with a multiple-choice question. Select the option letter that you believe provides the best answer to the question. Keep your response concise by simply stating the letter of your chosen answer without providing any additional explanation.

        Question:

        {question_string}

        Response:\n
        """
    return prompt


def calculate_item_uncertainty(row, generator):
    ''' Returns: first_token_probability, order_probability, full_answer, full_prompt, model_is_correct'''
    def get_prompt_permutations(question, choices):
        # Returns a list of all possible questions with the different permutations of the answer choices
        # list of permutations and for each, the original order of the answer choices
        prompts = []
        original_order = []
        for perm in itertools.permutations(choices):
            prompts.append(format_question_string(question, perm))
            original_order.append([choices.index(p) for p in perm])
        return prompts, original_order

    # 1. Prepare the prompt
    question = row['Question']
    choices = [row['Answer_A'], row['Answer_B'], row['Answer_C'], row['Answer_D'], row['Answer_E'],
               row['Answer_F'], row['Answer_G'], row['Answer_H'], row['Answer_I'], row['Answer_J']]
    # convert all to strings
    choices = [str(c) for c in choices]
    # Remove empty choices
    choices = [c for c in choices if c != 'nan']

    # 2. Prepare the permutations and sample NUM_CHOICE_PERMUTATIONS of them
    prompts, original_order = get_prompt_permutations(question, choices)
    # Only keep NUM_CHOICE_PERMUTATIONS random permutations, otherwise it takes too long. If there are fewer permutations, keep all.
    combinations_to_keep = min(NUM_CHOICE_PERMUTATIONS, len(prompts))
    random_indices = random.sample(range(len(prompts)), combinations_to_keep)
    prompts = [prompts[i] for i in random_indices]
    original_order = [original_order[i] for i in random_indices]

    # 3. Prepare stored variables
    probs = []  # needed?!
    eventual_choices = []  # needed?!
    std_eventual_choices_order = []  # needed?!
    probability_eventual_choices_order = []  # needed?!

    full_answer = ""  # in case of permutations, only the 1st one is kept
    full_prompt = ""  # in case of permutations, only the 1st one is kept
    model_is_correct = False

    # 4. For each permutation, calculate the probabilities
    for i, p in enumerate(tqdm(prompts, desc="Calculating probabilities", leave=False)):
        order_for_this_prompt = original_order[i]
        probs_not_in_original_order, full_answer_temp = generator.calc_probabilities_for_prompt(
            p, len(choices))
        if full_answer == "":  # Only the first generated answer is kept
            full_answer = full_answer_temp
        # original_order is a list of the original order of the answer choices, e.g., [1,3,0,2]
        # probs_not_in_original_order is a list of probabilities, e.g., [0.5, 0.32, 0.53, 0.24]
        # reorder the probabilities to match the original order of the answer choices. 0.53, 0.5, 0.24, 0.32
        reordered_probs = [0] * len(probs_not_in_original_order)
        reordered_eventual_choices = [0] * len(reordered_probs)
        for j, idx in enumerate(order_for_this_prompt):
            reordered_probs[idx] = probs_not_in_original_order[j]
        # based on reordered_probs, the eventual_choices is just the maximum element turned to one. For example, from [0.4, 0.3, 0.2, 0.1] to [1, 0, 0, 0]
        reordered_eventual_choices[reordered_probs.index(
            max(reordered_probs))] = 1
        # Store the results of this ordering
        eventual_choices.append(reordered_eventual_choices)
        probs.append(reordered_probs)

    # 5. Average over orderings
    # calculate the standard deviation for each answer choice
    # std_eventual_choices_order = [np.std(x) for x in zip(*eventual_choices)]
    probability_eventual_choices_order = [
        np.mean(x) for x in zip(*eventual_choices)]
    # calculate the average probability for each answer choice
    probs = [sum(x) / len(probs) for x in zip(*probs)]

    # 6. Only keep the ones corresponding to the correct answer
    # Get the index of the correct answer, (e.g., A->0, B->1, C->2)
    index = string.ascii_lowercase.index(row['Answer_Key'].lower())
    first_token_probability = probs[index]
    order_probability = probability_eventual_choices_order[index]

    # 7. Check if the model chose the correct answer
    if first_token_probability == max(probs):
        model_is_correct = True

    # 8. Extract the probabilities of the selsected choice (not only the probability of the correct answer)
    first_token_probability_selected_choice = max(probs)
    order_probability_selected_choice = max(probability_eventual_choices_order)

    # 9. Also extract the prompt
    full_prompt = prompts[0]  # Only the first prompt is kept

    return first_token_probability, order_probability, first_token_probability_selected_choice, order_probability_selected_choice, full_answer, full_prompt, model_is_correct


def generate_uncertainty_for_set(generator, questions_set):
    """Given a set of questions, calculates the uncertainty metrics for each question and adds them to the dataframe"""
    for index, row in tqdm(list(questions_set.iterrows()), total=len(questions_set)):
        # first_token_probability is the probability of token corresponding to the correct answer
        # order_probability is the probability of the correct answer being chosen across orderings
        # full_answer is the generated answer text
        # full_prompt is the full prompt text
        # model_is_correct is a boolean indicating if the model chose the correct answer, based on highest 1st token probability across orderings
        first_token_probability, order_probability, first_token_probability_selected_choice, order_probability_selected_choice, full_answer, full_prompt, model_is_correct = calculate_item_uncertainty(
            row, generator)

        # Store results in the original dataframe
        questions_set.at[index,
                         'first_token_probability'] = first_token_probability

        questions_set.at[index, 'order_probability'] = order_probability

        questions_set.at[index,
                         'first_token_probability_selected_choice'] = first_token_probability_selected_choice

        questions_set.at[index,
                         'order_probability_selected_choice'] = order_probability_selected_choice

        questions_set.at[index, 'full_answer'] = full_answer

        questions_set.at[index, 'full_prompt'] = full_prompt

        questions_set.at[index, 'model_is_correct'] = model_is_correct

    return questions_set  # This now also contains the uncertainty metrics


# Main function
if __name__ == "__main__":
    # 1. Setup
    print("---------------SETTING UP---------------")

    set_globals(parse_args())
    set_warnings()

    files_to_process = [ROOT_DIR + 'data/' + DATASET + '/raw/train.csv',
                        ROOT_DIR + 'data/' + DATASET + '/raw/test.csv']

    # 2. Load the LLM
    print("----------INITIALISING GENERATOR MODEL---------")
    generator = Generator()

    # 3. Calculate uncertainty metrics for each dataset
    for dataset in files_to_process:
        print("--------------EXECUTING INFERENCE--------------")
        print("Calculating uncertainty for ", dataset)
        questions_set = pd.read_csv(dataset)
        if TEST_MODE:
            # Keep only the first 10 items
            questions_set = questions_set.head(10)

        # Calculate model uncertainty for the dataset
        modified_set = generate_uncertainty_for_set(generator, questions_set)

        print("---------------EXPORTING RESULTS---------------")
        # Add 'train' or 'test' to the output file name
        suffix = '_train_set' if 'train' in dataset else '_test_set'
        # Construct the output file name
        output_file_name = OUTPUT_FILE_NAME.replace(".csv", f"{suffix}.csv")
        # Save the modified set to the output file
        modified_set.to_csv(output_file_name, index=False)
        print('Results saved to ', output_file_name)

    # 4. Done
    print("---------------------DONE!---------------------")
