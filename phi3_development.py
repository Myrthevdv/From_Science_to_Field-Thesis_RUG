import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import pandas as pd

# Set the random seed for reproducibility
torch.random.manual_seed(0)

#Setup the model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-128k-instruct",
    device_map="cuda",
    torch_dtype="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-128k-instruct")

# Read the Excel data file
development_df = pd.read_excel('Development_set.xlsx')
# Extract the text to be summarized
text_to_be_summarized = development_df["AIC"].iloc[-1]
# Define the prompt question
prompt_question = "\n\nCreate one concise and easily understandable summary without bullet points of a scientific article within the domain of of sports science. Ensure the summary is relevant for athletes of all levels and backgrounds, avoiding technical jargon, and highlight its implications for athletes"

# Initialize the pipeline for text generation
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)

# Define the arguments for text generation
generation_args = {
    "max_new_tokens": 1000,
    "return_full_text": False,
    "temperature": 0.0,
    "do_sample": False,
}

# Set the index for accessing the DataFrame, for selecting the article that is used as example in the few-shot prompt
index = 1


# Zero_shot prompt
zero_shot_prompt = [{"role": "user", "content":text_to_be_summarized + prompt_question}]
output_zero_shot = pipe(zero_shot_prompt, **generation_args)
print(output_zero_shot[0]['generated_text'])


# Article-top-summary

best_summary = ""
if development_df["Rank ChatGPT"].iloc[index] == 1:
    best_summary = development_df["Summary ChatGPT"].iloc[index]
elif development_df["Rank Cohere"].iloc[index] == 1:
    best_summary = development_df["Summary Cohere"].iloc[index]
else:
    best_summary = development_df["Summary Gimini"].iloc[index]

article_top_prompt = [
    {"role": "user", "content": development_df["AIC"].iloc[index] + prompt_question },
    {"role": "assistant", "content": best_summary},
    {"role": "user", "content":text_to_be_summarized + prompt_question}]

output_article_top = pipe(article_top_prompt, **generation_args)
print(output_article_top[0]['generated_text'])


#Article-all-summary


article_all_prompt = [
    {"role": "user", "content":
    "I rated three summaries relative to each other, giving the best summary a 1 and the worst summary a 3\n"
    "Relative rank: " + str(development_df["Rank ChatGPT"].iloc[index]) + "\n"
    "Summary: " + str(development_df["Summary ChatGPT"].iloc[index]) + "\n\n"
    "Relative rank: " + str(development_df["Rank Cohere"].iloc[index]) + "\n"
    "Summary: " + str(development_df["Summary Cohere"].iloc[index]) + "\n\n"
    "Relative rank: " + str(development_df["Rank Gimini"].iloc[index]) + "\n"
    "Summary: " + str(development_df["Summary Gimini"].iloc[index]) + "\n\n"
    },
    {"role": "user", "content":text_to_be_summarized + prompt_question}
]

output_article_all = pipe(article_all_prompt, **generation_args)
print(output_article_all[0]['generated_text'])


# Article-all-explanation-summary
question = (
    "I rated three summaries of a scientific article on sport science based on the following criteria on a scale of 1 (Very poor) till 5 (Excellent):\n"
    "1. Clarity and Accessibility: Evaluate if the summary is clear, readable, and understandable for athletes of all levels, avoiding technical jargon and complex sentences.\n"
    "2. Inclusion of Key Findings (Quality): Evaluate if the summary includes the most important findings from the scientific article related to sports injuries/sports science. Ensure there are no hallucinations and all information can be found in the source text.\n"
    "3. Relevance for Athletes: Evaluate if the findings stated in the summary are relevant for athletes, and if they can directly benefit athletes in their training, performance, or injury prevention efforts.\n\n"
    "Article " + development_df["AIC"].iloc[index] + "\n\n"
    "I rated the following summaries:\n"
    "Summary: " + str(development_df["Summary ChatGPT"].iloc[index]) + "\n"
    "Overall Relative Rank: " + str(development_df["Rank ChatGPT"].iloc[index]) + "\n"
    "Clarity and Accessibility Rank: " + str(development_df["Clarity and accessibility rank ChatGPT"].iloc[index]) + "\n"
    "Clarity and Accessibility Rank Explanation: " + str(development_df["Clarity and accessibility explanation ChatGPT"].iloc[index]) + "\n"
    "Inclusion of Key Findings (Quality) Rank: " + str(development_df["Inclusion of key findings rank ChatGPT"].iloc[index]) + "\n"
    "Inclusion of Key Findings (Quality) Rank Explanation: " + str(development_df["Inclusion of key findings explanation ChatGPT"].iloc[index]) + "\n"
    "Relevance for Athletes Rank: " + str(development_df["relevance for athletes rank ChatGPT"].iloc[index]) + "\n"
    "Relevance for Athletes Rank Explanation: " + str(development_df["relevance for athletes explanation ChatGPT"].iloc[index]) + "\n\n"
    "Summary: " + str(development_df["Summary Cohere"].iloc[index]) + "\n"
    "Overall Relative Rank: " + str(development_df["Rank Cohere"].iloc[index]) + "\n"
    "Clarity and Accessibility Rank: " + str(development_df["Clarity and accessibility rank Cohere"].iloc[index]) + "\n"
    "Clarity and Accessibility Rank Explanation: " + str(development_df["Clarity and accessibility explanation Cohere"].iloc[index]) + "\n"
    "Inclusion of Key Findings (Quality) Rank: " + str(development_df["Inclusion of key findings rank Cohere"].iloc[index]) + "\n"
    "Inclusion of Key Findings (Quality) Rank Explanation: " + str(development_df["Inclusion of key findings explanation Cohere"].iloc[index]) + "\n"
    "Relevance for Athletes Rank: " + str(development_df["relevance for athletes rank Cohere"].iloc[index]) + "\n"
    "Relevance for Athletes Rank Explanation: " + str(development_df["relevance for athletes explanation Cohere"].iloc[index]) + "\n\n"
    "Summary: " + str(development_df["Summary Gimini"].iloc[index]) + "\n"
    "Overall Relative Rank: " + str(development_df["Rank Gimini"].iloc[index]) + "\n"
    "Clarity and Accessibility Rank: " + str(development_df["Clarity and accessibility rank Gimini"].iloc[index]) + "\n"
    "Clarity and Accessibility Rank Explanation: " + str(development_df["Clarity and accessibility explanation Gimini"].iloc[index]) + "\n"
    "Inclusion of Key Findings (Quality) Rank: " + str(development_df["Inclusion of key findings rank Gimini"].iloc[index]) + "\n"
    "Inclusion of Key Findings (Quality) Rank Explanation: " + str(development_df["Inclusion of key findings explanation Gimini"].iloc[index]) + "\n"
    "Relevance for Athletes Rank: " + str(development_df["relevance for athletes rank Gimini"].iloc[index]) + "\n"
    "Relevance for Athletes Rank Explanation: " + str(development_df["relevance for athletes explanation Gimini"].iloc[index]) + "\n\n"
)

article_all_summary_explanation_prompt = [
    {"role": "user", "content": question},
    {"role": "user", "content": text_to_be_summarized + prompt_question}
]


output_article_all_summary_explanation = pipe(article_all_summary_explanation_prompt, **generation_args)
print(output_article_all_summary_explanation[0]['generated_text'])
