Cell 1

!pip install sagemaker --quiet --upgrade --force-reinstall
!pip install ipywidgets --quiet

model_id, model_version, = (
    "huggingface-llm-falcon-7b-instruct-bf16",
    "*",
)

Cell 2

import ipywidgets as widgets

model_ids = [
    "huggingface-llm-falcon-40b-bf16",
    "huggingface-llm-falcon-40b-instruct-bf16",
    "huggingface-llm-falcon-7b-bf16",
    "huggingface-llm-falcon-7b-instruct-bf16",
]

Cell 3

# display the model-ids in a dropdown to select a model for inference.
model_dropdown = widgets.Dropdown(
    options=model_ids,
    value=model_id,
    description="Select a model",
    style={"description_width": "initial"},
    layout={"width": "max-content"},
)
display(model_dropdown)

Cell 4

model_id = model_dropdown.value

Cell 5

%%time
from sagemaker.jumpstart.model import JumpStartModel

my_model = JumpStartModel(model_id=model_id)
predictor = my_model.deploy()

Cell 6

%%time
from sagemaker.jumpstart.model import JumpStartModel

my_model = JumpStartModel(model_id=model_id)
predictor = my_model.deploy()

Cell 7

%%time


prompt = "Tell me about Amazon SageMaker."

payload = {
    "inputs": prompt,
    "parameters": {
        "do_sample": True,
        "top_p": 0.9,
        "temperature": 0.8,
        "max_new_tokens": 1024,
        "stop": ["<|endoftext|>", "</s>"],
    },
}

response = predictor.predict(payload)
print(response[0]["generated_text"])

Cell 8

def query_endpoint(payload):
    """Query endpoint and print the response"""
    response = predictor.predict(payload)
    print(f"\033[1m Input:\033[0m {payload['inputs']}")
    print(f"\033[1m Output:\033[0m {response[0]['generated_text']}")

Cell 9

# Code generation
payload = {
    "inputs": "Write a program to compute factorial in python:",
    "parameters": {"max_new_tokens": 200},
}
query_endpoint(payload)

Cell 10

payload = {
    "inputs": "Building a website can be done in 10 simple steps:",
    "parameters": {"max_new_tokens": 110, "no_repeat_ngram_size": 3},
}
query_endpoint(payload)

Cell 11

# Translation
payload = {
    "inputs": """Translate English to French:

    sea otter => loutre de mer

    peppermint => menthe poivrée

    plush girafe => girafe peluche

    cheese =>""",
    "parameters": {"max_new_tokens": 3},
}

query_endpoint(payload)

Cell 12

# Sentiment-analysis
payload = {
    "inputs": """"I hate it when my phone battery dies."
                Sentiment: Negative
                ###
                Tweet: "My day has been :+1:"
                Sentiment: Positive
                ###
                Tweet: "This is the link to the article"
                Sentiment: Neutral
                ###
                Tweet: "This new music video was incredibile"
                Sentiment:""",
    "parameters": {"max_new_tokens": 2},
}
query_endpoint(payload)

Cell 13

# Question answering
payload = {
    "inputs": "Could you remind me when was the C programming language invented?",
    "parameters": {"max_new_tokens": 50},
}
query_endpoint(payload)

Cell 14

# Recipe generation
payload = {
    "inputs": "What is the recipe for a delicious lemon cheesecake?",
    "parameters": {"max_new_tokens": 400},
}
query_endpoint(payload)

Cell 15

# Summarization

payload = {
    "inputs": """Starting today, the state-of-the-art Falcon 40B foundation model from Technology
    Innovation Institute (TII) is available on Amazon SageMaker JumpStart, SageMaker's machine learning (ML) hub
    that offers pre-trained models, built-in algorithms, and pre-built solution templates to help you quickly get
    started with ML. You can deploy and use this Falcon LLM with a few clicks in SageMaker Studio or
    programmatically through the SageMaker Python SDK.
    Falcon 40B is a 40-billion-parameter large language model (LLM) available under the Apache 2.0 license that
    ranked #1 in Hugging Face Open LLM leaderboard, which tracks, ranks, and evaluates LLMs across multiple
    benchmarks to identify top performing models. Since its release in May 2023, Falcon 40B has demonstrated
    exceptional performance without specialized fine-tuning. To make it easier for customers to access this
    state-of-the-art model, AWS has made Falcon 40B available to customers via Amazon SageMaker JumpStart.
    Now customers can quickly and easily deploy their own Falcon 40B model and customize it to fit their specific
    needs for applications such as translation, question answering, and summarizing information.
    Falcon 40B are generally available today through Amazon SageMaker JumpStart in US East (Ohio),
    US East (N. Virginia), US West (Oregon), Asia Pacific (Tokyo), Asia Pacific (Seoul), Asia Pacific (Mumbai),
    Europe (London), Europe (Frankfurt), Europe (Ireland), and Canada (Central),
    with availability in additional AWS Regions coming soon. To learn how to use this new feature,
    please see SageMaker JumpStart documentation, the Introduction to SageMaker JumpStart –
    Text Generation with Falcon LLMs example notebook, and the blog Technology Innovation Institute trainsthe
    state-of-the-art Falcon LLM 40B foundation model on Amazon SageMaker. Summarize the article above:""",
    "parameters": {"max_new_tokens": 200},
}
query_endpoint(payload)

Cell 16

max_new_tokens = 1000
max_new_tokens_single_iteration = 100

payload = {
    "inputs": "List down all the services by Amazon and a detailed description of each of the service. Tell me how to use Kendra. Tell me how to use AWS. Recite the guide to get started with SageMaker?",
    "parameters": {"max_new_tokens": max_new_tokens_single_iteration},
}

print(f"Input Text: {payload['inputs']}")

for i, _ in enumerate(range(0, max_new_tokens, max_new_tokens_single_iteration)):
    response = predictor.predict(payload)
    generated_text = response[0]["generated_text"]
    full_text = payload["inputs"] + generated_text
    print(f"\033[1mIteration {i+1}:\033[0m\n {generated_text}\n")
    payload["inputs"] = full_text

Cell 17

# Delete the SageMaker endpoint
predictor.delete_model()
predictor.delete_endpoint()
