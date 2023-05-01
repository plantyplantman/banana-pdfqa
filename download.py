import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


def download_model():
    """download model during docker image  setup
    so it will be in cache before app.py and http_api kick-in"""

    tokenizer = AutoTokenizer.from_pretrained(
        "TheBloke/medalpaca-13B-GPTQ-4bit")
    model = AutoModelForCausalLM.from_pretrained(
        "TheBloke/medalpaca-13B-GPTQ-4bit",
        device_map="auto",
        torch_dtype=torch.float16
    )

    pipeline(
        "question-answering", model=model, tokenizer=tokenizer
    )


if __name__ == "__main__":
    download_model()
