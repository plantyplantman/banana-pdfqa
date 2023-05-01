import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from langchain import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from decorators import logger, timeit


@timeit
def init():
    global model
    global qa_chain

    logger.info("initializing medalpaca-13B-GPTQ-4bit...")

    tokenizer = AutoTokenizer.from_pretrained(
        "TheBloke/medalpaca-13B-GPTQ-4bit")
    device = 0 if torch.cuda.is_available() else -1
    model = AutoModelForCausalLM.from_pretrained(
        "TheBloke/medalpaca-13B-GPTQ-4bit",
        device_map="auto",
        torch_dtype=torch.float16
    )

    model = pipeline(
        "question-answering", model=model,
        tokenizer=tokenizer,
        device=device
    )

    logger.info("initializing medalpaca-13B-GPTQ-4bit complete!")


@timeit
def inference(model_inputs: dict) -> dict:
    global model

    context = model_inputs.get("context", None)
    question = model_inputs.get("question", None)

    if not all([context, question]):
        return {
            "msg": "question and context are required!"
        }

    result = model({"question": question, "context": context})
    return result
