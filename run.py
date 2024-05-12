import os
import openai
openai.api_key = ""
import json
from raptor import RetrievalAugmentation 
from raptor import BaseSummarizationModel, BaseQAModel, BaseEmbeddingModel, RetrievalAugmentationConfig
from raptor import SBertEmbeddingModel

###
from langchain_community.llms import Ollama
from raptor import BaseSummarizationModel
class OllamaSummarizationModel(BaseSummarizationModel, BaseQAModel):
    def __init__(self, model_name="google/gemma-2b-it"):
        # Initialize the tokenizer and the pipeline for the GEMMA model
        # self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # self.summarization_pipeline = pipeline(
        #     "text-generation",
        #     model=model_name,
        #     model_kwargs={"torch_dtype": torch.bfloat16},
        #     device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),  # Use "cpu" if CUDA is not available
        # )
        
        self.model = Ollama(model="llama3:8b", num_predict=150)

    def summarize(self, context, max_tokens=150):
        # Format the prompt for summarization
        # messages=[
        #     {"role": "user", "content": f"Write a summary of the following, including as many key details as possible: {context}:"}
        # ]
        
        # prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # # Generate the summary using the pipeline
        # outputs = self.summarization_pipeline(
        #     prompt,
        #     max_new_tokens=max_tokens,
        #     do_sample=True,
        #     temperature=0.7,
        #     top_k=50,
        #     top_p=0.95
        # )
        
        # # Extracting and returning the generated summary
        # summary = outputs[0]["generated_text"].strip()
        
        message = f'wrtie a summary of the following, including as many key details as possilbe: {context}'
        
        summary = self.model.invoke(message)
        
        return summary
    
    def answer_question(self, context, question):
            # Apply the chat template for the context and question
        # messages=[
        #       {"role": "user", "content": f"Given Context: {context} Give the best full answer amongst the option to question {question}"}
        # ]
        # prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # # Generate the answer using the pipeline
        # outputs = self.qa_pipeline(
        #     prompt,
        #     max_new_tokens=256,
        #     do_sample=True,
        #     temperature=0.7,
        #     top_k=50,
        #     top_p=0.95
        # )
        
        # Extracting and returning the generated answer
        # answer = outputs[0]["generated_text"][len(prompt):]
        message =f"Given Context: {context} Give the best full answer amongst the option to question {question}"
        
        answer = self.model.invoke(message)

        return answer
    

# model = OllamaSummarizationModel()
# k = model.summarize("Instruction tuning large language models (LLMs) using machine-generated \
# instruction-following data has been shown to improve zero-shot capabilities on \
# new tasks, but the idea is less explored in the multimodal field. We present the \
# first attempt to use language-only GPT-4 to generate multimodal language-image \
# instruction-following data.")
# print(k)

# RAC = RetrievalAugmentationConfig(summarization_model=OllamaSummarizationModel(), qa_model=OllamaSummarizationModel(), embedding_model=SBertEmbeddingModel())
# RAC = RetrievalAugmentationConfig(summarization_model=OllamaSummarizationModel(), qa_model=OllamaSummarizationModel())
# RA = RetrievalAugmentation(config=RAC)
# with open('demo/llava.txt', 'r', encoding='utf-8') as file:
#     text = file.read()
# RA.add_documents(text)

##############
# Cinderella story defined in sample.txt
with open('demo/llava.txt', 'r', encoding='utf-8') as file:
    text = file.read()

print(text[:100])
RAC = RetrievalAugmentationConfig(embedding_model=SBertEmbeddingModel())
RA = RetrievalAugmentation(config=RAC)

# construct the tree
RA.add_documents(text)


with open('./answer.json','r') as f:
    data = json.load(f)

while True:
    qs = input('Q :')
    ans = RA.answer_question(question=qs)
    print(ans)
    
    tmp = dict()
    tmp["q"] = qs
    tmp["ans"]=ans
    data.append(tmp)
    with open('./answer.json','w') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    

# question = "what are the contributions of this paper?"
# answer = RA.answer_question(question=question)
# print("Answer: ", answer)