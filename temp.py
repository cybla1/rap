import torch
from raptor import BaseSummarizationModel, BaseQAModel, BaseEmbeddingModel, RetrievalAugmentationConfig
from transformers import AutoTokenizer
# You can define your own Summarization model by extending the base Summarization Class. 
class GEMMASummarizationModel(BaseSummarizationModel):
    def __init__(self, model_name="google/gemma-2b-it"):
        # Initialize the tokenizer and the pipeline for the GEMMA model
        
        from langchain_community.embeddings import HuggingFaceEmbeddings

        embeddings_model = HuggingFaceEmbeddings(
            model_name='BAAI/bge-large-en-v1.5',
            model_kwargs={'device':'cpu'},
            encode_kwargs={'normalize_embeddings':True},
        )
        
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("sentence-transformers/multi-qa-mpnet-base-cos-v1")
        
        print(3)
        #self.tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-large-en-v1.5")
        # self.summarization_pipeline = pipeline(
        #     "text-generation",
        #     model=model_name,
        #     model_kwargs={"torch_dtype": torch.bfloat16},
        #     device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),  # Use "cpu" if CUDA is not available
        # )

    def summarize(self, context, max_tokens=150):
        # Format the prompt for summarization
        messages=[
            {"role": "user", "content": f"Write a summary of the following, including as many key details as possible: {context}:"}
        ]
        
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # Generate the summary using the pipeline
        outputs = self.summarization_pipeline(
            prompt,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            top_p=0.95
        )
        
        # Extracting and returning the generated summary
        summary = outputs[0]["generated_text"].strip()
        return summary


model = GEMMASummarizationModel()
print(1)