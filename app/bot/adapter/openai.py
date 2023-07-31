
import pinecone
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone
import os
import openai


class ChatService:
    def __init__(self, aiapi_key, pinecone_key, pinecone_env):
        openai.aiapi_key = aiapi_key
        self.aiapi_key = aiapi_key
        self.pinecone_key = pinecone_key
        self.pinecone_env = pinecone_env 
       
    loader = TextLoader("app/output.txt")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

    pinecone.init(
        api_key=os.getenv("PINECONE_API_KEY"),  # find at app.pinecone.io
        environment=os.getenv("PINECONE_ENV"),  # next to api key in console
        )
    
    docsearch = Pinecone.from_documents(docs, embeddings, index_name="yerke")
    llm = OpenAI(temperature=0.5, openai_api_key=os.getenv("OPENAI_API_KEY"))
    chain = load_qa_chain(llm, chain_type="stuff")
        
    def get_response(self, prompt):
        doc = ChatService.docsearch.similarity_search(prompt, 1)
        answer = ChatService.chain.run(input_documents=doc, question=prompt)
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": f"""
                    You're AI assistant which suggest to user bars or pubs
                    According to this {answer} recommend to the user first 3 places
                    Do not answer on questions which is not related to choosing bars or pub or about alchocol
                    Response in russian, if user writes on anther language write that you can not understand
                    You only suggest Almaty's bars and pubs
                    Also, when user asks question about average check answer to him
                    You're also barthender, which is familiar about alcohol 
                    """},
            ], 
            max_tokens=1000, 
            temperature=0.5,
        )
        return completion.choices[0].message

