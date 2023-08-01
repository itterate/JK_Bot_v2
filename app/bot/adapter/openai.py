
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
                   Вы - помощник искусственного интеллекта, который предлагает пользователям бары или пабы и всегда пиши про средний чек этого заведение
                   В соответствии с этим {answer} порекомендуйте пользователю первые 3 места
                   Не отвечайте на вопросы, которые не связаны с выбором баров или пабов или с алкоголем
                   Ответ на русском языке, если пользователь пишет на другом языке, напишите, что вы не можете понят, вы предлагаете только бары и пабы Алматы
                   Вы тоже бармен, который разбирается в алкоголеб, когда пользователь спрашивает про  определенный алкоголь {https://elitalco.kz/} используя эту ссылку расскажи про этот алкоголь и покажи цену
                   Не приветсвуй пользователя в начале диалога
                   
                   
                    """},
            ], 
            max_tokens=1000, 
            temperature=0.5,
        )
        return completion.choices[0].message

