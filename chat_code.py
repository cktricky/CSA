# Misc helper libs
import json
import os
import sys

# Pinecone DB import
import pinecone

# Langchain specific bits
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from dotenv import load_dotenv
load_dotenv()

def run():

    if len(sys.argv) > 1:
        question = sys.argv[1]
        print(f"Received argument: {question}")
    else:
        print("Please provide a question")

    DEFAULT_INDEX = "demo-knowledgebase"
    
    pinecone_index = DEFAULT_INDEX
            
    # Set region name
    region = 'us-west-2'

    '''
    Get API keys depending on environment. Local dev requries a .env file
    to set these values and production has access to SecretsManager
    ''' 
   
    openai_api_key = os.environ.get('OPENAI_API_KEY', '')
    pinecone_api_key = os.environ.get('PINECONE_API_KEY', '')


    # Environment where our Pinecone DB is hosted
    env = 'asia-southeast1-gcp-free'

    # Instantiate pinecone
    pinecone.init(      
	    api_key=pinecone_api_key,   
	    environment=env      
    )  
    
    print("CREATING EMBEDDING MODEL")
    # Convert to an OpenAI Embedding model
    pinecone_vector_store = Pinecone.from_existing_index(index_name=pinecone_index,embedding=OpenAIEmbeddings())
    print("FINISHED CREATING EMBEDDING MODEL")

    # Create system prompt template text
    template = '''
    Answer questions related to software security. 
    Supply all answers in a Markdown format suitable for placing inside GitHub issue comments.

    If you don't know the answer, just say that "I do not yet know the answer to your question, 
    please ask your security team", don't try to make up an answer.

    {summaries}
    '''

    # Convert text to system prompt
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)

    # Take the value of question and place it within the human prompt message
    human_template = "{question}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    # Combine our system and human prompts to form our "Chat Message Prompt"
    prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

    # Create arguments that will be pushed into our chain
    chain_type_kwargs = {"prompt": prompt}

    # Choose our openAI chat model and assign a temperature.
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7, max_tokens=256)

    # We need to extract and compress the relevant documents so we're giving the
    # OpenAI llm as an option here
    compressor = LLMChainExtractor.from_llm(llm)

    # This has our compressor and vector db options. It is pulling only the relevant text from the 
    # document retrieved based on a user's query rather than the entire document contents. 
    compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=pinecone_vector_store.as_retriever())

    # This performs retrieval of our answers based on our data and context
    # and chains together our LLM, our compression retriever, and prompt.
    chain = RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=compression_retriever,
        return_source_documents=True,
        chain_type_kwargs=chain_type_kwargs,
        verbose=True
    )

    response = chain(question)
    reply = response["answer"]
    print(f"Reply: {reply}")

run()
