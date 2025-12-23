#eNGINE

from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

#Prompt Template
from langchain.prompts import PromptTempalte
prompt  = PromptTempalte(
    input_variable = ["topic"]
    template = "Explain {topic} in simple words."
)

#chain(LLMChain)
# a chain = Prompt + LLm

from langchain.chains import LLMChain

chain = LLMChain(
    llm = llm
    prompt = prompt
)

response = chain.run(topic = "Langchain")
print(response)

#sequantial chains(multi-step reasoning)
# output of step 1 --> output of step 2

from langchain.chains import SequentialChain

# genreally LLMs are stateless by default
# Memory adds chat History

from langchain.memory import ConversationalBufferMemory

memory = ConversationalBufferMemory()

#Tools
# Tools lets the LLms do Calculations, call API, Search etc

from langchain.tools import Tool

#agents( decision Makers)
# which tool i need to use right now

from langchain.agents import initialize_agent

# document loaders
# used for Rag for loading the documents

from langchain.document_loaders import TextLoader

# Text Splitter LLms has the token Limits

from langchain.text_splitter import RecusriveCharacterTextsplitter

# embeddings convert the text to vectors
#used for the semantic seach, RAG, similarity matching

from langchain_openai import OpenAIEmbeddings

# vector stores
# common things we use in this is FiAss, Chroma , PineCone

from lanchain.vectorstores import FAISS

# Retrieval in RAG
# user query, relevant documents and llm answers

from langchain.chains import RetrievalQA
