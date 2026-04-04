from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI 
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import ChatPromptTemplate


load_dotenv()
data = TextLoader("documnet_Loader/notes.txt")
docs = data.load()
template = ChatPromptTemplate.from_messages([
    ("system", "You are an AI that summarizes text"),
    ("human" , "{data}")
    ]
    )
model = ChatMistralAI(model = "mistral-small-2506")

prompt = template.format_messages(data=docs[0].page_content)
result = model.invoke(prompt)
print(result.content)
