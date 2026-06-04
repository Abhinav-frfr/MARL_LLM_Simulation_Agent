from langchain_ollama import ChatOllama

llm = ChatOllama(model="llama3.1:8b")
response = llm.invoke("Tell me about India pakistan war")
print(response.content)