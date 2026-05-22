from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model='gpt-5.4-mini', temperature=0)

res = model.invoke("What is the capial of India???")

print(res.content)