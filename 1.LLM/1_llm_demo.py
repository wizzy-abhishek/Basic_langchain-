from langchain_openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

llm = OpenAI(model='gpt-5.4-mini')

answer = llm.invoke('What is the capital of India??')

print(answer)