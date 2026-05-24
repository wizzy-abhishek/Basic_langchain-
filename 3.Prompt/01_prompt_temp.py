from langchain_core.prompts import PromptTemplate, load_prompt
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI()

topic = input('Research topic\n')
tone = input('tone like easy, techincal, noob, etc\n')

template = load_prompt('prompt1.json')

#template.save('prompt1.json')

prompt = template.invoke({
    'topic': topic, 
    'tone': tone
})

print(model.invoke(prompt).content)
