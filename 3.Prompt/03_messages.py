from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

load_dotenv()

model = ChatOpenAI()

chat_history = [
    SystemMessage("You are a helpful assistant that tell me if I am doing write or wrong in any financial situation"), 
]

while(True):
    user_input = input("user: ")
    if 'exit' == user_input:
        break
    human = HumanMessage(user_input)
    chat_history.append(human)
    ai = AIMessage(model.invoke(chat_history).content)
    print(ai)
    chat_history.append(ai)
