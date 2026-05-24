from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI()
chat_history = []

chat_te = ChatPromptTemplate.from_messages(
    [
        ('system', 'You are a helpful assistant of {domain} domain'),
        MessagesPlaceholder(variable_name='chat_history'),
        ('human', '{user_input}')
    ]
)

domain = input('domain: ')

while True:
    user = input('human: ')
    if user == 'exit':
        break

    chat_history.append(("human", user))

    messages = chat_te.format_messages(
        domain=domain,
        chat_history=chat_history,
        user_input=user
    )

    ai_response = model.invoke(messages)

    print("AI:", ai_response.content)

    chat_history.append(("ai", ai_response.content))