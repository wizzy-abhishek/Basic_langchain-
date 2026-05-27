import csv, os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableBranch, RunnableLambda
from pydantic import BaseModel
from typing import Annotated, Optional

load_dotenv()

class Review(BaseModel):
    sentiment: Annotated[str, """Return the sentiment of the text provided. 
                        Say positive, negative, constructive, or neutral. 
                        Give output in these 4 expression ONLY. REMEMBER NO OTHER THAN THESE 4. 
                        Exactly like that don't start with a capital letter.
                        Just give one-word answer.  
                        """]
    
    additional_information: Optional[Annotated[list , """
    IMPORTANT:- Pick all the key point suggestions from the review
                                    """]]
    
model = ChatOpenAI(temperature=0.0)

prompt = PromptTemplate(template="""You are an AI assistant that classifies the text into feedback type segment. 
                                    review = {review}
                                    """, input_variables=['review'])

struc_model = model.with_structured_output(Review)

chain = prompt | struc_model | RunnableLambda(lambda x: dict(x))

input_text = """
The content is too vague and not detailed enough. 
Try expanding each point and including definitions to make it clearer.
"""

def save_feedback_to_csv(data):
    """
    Appends the extracted sentiment and additional information directly to a CSV file.
    """
    file_name = "Extra Data/Review_analysis.csv"
    file_exists = os.path.isfile(file_name)
    
    with open(file_name, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["Sentiment", "Additional Information"])
        additional_info_str = ", ".join(data.get('additional_information', [])) if data.get('additional_information') else "None"
        writer.writerow([data['sentiment'], additional_info_str])

def handle_positive_constructive(x):
    save_feedback_to_csv(x)
    return f"Thank you for your {x['sentiment']} feedback!"

def handle_negative(x):
    save_feedback_to_csv(x)
    info = x.get('additional_information') or "No additional details provided."
    return (f"Feedback type: {x['sentiment']}\n"
            f"Details: {info}\n"
            "We have noted the feedback and we will work on it.")

def handle_fallback(x):
    return f"Thank you for your feedback."

branch_chain = RunnableBranch(
    (lambda x: x['sentiment'] in ['positive', 'constructive'], RunnableLambda(handle_positive_constructive)),
    (lambda x: x['sentiment'] == 'negative', RunnableLambda(handle_negative)),
    RunnableLambda(handle_fallback)
)

full_chain = chain | branch_chain

res = full_chain.invoke({'review': input_text})
print(res)

