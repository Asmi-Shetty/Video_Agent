#Actionableitems, decision, questions

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

import os

def get_llm():
    return ChatMistralAI(model = "mistral-small-latest", mistral_api_key = os.getenv("MISTRAL_API_KEY"), temperature = 0.2)

def build_chain(system_prompt : str):
    llm = get_llm()
    return (
        RunnablePassthrough() | RunnableLambda(lambda x : {"text" : x}) | ChatPromptTemplate.from_messages(

            [ 
                ("system", system_prompt),
                ("human", "{text}")

            ]
        ) | llm | StrOutputParser()
    )

def extract_action_items(transcript : str)->str:
    chain = build_chain(
        "You are an expert meeting analyst. From the meeting transcript,"
        "Extracted all the actions items from the transcript:\n"
        "-Task description\n"
        "-Owner (who is responsible)\n"
        "-Deadline (if mentioned , else write 'Not specified')\n\n"
        "Format as a numbered list. If none found say 'No actin items found.'"
    )

    return chain.invoke(transcript)

def extract_key_decisions(transcript: str) -> str:
    chain = build_chain(
        """
        You are an expert meeting analyst. From the meeting transcript, extract all key decisions made.
        Return each decision as a concise bullet point.
        Format: "- [Description of the decision]"
        If no decisions were made, return: "No key decisions were made."
        """
    )
    return chain.invoke(transcript)

def extract_questions(transcript: str) -> str:
    chain = build_chain(
        """
        You are an expert meeting analyst. From the meeting transcript, extract all open questions.
        Return each question as a concise bullet point.
        Format: "- [Question text]"
        If no questions were found, return: "No questions were found."
        """
    )
    return chain.invoke(transcript)

def extract_topics(transcript: str) -> str:
    chain = build_chain(
        """
        You are an expert meeting analyst. From the meeting transcript, list all topics discussed.
        Format each topic as: "- [Topic name]"
        Return only the topics, nothing else.
        """
    )
    return chain.invoke(transcript)


