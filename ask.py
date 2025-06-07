#!/usr/bin/env python3
from survey import LLMSession
import os

llm_config = {}

api_key = os.getenv("OPENAI_API_KEY") or input("Enter your OpenAI API key (or empty): ")

if api_key:
    llm_config["api_key"] = api_key

base_url = os.getenv("OPENAI_BASE_URL") or input("Enter your OpenAI base URL (or empty): ")

if base_url:
    llm_config["base_url"] = base_url

model = os.getenv("OPENAI_MODEL") or input("Enter your model: ")

if model:
    llm_config["model"] = model

session = LLMSession("Please answer the questions honestly.", llm_config)

question1 = "Based on your training and knowledge, can you list any academic articles that are relevant to the topic of consumer responses to price discrimination, particularly in the context of fairness perceptions and personalized pricing?"

print(f"Question 1: {question1}")

reply = session.query(question1)

print(f"Reply: {reply}")

question2 = "The following article is titled \"Seeking the perfect price: Consumer responses to personalized price discrimination in e-commerce\". Who are the authors of this paper?Please choose one of the options. A) Gerrit Hufnagel, Manfred Schwaiger, Louisa Weritz; B) George Gui, Olivier Toubia; C) James Brand, Ayelet Israeli, Donald Ngwe; D) Liying Qiu, Param Vir Singh, Kannan Srinivasan"

print(f"Question 2: {question2}")

reply2 = session.query(question2)

print(f"Reply: {reply2}")