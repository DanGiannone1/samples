#evaluate.py
# This script is used to evaluate the quality of the generated answers from the API using the LLM.
# It uses the LLM to evaluate the quality of the generated answers based on various metrics such as correctness, quality, focus, retrieval relevance, and I-don't-know-ness.
# The script loads the questions from a JSON file, calls the chat API to get the response, and then evaluates the response, the context, and the ground truth answer using the LLM.

import requests
import json
from typing import Dict, Any, List
from openai import AzureOpenAI
import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI

load_dotenv()

aoai_deployment = os.getenv("AOAI_DEPLOYMENT")
aoai_key = os.getenv("AOAI_KEY")
aoai_endpoint = os.getenv("AOAI_ENDPOINT")

aoai_client = AzureOpenAI(  
    azure_endpoint=aoai_endpoint,  
    api_key=aoai_key,  
    api_version="2024-05-01-preview"  
)

primary_llm = AzureChatOpenAI(
    azure_deployment=aoai_deployment,
    api_version="2024-05-01-preview",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key=aoai_key,
    azure_endpoint=aoai_endpoint
)

# Prompty content as string variables
quality_prompt = """You are an AI evaluator. 
The "quality" metric is a measure of how well the generated answer adheres to the quality standards. The quality standards are as follows:
1. Clarity: The information should be presented in a clear, concise, and understandable manner, avoiding unnecessary jargon or complexity.
2. Tone: The tone of the answer should be appropriate for the context and audience, maintaining a professional and respectful demeanor.
3. Persona: The answer should be consistent with the persona of the AI assistant, reflecting the expected behavior and characteristics. In our case, the AI assistant is here to help with healthcare and employee-handbook questions.
4. Helpfulness: The answer should try to be as helpful as possible. It should provide relevant information from the context and never be lazy. 

Score the answer between one to five stars. One star indicates poor quality, while five stars indicate excellent quality.


question: What does my healthcare plan cost per month?
context: Your healthcare plan costs $100 per month. Your dental coverage includes two cleanings per year, and your vision coverage includes an annual eye exam and a $150 allowance for frames or contact lenses. You can choose between a PPO and an HMO plan, with the PPO offering more flexibility in choosing healthcare providers and the HMO providing lower out-of-pocket costs. The plan also includes a prescription drug benefit with a $10 copay for generic drugs and a $30 copay for brand-name drugs. In addition, you have access to a telemedicine service for virtual doctor visits at no additional cost.
answer: $100, check the sources for more information.
thoughts: The answer is too short and does not adequetly address the question. The answer sounds terse and unhelpful. It is also being lazy and directing the user to check the sources even though it has all the relevant context. It should be rated 1 star.
stars: 1

question: What does my healthcare plan cost per month?
context: Your healthcare plan costs $100 per month. In the bustling streets of Tokyo, the latest advancements in robotics are seamlessly integrated into daily life, from automated restaurant servers to sophisticated cleaning robots in public spaces. Meanwhile, halfway across the world, the serene landscapes of southern Pennsylvania are a haven for nature enthusiasts, boasting picturesque hiking trails and abundant wildlife. As the world becomes more interconnected, the field of renewable energy continues to evolve, with innovations like floating solar farms and advanced wind turbines pushing the boundaries of sustainable power generation. Amidst these technological and environmental shifts, the art world remains ever vibrant, with contemporary artists drawing inspiration from these changes to create thought-provoking works that challenge societal norms and provoke deep reflection. 
answer: Your healthcare plan costs $100 per month. It includes dental coverage for two cleanings per year and vision coverage with an annual eye exam and a $150 allowance for frames or contact lenses. You can choose between a PPO and an HMO plan; the PPO offers more flexibility in selecting healthcare providers, while the HMO has lower out-of-pocket costs. The plan also includes a prescription drug benefit, with a $10 copay for generic drugs and a $30 copay for brand-name drugs. Additionally, you have access to a telemedicine service for virtual doctor visits at no extra cost.
thoughts: The answer is clear, detailed, and provides all the relevant information from the context. It is helpful and addresses the question effectively. It should be rated 5 stars.
stars: 5


question: {{question}}
context: {{context}}
answer: {{answer}}
thoughts:
stars:
"""

correctness_prompt = """You are an AI evaluator. 
The "correctness metric" is a measure of if the generated answer is correct based on the ground truth answer. You will be given the generated answer and the ground truth answer. You need to compare them and score the content between one to five stars using the following rating scale:
One star: The answer is incorrect
Three stars: The answer is partially correct, but could be missing some key context or nuance that makes it potentially misleading or incomplete compared with the ground truth.  
Five stars: The answer is correct

This rating value should always be either 1, 3, or 5. 


question: What does my healthcare plan cost per month?
truth: Your healthcare plan costs $200 per month
answer: Your healthcare plan costs $200 per month
thoughts: The answer is correct, so it should be rated 5 stars.
stars: 5

question: What does my healthcare plan cost per month?
truth: Your healthcare plan costs $100 per month
answer: Your healthcare plan costs $200 per month
thoughts: The costs differ between the truth and the answer, so it is completely incorrect, so it should be rated 1 star.
stars: 1


question: {{question}}
ground_truth: {{ground_truth}}
answer: {{answer}}
thoughts:
stars:
"""

dont_know_prompt = """You are an AI evaluator. 
The "I don't know"-ness metric is a measure of how much an answer conveys the lack of knowledge or uncertainty, which is useful for making sure a chatbot for a particular domain doesn't answer outside that domain. Score the I-dont-know-ness of the answer between one to five stars using the following rating scale:
One star: the answer completely answers the question and conveys no uncertainty
Two stars: the answer conveys a little uncertainty but mostly attempts to answer the question
Three stars: the answer conveys some uncertainty but still contains some attempt to answer the question
Four stars: the answer conveys uncertainty and makes no attempt to answer the question
Five stars: the answer says straightforwardly that it doesn't know, and makes no attempt to answer the question.

This rating value should always be an integer between 1 and 5. So the rating produced should be 1 or 2 or 3 or 4 or 5.


question: What are the main goals of Perseverance Mars rover mission?
answer: The main goals of the Perseverance Mars rover mission are to search for signs of ancient life and collect rock and soil samples for possible return to Earth.
stars: 1

question: What field did Marie Curie excel in?
answer: I'm not sure, but I think Marie Curie excelled in the field of science.
stars: 2

question: What are the main components of the Mediterranean diet?
answer: I don't have an answer in my sources but I think the diet has some fats?
stars: 3

question: What are the main attractions of the Queen's Royal Castle?
answer: I'm not certain. Perhaps try rephrasing the question?
stars: 4

question: Where were The Beatles formed?
answer: I'm sorry, I don't know, that answer is not in my sources.
stars: 5


question: {{question}}
answer: {{answer}}
stars:
"""

focus_prompt = """You are an AI evaluator.
The "focus" metric is a measure of how well the generated answer ignores irrelevant information in the context and focuses on the relevant content. Score the answer between one to five stars using the following rating scale:
One star: The answer contains a significant amount of irrelevant information. Almost all of the irrelevant information from the context is included in the answer. 
Two stars: The answer contains some irrelevant information. Approximately half of the irrelevant information from the context is included in the answer.
Three stars: The answer contains a moderate amount of irrelevant information. A few pieces of irrelevant information from the context are included in the answer, but the majority of the answer is relevant.
Four stars: The answer ignores almost all of the irrelevant content, but some minor irrelevant information is included. 
Five stars: The answer ignores all irrelevant information and focuses only on the relevant content. 


question: What does my healthcare plan cost per month?
context: Your healthcare plan costs $100 per month. In the bustling streets of Tokyo, the latest advancements in robotics are seamlessly integrated into daily life, from automated restaurant servers to sophisticated cleaning robots in public spaces. Meanwhile, halfway across the world, the serene landscapes of southern Pennsylvania are a haven for nature enthusiasts, boasting picturesque hiking trails and abundant wildlife. As the world becomes more interconnected, the field of renewable energy continues to evolve, with innovations like floating solar farms and advanced wind turbines pushing the boundaries of sustainable power generation. Amidst these technological and environmental shifts, the art world remains ever vibrant, with contemporary artists drawing inspiration from these changes to create thought-provoking works that challenge societal norms and provoke deep reflection. 
answer: Your healthcare plan costs $100 per month. The context also mentions various other topics: the integration of robotics into daily life in Tokyo, the natural beauty of southern Pennsylvania, advancements in renewable energy like floating solar farms and advanced wind turbines, and the influence of these technological and environmental changes on contemporary art.
thoughts: The context contains an answer to the healthcare plan cost question which is relevant, but also a large amount of irrelevant content. The answer mentions of all the irrelevant information from the context, so it should be rated 1 star.
stars: 1

question: What does my healthcare plan cost per month?
context: Your healthcare plan costs $100 per month. In the bustling streets of Tokyo, the latest advancements in robotics are seamlessly integrated into daily life, from automated restaurant servers to sophisticated cleaning robots in public spaces. Meanwhile, halfway across the world, the serene landscapes of southern Pennsylvania are a haven for nature enthusiasts, boasting picturesque hiking trails and abundant wildlife. As the world becomes more interconnected, the field of renewable energy continues to evolve, with innovations like floating solar farms and advanced wind turbines pushing the boundaries of sustainable power generation. Amidst these technological and environmental shifts, the art world remains ever vibrant, with contemporary artists drawing inspiration from these changes to create thought-provoking works that challenge societal norms and provoke deep reflection. 
answer: Your healthcare plan costs $100 per month. The field of renewable energy continues to evolve, which may have an impact on the cost of your healthcare plan in the future.
thoughts: The context contains an answer to the healthcare plan cost question which is relevant, but also some irrelevant content. The answer includes a piece of irrelevant information about renewable energy, so it should be rated 2 stars.
stars: 2

question: What does my healthcare plan cost per month?
context: Your healthcare plan costs $100 per month. In the bustling streets of Tokyo, the latest advancements in robotics are seamlessly integrated into daily life, from automated restaurant servers to sophisticated cleaning robots in public spaces. Meanwhile, halfway across the world, the serene landscapes of southern Pennsylvania are a haven for nature enthusiasts, boasting picturesque hiking trails and abundant wildlife. As the world becomes more interconnected, the field of renewable energy continues to evolve, with innovations like floating solar farms and advanced wind turbines pushing the boundaries of sustainable power generation. Amidst these technological and environmental shifts, the art world remains ever vibrant, with contemporary artists drawing inspiration from these changes to create thought-provoking works that challenge societal norms and provoke deep reflection. 
answer: Your healthcare plan costs $100 per month. 
thoughts: The context contains an answer to the healthcare plan cost question which is relevant and the answer is focused only on the relevant content. No irrelevant content made it through. It should be rated 5 stars.
stars: 5



question: {{question}}
context: {{context}}
answer: {{answer}}
thoughts:
stars:
"""

retrieval_relevance_prompt = """You are an AI evaluator. 
The "Retrieval Relevance metric is a measure of how relevant the provided context is to the user question. Score the content between one to five stars using the following rating scale:
One star: None of the content is relevant
Two stars: A small portion of the content is relevant
Three stars: Approximately half of the content is relevant
Four stars: The majority of the content is relevant
Five stars: All of the content is relevant

This rating value should always be an integer between 1 and 5. So the rating produced should be 1 or 2 or 3 or 4 or 5.

question: What does my healthcare plan say about Eye & Dental?
Context: 
Source 1 - The main goals of the Perseverance Mars rover mission are to search for signs of ancient life and collect rock and soil samples for possible return to Earth.
Source 2 - Marie Curie excelled in the field of science.
Source 3 - The sky is blue
thoughts: The first source is completely unrelated to the question. The second source is also irrelevant. The third source is also irrelevant. Therefore, the rating should be 1 star.
stars: 1

question: What does my healthcare plan say about Eye & Dental?
Context: 
Source 1 - Your healthcare plan covers eye and dental care [employee benefits handbook]
Source 2 - You receive benefits up to $500 for eye and dental care annually [employee benefits handbook]
Source 3 - You can visit any dentist or optometrist in the network for eye and dental care [employee benefits handbook]
thoughts: The first source talks about eye and dental care which is relevant to the question.  The second source also talks about eye and dental care. The third source also talks about eye and dental care. Therefore, the rating should be 5 stars.
stars: 5


question: {{question}}
context: {{context}}
stars:
"""
def load_questions(file_path: str) -> List[Dict[str, str]]:
    with open(file_path, 'r') as file:
        return json.load(file)

def call_chat_api(user_input: str, api_url: str = "http://localhost:5000/chat") -> Dict[str, Any]:
    payload = {"user_input": user_input}
    response = requests.post(api_url, json=payload)
    return response.json()

def run_evaluation(prompty: str, question: str, context: str, answer: str, ground_truth: str = "", metric: str = "") -> str:
    formatted_prompt = prompty.replace("{question}", question)
    formatted_prompt = formatted_prompt.replace("{context}", context)
    formatted_prompt = formatted_prompt.replace("{answer}", answer)
    formatted_prompt = formatted_prompt.replace("{ground_truth}", ground_truth)
    
    messages = [
        {"role": "system", "content": "You are an AI assistant evaluating the quality of answers."},
        {"role": "user", "content": formatted_prompt}
    ]
    
    response = primary_llm.invoke(messages)
    print(f"Metric: {metric}")
    print(response.content)
    return response.content

import colorama
from colorama import Fore, Back, Style

colorama.init(autoreset=True)

def run_evaluations(question: str, ground_truth: str, api_response: Dict[str, Any]) -> Dict[str, str]:
    context = "\n".join([f"Source {i+1} - {item['content']}" for i, item in enumerate(api_response['context'])])
    
    print(f"\n{Fore.CYAN}{'='*80}")
    print(f"{Fore.CYAN}Question: {Style.RESET_ALL}{question}")
    print(f"\n{Fore.CYAN}API Response: {Style.RESET_ALL}{api_response['response']}")
    print(f"\n{Fore.CYAN}Ground Truth: {Style.RESET_ALL}{ground_truth}")
    print(f"\n{Fore.CYAN}Context:")
    for i, item in enumerate(api_response['context']):
        print(f"{Style.RESET_ALL}Source {i+1} - {item['filename']}")
    
    print(f"\n{Fore.YELLOW}Evaluations:")

    evaluations = {
        'Quality': run_evaluation(quality_prompt, question, context, api_response['response'], metric="Quality"),
        'Correctness': run_evaluation(correctness_prompt, question, context, api_response['response'], ground_truth, metric="Correctness"),
        #'Don\'t Know': run_evaluation(dont_know_prompt, question, context, api_response['response'], metric="Don't Know"),
        'Focus': run_evaluation(focus_prompt, question, context, api_response['response'], metric="Focus"),
        'Retrieval Relevance': run_evaluation(retrieval_relevance_prompt, question, context, api_response['response'], metric="Retrieval Relevance")
    }

    for metric, result in evaluations.items():
        print(f"\n  {Fore.GREEN}{metric}")
        thoughts = result.split('thoughts:', 1)[-1].split('stars:', 1)[0].strip()
        stars = result.split('stars:', 1)[-1].strip()
        print(f"    {Fore.BLUE}Thoughts: {Style.RESET_ALL}{thoughts}")
        print(f"    {Fore.MAGENTA}Stars: {Style.RESET_ALL}{stars}")

    print(f"\n{Fore.YELLOW}Summary:")
    for metric, result in evaluations.items():
        stars = result.split('stars:', 1)[-1].strip()
        print(f"  {Fore.GREEN}{metric}: {Style.RESET_ALL}{stars} stars")

    print(f"\n{Fore.CYAN}{'='*80}")
    return evaluations

def main():
    questions = load_questions('tests.json')
    
    for item in questions:
        print(f"\n{Fore.CYAN}Evaluating: {Style.RESET_ALL}{item['question']}")
        response = call_chat_api(item['question'])
        run_evaluations(item['question'], item['ground_truth'], response)
        input(f"\n{Fore.YELLOW}Press Enter to continue to the next question...{Style.RESET_ALL}")

def extract_evaluation_output(evaluation: str) -> Dict[str, str]:
    thoughts = ""
    stars = "N/A"
    
    if "thoughts:" in evaluation.lower():
        thoughts = evaluation.split("thoughts:", 1)[1].split("stars:", 1)[0].strip()
    
    if "stars:" in evaluation.lower():
        stars = evaluation.split("stars:", 1)[1].strip()
    
    return {"thoughts": thoughts, "stars": stars}



if __name__ == "__main__":
    main()