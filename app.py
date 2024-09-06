# app.py
# Basic RAG chatbot application using Azure OpenAI and Azure AI Search

import os
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from azure.core.credentials import AzureKeyCredential
from openai import AzureOpenAI

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Azure OpenAI configuration
aoai_deployment = os.getenv("AOAI_DEPLOYMENT")
aoai_key = os.getenv("AOAI_KEY")
aoai_endpoint = os.getenv("AOAI_ENDPOINT")

# Azure AI Search configuration
search_endpoint = os.getenv('AZURE_SEARCH_ENDPOINT')
search_key = os.getenv('AZURE_SEARCH_KEY')
search_index = os.getenv('AZURE_SEARCH_INDEX')

# Initialize Azure OpenAI client
aoai_client = AzureOpenAI(
    azure_endpoint=aoai_endpoint,
    api_key=aoai_key,
    api_version="2024-05-01-preview"
)

# Initialize LangChain Azure Chat OpenAI
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

# Prompts
QUERY_TRANSLATION_PROMPT = """
You are an AI assistant tasked with translating user queries into effective search queries. 
Your goal is to create a search query that will retrieve the most relevant documents from a search index.
Analyze the user's input and generate a concise, relevant search query.
"""

RAG_SYSTEM_PROMPT = """
You are a helpful AI assistant. You are given a user input and some context, it is your job to answer the question based on the context. 
You can use the context to generate a response that is relevant and informative. The context may not always be relevant to the user input, you should use your best judgment to determine the most appropriate response.
Do not provide answers that are not included in the context. Your goal is to provide accurate and helpful responses based on the information provided only.
"""

def generate_embeddings(text, model="text-embedding-ada-002"):
    """Generate embeddings for the given text using Azure OpenAI."""
    return aoai_client.embeddings.create(input=[text], model=model).data[0].embedding

def get_context(user_input):
    """Retrieve relevant context from Azure AI Search based on the user input."""
    print("Running DocStore Agent")
    
    # Generate search query
    messages = [
        {"role": "system", "content": QUERY_TRANSLATION_PROMPT},
        {"role": "user", "content": user_input},
    ]
    search_query = primary_llm.invoke(messages).content
    print(f"Search query: {search_query}")
    
    # Set up search client
    search_client = SearchClient(
        search_endpoint,
        search_index,
        AzureKeyCredential(search_key)
    )
    
    # Generate query vector
    query_vector = generate_embeddings(search_query)
    vector_query = VectorizedQuery(vector=query_vector, k_nearest_neighbors=3, fields="contentVector")
    
    # Perform search
    results = search_client.search(
        search_text=search_query,
        vector_queries=[vector_query],
        top=3
    )
    
    # Process results
    context = []
    for source in results:
        print(f"Score: {source['@search.score']} {source['id']}")
        context.append({
            "filename": source['id'],
            "content": source['content']
        })
    
    return context

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat requests and return AI-generated responses."""
    user_input = request.json.get('user_input', '')
    print(f"User input: {user_input}")
    
    # Get context
    context = get_context(user_input)
    
    # Prepare context for LLM
    context_text = "\n".join([f"{item['filename']}: {item['content']}" for item in context])
    print("Context: ", context_text)
    
    llm_input = f"Context: {context_text}\n\nUser Input: {user_input}"
    
    messages = [
        {"role": "system", "content": RAG_SYSTEM_PROMPT},
        {"role": "user", "content": llm_input}
    ]

    # Generate response
    raw_response = primary_llm.invoke(messages)
    response = raw_response.content
    
    # Prepare the final response
    final_response = {
        "response": response,
        "context": context
    }
    
    return jsonify(final_response)

if __name__ == '__main__':
    app.run(debug=True)