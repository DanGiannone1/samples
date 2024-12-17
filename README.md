# Simple Code Samples for Gen AI in Azure

This repository contains a collection of code samples demonstrating various techniques and patterns using different technologies and Azure services.

## Project Information

A curated collection of code samples for quick learning and prototyping generative AI projects with Azure Services. This repo contains simple samples for various Azure services like Azure AI Search, Azure OpenAI, Cosmos DB, and others. It also contains orchestrator examples such as LangChain, as well as a simple RAG evaluation script and RAG chatbot Flask app.

- **Programming Languages:** Python
- **Frameworks:** LangChain
- **Azure Services:**
  - Azure Cosmos DB
  - Azure Document Intelligence
  - Azure OpenAI
  - Azure Data Lake Storage
  - Azure AI Search
- **Project Type:** Educational/Reference
- **Code Complexity:** Beginner
- **Business Value:** Many repositories and accelerators out there are very complex, which is a roadblock to developers who are just getting started with Azure or Gen AI. This repo can help beginners rapidly learn and prototype solutions. For experienced developers, these examples of simple syntax with the SDK versions specified can also prove valuable. 
- **Target Audience:** Beginner to intermediate Python developers who want to build generative AI solutions with Azure Services. Experienced developers who want to quickly prototype.

## Key Features

- Clean, modular code structure
- Very clear on what model versions, API/SDK versions, and library versions are necessary to make each module run.
- Authentication examples using both key-based and Azure Identity
- Integration patterns between multiple Azure services
- RAG (Retrieval Augmented Generation) implementation examples
- Evaluation framework for RAG applications

## Key Components

- **Document Processing Pipeline:** Complete pipeline for document ingestion, analysis, and storage
- **Vector Search Implementation:** Multiple approaches to vector search including hybrid search
- **RAG Implementation:** End-to-end RAG implementation with evaluation framework
- **Multiple Authentication Methods:** Support for both key-based and Azure Identity authentication
- **Chunking Strategies:** Various text chunking implementations for optimal document processing
- **Evaluation Framework:** Tools for evaluating RAG implementation quality
- **Multimodal Processing:** Support for processing both text and image content

### Simplistic RAG Flask App

- RAG app in 150 lines of code, no orchestrator
- Simple & easy to understand

### Evaluation Script

- Showcases the basics of LLM-based evaluation of RAG systems
- Easily configurable
- Test with the simple RAG app

### Azure OpenAI Integration (`azure_openai/`)

- Direct SDK integration examples
- Embedding generation
- Completion and chat completion examples
- Structured output parsing

### Azure AI Search (`ai_search/`)

- Vector, keyword, and hybrid search implementations
- Index management
- Document upload and retrieval
- Custom scoring profiles

### Document Processing (`document_ingestion/`)

- PDF processing
- Text chunking strategies
- Multimodal document handling
- Metadata extraction

### RAG Application (`simple_rag_app/`)

- Flask-based web application
- Vector search integration
- Context retrieval and injection
- Response generation

### Document Intelligence (`azure_document_intelligence/`)

- PDF text extraction
- Layout analysis
- Image content processing
- OCR capabilities

### Storage Integration (`storage/`)

- Azure Data Lake Storage operations
- Blob storage management
- Container operations
- File upload/download utilities

### Database Integration (`azure_cosmos_db/`)

- Document metadata storage
- CRUD operations
- Query operations
- Bulk operations

## Getting Started

### Prerequisites

- Python 3.9 or higher
- An Azure subscription with access to:
  - Azure OpenAI
  - Azure AI Search
  - Azure Document Intelligence
  - Azure Cosmos DB
  - Azure Storage Account
- Visual Studio Code (recommended) or another Python IDE

### Development Environment Setup

1. Clone the repository:
    ```bash
    git clone [your-repo-url]
    cd [repo-directory]
    ```
2. Create and activate a virtual environment:

    **Windows:**
    ```bash
    python -m venv venv
    venv\Scripts\activate
    ```

    **Linux/Mac:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
4. Create an `.env` file in the root directory with your Azure configurations (see Environment Variables section above)