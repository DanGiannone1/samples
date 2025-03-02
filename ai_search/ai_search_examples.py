from ai_search import *
from azure_openai.aoai import generate_embeddings_aoai
import time
import json
import csv
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def create_simple_index():

    search_manager = AISearchManager(search_index_name='test2')

    # Load index configuration from JSON file

    # Prepare sample documents with embeddings
    sample_documents = [
        {
            "id": "1",
            "content": "This is a sample document about artificial intelligence and machine learning."
        },
        {
            "id": "2",
            "content": "Natural language processing is a subfield of artificial intelligence."
        }
    ]

    # Generate embeddings for documents
    for doc in sample_documents:
        content_vector = generate_embeddings_aoai(doc['content'])
        if content_vector:
            doc['content_vector'] = content_vector
            doc['content_vector2'] = content_vector # Using the same embedding for demonstration
        else:
            print(f"Failed to generate embedding for document: {doc['id']}")

    # Upload documents
   
    search_manager.upload_documents(sample_documents)



def create_index_with_scoring_profile():

    search_manager = AISearchManager()

    category_scoring_profile = ScoringProfile(
        name="categoryScoringProfile",
        functions=[
            TagScoringFunction(
                field_name="category",
                boost=5,
                parameters=TagScoringParameters(tags_parameter="category"),
                interpolation=ScoringFunctionInterpolation.LINEAR
            )
        ],
        function_aggregation=ScoringFunctionAggregation.SUM
        )


    temporalId_scoring_profile = ScoringProfile(
        name="temporalIdScoringProfile",
        functions=[
            FreshnessScoringFunction(
                field_name="temporalId",
                boost=2,
                parameters=FreshnessScoringParameters(boosting_duration="P1095D"),
                interpolation=ScoringFunctionInterpolation.LINEAR
            )
    ],
    function_aggregation=ScoringFunctionAggregation.SUM
)

    config = {
            "name": "djg_with_scoring_profile",
            "fields": [
                SimpleField(name="id", type=SearchFieldDataType.String, key=True, filterable=True),
                SimpleField(name="temporalId", type=SearchFieldDataType.DateTimeOffset, filterable=True, facetable=True),
                SimpleField(name="category", type=SearchFieldDataType.String, filterable=True, facetable=True),
                SimpleField(name="sourceFileName", type=SearchFieldDataType.String, filterable=True),
                SimpleField(name="sourcePages", type=SearchFieldDataType.String, filterable=True),
                SearchableField(name="content", type=SearchFieldDataType.String),
                SearchField(name="contentVector", type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                            searchable=True, vector_search_dimensions=1536, vector_search_profile_name="myHnswProfile")
            ],
            "vector_search": VectorSearch(
                algorithms=[
                    HnswAlgorithmConfiguration(name="myHnsw")
                ],
                profiles=[
                    VectorSearchProfile(
                        name="myHnswProfile",
                        algorithm_configuration_name="myHnsw",
                    )
                ]
            ),
            'scoring_profiles': [category_scoring_profile, temporalId_scoring_profile]
        }



    search_manager.create_search_index_from_config(config)

###Example 1### 
# Create a simple index, vectorize json documents, upload to index, run simple search queries
def example_1():

    index_name = 'test_index'

    #Create the search manager
    search_manager = AISearchManager(search_index_name=index_name)


    #Define and create a simple index
    config = {
            "name": index_name,
            "fields": [
                SimpleField(name="id", type=SearchFieldDataType.String, key=True, filterable=True),
                SimpleField(name="date", type=SearchFieldDataType.DateTimeOffset, filterable=True, facetable=True),
                SimpleField(name="category", type=SearchFieldDataType.String, filterable=True, facetable=True),
                SimpleField(name="sourceFileName", type=SearchFieldDataType.String, filterable=True),
                SimpleField(name="sourcePage", type=SearchFieldDataType.String, filterable=True),
                SearchableField(name="content", type=SearchFieldDataType.String),
                SearchField(name="contentVector", type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                            searchable=True, vector_search_dimensions=1536, vector_search_profile_name="myHnswProfile")
            ],
            "vector_search": VectorSearch(
                algorithms=[
                    HnswAlgorithmConfiguration(name="myHnsw")
                ],
                profiles=[
                    VectorSearchProfile(
                        name="myHnswProfile",
                        algorithm_configuration_name="myHnsw",
                    )
                ]
            )
        }

    search_manager.create_search_index_from_config(config)

    #Read in documents from a json file
    with open('ai_search/sample_data.json', 'r') as f:
        documents = json.load(f)

    #Generate an embedding for each document's content
    for doc in documents:
        print(f"Generating embedding for document: {doc['id']}")
        content_vector = generate_embeddings_aoai(doc['content'])
        doc['contentVector'] = content_vector    

    #Upload the documents to the index
    search_manager.upload_documents(documents)

    #Read in documents from a CSV file 
    with open('ai_search/sample_data.csv', 'r') as f:
        documents = []
        for line in f:
            line = line.strip().split(',')
            doc = {
                "id": line[0],
                "date": line[1],
                "category": line[2],
                "sourceFileName": line[3],
                "sourcePage": line[4],
                "content": line[5]
            }
            documents.append(doc)

    #Generate an embedding for each document's content
    for doc in documents:
        print(f"Generating embedding for document: {doc['id']}")
        content_vector = generate_embeddings_aoai(doc['content'])
        doc['contentVector'] = content_vector    

    #Upload the documents to the index
    search_manager.upload_documents(documents)

    time.sleep(2)  # Wait for document indexing

    # Perform simple searches
    query = "artificial intelligence"

    

    print("\nSimple Text Search Results:")
    text_results = search_manager.simple_text_search(query)
    for result in text_results:
        print(f"ID: {result['id']}, Content: {result['content']}")

    print("\nSimple Vector Search Results:")
    vector_results = search_manager.simple_vector_search(query)
    for result in vector_results:
        print(f"ID: {result['id']}, Content: {result['content']}")

    print("\nSimple Hybrid Search Results:")
    hybrid_results = search_manager.simple_hybrid_search(query)
    for result in hybrid_results:
        print(f"ID: {result['id']}, Content: {result['content']}")

    #delete the index - comment/uncomment as needed
    search_manager.delete_index()
    print("\n\nIndex deleted successfully")


def example_2():

    index_name = 'test_index'

    #Create the search manager
    search_manager = AISearchManager(search_index_name=index_name)


    #Delete the index if it already exists
    search_manager.delete_index()

    config = {
            "name": index_name,
            "fields": [
                SimpleField(name="id", type=SearchFieldDataType.String, key=True, filterable=True),
                SimpleField(name="date", type=SearchFieldDataType.DateTimeOffset, filterable=True, facetable=True),
                SimpleField(name="category", type=SearchFieldDataType.String, filterable=True, facetable=True),
                SimpleField(name="sourceFileName", type=SearchFieldDataType.String, filterable=True),
                SimpleField(name="sourcePage", type=SearchFieldDataType.String, filterable=True),
                SearchableField(name="content", type=SearchFieldDataType.String),
                SearchField(name="fileNameVector", type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                            searchable=True, vector_search_dimensions=1536, vector_search_profile_name="myHnswProfile"),
                SearchField(name="contentVector", type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                            searchable=True, vector_search_dimensions=1536, vector_search_profile_name="myHnswProfile")
            ],
            "vector_search": VectorSearch(
                algorithms=[
                    HnswAlgorithmConfiguration(name="myHnsw")
                ],
                profiles=[
                    VectorSearchProfile(
                        name="myHnswProfile",
                        algorithm_configuration_name="myHnsw",
                    )
                ]
            )
        }

    search_manager.create_search_index_from_config(config)
    

    #Read in documents from a json file
    with open('ai_search/sample_data.json', 'r') as f:
        documents = json.load(f)

    #Generate an embedding for each document's content
    for doc in documents:
        print(f"Generating embedding for document: {doc['id']}")
        content_vector = generate_embeddings_aoai(doc['content'])
        filename_vector = generate_embeddings_aoai(doc['sourceFileName'])
        doc['contentVector'] = content_vector    
        doc['fileNameVector'] = filename_vector

    #Upload the documents to the index
    search_manager.upload_documents(documents)

    print("Waiting for document indexing...")
    time.sleep(2)  # Wait for document indexing


    #Run searches via a config file
    with open('ai_search/hybrid_search.json', 'r') as config_file:
        hybrid_search_config = json.load(config_file)

    

    # Define your search query
    query = "artificial intelligence"

    # Perform the dynamic search
    results = search_manager.dynamic_search(query, hybrid_search_config)

    # Print the results
    print("\nDynamic Search Results:")
    for result in results:
        print(f"ID: {result['id']}, Content: {result['content']}")


def create_indexes():



    search_manager = AISearchManager()


    #Index 1 - Simple index with some metadata fields, a content field, and two vector fields
    config = {
            "name": "test_index",
            "fields": [
                SimpleField(name="id", type=SearchFieldDataType.String, key=True, filterable=True),
                SimpleField(name="date", type=SearchFieldDataType.DateTimeOffset, filterable=True, facetable=True),
                SimpleField(name="category", type=SearchFieldDataType.String, filterable=True, facetable=True),
                SimpleField(name="sourceFileName", type=SearchFieldDataType.String, filterable=True),
                SimpleField(name="sourcePage", type=SearchFieldDataType.String, filterable=True),
                SearchableField(name="content", type=SearchFieldDataType.String),
                SearchField(name="fileNameVector", type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                            searchable=True, vector_search_dimensions=1536, vector_search_profile_name="myHnswProfile"),
                SearchField(name="contentVector", type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                            searchable=True, vector_search_dimensions=1536, vector_search_profile_name="myHnswProfile")
            ],
            "vector_search": VectorSearch(
                algorithms=[
                    HnswAlgorithmConfiguration(name="myHnsw")
                ],
                profiles=[
                    VectorSearchProfile(
                        name="myHnswProfile",
                        algorithm_configuration_name="myHnsw",
                    )
                ]
            )
        }

    search_manager.create_search_index_from_config(config)



    #Index 2 - an index with custom scoring profiles
    category_scoring_profile = ScoringProfile(
        name="categoryScoringProfile",
        functions=[
            TagScoringFunction(
                field_name="category",
                boost=5,
                parameters=TagScoringParameters(tags_parameter="category"),
                interpolation=ScoringFunctionInterpolation.LINEAR
            )
        ],
        function_aggregation=ScoringFunctionAggregation.SUM
        )


    temporalId_scoring_profile = ScoringProfile(
        name="temporalIdScoringProfile",
        functions=[
            FreshnessScoringFunction(
                field_name="date",
                boost=2,
                parameters=FreshnessScoringParameters(boosting_duration="P1095D"),
                interpolation=ScoringFunctionInterpolation.LINEAR
            )
    ],
    function_aggregation=ScoringFunctionAggregation.SUM
)

    config = {
            "name": "test_index_with_scoring_profile",
            "fields": [
                SimpleField(name="id", type=SearchFieldDataType.String, key=True, filterable=True),
                SimpleField(name="date", type=SearchFieldDataType.DateTimeOffset, filterable=True, facetable=True),
                SimpleField(name="category", type=SearchFieldDataType.String, filterable=True, facetable=True),
                SimpleField(name="sourceFileName", type=SearchFieldDataType.String, filterable=True),
                SimpleField(name="sourcePage", type=SearchFieldDataType.String, filterable=True),
                SearchableField(name="content", type=SearchFieldDataType.String),
                SearchField(name="contentVector", type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                            searchable=True, vector_search_dimensions=1536, vector_search_profile_name="myHnswProfile")
            ],
            "vector_search": VectorSearch(
                algorithms=[
                    HnswAlgorithmConfiguration(name="myHnsw")
                ],
                profiles=[
                    VectorSearchProfile(
                        name="myHnswProfile",
                        algorithm_configuration_name="myHnsw",
                    )
                ]
            ),
            'scoring_profiles': [category_scoring_profile, temporalId_scoring_profile]
        }



    search_manager.create_search_index_from_config(config)



def upload_documents():

    search_manager = AISearchManager(search_index_name='test_index')

    #Read in documents from a json file
    with open('ai_search/sample_data.json', 'r') as f:
        documents_1 = json.load(f)

    # Generate an embedding for each document's content
    for doc in documents_1:
        print(f"Generating embedding for document: {doc['id']}")
        content_vector = generate_embeddings_aoai(doc['content'])
        filename_vector = generate_embeddings_aoai(doc['sourceFileName'])
        doc['contentVector'] = content_vector    
        doc['fileNameVector'] = filename_vector 

    #Upload the documents to the index
    search_manager.upload_documents(documents_1)


    # Read in documents from a CSV file
    with open('ai_search/sample_data.csv', 'r', newline='') as f:
        reader = csv.DictReader(f)
        documents_2 = []
        for row in reader:
            doc = {
                "id": row["id"],
                "date": row["date"],
                "category": row["category"],
                "sourceFileName": row["sourceFileName"],
                "sourcePage": row["sourcePage"],
                "content": row["content"]
            }
            documents_2.append(doc)


    # Generate an embedding for each document's content
    for doc in documents_2:
        print(f"Generating embedding for document: {doc['id']}")
        content_vector = generate_embeddings_aoai(doc['content'])
        filename_vector = generate_embeddings_aoai(doc['sourceFileName'])
        doc['contentVector'] = content_vector    
        doc['fileNameVector'] = filename_vector

    #Upload the documents to the index
    search_manager.upload_documents(documents_2)

    #Upload the documents to the other indexes
    search_manager.search_index_name = 'test_index_with_scoring_profile'
    search_manager.upload_documents(documents_1)
    search_manager.upload_documents(documents_2)


def simple_searches():

    search_manager = AISearchManager(search_index_name='test_index')

    query = "artificial intelligence"


    print("\nSimple Text Search Results:")
    text_results = search_manager.simple_text_search(query)
    for result in text_results:
        print(f"ID: {result['id']}, Content: {result['content']}")

    print("\nSimple Vector Search Results:")
    vector_results = search_manager.simple_vector_search(query)
    for result in vector_results:
        print(f"ID: {result['id']}, Content: {result['content']}")

    print("\nSimple Hybrid Search Results:")
    hybrid_results = search_manager.simple_hybrid_search(query)
    for result in hybrid_results:
        print(f"ID: {result['id']}, Content: {result['content']}")

    print("\n\nSimple Hybrid search with reranker Results:")
    hybrid_with_reranker_results = search_manager.hybrid_search_simple_reranker(query)
    for result in hybrid_with_reranker_results:
        print(f"ID: {result['id']}, Content: {result['content']}")

if __name__ == "__main__":




    #create_indexes()

    #upload_documents()

    simple_searches()