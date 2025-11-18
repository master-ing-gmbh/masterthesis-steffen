
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Neo4jVector

def get_retriever(amountResults):
    
    # Initialize the same embeddings configuration
    embeddings = OllamaEmbeddings(
        model="mxbai-embed-large",
    )
    
    # Connect to existing index 
    #vector_store = Neo4jVector.from_existing_index(
        #embeddings,
        #search_type="hybrid",
        #index_name="failure_mode_context_index",
        #keyword_index_name="failure_mode_keyword_index")

    # Only use vector search
    vector_store = Neo4jVector.from_existing_index(
        embeddings,
        search_type="vector",
        node_label="VectorEmbedding", 
        index_name="failure_mode_context_index",
        retrieval_query="""
        RETURN node.text_chunk AS text, score, node {.*, text_chunk: Null, embedding: Null, id: Null} AS metadata
        """
    )
    
    return vector_store.as_retriever(search_kwargs={"k": amountResults})

def retrieve_functions_from_vector(element_context: dict, retriever, debug):
    product = element_context.get('Product')
    subsystem = element_context.get('Subsystem')
    system_element = element_context.get('SystemElement')

    query = f"Functions of {system_element}"

    results = retriever.invoke(query)
    if debug:
        print(f"Vector retrieval results for query '{query}': {results}")
    return results

def retrieve_failures_from_vector(element_context: dict, retriever, debug):
    # Construct a query based on the element context
    function = element_context.get('Function')

    query = f"Failure Modes of Function {function} with failure effects and failure causes"

    results = retriever.invoke(query)
    if debug:
        print(f"Vector retrieval results for query '{query}': {results}")
    return results

def retrieve_existing_measures_from_vector(element_context: dict, retriever, debug):
    failure_mode = element_context.get('FailureMode')
    failure_cause = element_context.get('FailureCause')
    all_results = []
    query_detective = f"Detective Measures for Failure cause {failure_cause}"
    query_preventive = f"Preventive Measures for Failure cause {failure_cause} in the context of failure mode {failure_mode}"

    results_detective = retriever.invoke(query_detective)
    results_preventive = retriever.invoke(query_preventive)

    all_results.extend(results_detective)
    all_results.extend(results_preventive)

    seen_content = set()
    unique_results = []
    for doc in all_results:
        if doc.page_content not in seen_content:
            unique_results.append(doc)
            seen_content.add(doc.page_content)
    
    return unique_results

def retrieve_risk_ratings_from_vector(element_context: dict, retriever, debug):
    # Construct a query based on the element context
    failure_mode = element_context.get('FailureMode')
    failure_cause = element_context.get('FailureCause')
    failure_effect = element_context.get('FailureEffect')
    all_results = []
    
    # failure cause
    if failure_cause:
        cause_query_occurrence = f"Occurrence of failure cause {failure_cause}"
        cause_results_occurrence = retriever.invoke(cause_query_occurrence)
        cause_query_detection = f"Detection of failure cause {failure_cause} in the context of failure mode {failure_mode}"
        cause_results_detection = retriever.invoke(cause_query_detection)

        all_results.extend(cause_results_detection)
        all_results.extend(cause_results_occurrence)
    
    # failure effect 
    if failure_effect:
        effect_query = f"Severity of failure effect {failure_effect}"
        effect_results = retriever.invoke(effect_query)
        all_results.extend(effect_results)
    
    seen_content = set()
    unique_results = []
    for doc in all_results:
        if doc.page_content not in seen_content:
            unique_results.append(doc)
            seen_content.add(doc.page_content)
    
    return unique_results