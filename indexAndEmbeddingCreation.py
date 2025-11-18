
from neo4j import GraphDatabase
import os
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Neo4jVector


def create_failure_mode_embeddings():
    
    driver = GraphDatabase.driver(
        uri=os.environ["NEO4J_URI"],
        auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"])
    )
    
    with driver.session() as session:
        # Get all FailureMode nodes with their rich context
        failure_modes = session.execute_read(get_failure_modes_with_context)
        
        print(f"Found {len(failure_modes)} failure modes to process")
        
        # Generate text chunks for each failure mode
        for failure_mode in failure_modes:
            text_chunk = generate_failure_mode_text_chunk(failure_mode)
            
            session.execute_write(create_vector_embedding_node, 
                                failure_mode['failure_mode_id'], 
                                text_chunk)
            
            print(f"Created embedding for: {failure_mode['failure_mode_name']}")
    
    driver.close()
    print("Embedding generation completed!")

def get_failure_modes_with_context(tx):
    
    query = """
    MATCH (p:Product)-[:hasSubsystem]->(s:Subsystem)-[:hasSystemElement]->(se:SystemElement)
          -[:hasFunction]->(f:Function)-[:hasFailureMode]->(fm:FailureMode)
    
    OPTIONAL MATCH (fm)-[r:isDueToFailureCause]->(fc:FailureCause)
    OPTIONAL MATCH (fc)-[:isImprovedByPreventiveMeasure]->(pm:Measure {type: 'preventive'})
    OPTIONAL MATCH (fc)-[:isImprovedByDetectiveMeasure]->(dm:Measure {type: 'detective'})
    OPTIONAL MATCH (dm)-[:improvesDetectionFor]->(fm) WHERE (fc)-[:isImprovedByDetectiveMeasure]->(dm)
    OPTIONAL MATCH (fm)-[:resultsInFailureEffect]->(fe:FailureEffect)
    
    WITH fm, p, s, se, f, fc, fe, r,
         collect(DISTINCT pm.name) as preventive_measures_for_cause,
         collect(DISTINCT dm.name) as detective_measures_for_cause
    
    WITH fm, p, s, se, f,
         collect(DISTINCT {
             name: fc.name,
             occurrence_rating: fc.occurrence_rating,
             detection_rating: r.detection_rating,
             preventive_measures: preventive_measures_for_cause,
             detective_measures: detective_measures_for_cause
         }) as causes_with_measures,
         collect(DISTINCT {
             name: fe.name,
             severity_rating: fe.severity_rating
         }) as effects_data
    
    RETURN fm.id as failure_mode_id,
           fm.name as failure_mode_name,
           p.name as product_name,
           s.name as subsystem_name,
           se.name as system_element_name,
           f.name as function_name,
           causes_with_measures,
           effects_data
    """
    
    result = tx.run(query)
    
    transformed_results = []
    for record in result:
        data = record.data()
        
        # Clean up causes data
        causes = []
        for cause_data in data['causes_with_measures']:
            if cause_data['name'] is not None:  
                formatted_cause = {
                    'cause_name': cause_data['name'],
                    'occurrence_rating': cause_data['occurrence_rating'],
                    'detection_rating': cause_data['detection_rating'],
                    'preventive_measures': [m for m in cause_data['preventive_measures'] if m is not None],
                    'detective_measures': [m for m in cause_data['detective_measures'] if m is not None]
                }
                causes.append(formatted_cause)
        
        # Clean up effects data
        effects = []
        for effect_data in data['effects_data']:
            if effect_data['name'] is not None: 
                formatted_effect = {
                    'effect_name': effect_data['name'],
                    'severity_rating': effect_data['severity_rating']
                }
                effects.append(formatted_effect)
        
        transformed_data = {
            'failure_mode_id': data['failure_mode_id'],
            'failure_mode_name': data['failure_mode_name'],
            'product_name': data['product_name'],
            'subsystem_name': data['subsystem_name'],
            'system_element_name': data['system_element_name'],
            'function_name': data['function_name'],
            'causes': causes,
            'effects': effects
        }
        
        transformed_results.append(transformed_data)
    
    return transformed_results

def generate_failure_mode_text_chunk(failure_mode_data):
    
    # Extract basic information
    fm_name = failure_mode_data['failure_mode_name']
    product = failure_mode_data['product_name']
    subsystem = failure_mode_data['subsystem_name']
    system_element = failure_mode_data['system_element_name']
    function = failure_mode_data['function_name']
    causes = failure_mode_data['causes']
    effects = failure_mode_data['effects']
    
    text_parts = []
    
    # System hierarchy context
    text_parts.append(f"The failure mode '{fm_name}' occurs in the '{system_element}' component, "
                     f"which is part of the '{subsystem}' subsystem in the '{product}' system.")
    
    # Function context
    text_parts.append(f"This failure affects the '{function}' function of the '{system_element}'.")
    
     # Failure causes context
    if causes and any(cause['cause_name'] for cause in causes if cause['cause_name']):
        text_parts.append(f"The failure mode '{fm_name}' can be caused by the following failure causes: ")
        for cause in causes:
            if cause['cause_name']:
                cause_text = f" Failure cause '{cause['cause_name']}'"
                if cause['occurrence_rating']:
                    cause_text += f" with an occurrence rating of {cause['occurrence_rating']}"
                if cause['detection_rating']:
                    cause_text += f" and a detection rating of {cause['detection_rating']} in the context of the failure mode '{fm_name}'."
                elif cause['occurrence_rating']:
                    cause_text += "."
                
                # Add preventive measures
                preventive_measures = [m for m in cause['preventive_measures'] if m]
                if preventive_measures:
                    cause_text += f" Preventive measures for the failure cause '{cause['cause_name']}' are: '{', '.join(preventive_measures)}'."
                
                # Add detective measures
                detective_measures = [m for m in cause['detective_measures'] if m]
                if detective_measures:
                    cause_text += f" Detective measures for detecting the failure cause '{cause['cause_name']}' in the context of failure mode '{fm_name}' are: '{', '.join(detective_measures)}'."
                
                text_parts.append(cause_text)
    
    # Failure effects context
    if effects and any(effect['effect_name'] for effect in effects if effect['effect_name']):
        text_parts.append(f" The failure mode '{fm_name}' results in the following failure effects:")
        for effect in effects:
            if effect['effect_name']:
                effect_text = f" Failure effect '{effect['effect_name']}'"
                if effect['severity_rating']:
                    effect_text += f" with a severity rating of {effect['severity_rating']}."
                text_parts.append(effect_text)
    
    full_text = " ".join(text_parts)
    
    return full_text

def create_vector_embedding_node(tx, failure_mode_id, text_chunk):
    
    query = """
    MATCH (fm:FailureMode {id: $failure_mode_id})
    MERGE (ve:VectorEmbedding {
        failure_mode_id: $failure_mode_id,
        text_chunk: $text_chunk
    })
    MERGE (fm)-[:HAS_EMBEDDING]->(ve)
    """
    
    tx.run(query, failure_mode_id=failure_mode_id, text_chunk=text_chunk)

def create_vector_index():
    # Initialize embeddings
    # This is where I could set more parameters for the embeddings like: model='mxbai-embed-large' validate_model_on_init=False base_url=None client_kwargs={} async_client_kwargs={} sync_client_kwargs={} mirostat=None mirostat_eta=None mirostat_tau=None num_ctx=None num_gpu=None keep_alive=None num_thread=None repeat_last_n=None repeat_penalty=None temperature=None stop=None tfs_z=None top_k=None top_p=None
    embeddings = OllamaEmbeddings(
        model="mxbai-embed-large",
        # Add other params as needed for production
    )
    
    # Create the actual vector index in Neo4j
    # This creates the index structure in the database
   # Neo4jVector.from_existing_graph(
       # embeddings,
        #search_type="hybrid",
        #node_label="VectorEmbedding", 
       # text_node_properties=["text_chunk"],
       # embedding_node_property="embedding",
        #index_name="failure_mode_context_index",
        #keyword_index_name="failure_mode_keyword_index")
    # Only use vector search instead of hybrid
    
    Neo4jVector.from_existing_graph(
        embeddings,
        search_type="vector",
        node_label="VectorEmbedding", 
        text_node_properties=["text_chunk"],
        embedding_node_property="embedding",
        index_name="failure_mode_context_index",
    )
    
    print("Vector index 'failure_mode_context_index' created successfully!")
    return True

def create_fmea_fulltext_indexes():
    
    driver = GraphDatabase.driver(
        uri=os.environ["NEO4J_URI"],
        auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"])
    )
    
    def create_fulltext_index(tx):
        query = '''
        CREATE FULLTEXT INDEX `fulltext_entity_id` IF NOT EXISTS
        FOR (n:SystemElement|Subsystem|FailureMode|Function|FailureCause|FailureEffect|Measure|Product) 
        ON EACH [n.name]
        '''
        tx.run(query)
    
    try:
        with driver.session() as session:
            session.execute_write(create_fulltext_index)
            print("FMEA fulltext index created successfully.")
    except Exception as e:
        print(f"Index creation failed or already exists: {e}")
    finally:
        driver.close()


def create_vector_embedding_fulltext_index():
    
    driver = GraphDatabase.driver(
        uri=os.environ["NEO4J_URI"],
        auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"])
    )
    
    def create_fulltext_index(tx):
        query = '''
        CREATE FULLTEXT INDEX `fulltext_vector_text_chunk` IF NOT EXISTS
        FOR (n:VectorEmbedding) 
        ON EACH [n.text_chunk]
        '''
        tx.run(query)
    
    try:
        with driver.session() as session:
            session.execute_write(create_fulltext_index)
            print("VectorEmbedding fulltext index created successfully.")
    except Exception as e:
        print(f"Index creation failed or already exists: {e}")
    finally:
        driver.close()
