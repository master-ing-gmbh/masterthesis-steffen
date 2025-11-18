from langchain_core.prompts import ChatPromptTemplate

def qa_system_generation_query(entities: dict) -> str:
    only_product_has_entities = (
    entities.get('Product', []) and  
    all(not entity_list for entity_type, entity_list in entities.items() if entity_type != 'Product') 
)
    if only_product_has_entities:
        product_entities = entities['Product']
        union_parts = []
        
        for product_entity in product_entities:
            clean_product = product_entity.replace("'", "\\'")
            
            # Product to Subsystem relationships
            product_subsystem_query = f"""
            MATCH (p:Product)-[r1]-(s:Subsystem)
            WHERE toLower(p.name) CONTAINS toLower('{clean_product}')
            RETURN p.name as main_node_name,
                ['Product'] as main_node_type,
                {{}} as main_node_properties,
                type(r1) as relationship,
                s.name as connected_node_name,
                ['Subsystem'] as connected_node_type,
                {{}} as connected_node_properties,
                p.name as product_context
            """
            
            # Subsystem to SystemElement relationships
            subsystem_element_query = f"""
            MATCH (p:Product)-[]-(s:Subsystem)-[r2]-(se:SystemElement)
            WHERE toLower(p.name) CONTAINS toLower('{clean_product}')
            RETURN s.name as main_node_name,
                ['Subsystem'] as main_node_type,
                {{}} as main_node_properties,
                type(r2) as relationship,
                se.name as connected_node_name,
                ['SystemElement'] as connected_node_type,
                {{}} as connected_node_properties,
                p.name as product_context
            """
            
            union_parts.append(product_subsystem_query.strip())
            union_parts.append(subsystem_element_query.strip())
        
        final_query = "\nUNION\n".join(union_parts) + " ORDER BY product_context, main_node_type DESC"
        
        return final_query
    else:
        union_parts = []
        
        for entity_type, entity_list in entities.items():
            if entity_list:
                for entity in entity_list:
                    clean_entity = entity.replace("'", "\\'")
                    
                    # entity search
                    entity_query = f"""
                    MATCH (n:{entity_type})
                    WHERE toLower(n.name) CONTAINS toLower('{clean_entity}')
                    OPTIONAL MATCH (n)-[r]-(connected)
                    WHERE NOT 'VectorEmbedding' IN labels(connected)
                    RETURN n.name as main_node_name,
                        labels(n) as main_node_type,
                        CASE 
                            WHEN 'VectorEmbedding' IN labels(n) 
                            THEN {{text_chunk: n.text_chunk}}
                            ELSE apoc.map.removeKeys(properties(n), ['id', 'name', 'embedding', 'failure_mode_id'])
                        END as main_node_properties,
                        type(r) as relationship,
                        properties(r) as relationship_properties,
                        connected.name as connected_node_name,
                        labels(connected) as connected_node_type,
                        apoc.map.removeKeys(properties(connected), ['id', 'name', 'embedding', 'failure_mode_id']) as connected_node_properties
                    """
                    
                    # VectorEmbedding text_chunk search
                    vector_query = f"""
                    MATCH (v:VectorEmbedding)
                    WHERE toLower(v.text_chunk) CONTAINS toLower('{clean_entity}')
                    RETURN null as main_node_name,
                        ['BroaderContext'] as main_node_type,
                        {{text_chunk: v.text_chunk}} as main_node_properties,
                        null as relationship,
                        null as relationship_properties,
                        null as connected_node_name,
                        null as connected_node_type,
                        null as connected_node_properties
                    """
                    
                    union_parts.append(entity_query.strip())
                    union_parts.append(vector_query.strip())
        
        if not union_parts:
            return []
        
        final_query = "\nUNION\n".join(union_parts) + " LIMIT 50"
        
        return final_query

def retrieve_qa_system_generation_data(question: str, entities: dict, llm, graph, debug: bool) -> dict:
    if debug:
        print("Entities for query: ", entities)
    
    query = qa_system_generation_query(entities)
        
    try:
        results = graph.query(query)
        results = format_qa_system_generation_results(results)
        if debug:
            print(query)
            print(results)
        return results
    except Exception as fallback_error:
        if debug: print("Fallback query failed with error: ", fallback_error)
        return []


def format_qa_system_generation_results(fallback_results: list) -> list:
    """
    Format fallback results according to specified rules:
    - Replace node names with node types as keys
    - Remove node type fields
    - Remove empty property dictionaries
    - Rename property keys to include node type
    - Special handling for BroaderContext type
    """
    
    formatted_results = []
    
    for result in fallback_results:
        formatted_result = {}
        
        # Get node types for key naming
        main_type = result.get('main_node_type', [None])[0] if result.get('main_node_type') else None
        connected_type = result.get('connected_node_type', [None])[0] if result.get('connected_node_type') else None
        
        if main_type == "BroaderContext":
            main_props = result.get('main_node_properties', {})
            if main_props and 'text_chunk' in main_props:
                formatted_result['BroaderContext'] = main_props['text_chunk']
        else:
            if main_type and result.get('main_node_name'):
                formatted_result[main_type] = result['main_node_name']
            
            main_props = result.get('main_node_properties', {})
            if main_props:
                if main_type:
                    formatted_result[f"{main_type}_properties"] = main_props
        
        if result.get('relationship'):
            formatted_result['relationship'] = result['relationship']
        
        relationship_props = result.get('relationship_properties', {})
        if relationship_props:
            for key, value in relationship_props.items():
                formatted_result[key] = value

        if connected_type and result.get('connected_node_name'):
            formatted_result[connected_type] = result['connected_node_name']
        
        connected_props = result.get('connected_node_properties', {})
        if connected_props:
            if connected_type:
                formatted_result[f"{connected_type}_properties"] = connected_props
        
        formatted_results.append(formatted_result)
    
    return formatted_results

def retrieve_functions_from_graph(element_context: dict, graph, debug):
    try:
        # Extract values from element context
        system_element = element_context.get('SystemElement')
        clean_system_element = system_element.replace("'", "\\'")

        cypher_query = f"""
        MATCH (se:SystemElement)
        WHERE toLower(se.name) CONTAINS toLower('{clean_system_element}')
        OPTIONAL MATCH (se)-[]-(f:Function)
        OPTIONAL MATCH (se)-[]-(ss:Subsystem)
        OPTIONAL MATCH (ss)-[]-(p:Product)
        RETURN se.name as system_element_name,
               f.name as function_name,
               ss.name as subsystem_name,
               p.name as product_name
        """
        raw_results = graph.query(cypher_query)
        
        function_hierarchy = []
        
        for result in raw_results:
            function_entry = {
                'Product': result.get('product_name'),
                'Subsystem': result.get('subsystem_name'), 
                'SystemElement': result.get('system_element_name'),
                'Function': result.get('function_name')
            }
            
            function_hierarchy.append(function_entry)
            
        if debug:
            print(f"Functions hierarchy for {system_element}: ", function_hierarchy)

        return function_hierarchy
     
    except Exception as e:
        if debug:
            print(f"Error retrieving functions from graph: {type(e).__name__}: {e}")
        return []
    

def retrieve_failures_from_graph(element_context: dict, graph, debug):
    try:
        # Extract values from element context
        #system_element = element_context.get('SystemElement')
        #clean_system_element = system_element.replace("'", "\\'")
        function = element_context.get('Function')
        clean_function = function.replace("'", "\\'")
        
        cypher_query = f"""
        MATCH (f:Function)
        WHERE toLower(trim(f.name)) CONTAINS toLower(trim('{clean_function}'))
        OPTIONAL MATCH (f)-[]-(se:SystemElement)
        OPTIONAL MATCH (f)-[]-(fm:FailureMode)
        OPTIONAL MATCH (fm)-[]-(fc:FailureCause)
        OPTIONAL MATCH (fm)-[]-(fe:FailureEffect)
        OPTIONAL MATCH (se)-[]-(s:Subsystem)
        OPTIONAL MATCH (s)-[]-(p:Product)
        RETURN se.name as system_element_name,
               f.name as function_name,
               fm.name as failure_mode_name,
               fc.name as failure_cause_name,
               fe.name as failure_effect_name,
               s.name as subsystem_name,
               p.name as product_name
        """
        
        raw_results = graph.query(cypher_query)
        
        failure_list = []
        
        for result in raw_results:
            failure_entry = {
                'Product': result.get('product_name'),
                'Subsystem': result.get('subsystem_name'),
                'SystemElement': result.get('system_element_name'),
                'Function': result.get('function_name'),
                'FailureMode': result.get('failure_mode_name'),
                'FailureEffect': result.get('failure_effect_name'),
                'FailureCause': result.get('failure_cause_name')
            }
            
            failure_list.append(failure_entry)
            
        if debug:
            print(f"Total Failure entries for {clean_function}: ", failure_list)

        return failure_list
     
    except Exception as e:
        if debug:
            print(f"Error retrieving failures from graph: {type(e).__name__}: {e}")
        return []
    

def retrieve_existing_measures_from_graph(element_context: dict, graph, debug):

    try:
        # Extract values from element context
        failure_cause = element_context.get('FailureCause')
        failure_mode = element_context.get('FailureMode')

        clean_failure_cause = failure_cause.replace("'", "\\'")
        clean_failure_mode = failure_mode.replace("'", "\\'")

        cypher_query = f"""
        MATCH (fc:FailureCause)-[r:isDueToFailureCause]-(fm:FailureMode)
        WHERE toLower(fc.name) CONTAINS toLower('{clean_failure_cause}')
          AND toLower(fm.name) CONTAINS toLower('{clean_failure_mode}')
        OPTIONAL MATCH (fc)-[]-(m:Measure)
        WHERE (m.type <> 'detective' OR (m)-[:improvesDetectionFor]->(fm))
        OPTIONAL MATCH (fm)-[]-(fe:FailureEffect)
        OPTIONAL MATCH (fm)-[]-(f:Function)
        OPTIONAL MATCH (f)-[]-(se:SystemElement)
        OPTIONAL MATCH (se)-[]-(s:Subsystem)
        OPTIONAL MATCH (s)-[]-(p:Product)
        RETURN fc.name as failure_cause_name,
               m.name as measure_name,
               m.type as measure_type,
               fm.name as failure_mode_name,
               fe.name as failure_effect_name,
               f.name as function_name,
               se.name as system_element_name,
               s.name as subsystem_name,
               p.name as product_name
        """

        raw_results = graph.query(cypher_query)

        measure_list = []
        
        for result in raw_results:
            measure_type = result.get('measure_type', '').lower()
            
            # Create base entry structure
            measure_entry = {
                'Product': result.get('product_name'),
                'Subsystem': result.get('subsystem_name'),
                'SystemElement': result.get('system_element_name'),
                'Function': result.get('function_name'),
                'FailureMode': result.get('failure_mode_name'),
                'FailureCause': result.get('failure_cause_name'),
                'FailureEffect': result.get('failure_effect_name'),
                'PreventiveMeasure': None,
                'DetectiveMeasure': None
            }
            
            measure_name = result.get('measure_name')
            if measure_type == 'preventive':
                measure_entry['PreventiveMeasure'] = measure_name
            elif measure_type == 'detective':
                measure_entry['DetectiveMeasure'] = measure_name
            else:
                measure_entry['PreventiveMeasure'] = measure_name
                if debug:
                    print(f"Unknown measure type '{measure_type}' for measure '{measure_name}', defaulting to preventive")
            
            measure_list.append(measure_entry)
        if debug:
            print(f"Total Failure entries for {failure_cause}: ", measure_list)

        return measure_list
     
    except Exception as e:
        if debug:
            print(f"Error retrieving failures from graph: {type(e).__name__}: {e}")
        return []
    

def retrieve_risk_ratings_from_graph(element_context: dict, graph, debug):

    try:
        # Extract values from element context
        failure_cause = element_context.get('FailureCause')
        failure_mode = element_context.get('FailureMode')
        failure_effect = element_context.get('FailureEffect')

        clean_failure_cause = failure_cause.replace("'", "\\'")
        clean_failure_effect = failure_effect.replace("'", "\\'")
        clean_failure_mode = failure_mode.replace("'", "\\'")

        cypher_query = f"""
        MATCH (fc:FailureCause)-[r:isDueToFailureCause]-(fm:FailureMode)-[]->(fe:FailureEffect)
        WHERE toLower(fc.name) CONTAINS toLower('{clean_failure_cause}')
          AND toLower(fm.name) CONTAINS toLower('{clean_failure_mode}')
          AND toLower(fe.name) CONTAINS toLower('{clean_failure_effect}')
        OPTIONAL MATCH (fc)-[]-(m:Measure)
        WHERE (m.type <> 'detective' OR (m)-[:improvesDetectionFor]->(fm))
        OPTIONAL MATCH (fm)-[]-(f:Function)
        OPTIONAL MATCH (f)-[]-(se:SystemElement)
        OPTIONAL MATCH (se)-[]-(s:Subsystem)
        OPTIONAL MATCH (s)-[]-(p:Product)
        RETURN fc.name as failure_cause_name,
                r.detection_rating as failure_cause_detection,
                fc.occurrence_rating as failure_cause_occurrence,
               m.name as measure_name,
               m.type as measure_type,
               fm.name as failure_mode_name,
               fe.name as failure_effect_name,
               fe.severity_rating as failure_effect_severity,
               f.name as function_name,
               se.name as system_element_name,
               s.name as subsystem_name,
               p.name as product_name
        """
        
        raw_results = graph.query(cypher_query)
        
        grouped_results = {}
        
        for result in raw_results:
            # Create unique key for grouping
            key = (
                result.get('product_name'),
                result.get('subsystem_name'),
                result.get('system_element_name'),
                result.get('function_name'),
                result.get('failure_mode_name'),
                result.get('failure_cause_name'),
                result.get('failure_effect_name')
            )
            
            if key not in grouped_results:
                grouped_results[key] = {
                    'Product': result.get('product_name'),
                    'Subsystem': result.get('subsystem_name'),
                    'SystemElement': result.get('system_element_name'),
                    'Function': result.get('function_name'),
                    'FailureMode': result.get('failure_mode_name'),
                    'FailureCause': result.get('failure_cause_name'),
                    'FailureEffect': result.get('failure_effect_name'),
                    'PreventiveMeasure': [],
                    'DetectiveMeasure': [],
                    'Severity': result.get('failure_effect_severity'),
                    'Detection': result.get('failure_cause_detection'),
                    'Occurrence': result.get('failure_cause_occurrence'),
                }
            
            measure_name = result.get('measure_name')
            measure_type = result.get('measure_type', '').lower()
            
            if measure_name: 
                if measure_type == 'preventive':
                    if measure_name not in grouped_results[key]['PreventiveMeasure']:
                        grouped_results[key]['PreventiveMeasure'].append(measure_name)
                elif measure_type == 'detective':
                    if measure_name not in grouped_results[key]['DetectiveMeasure']:
                        grouped_results[key]['DetectiveMeasure'].append(measure_name)
                else:
                    if measure_name not in grouped_results[key]['PreventiveMeasure']:
                        grouped_results[key]['PreventiveMeasure'].append(measure_name)
                    if debug:
                        print(f"Unknown measure type '{measure_type}' for measure '{measure_name}', defaulting to preventive")
        
        measure_list = list(grouped_results.values())
        
        if debug:
            print(f"Grouped into {len(measure_list)} unique entries")
            for entry in measure_list:
                print(f"Entry: {entry}")

        return measure_list
     
    except Exception as e:
        if debug:
            print(f"Error retrieving failures from graph: {type(e).__name__}: {e}")
        return []