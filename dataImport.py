import pandas as pd
import os
from neo4j import GraphDatabase

def data_upload_and_mapping_to_graph():
    # Initialize Neo4j driver
    driver = GraphDatabase.driver(
        uri=os.environ["NEO4J_URI"],
        auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"])
    )

    def clear_database_completely(session):
        # Drop constraints in separate transaction
        with session.begin_transaction() as tx:
            constraints = list(tx.run("SHOW CONSTRAINTS"))
            tx.commit()
        
        for constraint in constraints:
            session.run(f"DROP CONSTRAINT `{constraint['name']}` IF EXISTS")
        
        # Drop indexes in separate transaction
        with session.begin_transaction() as tx:
            indexes = list(tx.run("SHOW INDEXES"))
            tx.commit()
        
        for index in indexes:
            if index.get('owningConstraint') is None:
                session.run(f"DROP INDEX `{index['name']}` IF EXISTS")
        
        # Delete all data
        session.run("MATCH (n) DETACH DELETE n")

    def clean_name(name):
        if name is None:
            return None
        cleaned = str(name).replace('\t', ' ').replace('\n', ' ').replace('\r', ' ')
        import re
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        return cleaned
        
    def clean_all_node_names(session):
        query = """
        MATCH (n)
        WHERE n.name IS NOT NULL
        SET n.name = trim(
            replace(
                replace(
                    replace(n.name, "  ", " "), 
                    "\n", " "
                ), 
                "\t", " "
            )
        )
        """
        session.run(query)
        print("Cleaned all node names in database")
        
    def import_fmea_data(csv_file_path):
        df = pd.read_csv(
                csv_file_path,
                sep=';',                    
                encoding='utf-8',           
                quotechar='"',              
                on_bad_lines='warn',        
                engine='python',            
                skipinitialspace=True,      
                doublequote=True            
)
        
        entity_counters = {
            'product': 0,
            'subsystem': 0,
            'system_element': 0,
            'function': 0,
            'failure_mode': 0,
            'failure_cause': 0,
            'failure_effect': 0,
            'measure': 0,
        }
        
        # Track existing entities to avoid duplicates and ID conflicts
        existing_entities = {
            'product': {},
            'subsystem': {},
            'system_element': {},
            'function': {},
            'failure_mode': {},
            'failure_cause': {},
            'failure_effect': {},
            'measure': {},
        }
        
        
        with driver.session() as session:
            # Clear existing data
            clear_database_completely(session)
            print("Database cleared.")
            
            for index, row in df.iterrows():
                print(f"Processing row {index + 1}/{len(df)}")
                
                cleaned_row = {
                    'product': clean_name(row['product']),
                    'subsystem': clean_name(row['subsystem']),
                    'system_element': clean_name(row['system_element']),
                    'function': clean_name(row['function']),
                    'failure_mode': clean_name(row['failure_mode']),
                    'failure_effect': clean_name(row['failure_effect']),
                    'failure_cause': clean_name(row['failure_cause']),
                    'measure_name': clean_name(row['measure_name']),
                    'measure_type': row['measure_type'], 
                    'severity': row['severity'],
                    'occurrence': row['occurrence'],
                    'detection': row['detection']
                }
                
                # Create/get Product
                product_key = cleaned_row['product']
                if product_key not in existing_entities['product']:
                    entity_counters['product'] += 1
                    product_id = entity_counters['product']
                    existing_entities['product'][product_key] = product_id
                    session.execute_write(create_product_node, product_id, cleaned_row['product'])
                else:
                    product_id = existing_entities['product'][product_key]
                
                # Create/get Subsystem
                subsystem_key = f"{product_key}_{cleaned_row['subsystem']}"
                if subsystem_key not in existing_entities['subsystem']:
                    entity_counters['subsystem'] += 1
                    subsystem_id = entity_counters['subsystem']
                    existing_entities['subsystem'][subsystem_key] = subsystem_id
                    session.execute_write(create_subsystem_node, subsystem_id, cleaned_row['subsystem'], product_id)
                else:
                    subsystem_id = existing_entities['subsystem'][subsystem_key]
                
                # Create/get SystemElement
                system_element_key = f"{subsystem_key}_{cleaned_row['system_element']}"
                if system_element_key not in existing_entities['system_element']:
                    entity_counters['system_element'] += 1
                    system_element_id = entity_counters['system_element']
                    existing_entities['system_element'][system_element_key] = system_element_id
                    session.execute_write(create_system_element_node, system_element_id, cleaned_row['system_element'], subsystem_id)
                else:
                    system_element_id = existing_entities['system_element'][system_element_key]
                
                # Create/get Function
                function_key = f"{system_element_key}_{cleaned_row['function']}"
                if function_key not in existing_entities['function']:
                    entity_counters['function'] += 1
                    function_id = entity_counters['function']
                    existing_entities['function'][function_key] = function_id
                    session.execute_write(create_function_node, function_id, cleaned_row['function'], system_element_id)
                else:
                    function_id = existing_entities['function'][function_key]
                
                # Create/get FailureMode
                failure_mode_key = f"{system_element_key}_{cleaned_row['failure_mode']}"
                if failure_mode_key not in existing_entities['failure_mode']:
                    entity_counters['failure_mode'] += 1
                    failure_mode_id = entity_counters['failure_mode']
                    existing_entities['failure_mode'][failure_mode_key] = failure_mode_id
                    session.execute_write(create_failure_mode_node, failure_mode_id, cleaned_row['failure_mode'], function_id)
                else:
                    failure_mode_id = existing_entities['failure_mode'][failure_mode_key]
                    session.execute_write(create_failure_mode_function_relationship, function_id, failure_mode_id)
                
                # Create/get FailureEffect 
                failure_effect_key = f"{system_element_key}_{cleaned_row['failure_effect']}"
                if failure_effect_key not in existing_entities['failure_effect']:
                    entity_counters['failure_effect'] += 1
                    failure_effect_id = entity_counters['failure_effect']
                    existing_entities['failure_effect'][failure_effect_key] = failure_effect_id
                    session.execute_write(create_failure_effect_node, failure_effect_id, cleaned_row['failure_effect'], 
                                        cleaned_row['severity'], failure_mode_id)
                else:
                    failure_effect_id = existing_entities['failure_effect'][failure_effect_key]
                    session.execute_write(create_failure_mode_effect_relationship, failure_mode_id, failure_effect_id)
                
                # Create/get FailureCause
                failure_cause_key = f"{system_element_key}_{cleaned_row['failure_cause']}"
                if failure_cause_key not in existing_entities['failure_cause']:
                    entity_counters['failure_cause'] += 1
                    failure_cause_id = entity_counters['failure_cause']
                    existing_entities['failure_cause'][failure_cause_key] = failure_cause_id
                    
                    session.execute_write(create_failure_cause_node, failure_cause_id, cleaned_row['failure_cause'], 
                                        cleaned_row['occurrence'], failure_mode_id, cleaned_row['detection'])
                else:
                    failure_cause_id = existing_entities['failure_cause'][failure_cause_key]
                    
                    session.execute_write(create_failure_mode_cause_relationship, failure_mode_id, failure_cause_id, cleaned_row['detection'])
                
                # Create/get Measure with type-specific uniqueness
                measure_key = f"{system_element_key}_{cleaned_row['measure_name']}_{cleaned_row['measure_type']}"
                if measure_key not in existing_entities['measure']:
                    entity_counters['measure'] += 1
                    measure_id = entity_counters['measure']
                    existing_entities['measure'][measure_key] = measure_id
                    session.execute_write(create_measure_node, measure_id, cleaned_row['measure_name'], 
                                        cleaned_row['measure_type'], failure_cause_id, failure_mode_id)
                else:
                    measure_id = existing_entities['measure'][measure_key]
                    session.execute_write(create_measure_relationship, failure_cause_id, measure_id, 
                                        cleaned_row['measure_type'], failure_mode_id)
                
            clean_all_node_names(session)
                
        print(f"\nImport completed successfully!")
        print(f"Created {entity_counters['product']} products")
        print(f"Created {entity_counters['subsystem']} subsystems")
        print(f"Created {entity_counters['system_element']} system elements")
        print(f"Created {entity_counters['function']} functions")
        print(f"Created {entity_counters['failure_mode']} failure modes")
        print(f"Created {entity_counters['failure_cause']} failure causes")
        print(f"Created {entity_counters['failure_effect']} failure effects")
        print(f"Created {entity_counters['measure']} measures")

    

    # Node creation functions
    def create_product_node(tx, product_id, name):
        query = """
        MERGE (p:Product {id: $product_id, name: $name})
        """
        tx.run(query, product_id=product_id, name=name)

    def create_subsystem_node(tx, subsystem_id, name, product_id):
        query = """
        MERGE (s:Subsystem {id: $subsystem_id, name: $name})
        MERGE (p:Product {id: $product_id})
        MERGE (p)-[:hasSubsystem]->(s)
        """
        tx.run(query, subsystem_id=subsystem_id, name=name, product_id=product_id)

    def create_system_element_node(tx, system_element_id, name, subsystem_id):
        query = """
        MERGE (se:SystemElement {id: $system_element_id, name: $name})
        MERGE (s:Subsystem {id: $subsystem_id})
        MERGE (s)-[:hasSystemElement]->(se)
        """
        tx.run(query, system_element_id=system_element_id, name=name, subsystem_id=subsystem_id)

    def create_function_node(tx, function_id, name, system_element_id):
        query = """
        MERGE (f:Function {id: $function_id, name: $name})
        MERGE (se:SystemElement {id: $system_element_id})
        MERGE (se)-[:hasFunction]->(f)
        """
        tx.run(query, function_id=function_id, name=name, system_element_id=system_element_id)

    def create_failure_mode_node(tx, failure_mode_id, name, function_id):
        query = """
        MERGE (fm:FailureMode {id: $failure_mode_id, name: $name})
        MERGE (f:Function {id: $function_id})
        MERGE (f)-[:hasFailureMode]->(fm)
        """
        tx.run(query, failure_mode_id=failure_mode_id, name=name, function_id=function_id)

    def create_failure_effect_node(tx, failure_effect_id, name, severity_rating, failure_mode_id):
        query = """
        MERGE (fe:FailureEffect {id: $failure_effect_id, name: $name, severity_rating: $severity_rating})
        MERGE (fm:FailureMode {id: $failure_mode_id})
        MERGE (fm)-[:resultsInFailureEffect]->(fe)
        """
        tx.run(query, failure_effect_id=failure_effect_id, name=name, 
            severity_rating=severity_rating, failure_mode_id=failure_mode_id)

    def create_failure_cause_node(tx, failure_cause_id, name, occurrence_rating, failure_mode_id, detection_rating):
        query = """
        MERGE (fc:FailureCause {id: $failure_cause_id, name: $name, occurrence_rating: $occurrence_rating})
        MERGE (fm:FailureMode {id: $failure_mode_id})
        MERGE (fm)-[:isDueToFailureCause {detection_rating: $detection_rating}]->(fc)
        """
        tx.run(query, failure_cause_id=failure_cause_id, name=name, 
            occurrence_rating=occurrence_rating, failure_mode_id=failure_mode_id,
            detection_rating=detection_rating)

    def create_measure_node(tx, measure_id, name, measure_type, failure_cause_id, failure_mode_id):
        if measure_type == 'preventive':
            query = """
            MERGE (m:Measure {id: $measure_id, name: $name, type: $measure_type})
            MERGE (fc:FailureCause {id: $failure_cause_id})
            MERGE (fc)-[:isImprovedByPreventiveMeasure]->(m)
            """
            tx.run(query, measure_id=measure_id, name=name, 
                measure_type=measure_type, failure_cause_id=failure_cause_id)
        else:  
            query = """
            MERGE (m:Measure {id: $measure_id, name: $name, type: $measure_type})
            MERGE (fc:FailureCause {id: $failure_cause_id})
            MERGE (fm:FailureMode {id: $failure_mode_id})
            MERGE (fc)-[:isImprovedByDetectiveMeasure]->(m)
            MERGE (m)-[:improvesDetectionFor]->(fm)
            """
            tx.run(query, measure_id=measure_id, name=name, 
                measure_type=measure_type, failure_cause_id=failure_cause_id,
                failure_mode_id=failure_mode_id)

    # Relationship creation functions for existing nodes
    def create_failure_mode_function_relationship(tx, function_id, failure_mode_id):
        query = """
        MERGE (f:Function {id: $function_id})
        MERGE (fm:FailureMode {id: $failure_mode_id})
        MERGE (f)-[:hasFailureMode]->(fm)
        """
        tx.run(query, function_id=function_id, failure_mode_id=failure_mode_id)

    def create_failure_mode_effect_relationship(tx, failure_mode_id, failure_effect_id):
        query = """
        MERGE (fm:FailureMode {id: $failure_mode_id})
        MERGE (fe:FailureEffect {id: $failure_effect_id})
        MERGE (fm)-[:resultsInFailureEffect]->(fe)
        """
        tx.run(query, failure_mode_id=failure_mode_id, failure_effect_id=failure_effect_id)

    def create_failure_mode_cause_relationship(tx, failure_mode_id, failure_cause_id, detection_rating):
        query = """
        MERGE (fm:FailureMode {id: $failure_mode_id})
        MERGE (fc:FailureCause {id: $failure_cause_id})
        MERGE (fm)-[:isDueToFailureCause {detection_rating: $detection_rating}]->(fc)
        """
        tx.run(query, failure_mode_id=failure_mode_id, failure_cause_id=failure_cause_id,
            detection_rating=detection_rating)
    
    def create_measure_relationship(tx, failure_cause_id, measure_id, measure_type, failure_mode_id):
        if measure_type == 'preventive':
            query = """
            MERGE (fc:FailureCause {id: $failure_cause_id})
            MERGE (m:Measure {id: $measure_id})
            MERGE (fc)-[:isImprovedByPreventiveMeasure]->(m)
            """
            tx.run(query, failure_cause_id=failure_cause_id, measure_id=measure_id)
        else: 
            query = """
            MERGE (fc:FailureCause {id: $failure_cause_id})
            MERGE (fm:FailureMode {id: $failure_mode_id})
            MERGE (m:Measure {id: $measure_id})
            MERGE (fc)-[:isImprovedByDetectiveMeasure]->(m)
            MERGE (m)-[:improvesDetectionFor]->(fm)
            """
            tx.run(query, failure_cause_id=failure_cause_id, measure_id=measure_id,
                failure_mode_id=failure_mode_id)



    csv_file_path = "data/EngineBlockCleaned.csv"
    import_fmea_data(csv_file_path)
    driver.close()