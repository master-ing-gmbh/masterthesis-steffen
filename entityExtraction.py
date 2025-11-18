from langchain_core.prompts import ChatPromptTemplate
import json

def extract_entities_from_question(question: str, llm, debug: bool) -> dict:
    fmea_entity_prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "You are an FMEA entity extraction specialist. Extract relevant entities from user questions "
         "to enable precise knowledge graph queries. Map entities to these exact types: "
         "FailureCause, FailureEffect, FailureMode, Function, Measure, Product, Subsystem, SystemElement. "
         "\n\nPRODUCT ENTITY SPECIAL RULES:"
         "\n- When user explicitly requests system structure generation ONLY the Product entity should be extracted."
         "\n- Generation keywords: 'create', 'develop', 'build', 'generate', 'design', 'structure'"
         "\n- For Product entities, provide 2 additional linguistic variations: synonyms and abstraction levels"
         "\n- Product entities should NOT be in additional list like here: [['Washing Machine', 'Laundry Appliance', 'Household Washer']]"
         "\n- Example: ['Electric Vehicle', 'EV', 'Electric Car'] (extracted entity, synonym, abstraction)"
         "\n\nENTITY MAPPING GUIDELINES:"
         "\n- SystemElement: Physical components (motor, pump, brake, sensor, valve)"
         "\n- Subsystem: Groups of components (brake system, hydraulic system, control system)"
         "\n- Product: Complete systems or end products"
         "\n- Function: What something does (braking, pumping, cooling, monitoring)"
         "\n- FailureMode: Ways things can fail (overheating, leakage, fracture, jamming)"
         "\n- FailureCause: Root causes (wear, corrosion, overload, contamination)" 
         "\n- FailureEffect: Consequences (loss of function, safety risk, performance degradation)"
         "\n- Measure: Preventive/corrective actions (inspection, maintenance, design change)"
         "\n\nSYNONYM HANDLING:"
         "\n- Technical variants: breakdown→failure, component→SystemElement"
         "\n- Context-aware: 'failure' alone → look for context clues"
         "\n\nFORMAT REQUIREMENTS:"
         "\n- Return ONLY valid JSON, no explanations or additional text"
         "\n- Use empty arrays [] for entity types with no matches"
         "\n- All entity type keys must be present in output"
         "\n- Entity names should be clean and standardized"
         "\n- Entity types must match the defined categories exactly and are the key of the output, following a list with extracted entities."
        ),
        ("human", 
         "Extract FMEA entities from this question: {question}"
         "\nReturn only the JSON with all entity types, using empty arrays for unmatched types."
        )
    ])
    
    
    try:
        response = llm.invoke(fmea_entity_prompt.format(question=question))
        if debug: print(response)
        json_text = response.content.strip()
        if debug:
            print(f"Raw LLM response: '{json_text}'") 
            print(f"Response length: {len(json_text)}")
        
        json_text = json_text.replace('```json', '').replace('```', '')
        json_text = json_text.strip()
        
        entities = json.loads(json_text)
        if debug:
            print("Entities: ", entities)
            print("Successful entity extraction")
        return entities
    
    except json.JSONDecodeError as e:
        if debug:
            print(f"JSON parsing failed: {e}")
            print(f"Problematic text: '{json_text}'") 
        return {
            "FailureCause": [],
            "FailureEffect": [],
            "FailureMode": [],
            "Function": [],
            "Measure": [],
            "Product": [],
            "Subsystem": [],
            "SystemElement": []
        }
    except Exception as e:
        if debug:
            print(f"General entity extraction failed: {e}")
        return {
            "FailureCause": [],
            "FailureEffect": [],
            "FailureMode": [],
            "Function": [],
            "Measure": [],
            "Product": [],
            "Subsystem": [],
            "SystemElement": []
        }

def extract_system_elements(data, debug):
    system_structure_list = []
    try:
        for index, item in enumerate(data):

            product = item.get('Product')
            subsystem = item.get('Subsystem')
            system_element = item.get('SystemElement')

            if not all([product, subsystem, system_element]):
                if debug:
                    print(f"  Skipping item {index + 1}: Missing required field(s)")
                continue
            
            hierarchy_entry = {
                'Product': product,
                'Subsystem': subsystem,
                'SystemElement': system_element
            }
            
            system_structure_list.append(hierarchy_entry)
        if debug:
            print("Extracted system structure: ", system_structure_list)
        return system_structure_list

    except Exception as e:
        if debug:
            print(f"Error in Extraction of system structure: {type(e).__name__}: {e}")
        return []
    

def extract_system_elements_with_functions(data, debug):
    functions_list = []
    try:
        for index, item in enumerate(data):

            # Extract the required fields
            product = item.get('Product')
            subsystem = item.get('Subsystem')
            system_element = item.get('SystemElement')
            function = item.get('Function')

            if not all([product, subsystem, system_element, function]):
                if debug:
                    print(f"  Skipping item {index + 1}: Missing required field(s)")
                continue
            
            hierarchy_entry = {
                'Product': product,
                'Subsystem': subsystem,
                'SystemElement': system_element,
                'Function': function
            }
            
            functions_list.append(hierarchy_entry)
        if debug:
            print("Extracted functions: ", functions_list)
        return functions_list

    except Exception as e:
        if debug:
            print(f"Error in Extraction of functions: {type(e).__name__}: {e}")
        return []

def extract_failure_chains(data, debug):
    grouped_results = {}
    
    try:
        for index, item in enumerate(data):
            # Extract the required fields
            product = item.get('Product')
            subsystem = item.get('Subsystem')
            system_element = item.get('SystemElement')
            function = item.get('Function')
            failure_mode = item.get('FailureMode')
            failure_cause = item.get('FailureCause')
            failure_effect = item.get('FailureEffect')

            if not all([product, subsystem, system_element, function, failure_mode, failure_cause, failure_effect]) or \
               failure_mode == "None" or failure_cause == "None" or failure_effect == "None":
                if debug:
                    print(f"Skipping item {index + 1}: Missing required field(s) or contains 'None' values")
                continue
            
            key = (
                str(product) if product is not None else "",
                str(subsystem) if subsystem is not None else "",
                str(system_element) if system_element is not None else "",
                str(function) if function is not None else "",
                str(failure_mode) if failure_mode is not None else "",
                str(failure_cause) if failure_cause is not None else "",
                str(failure_effect) if failure_effect is not None else ""
            )
            
            if key not in grouped_results:
                grouped_results[key] = {
                    'Product': product,
                    'Subsystem': subsystem,
                    'SystemElement': system_element,
                    'Function': function,
                    'FailureMode': failure_mode,
                    'FailureCause': failure_cause,
                    'FailureEffect': failure_effect,
                    'PreventiveMeasure': [],
                    'DetectiveMeasure': [],
                }
            
            # Add measures to appropriate lists if they are not None
            preventive_measure = item.get('PreventiveMeasure')
            detective_measure = item.get('DetectiveMeasure')
            
            if preventive_measure and preventive_measure != "None" and preventive_measure not in grouped_results[key]['PreventiveMeasure']:
                grouped_results[key]['PreventiveMeasure'].append(preventive_measure)
            
            if detective_measure and detective_measure != "None" and detective_measure not in grouped_results[key]['DetectiveMeasure']:
                grouped_results[key]['DetectiveMeasure'].append(detective_measure)
        
        failure_chains = list(grouped_results.values())
        
        if debug:
            print(f"Extracted {len(failure_chains)} unique failure chains")
        
        return failure_chains

    except Exception as e:
        if debug:
            print(f"Error in extraction of failure chains: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
        return []

def extract_failure_chains_with_risk_ratings(data, debug):
    grouped_results = {}
    
    try:
        for index, item in enumerate(data):
            # Extract the required fields
            product = item.get('Product')
            subsystem = item.get('Subsystem')
            system_element = item.get('SystemElement')
            function = item.get('Function')
            failure_mode = item.get('FailureMode')
            failure_cause = item.get('FailureCause')
            failure_effect = item.get('FailureEffect')
            occurrence_rating = item.get('Occurrence')
            detection_rating = item.get('Detection')
            severity_rating = item.get('Severity')

            if not all([product, subsystem, system_element, function, failure_mode, failure_cause, failure_effect]):
                if debug:
                    print(f"Skipping item {index + 1}: Missing required field(s)")
                continue
            
            key = (
                product, subsystem, system_element, function, 
                failure_mode, failure_cause, failure_effect, occurrence_rating, detection_rating, severity_rating
            )
            
            if key not in grouped_results:
                grouped_results[key] = {
                    'Product': product,
                    'Subsystem': subsystem,
                    'SystemElement': system_element,
                    'Function': function,
                    'FailureMode': failure_mode,
                    'FailureCause': failure_cause,
                    'FailureEffect': failure_effect,
                    'PreventiveMeasure': [],
                    'DetectiveMeasure': [],
                    'Occurrence': occurrence_rating,
                    'Detection': detection_rating,
                    'Severity': severity_rating,
                }
            
            # Add measures to appropriate lists
            preventive_measure = item.get('PreventiveMeasure')
            detective_measure = item.get('DetectiveMeasure')
            
            if preventive_measure and preventive_measure not in grouped_results[key]['PreventiveMeasure']:
                grouped_results[key]['PreventiveMeasure'].append(preventive_measure)
            
            if detective_measure and detective_measure not in grouped_results[key]['DetectiveMeasure']:
                grouped_results[key]['DetectiveMeasure'].append(detective_measure)
        
        failure_chains = list(grouped_results.values())
        
        
        return failure_chains

    except Exception as e:
        if debug:
            print(f"Error in extraction of failure chains: {type(e).__name__}: {e}")
        return []