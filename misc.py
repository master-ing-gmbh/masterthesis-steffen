
def comprehensive_retriever(graph_data: list, vector_data: list, debug: bool) -> str:
    
    # Structure the combined data for the language model
    final_context = f"""
    === GRAPH QUERY DATA ===
    {graph_data}

    === VECTOR QUERY DATA ===
    {vector_data}
    """
    if debug: print("Final combined context for output generation:", final_context)
    return final_context

def add_functions_to_table_structure(table_structure, system_structure_context: dict, generated_functions: list, debug):
    try:
        # Extract context values
        product = system_structure_context.get('Product')
        subsystem = system_structure_context.get('Subsystem')
        system_element = system_structure_context.get('SystemElement')
        
        if table_structure:
            max_row_id = max(row.get('row_id', 0) for row in table_structure)
            next_row_id = max_row_id + 1
        else:
            next_row_id = 1
        for index, function in enumerate(generated_functions):

            # Create structure with function and null fields
            function_row = {
                "row_id": next_row_id + index,
                "Product": product,
                "Subsystem": subsystem,
                "SystemElement": system_element,
                "Function": function,
                "FailureMode": None,
                "FailureCause": None,
                "FailureEffect": None,
                "PreventiveMeasure": None,
                "DetectiveMeasure": None,
                "SeverityRating": None,
                "OccurrenceRating": None,
                "DetectionRating": None,
            }
            
            table_structure.append(function_row)
            
            
    except Exception as e:
        if debug:
            print(f"Error creating function rows: {type(e).__name__}: {e}")
        return table_structure
    
    return table_structure

def add_failure_modes_to_table_structure(table_structure, system_structure_context: dict, generated_failure_modes: list, debug):
    try:
        # Extract context values
        product = system_structure_context.get('Product')
        subsystem = system_structure_context.get('Subsystem')
        system_element = system_structure_context.get('SystemElement')
        function = system_structure_context.get('Function')
        
        if table_structure:
            max_row_id = max(row.get('row_id', 0) for row in table_structure)
            next_row_id = max_row_id + 1
        else:
            next_row_id = 1
        for index, failure in enumerate(generated_failure_modes):

            function_row = {
                "row_id": next_row_id + index,
                "Product": product,
                "Subsystem": subsystem,
                "SystemElement": system_element,
                "Function": function,
                "FailureMode": failure.get('FailureMode'),
                "FailureCause": failure.get('FailureCause'),
                "FailureEffect": failure.get('FailureEffect'),
                "PreventiveMeasure": None,
                "DetectiveMeasure": None,
                "SeverityRating": None,
                "OccurrenceRating": None,
                "DetectionRating": None,
            }
            
            table_structure.append(function_row)
            
            
    except Exception as e:
        if debug:
            print(f"Error creating function rows: {type(e).__name__}: {e}")
        return table_structure
    
    return table_structure

def add_existing_measures_to_table_structure(table_structure, system_structure_context: dict, generated_measures: list, debug):
    try:
        # Extract context values
        product = system_structure_context.get('Product')
        subsystem = system_structure_context.get('Subsystem')
        system_element = system_structure_context.get('SystemElement')
        function = system_structure_context.get('Function')
        failure_mode = system_structure_context.get('FailureMode')
        failure_effect = system_structure_context.get('FailureEffect')
        failure_cause = system_structure_context.get('FailureCause')

        if table_structure:
            max_row_id = max(row.get('row_id', 0) for row in table_structure)
            next_row_id = max_row_id + 1
        else:
            next_row_id = 1
        
        for measure_dict in generated_measures:
            if not isinstance(measure_dict, dict):
                if debug:
                    print(f"Skipping non-dict item: {measure_dict}")
                continue
                
            if "PreventiveMeasure" in measure_dict:
                preventive_measure = measure_dict["PreventiveMeasure"]
                if isinstance(preventive_measure, str) and preventive_measure.strip():
                    row = {
                        "row_id": next_row_id,
                        "Product": product,
                        "Subsystem": subsystem,
                        "SystemElement": system_element,
                        "Function": function,
                        "FailureMode": failure_mode,
                        "FailureCause": failure_cause,
                        "FailureEffect": failure_effect,
                        "PreventiveMeasure": preventive_measure,
                        "DetectiveMeasure": None,
                        "SeverityRating": None,
                        "OccurrenceRating": None,
                        "DetectionRating": None,
                    }
                    table_structure.append(row)
                    next_row_id += 1

            elif "DetectiveMeasure" in measure_dict:
                detective_measure = measure_dict["DetectiveMeasure"]
                if isinstance(detective_measure, str) and detective_measure.strip():
                    row = {
                        "row_id": next_row_id,
                        "Product": product,
                        "Subsystem": subsystem,
                        "SystemElement": system_element,
                        "Function": function,
                        "FailureMode": failure_mode,
                        "FailureCause": failure_cause,
                        "FailureEffect": failure_effect,
                        "PreventiveMeasure": None,
                        "DetectiveMeasure": detective_measure,
                        "SeverityRating": None,
                        "OccurrenceRating": None,
                        "DetectionRating": None,
                    }
                    table_structure.append(row)
                    next_row_id += 1
        
    except Exception as e:
        if debug:
            print(f"Error creating function rows: {type(e).__name__}: {e}")
        return table_structure
    
    return table_structure

def add_risk_rating_to_table_structure(table_structure, system_structure_context: dict, generated_risk_ratings: dict, debug):
    try:
        # Extract context values
        product = system_structure_context.get('Product')
        subsystem = system_structure_context.get('Subsystem')
        system_element = system_structure_context.get('SystemElement')
        function = system_structure_context.get('Function')
        failure_mode = system_structure_context.get('FailureMode')
        failure_cause = system_structure_context.get('FailureCause')
        failure_effect = system_structure_context.get('FailureEffect')
        detective_measures = system_structure_context.get('DetectiveMeasure', [])
        preventive_measures = system_structure_context.get('PreventiveMeasure', [])

        # Extract risk ratings
        severity_rating = generated_risk_ratings.get('Severity')
        occurrence_rating = generated_risk_ratings.get('Occurrence')
        detection_rating = generated_risk_ratings.get('Detection')

        if table_structure:
            max_row_id = max(row.get('row_id', 0) for row in table_structure)
            next_row_id = max_row_id + 1
        else:
            next_row_id = 1
        
        current_row_id = next_row_id

        # Create rows for detective measures
        for detective_measure in detective_measures:
            function_row = {
                "row_id": current_row_id,
                "Product": product,
                "Subsystem": subsystem,
                "SystemElement": system_element,
                "Function": function,
                "FailureMode": failure_mode,
                "FailureCause": failure_cause,
                "FailureEffect": failure_effect,
                "PreventiveMeasure": None,
                "DetectiveMeasure": detective_measure,
                "Severity": severity_rating,
                "Occurrence": occurrence_rating,
                "Detection": detection_rating,
            }
            
            table_structure.append(function_row)
            current_row_id += 1
            
        # Create rows for preventive measures
        for preventive_measure in preventive_measures:
            function_row = {
                "row_id": current_row_id,
                "Product": product,
                "Subsystem": subsystem,
                "SystemElement": system_element,
                "Function": function,
                "FailureMode": failure_mode,
                "FailureCause": failure_cause,
                "FailureEffect": failure_effect,
                "PreventiveMeasure": preventive_measure,
                "DetectiveMeasure": None,
                "Severity": severity_rating,
                "Occurrence": occurrence_rating,
                "Detection": detection_rating,
            }
            
            table_structure.append(function_row)
            current_row_id += 1
            
    except Exception as e:
        if debug:
            print(f"Error creating risk rating rows: {type(e).__name__}: {e}")
        return table_structure
    
    return table_structure

def add_new_measures_to_table_structure(table_structure, system_structure_context: dict, generated_new_measures: dict, debug):
    try:
        # Extract context values
        product = system_structure_context.get('Product')
        subsystem = system_structure_context.get('Subsystem')
        system_element = system_structure_context.get('SystemElement')
        function = system_structure_context.get('Function')
        failure_mode = system_structure_context.get('FailureMode')
        failure_effect = system_structure_context.get('FailureEffect')
        failure_cause = system_structure_context.get('FailureCause')
        occurrence_rating = system_structure_context.get('Occurrence')
        detection_rating = system_structure_context.get('Detection')
        severity_rating = system_structure_context.get('Severity')
        
        # Get existing measures from context
        existing_preventive_measures = system_structure_context.get('PreventiveMeasure', [])
        existing_detective_measures = system_structure_context.get('DetectiveMeasure', [])

        if table_structure:
            max_row_id = max(row.get('row_id', 0) for row in table_structure)
            next_row_id = max_row_id + 1
        else:
            next_row_id = 1

        new_preventive_measures = []
        new_detective_measures = []
        
        for measure_dict in generated_new_measures:
            if 'PreventiveMeasure' in measure_dict:
                new_preventive_measures.append(measure_dict['PreventiveMeasure'])
            elif 'DetectiveMeasure' in measure_dict:
                new_detective_measures.append(measure_dict['DetectiveMeasure'])

        row = {
            "row_id": next_row_id,
            "Product": product,
            "Subsystem": subsystem,
            "SystemElement": system_element,
            "Function": function,
            "FailureMode": failure_mode,
            "FailureCause": failure_cause,
            "FailureEffect": failure_effect,
            "PreventiveMeasure": ", ".join(existing_preventive_measures) if existing_preventive_measures else None,
            "DetectiveMeasure": ", ".join(existing_detective_measures) if existing_detective_measures else None,
            "Severity": severity_rating,
            "Occurrence": occurrence_rating,
            "Detection": detection_rating,
            "NewPreventiveMeasure": ", ".join(new_preventive_measures) if new_preventive_measures else None,
            "NewDetectiveMeasure": ", ".join(new_detective_measures) if new_detective_measures else None,
        }
        
        table_structure.append(row)
        
        
    except Exception as e:
        if debug:
            print(f"Error creating measure row: {type(e).__name__}: {e}")
        return table_structure
    
    return table_structure