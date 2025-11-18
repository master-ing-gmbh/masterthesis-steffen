from langchain_core.prompts import ChatPromptTemplate
import json


def generate_answer_system_structure(question, context, llm, debug: bool):
    comprehensive_prompt = ChatPromptTemplate.from_messages([
        (
            "system", 
            "You are an expert FMEA analyst specializing in technical risk analysis and system decomposition.\n\n"

            "## SYSTEM SIMILARITY DEFINITIONS\n"
            "**Product Category**: Group of products serving similar purposes (e.g., home appliances, automotive systems, industrial equipment)\n"
            "**System Element Identity**: Core functional component regardless of naming variations (motor = engine, pump = compressor)\n"
            "**Subsystem Context**: Functional grouping within product (motor system, control system, fluid system)\n\n"
            
            "## CONTEXT ANALYSIS DECISION MATRIX\n"
            
            "### 1. EXACT MATCH\n"
            "**Criteria**: Question context matches retrieved data context exactly\n"
            "- Allow naming variations (motor/engine, brake/braking system)\n"
            "- Same product category and functional context\n\n"
   
            "### 2. SIMILAR CONTEXT\n"
            "**Criteria**: Same SystemElement in different Subsystem OR different Product within same category\n"
            "- Examples: Motor in washing machine vs dishwasher, Brake system in car vs truck\n"
            "- Same product category but different application context\n\n"

            "### 3. DIFFERENT CONTEXT\n"
            "**Criteria**: Different product category OR no relevant retrieved data\n"
            "- Examples: Hydraulic pump vs electric motor, automotive vs aerospace systems\n"
            "- Fundamentally different operational contexts\n\n"

            "## REQUEST TYPE DETECTION\n"
            "Analyze user request and respond according to type:\n\n"

            "### TYPE 1: SYSTEM STRUCTURE GENERATION\n"
            "**Detection Keywords**: 'create', 'develop', 'build', 'generate', 'design', 'structure'\n\n"
            
            "**Goal**: Detailed identification and decomposition of FMEA scope into system, subsystem, system elements for technical risk analysis\n\n"

            "**Requirements**:\n"
            "- Create modular structure supporting reuse in future analyses\n"
            "- Define and delimit constructive interfaces\n"
            "- Generate a complete system structure with all system elements required for the product to function\n"
            "- Use the exact name properties for products, subsystems and system elements from the retrieved data. Do NOT rename any retrieved system elements. Be careful with similar names, even these should not be altered. The exact names are important for the further graph query.\n"
            "- Generate Product-Subsystem-SystemElement hierarchy\n"
            "- When in doubt, data from the graph retrieval should be preferred.\n"
            "- Each SystemElement gets its own row. Use NONE for Function, FailureMode, FailureCause (filled in later)\n"
            
            
            "**Context-Based Generation Strategy**:\n"
            "- **EXACT MATCH**: Use retrieved data structures with the same context as question context as only data source. Do NOT rename any retrieved subsystems or system elements.\n"
            "- **SIMILAR CONTEXT**: Adapt retrieved structures to target product context. Verify retrieved information still makes sense for target product and is applicable. Use creative techniques for generating additional subsystems and system elements if needed\n"
            "- **DIFFERENT CONTEXT**: Generate novel structure using engineering principles and technical knowledge\n\n"

            "'**OUTPUT**: JSON structure only\n"
            "[\n"
                    "  {{\n"
                    "    \"row_id\": 1,\n"
                    "    \"Product\": \"ProductName\",\n"
                    "    \"Subsystem\": \"SubsystemName\",\n"
                    "    \"SystemElement\": \"ElementName\",\n"
                    "    \"Function\": None,\n"
                    "    \"FailureMode\": None,\n"
                    "    \"FailureCause\": None,\n"
                    "    \"FailureEffect\": None,\n"
                    "    \"PreventiveMeasure\": None,\n"
                    "    \"DetectiveMeasure\": None,\n"
                    "    \"SeverityRating\": None,\n"
                    "    \"OccurrenceRating\": None,\n"
                    "    \"DetectionRating\": None,\n"
                    "  }}\n"
                    "]\n\n"

            "### TYPE 2: QUESTION ANSWERING\n"

            "**Analysis Methodology**:\n"
            "1. Apply context analysis decision matrix to determine data relevance\n"
            "2. Prioritize graph query data over vector query data\n"
            "3. Remove duplicate information, favoring graph data when overlaps occur\n"
            "4. Extract and prominently feature risk ratings (severity, occurrence, detection)\n"
            "5. Distinguish between preventive and detective measures\n\n"
            
            "**Response Requirements**:\n"
            "- Use natural language that is precise and technical\n"
            "- Include only necessary information directly relevant to the question\n"
            "- Feature risk ratings prominently with specific numerical values\n"
            "- Clearly differentiate preventive measures from detective measures\n"
            "- Use specific component, subsystem, and system names from data\n"
            "- Apply context analysis to bridge gaps between question and retrieved data\n\n"
                    
            "**Context-Based Response Strategy**:\n"
            "- **EXACT MATCH**: Use only retrieved data of the same context as question context directly. Tag the Output as <High Confidence>\n"
            "- **SIMILAR CONTEXT**: Adapt retrieved data with analytical reasoning. Verify retrieved information still makes sense for question. Use creative techniques for generating additional information if needed. Tag output as <Medium Confidence>\n"
            "- **DIFFERENT CONTEXT**: Leverage any available retrieved data patterns and internal knowledge. Apply creative/generative skills with technical knowledge. Tag output as <Low Confidence>\n\n"
            
            "\n\n# OUTPUT FORMAT REQUIREMENT:\n"
            "Always return a JSON object with exactly 2 keys:\n"
            "{{\n"
            "  \"analysis_decision\": \"Context analysis with reasoning as string\",\n"
            "  \"content\": \"For TYPE 1: JSON array of system structure | For TYPE 2: Text answer as string\"\n"
            "}}\n"
        ),
        (
            "human",
            "## USER REQUEST\n"
            "Question: {question}\n\n"
            "## RETRIEVED CONTEXT FROM RAG\n"
            "{context}\n\n"
            "Analyze the request type and context similarity, then provide appropriate response following the established methodology."
        )
    ])

    response = llm.invoke(comprehensive_prompt.format(question=question, context=str(context)))
    if debug: 
        print("Final LLM response:", response.content)
    
    try:
        json_text = response.content.strip()
        json_text = json_text.replace('```json', '').replace('```', '').strip()
        
        parsed_response = json.loads(json_text)
        
        # Validate required structure
        if isinstance(parsed_response, dict) and 'analysis_decision' in parsed_response and 'content' in parsed_response:
            analysis_decision = parsed_response['analysis_decision']
            content = parsed_response['content']
            
            # For system structure generation, content should be JSON array
            # For question answering, content should be string
            if isinstance(content, str):
                if content.strip().startswith('[') and content.strip().endswith(']'):
                    try:
                        content = json.loads(content)
                    except:
                        pass  
            
            result = {
                'analysis_decision': analysis_decision,
                'content': content
            }
            
            if debug:
                print(f"Successfully parsed response with analysis_decision: {analysis_decision[:100]}...")
                print(f"Content type: {type(content)}, Content preview: {str(content)[:100]}...")
            
            return result
            
        else:
            if debug:
                print("Missing required keys 'analysis_decision' or 'content'")
            return {
                'analysis_decision': 'ERROR: Invalid response structure',
                'content': response.content
            }
        
    except json.JSONDecodeError as e:
        if debug:
            print(f"JSON parsing failed: {e}")
            print(f"Problematic text: '{response.content}'")
        return {
            'analysis_decision': 'ERROR: JSON parsing failed',
            'content': response.content
        }
    except Exception as e:
        if debug:
            print(f"Error processing response: {e}")
        return {
            'analysis_decision': 'ERROR: Processing failed',
            'content': response.content
        }

def generate_functions(comprehensive_context, element_context, llm, debug: bool):
    function_generation_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an FMEA function generation expert. Generate technical functions for system elements based on context analysis.\n\n"
        
        "## FUNCTION REQUIREMENTS:\n"
        "- Functions must be unambiguous, concrete, verifiable, and validatable\n"
        "- Functions describe intended purpose and input-output relationships of system elements\n"
        "- Include at least subject + verb with clear technical specification\n"
        "- Maintain FMEA standards and technical accuracy\n"
        "- Examples: 'Convert electrical energy to rotational motion', 'Filter hydraulic fluid', 'Control motor speed'\n\n"
        
        "## SYSTEM SIMILARITY DEFINITIONS\n"
        "**Product Category**: Group of products serving similar purposes (e.g., home appliances, automotive systems, industrial equipment)\n"
        "**System Element Identity**: Core functional component regardless of naming variations (motor = engine, pump = compressor)\n"
        "**Subsystem Context**: Functional grouping within product (motor system, control system, fluid system)\n\n"
        
        "## CONTEXT ANALYSIS DECISION MATRIX\n"
        
        "### 1. EXACT MATCH\n"
        "**Criteria**: Target Product, Subsystem and System Element match retrieved data exactly\n"
        "- Allow naming variations (motor/engine, brake/braking system)\n"
        "- Same product category and functional context\n"
        "**Action**: \n"
        "- Use ONLY retrieved context data of the same context for function generation\n"
        "- Do NOT generate new functions not found in retrieved data\n"
        "- Do NOT rename retrieved functions\n"
        
        "### 2. SIMILAR CONTEXT\n"
        "**Criteria**: Same System Element in different Subsystem OR different Product\n"
        "- Allow naming variations (motor/engine, brake/braking system)\n"
        "- Examples: Motor in washing machine vs dishwasher, Brake system in car vs truck\n"
        "- Same product category but different application context\n"
        "**Action**:\n"
        "- Transfer applicable functions from retrieved data\n"
        "- Analytically adapt functions to new system context\n"
        "- Verify retrieved functions still make sense for target product and are applicable\n"
        "- Do NOT rename retrieved functions\n"
        "- Use multi-perspective brainstorming to generate additional functions if needed\n\n"
        
        "### 3. DIFFERENT CONTEXT\n"
        "**Criteria**: Different product category OR no relevant retrieved data\n"
        "- Examples: Hydraulic pump vs electric motor, automotive vs aerospace systems\n"
        "- Fundamentally different operational contexts\n"
        "**Action**:\n"
        "- Use multi-perspective brainstorming for function generation\n"
        "- Apply creative/generative skills with technical knowledge\n"
        "- Leverage any available retrieved data patterns and internal knowledge\n\n"
        
        "## MULTI-PERSPECTIVE BRAINSTORMING PROCESS\n"
        "For SIMILAR and DIFFERENT contexts, analyze from these engineering roles:\n\n"
        "**Design Perspective**: \n"
        "- What are the design requirements and constraints?\n"
        "- How does this element contribute to overall product functionality?\n\n"
        "**Development Perspective**:\n"
        "- What technical implementations are required?\n"
        "- How does this element interface with other system components?\n\n"
        "**Testing Perspective**:\n"
        "- What validation and verification requirements exist?\n"
        "- How can functionality be measured and tested?\n\n"
        "**After Sales Perspective**:\n"
        "- What are practical operational scenarios?\n"
        "- How do customers actually use this element?\n\n"
        
        "## OUTPUT FORMAT:\n"
        "Output a json with 2 keys: analysis_decision and function_list. analysis_decision: Provide type of context and detailed reasoning for context analysis decision matrix. function_list: output ONLY a list of function strings:\n"
        "[\"Function 1\", \"Function 2\", \"Function 3\", \"Function 4\"]"
    ),
    (
        "human",
        "TARGET SYSTEM CONTEXT:\n"
        "Product: {product}\n"
        "Subsystem: {subsystem}\n"
        "System Element: {system_element}\n\n"
        "Retrieved Context from RAG:\n"
        "{context_data}\n\n"
        "Analyze the context similarity and generate appropriate technical functions following the decision matrix process."
    )
])

    response = llm.invoke(function_generation_prompt.format(
        product=element_context['Product'],
        subsystem=element_context['Subsystem'], 
        system_element=element_context['SystemElement'],
        context_data=comprehensive_context

    ))

    if debug: 
        print("Raw LLM response:", response.content)
    
    try:
        json_text = response.content.strip()
        json_text = json_text.replace('```json', '').replace('```', '').strip()
        
        parsed_response = json.loads(json_text)
        
        # Extract the function_list from the JSON
        if isinstance(parsed_response, dict) and 'function_list' in parsed_response:
            function_list = parsed_response['function_list']
        elif isinstance(parsed_response, list):
            function_list = parsed_response
        else:
            if debug:
                print("Unexpected JSON structure, returning empty list")
            return []
        
        return function_list
        
    except json.JSONDecodeError as e:
        if debug:
            print(f"JSON parsing failed: {e}")
            print(f"Problematic text: '{response.content}'")
        return []
    except Exception as e:
        if debug:
            print(f"Error processing function response: {e}")
        return []
    
def generate_failures(comprehensive_context, element_context, llm, debug: bool):
    failure_generation_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an FMEA failure generation expert. Generate technical failure modes including the failure effects and failure causes for functions of system elements based on context analysis.\n\n"

        "## FAILURE MODE REQUIREMENTS:\n"
        "- Failure modes describe the ways in which a system element could fail to perform its assigned function\n"
        "- These are derived from the defined functions\n"
        "- Possible types of failure modes include: loss of function, functional limitation, functional interruption, functional fluctuation, overfunction, unintended function, sticking at a certain value, incorrect direction\n"
        "- Each system element can fail in several different ways. All possible types of failure should be analyzed\n"
        "- Maintain FMEA standards and technical accuracy\n"
        "- Describe with technically correct terms\n\n"

        "## FAILURE EFFECTS REQUIREMENTS:\n"
        "- Failure effects are consequences of an failure mode and are therefore failure modes of a higher-level system\n"
        "- Failure effects occur either at the next higher level of the system structure or directly at the end user\n"
        "- A specific failure effect can be triggered by one or more failure modes simultaneously\n"
        "- Maintain FMEA standards and technical accuracy\n"
        "- Describe with technically correct terms\n\n"

        "## FAILURE CAUSES REQUIREMENTS:\n"
        "- Failure causes serve as an indication of why a particular type of failure mode may have occurred\n"
        "- Every possible error cause should be identified\n"
        "- An identified error cause may be the cause of several different types of failure modes in a system\n"
        "- Maintain FMEA standards and technical accuracy\n"
        "- Describe with technically correct terms\n\n"
        
        "## SYSTEM SIMILARITY DEFINITIONS\n"
        "**Product Category**: Group of products serving similar purposes (e.g., home appliances, automotive systems, industrial equipment)\n"
        "**System Element Identity**: Core functional component regardless of naming variations (motor = engine, pump = compressor)\n"
        "**Subsystem Context**: Functional grouping within product (motor system, control system, fluid system)\n\n"
        
        "## CONTEXT ANALYSIS DECISION MATRIX\n"
        
        "### 1. EXACT MATCH\n"
        "**Criteria**: Target Product, Subsystem, System Element and Function match parts of the retrieved data exactly\n"
        "- Allow naming variations (motor/engine, brake/braking system)\n"
        "- Same product category and functional context\n"
        "**Action**: \n"
        "- Use ONLY the parts of the retrieved context data which match the same context for failure mode, failure effect and failure cause generation\n"
        "- Do NOT generate new failure modes, causes or effects not found in retrieved data\n"

        "### 2. SIMILAR CONTEXT\n"
        "**Criteria**: Same combination of function and system element in different Subsystem OR different Product\n"
        "- Examples: Motor with function Convert electrical energy to motion in washing machine vs dishwasher, Hydraulic pump with function Generate hydraulic pressure in construction equipment vs agricultural machinery\n"
        "- Allow naming variations (motor/engine, brake/braking system)\n"
        "- Same product category but different application context\n"
        "**Action**:\n"
        "- Transfer applicable failure modes, effects and causes from retrieved data\n"
        "- Analytically adapt failure modes, effects and causes to new system context\n"
        "- Verify retrieved failure modes, effects and causes still make sense for target product\n"
        "- Use multi-perspective brainstorming to generate additional failure modes, effects and causes if needed\n\n"

        "### 3. DIFFERENT CONTEXT\n"
        "**Criteria**: Different product category OR no combination of function and system element was found in retrieved data\n"
        "- Examples: Hydraulic pump vs electric motor, automotive vs aerospace systems\n"
        "- Fundamentally different operational contexts\n"
        "**Action**:\n"
        "- Use multi-perspective brainstorming for failure modes, effects and causes generation\n"
        "- Apply creative/generative skills with technical knowledge\n"
        "- Leverage any available retrieved data patterns and internal knowledge\n\n"
        
        "## MULTI-PERSPECTIVE BRAINSTORMING PROCESS\n"
        "For SIMILAR and DIFFERENT contexts, analyze from these engineering roles:\n\n"
        "**Design Perspective**: \n"
        "- What design weaknesses could lead to failure modes?\n"
        "- What design constraints create potential failure scenarios?\n"
        "- How could design flaws cause cascading effects?\n\n"
        "**Development Perspective**:\n"
        "- What implementation errors could cause failures?\n"
        "- How could interface problems with other components create failure modes?\n"
        "- What manufacturing defects could lead to failure causes?\n\n"
        "**Testing Perspective**:\n"
        "- What failure modes are difficult to detect during testing?\n"
        "- What stress conditions could reveal hidden failure modes?\n"
        "- How could testing limitations miss critical failure scenarios?\n\n"
        "**After Sales Perspective**:\n"
        "- What real-world usage patterns could cause unexpected failures?\n"
        "- What customer behavior could lead to failure modes?\n"
        "- What maintenance neglect could result in failure causes?\n"
        "- What environmental conditions create failure scenarios?\n\n"
        
        "## OUTPUT FORMAT:\n"
        "Always return a JSON object with exactly 2 keys:\n"
        "{{\n"
        "  \"analysis_decision\": \"EXACT_MATCH/SIMILAR_CONTEXT/DIFFERENT_CONTEXT with detailed reasoning\",\n"
        "  \"content\": \"[{{FailureMode: failure_mode, FailureCause: failure_cause, FailureEffect: failure_effect}}]\"\n"
        "}}\n"
    ),
    (
        "human",
        "TARGET SYSTEM CONTEXT:\n"
        "Product: {product}\n"
        "Subsystem: {subsystem}\n"
        "System Element: {system_element}\n"
        "Function: {function}\n\n"
        "Retrieved Context from RAG:\n"
        "{context_data}\n\n"
        "Analyze the context similarity and generate appropriate technical failure modes, effects and causes following the decision matrix process."
    )
])

    response = llm.invoke(failure_generation_prompt.format(
        product=element_context['Product'],
        subsystem=element_context['Subsystem'], 
        system_element=element_context['SystemElement'],
        function=element_context['Function'],
        context_data=comprehensive_context

    ))

    if debug: 
        print("Raw LLM response:", response.content)
    
    try:
        json_text = response.content.strip()
        json_text = json_text.replace('```json', '').replace('```', '').strip()
        
        parsed_response = json.loads(json_text)
        
        if isinstance(parsed_response, dict) and 'analysis_decision' in parsed_response and 'content' in parsed_response:
            analysis_decision = parsed_response['analysis_decision']
            content = parsed_response['content']

            if isinstance(content, str):
                try:
                    content = json.loads(content)
                except json.JSONDecodeError:
                    if debug:
                        print(f"Failed to parse content as JSON: {content}")
                    content = []
            
            if not isinstance(content, list):
                if debug:
                    print(f"Content is not a list, converting: {type(content)}")
                content = []
            
            result = {
                'analysis_decision': analysis_decision,
                'content': content
            }

            return result
        
        else:
            if debug:
                print("Missing required keys 'analysis_decision' or 'content'")
            return {
                'analysis_decision': 'ERROR: Invalid response structure',
                'content': []
            }
        
        
        
    except json.JSONDecodeError as e:
        if debug:
            print(f"JSON parsing failed: {e}")
            print(f"Problematic text: '{response.content}'")
        return {
            'analysis_decision': 'ERROR: JSON parsing failed',
            'content': []
        }
    except Exception as e:
        if debug:
            print(f"Error processing response: {e}")
        return {
            'analysis_decision': 'ERROR: Processing failed',
            'content': []
        }
    
def generate_existing_measures(comprehensive_context, element_context, llm, debug: bool):
    existing_measure_generation_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an FMEA measure generation expert. Generate Preventive and Detective Measures "
        "for provided failure causes of system elements based on retrieved context data.\n\n"

        "## MEASURE REQUIREMENTS:\n"
        "- Preventive Measures: Actions that reduce or eliminate the likelihood of a failure cause occurring\n"
        "- Detective Measures: Actions that improve detection of a failure cause once it has occurred\n"
        "- Preventive Measures only address the failure cause, not the failure mode or failure effect\n"
        "- Detective Measures address a special combination of failure mode and failure cause\n"
        "- Use technically correct terms following FMEA standards\n"
        "- Do NOT create new measures beyond the retrieved context data\n\n"
        
        "## SYSTEM SIMILARITY DEFINITIONS\n"
        "**Product Category**: Group of products serving similar purposes (e.g., home appliances, automotive systems, industrial equipment)\n"
        "**System Element Identity**: Core functional component regardless of naming variations (motor = engine, pump = compressor)\n"
        "**Subsystem Context**: Functional grouping within product (motor system, control system, fluid system)\n\n"
        
        "## CONTEXT ANALYSIS DECISION MATRIX\n"
        "### 1. EXACT MATCH\n"
        "- Criteria: Target Product, Subsystem, System Element, Function, Failure Mode and Failure Cause match retrieved data exactly\n"
        "- Action: Use ONLY retrieved context data with the same context for preventive and detective measure generation\n\n"

        "### 2. SIMILAR CONTEXT\n"
        "- Criteria: Same combination of System Element, Function, Failure Mode and Failure Cause in different subsystem OR different product\n"
        "- Action: Transfer applicable preventive and detective measures from retrieved data\n"
        "- Allow naming variations\n"
        "- Same product category but different application context\n"
        "- Verify measures still make sense and are applicable for the new target system context\n\n"

        "### 3. DIFFERENT CONTEXT\n"
        "- Criteria: Different product category OR no relevant retrieved data\n"
        "- Action: Return an empty list for measures\n\n"
        
        "## OUTPUT FORMAT:\n"
        "For every Measure create a new list entry in the content value."
        "Always return a JSON object with exactly 2 keys:\n"
        "{{\n"
        "  \"analysis_decision\": \"EXACT_MATCH/SIMILAR_CONTEXT/DIFFERENT_CONTEXT with detailed reasoning\",\n"
        "  \"content\": \"[{{Preventive/DetectiveMeasure: preventive/detective measure}}]\"\n"
        "}}\n"

    ),
    (
        "human",
        "TARGET SYSTEM CONTEXT:\n"
        "Product: {product}\n"
        "Subsystem: {subsystem}\n"
        "System Element: {system_element}\n"
        "Function: {function}\n"
        "Failure Mode: {failure_mode}\n"
        "Failure Cause: {failure_cause}\n"
        "Failure Effect: {failure_effect}\n\n"
        "Retrieved Context from RAG:\n"
        "{context_data}\n\n"
        "Analyze the context similarity and generate preventive and detective measures accordingly."
    )
])

    response = llm.invoke(existing_measure_generation_prompt.format(
        product=element_context['Product'],
        subsystem=element_context['Subsystem'], 
        system_element=element_context['SystemElement'],
        function=element_context['Function'],
        failure_mode=element_context['FailureMode'],
        failure_cause=element_context['FailureCause'],
        failure_effect=element_context['FailureEffect'],
        context_data=comprehensive_context

    ))

    if debug: 
        print("Raw LLM response:", response.content)
    
    try:
        json_text = response.content.strip()
        json_text = json_text.replace('```json', '').replace('```', '').strip()
        
        parsed_response = json.loads(json_text)
        
        # Extract the information from the JSON
        if isinstance(parsed_response, dict) and 'analysis_decision' in parsed_response and 'content' in parsed_response:
            analysis_decision = parsed_response['analysis_decision']
            content = parsed_response['content']

            if isinstance(content, str):
                try:
                    content = json.loads(content)
                except json.JSONDecodeError:
                    if debug:
                        print(f"Failed to parse content as JSON: {content}")
                    content = []
            
            # Ensure content is a list
            if not isinstance(content, list):
                if debug:
                    print(f"Content is not a list, converting: {type(content)}")
                content = []
            
            result = {
                'analysis_decision': analysis_decision,
                'content': content
            }

            return result
        
        else:
            if debug:
                print("Missing required keys 'analysis_decision' or 'content'")
            return {
                'analysis_decision': 'ERROR: Invalid response structure',
                'content': []
            }
        
        
        
    except json.JSONDecodeError as e:
        if debug:
            print(f"JSON parsing failed: {e}")
            print(f"Problematic text: '{response.content}'")
        return {
            'analysis_decision': 'ERROR: JSON parsing failed',
            'content': []
        }
    except Exception as e:
        if debug:
            print(f"Error processing response: {e}")
        return {
            'analysis_decision': 'ERROR: Processing failed',
            'content': []
        }
    

async def generate_existing_measures_async(comprehensive_context, element_context, async_llm, debug: bool):
    """Async version of generate_existing_measures using the same logic but with async LLM calls"""
    existing_measure_generation_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an FMEA measure generation expert. Generate Preventive and Detective Measures "
        "for provided failure causes of system elements based on retrieved context data.\n\n"

        "## MEASURE REQUIREMENTS:\n"
        "- Preventive Measures: Actions that reduce or eliminate the likelihood of a failure cause occurring\n"
        "- Detective Measures: Actions that improve detection of a failure cause once it has occurred\n"
        "- Preventive Measures only address the failure cause, not the failure mode or failure effect\n"
        "- Detective Measures address a special combination of failure mode and failure cause\n"
        "- Use technically correct terms following FMEA standards\n"
        "- Do NOT create new measures beyond the retrieved context data\n\n"
        
        "## SYSTEM SIMILARITY DEFINITIONS\n"
        "**Product Category**: Group of products serving similar purposes (e.g., home appliances, automotive systems, industrial equipment)\n"
        "**System Element Identity**: Core functional component regardless of naming variations (motor = engine, pump = compressor)\n"
        "**Subsystem Context**: Functional grouping within product (motor system, control system, fluid system)\n\n"
        
        "## CONTEXT ANALYSIS DECISION MATRIX\n"
        "### 1. EXACT MATCH\n"
        "- Criteria: Target Product, Subsystem, System Element, Function, Failure Mode and Failure Cause match retrieved data exactly\n"
        "- Action: Use ONLY retrieved context data with the same context for preventive and detective measure generation\n\n"

        "### 2. SIMILAR CONTEXT\n"
        "- Criteria: Same combination of System Element, Function, Failure Mode and Failure Cause in different subsystem OR different product\n"
        "- Action: Transfer applicable preventive and detective measures from retrieved data\n"
        "- Allow naming variations\n"
        "- Same product category but different application context\n"
        "- Verify measures still make sense and are applicable for the new target system context\n\n"

        "### 3. DIFFERENT CONTEXT\n"
        "- Criteria: Different product category OR no relevant retrieved data\n"
        "- Action: Return an empty list for measures\n\n"
        
        "## OUTPUT FORMAT:\n"
        "For every Measure create a new list entry in the content value."
        "Always return a JSON object with exactly 2 keys:\n"
        "{{\n"
        "  \"analysis_decision\": \"EXACT_MATCH/SIMILAR_CONTEXT/DIFFERENT_CONTEXT with detailed reasoning\",\n"
        "  \"content\": \"[{{Preventive/DetectiveMeasure: preventive/detective measure}}]\"\n"
        "}}\n"

    ),
    (
        "human",
        "TARGET SYSTEM CONTEXT:\n"
        "Product: {product}\n"
        "Subsystem: {subsystem}\n"
        "System Element: {system_element}\n"
        "Function: {function}\n"
        "Failure Mode: {failure_mode}\n"
        "Failure Cause: {failure_cause}\n"
        "Failure Effect: {failure_effect}\n\n"
        "Retrieved Context from RAG:\n"
        "{context_data}\n\n"
        "Analyze the context similarity and generate preventive and detective measures accordingly."
    )
])

    # Use async invoke instead of sync invoke
    response = await async_llm.ainvoke(existing_measure_generation_prompt.format(
        product=element_context['Product'],
        subsystem=element_context['Subsystem'], 
        system_element=element_context['SystemElement'],
        function=element_context['Function'],
        failure_mode=element_context['FailureMode'],
        failure_cause=element_context['FailureCause'],
        failure_effect=element_context['FailureEffect'],
        context_data=comprehensive_context

    ))

    if debug: 
        print("Raw async LLM response:", response.content)
    
    try:
        json_text = response.content.strip()
        json_text = json_text.replace('```json', '').replace('```', '').strip()
        
        parsed_response = json.loads(json_text)
        
        if isinstance(parsed_response, dict) and 'analysis_decision' in parsed_response and 'content' in parsed_response:
            analysis_decision = parsed_response['analysis_decision']
            content = parsed_response['content']

            if isinstance(content, str):
                try:
                    content = json.loads(content)
                except json.JSONDecodeError:
                    if debug:
                        print(f"Failed to parse content as JSON: {content}")
                    content = []
            
            # Ensure content is a list
            if not isinstance(content, list):
                if debug:
                    print(f"Content is not a list, converting: {type(content)}")
                content = []
            
            result = {
                'analysis_decision': analysis_decision,
                'content': content
            }

            return result
        
        else:
            if debug:
                print("Missing required keys 'analysis_decision' or 'content'")
            return {
                'analysis_decision': 'ERROR: Invalid response structure',
                'content': []
            }
        
    except json.JSONDecodeError as e:
        if debug:
            print(f"JSON parsing failed: {e}")
            print(f"Problematic text: '{response.content}'")
        return {
            'analysis_decision': 'ERROR: JSON parsing failed',
            'content': []
        }
    except Exception as e:
        if debug:
            print(f"Error processing response: {e}")
        return {
            'analysis_decision': 'ERROR: Processing failed',
            'content': []
        }


def generate_risk_rating(comprehensive_context, element_context, llm, debug: bool):
    risk_rating_generation_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an FMEA risk assessment expert specializing in severity, occurrence, and detection rating evaluation.\n\n"
        
        "## RISK RATING OBJECTIVES\n"
        "Generate accurate risk ratings (1-10 scale) for FMEA failure analysis considering existing preventive and detective measures.\n\n"
        
        "## RATING DEFINITIONS\n"
        
        "### SEVERITY (S - Impact of Failure Effect)\n"
        "Rates the failure effect impact on next system level or end customer:\n"
        "- **1**: No effect, customer does not notice\n"
        "- **2-3**: Insignificant, customer only slightly disturbed\n"
        "- **4-6**: Process disturbance, problems for some customers\n"
        "- **7-8**: Reduced service, customers are dissatisfied\n"
        "- **9-10**: Violation of regulations, financial damage to company or customer\n"
        "**Note**: Rating must be consistent for all failure modes with same failure effect\n\n"
        
        "### OCCURRENCE (O - Likelihood of Failure Cause)\n"
        "Probability of failure cause occurrence considering existing preventive measures:\n"
        "- **1**: Nearly impossible (preventive measures completely eliminate cause)\n"
        "- **2**: Unlikely\n"
        "- **3**: Low likelihood\n"
        "- **4-6**: Occasional occurrence\n"
        "- **7-8**: Frequent occurrence\n"
        "- **9-10**: Very frequent/constant occurrence (measures ineffective)\n"
        "**Note**: Rating must be consistent for all failure modes with same failure causes and preventive measures\n\n"
        
        "### DETECTION (D - Ability to Detect Failure Cause)\n"
        "Probability of detecting failure cause considering existing detective measures:\n"
        "- **1-2**: Certain detection in subsequent processes\n"
        "- **3-4**: High likelihood of detection in subsequent processes\n"
        "- **5-6**: Detection only through targeted inspection\n"
        "- **7-8**: No detection before customer delivery, customer likely detects\n"
        "- **9**: Only expert customer will detect\n"
        "- **10**: Not immediately detectable, only over time\n"
        "**Note**: Rating must be consistent for the specific combination of failure mode, failure cause and detective measures (same causes can have different detection ratings when paired with different failure modes)\n\n"
        
        "## SYSTEM SIMILARITY DEFINITIONS\n"
        "**Product Category**: Group of products serving similar purposes (e.g., home appliances, automotive systems, industrial equipment)\n"
        "**System Element Identity**: Core functional component regardless of naming variations (motor = engine, pump = compressor)\n"
        "**Subsystem Context**: Functional grouping within product (motor system, control system, fluid system)\n\n"
        
        "## CONTEXT ANALYSIS DECISION MATRIX\n"
        
        "### EXACT MATCH\n"
        "**Criteria**: Target Product, Subsystem, System Element, Function, Failure Mode, Failure Cause, Failure Effect match retrieved data exactly\n"
        "- Allow naming variations (motor/engine, brake/braking system)\n"
        "- Same product category and functional context\n"
        "**Action**: Use retrieved risk ratings directly\n\n"
        
        "### SIMILAR CONTEXT\n"
        "**Criteria**: Same system element, function, failure mode, failure cause, failure effect in different Subsystem OR different Product\n"
        "- Examples: Motor overheating in washing machine vs dishwasher\n"
        "- Same product category but different application context\n"
        "**Action**: Adapt retrieved ratings to new system context with analytical reasoning\n\n"
        
        "### DIFFERENT CONTEXT\n"
        "**Criteria**: Different product category OR no relevant retrieved data\n"
        "- Examples: Automotive vs aerospace systems, no matching failure scenarios\n"
        "- Fundamentally different operational contexts\n"
        "**Action**: Generate ratings using rating schema and technical knowledge\n\n"
        
        "## RATING METHODOLOGY\n"
        
        "### For EXACT MATCH:\n"
        "- Extract severity, occurrence, and detection ratings from retrieved data of the same context\n"
        "- Verify ratings align with current preventive/detective measures\n\n"
        
        "### For SIMILAR CONTEXT:\n"
        "- Analyze retrieved ratings for applicability\n"
        "- Adjust ratings based on different system context\n"
        "- Consider differences in customer impact, operating environment, measures effectiveness\n\n"
        
        "### For DIFFERENT CONTEXT:\n"
        "- Apply rating schema systematically\n"
        "- Consider failure effect severity in target product context\n"
        "- Evaluate preventive measures effectiveness against failure cause for occurrence\n"
        "- Assess detective measures capability for failure detection\n\n"
        
        "## OUTPUT FORMAT\n"
        "Always return JSON object with exactly 2 keys:\n"
        "```json\n"
        "{{\n"
        "  \"analysis_decision\": \"EXACT_MATCH/SIMILAR_CONTEXT/DIFFERENT_CONTEXT with detailed reasoning for rating decisions\",\n"
        "  \"content\": {{\n"
        "    \"Severity\": severity_rating,\n"
        "    \"Occurrence\": occurrence_rating,\n"
        "    \"Detection\": detection_rating\n"
        "  }}\n"
        "}}\n"
        "```"
    ),
    (
        "human",
        "## TARGET SYSTEM CONTEXT\n"
        "Product: {product}\n"
        "Subsystem: {subsystem}\n"
        "SystemElement: {system_element}\n"
        "Function: {function}\n"
        "FailureMode: {failure_mode}\n"
        "FailureCause: {failure_cause}\n"
        "FailureEffect: {failure_effect}\n\n"
        "## EXISTING MEASURES\n"
        "Preventive Measures: {preventive_measures}\n"
        "Detective Measures: {detective_measures}\n\n"
        "## RETRIEVED CONTEXT FROM RAG\n"
        "{context_data}\n\n"
        "Analyze context similarity and generate appropriate risk ratings for severity, occurrence, and detection following the decision matrix process."
    )
])

    response = llm.invoke(risk_rating_generation_prompt.format(
        product=element_context['Product'],
        subsystem=element_context['Subsystem'], 
        system_element=element_context['SystemElement'],
        function=element_context['Function'],
        failure_mode=element_context['FailureMode'],
        failure_cause=element_context['FailureCause'],
        failure_effect=element_context['FailureEffect'],
        preventive_measures=element_context['PreventiveMeasure'],
        detective_measures=element_context['DetectiveMeasure'],
        context_data=comprehensive_context

    ))

    if debug: 
        print("Raw LLM response:", response.content)
    
    try:
        json_text = response.content.strip()
        json_text = json_text.replace('```json', '').replace('```', '').strip()
        
        parsed_response = json.loads(json_text)

 
        # Extract the information from the JSON
        if isinstance(parsed_response, dict) and 'analysis_decision' in parsed_response and 'content' in parsed_response:
            analysis_decision = parsed_response['analysis_decision']
            content = parsed_response['content']

            if isinstance(content, str):
                try:
                    content = json.loads(content)
                except json.JSONDecodeError:
                    if debug:
                        print(f"Failed to parse content as JSON: {content}")
                    content = {}
            
            if not isinstance(content, dict):
                if debug:
                    print(f"Content is not a dict, converting: {type(content)}")
                content = {}

            result = {
                'analysis_decision': analysis_decision,
                'content': content
            }

            return result
        
        else:
            if debug:
                print("Missing required keys 'analysis_decision' or 'content'")
            return {
                'analysis_decision': 'ERROR: Invalid response structure',
                'content': {}
            }
        
        
        
    except json.JSONDecodeError as e:
        if debug:
            print(f"JSON parsing failed: {e}")
            print(f"Problematic text: '{response.content}'")
        return {
            'analysis_decision': 'ERROR: JSON parsing failed',
            'content': {}
        }
    except Exception as e:
        if debug:
            print(f"Error processing response: {e}")
        return {
            'analysis_decision': 'ERROR: Processing failed',
            'content': {}
        }

async def generate_risk_rating_async(comprehensive_context, element_context, async_llm, debug: bool):
    """Async version of generate_risk_rating using the same logic but with async LLM calls"""
    risk_rating_generation_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an FMEA risk assessment expert specializing in severity, occurrence, and detection rating evaluation.\n\n"
        
        "## RISK RATING OBJECTIVES\n"
        "Generate accurate risk ratings (1-10 scale) for FMEA failure analysis considering existing preventive and detective measures.\n\n"
        
        "## RATING DEFINITIONS\n"
        
        "### SEVERITY (S - Impact of Failure Effect)\n"
        "Rates the failure effect impact on next system level or end customer:\n"
        "- **1**: No effect, customer does not notice\n"
        "- **2-3**: Insignificant, customer only slightly disturbed\n"
        "- **4-6**: Process disturbance, problems for some customers\n"
        "- **7-8**: Reduced service, customers are dissatisfied\n"
        "- **9-10**: Violation of regulations, financial damage to company or customer\n"
        "**Note**: Rating must be consistent for all failure modes with same failure effect\n\n"
        
        "### OCCURRENCE (O - Likelihood of Failure Cause)\n"
        "Probability of failure cause occurrence considering existing preventive measures:\n"
        "- **1**: Nearly impossible (preventive measures completely eliminate cause)\n"
        "- **2**: Unlikely\n"
        "- **3**: Low likelihood\n"
        "- **4-6**: Occasional occurrence\n"
        "- **7-8**: Frequent occurrence\n"
        "- **9-10**: Very frequent/constant occurrence (measures ineffective)\n"
        "**Note**: Rating must be consistent for all failure modes with same failure causes and preventive measures\n\n"
        
        "### DETECTION (D - Ability to Detect Failure Cause)\n"
        "Probability of detecting failure cause considering existing detective measures:\n"
        "- **1-2**: Certain detection in subsequent processes\n"
        "- **3-4**: High likelihood of detection in subsequent processes\n"
        "- **5-6**: Detection only through targeted inspection\n"
        "- **7-8**: No detection before customer delivery, customer likely detects\n"
        "- **9**: Only expert customer will detect\n"
        "- **10**: Not immediately detectable, only over time\n"
        "**Note**: Rating must be consistent for the specific combination of failure mode, failure cause and detective measures (same causes can have different detection ratings when paired with different failure modes)\n\n"
        
        "## SYSTEM SIMILARITY DEFINITIONS\n"
        "**Product Category**: Group of products serving similar purposes (e.g., home appliances, automotive systems, industrial equipment)\n"
        "**System Element Identity**: Core functional component regardless of naming variations (motor = engine, pump = compressor)\n"
        "**Subsystem Context**: Functional grouping within product (motor system, control system, fluid system)\n\n"
        
        "## CONTEXT ANALYSIS DECISION MATRIX\n"
        
        "### EXACT MATCH\n"
        "**Criteria**: Target Product, Subsystem, System Element, Function, Failure Mode, Failure Cause, Failure Effect match retrieved data exactly\n"
        "- Allow naming variations (motor/engine, brake/braking system)\n"
        "- Same product category and functional context\n"
        "**Action**: Use retrieved risk ratings directly\n\n"
        
        "### SIMILAR CONTEXT\n"
        "**Criteria**: Same system element, function, failure mode, failure cause, failure effect in different Subsystem OR different Product\n"
        "- Examples: Motor overheating in washing machine vs dishwasher\n"
        "- Same product category but different application context\n"
        "**Action**: Adapt retrieved ratings to new system context with analytical reasoning\n\n"
        
        "### DIFFERENT CONTEXT\n"
        "**Criteria**: Different product category OR no relevant retrieved data\n"
        "- Examples: Automotive vs aerospace systems, no matching failure scenarios\n"
        "- Fundamentally different operational contexts\n"
        "**Action**: Generate ratings using rating schema and technical knowledge\n\n"
        
        "## RATING METHODOLOGY\n"
        
        "### For EXACT MATCH:\n"
        "- Extract severity, occurrence, and detection ratings from retrieved data of the same context\n"
        "- Verify ratings align with current preventive/detective measures\n\n"
        
        "### For SIMILAR CONTEXT:\n"
        "- Analyze retrieved ratings for applicability\n"
        "- Adjust ratings based on different system context\n"
        "- Consider differences in customer impact, operating environment, measures effectiveness\n\n"
        
        "### For DIFFERENT CONTEXT:\n"
        "- Apply rating schema systematically\n"
        "- Consider failure effect severity in target product context\n"
        "- Evaluate preventive measures effectiveness against failure cause for occurrence\n"
        "- Assess detective measures capability for failure detection\n\n"
        
        "## OUTPUT FORMAT\n"
        "Always return JSON object with exactly 2 keys:\n"
        "```json\n"
        "{{\n"
        "  \"analysis_decision\": \"EXACT_MATCH/SIMILAR_CONTEXT/DIFFERENT_CONTEXT with detailed reasoning for rating decisions\",\n"
        "  \"content\": {{\n"
        "    \"Severity\": severity_rating,\n"
        "    \"Occurrence\": occurrence_rating,\n"
        "    \"Detection\": detection_rating\n"
        "  }}\n"
        "}}\n"
        "```"
    ),
    (
        "human",
        "## TARGET SYSTEM CONTEXT\n"
        "Product: {product}\n"
        "Subsystem: {subsystem}\n"
        "SystemElement: {system_element}\n"
        "Function: {function}\n"
        "FailureMode: {failure_mode}\n"
        "FailureCause: {failure_cause}\n"
        "FailureEffect: {failure_effect}\n\n"
        "## EXISTING MEASURES\n"
        "Preventive Measures: {preventive_measures}\n"
        "Detective Measures: {detective_measures}\n\n"
        "## RETRIEVED CONTEXT FROM RAG\n"
        "{context_data}\n\n"
        "Analyze context similarity and generate appropriate risk ratings for severity, occurrence, and detection following the decision matrix process."
    )
])

    # Use async invoke instead of sync invoke
    response = await async_llm.ainvoke(risk_rating_generation_prompt.format(
        product=element_context['Product'],
        subsystem=element_context['Subsystem'], 
        system_element=element_context['SystemElement'],
        function=element_context['Function'],
        failure_mode=element_context['FailureMode'],
        failure_cause=element_context['FailureCause'],
        failure_effect=element_context['FailureEffect'],
        preventive_measures=element_context['PreventiveMeasure'],
        detective_measures=element_context['DetectiveMeasure'],
        context_data=comprehensive_context

    ))

    if debug: 
        print("Raw LLM response:", response.content)
    
    try:
        json_text = response.content.strip()
        json_text = json_text.replace('```json', '').replace('```', '').strip()
        
        parsed_response = json.loads(json_text)

        if isinstance(parsed_response, dict) and 'analysis_decision' in parsed_response and 'content' in parsed_response:
            analysis_decision = parsed_response['analysis_decision']
            content = parsed_response['content']

            if isinstance(content, str):
                try:
                    content = json.loads(content)
                except json.JSONDecodeError:
                    if debug:
                        print(f"Failed to parse content as JSON: {content}")
                    content = {}
            
            if not isinstance(content, dict):
                if debug:
                    print(f"Content is not a dict, converting: {type(content)}")
                content = {}

            result = {
                'analysis_decision': analysis_decision,
                'content': content
            }

            return result
        
        else:
            if debug:
                print("Missing required keys 'analysis_decision' or 'content'")
            return {
                'analysis_decision': 'ERROR: Invalid response structure',
                'content': {}
            }
        
    except json.JSONDecodeError as e:
        if debug:
            print(f"JSON parsing failed: {e}")
            print(f"Problematic text: '{response.content}'")
        return {
            'analysis_decision': 'ERROR: JSON parsing failed',
            'content': {}
        }
    except Exception as e:
        if debug:
            print(f"Error processing response: {e}")
        return {
            'analysis_decision': 'ERROR: Processing failed',
            'content': {}
        }

def generate_new_measures(comprehensive_context, element_context, llm, debug: bool):
    new_measure_generation_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an FMEA measure generation expert specializing in creating innovative preventive and detective measures to reduce risk ratings.\n\n"
        
        "## MEASURE GENERATION OBJECTIVES\n"
        "Generate new measures that are NOT already implemented to lower occurrence and detection ratings:\n"
        "- **Preventive Measures**: Actions that reduce or eliminate likelihood of failure cause occurring\n"
        "- **Detective Measures**: Actions that improve detection of failure cause once it has occurred\n"
        "- Preventive Measures only address the failure cause, not the failure mode or failure effect\n"
        "- Detective Measures address a special combination of failure mode and failure cause\n"
        "- Use technically correct terms\n\n"
        
        "## RISK-BASED MEASURE STRATEGY\n"
        
        "### HIGH RISK RATINGS (7-10)\n"
        "Generate advanced, critical measures requiring significant investment:\n"
        "- Automated monitoring systems\n"
        "- Redundant safety systems\n"
        "- Advanced diagnostic technologies\n"
        "- Comprehensive redesign approaches\n\n"
        
        "### MEDIUM RISK RATINGS (4-6)\n"
        "Generate practical, cost-effective measures:\n"
        "- Regular inspection schedules\n"
        "- Training programs\n"
        "- Process improvements\n"
        "- Standard monitoring procedures\n\n"
        
        "### LOW RISK RATINGS (1-3)\n"
        "Generate basic, simple measures:\n"
        "- Basic maintenance tasks\n"
        "- Simple visual checks\n"
        "- Documentation procedures\n"
        "- User guidelines\n\n"
        
        "## MULTI-PERSPECTIVE BRAINSTORMING PROCESS\n"
        "Analyze from these engineering roles to generate comprehensive measures:\n\n"
        
        "**Design Perspective**:\n"
        "- What design improvements could prevent the failure cause?\n"
        "- How could design redundancy reduce occurrence?\n"
        "- What design features could enable better detection?\n\n"
        
        "**Development Perspective**:\n"
        "- What manufacturing controls could prevent the failure cause?\n"
        "- How could quality systems detect potential issues?\n"
        "- What process improvements reduce failure likelihood?\n\n"
        
        "**Testing Perspective**:\n"
        "- What testing methods could detect the failure cause early?\n"
        "- How could validation procedures prevent occurrence?\n"
        "- What diagnostic tools improve detection capability?\n\n"
        
        "**After Sales Perspective**:\n"
        "- What maintenance procedures prevent the failure cause?\n"
        "- How could customer training reduce occurrence?\n"
        "- What monitoring systems enable early detection?\n"
        "- What service procedures improve prevention?\n\n"
        
        "## CREATIVE GENERATION PROCESS\n"
        "1. **Analyze Current Situation**: Review existing measures and their effectiveness\n"
        "2. **Identify Gaps**: Find areas where current measures are insufficient\n"
        "3. **Risk-Based Prioritization**: Match measure sophistication to risk level\n"
        "4. **Multi-Perspective Analysis**: Apply all four engineering viewpoints\n"
        "5. **Innovation Focus**: Generate measures not currently implemented\n"
        "6. **Feasibility Consideration**: Ensure measures are technically implementable\n\n"
        
        "## MEASURE CATEGORIES FOR INSPIRATION\n"
        "Use retrieved context as inspiration for these measure types:\n"
        "- **Monitoring Systems**: Sensors, alarms, condition monitoring\n"
        "- **Maintenance Procedures**: Scheduled inspections, replacements, calibrations\n"
        "- **Design Modifications**: Material changes, redundancy, fail-safes\n"
        "- **Process Controls**: Quality checks, manufacturing standards, procedures\n"
        "- **Training Programs**: Operator education, maintenance training, awareness\n"
        "- **Technology Solutions**: Automated systems, diagnostic tools, software\n\n"
        
        "## OUTPUT FORMAT\n"
        "Generate separate entries for each preventive and detective measure. Output maximal 3 preventive measures and 3 detective measures\n"
        "```json\n"
        "{{\n"
        "  \"analysis_decision\": \"Analysis of current risk levels and measure gaps requiring new solutions\",\n"
        "  \"content\": [\n"
        "    {{\"PreventiveMeasure\": \"specific preventive action\"}},\n"
        "    {{\"DetectiveMeasure\": \"specific detective action\"}},\n"
        "    {{\"PreventiveMeasure\": \"another preventive action\"}}\n"
        "  ]\n"
        "}}\n"
        "```"
    ),
    (
        "human",
        "## TARGET FAILURE ANALYSIS\n"
        "Product: {product}\n"
        "Subsystem: {subsystem}\n"
        "SystemElement: {system_element}\n"
        "Function: {function}\n"
        "FailureMode: {failure_mode}\n"
        "FailureCause: {failure_cause}\n"
        "FailureEffect: {failure_effect}\n\n"
        
        "## CURRENT RISK PROFILE\n"
        "Occurrence Rating: {occurrence_rating} (target: reduce this)\n"
        "Detection Rating: {detection_rating} (target: reduce this)\n"
        "Severity Rating: {severity_rating} (reference for measure criticality)\n\n"
        
        "## EXISTING MEASURES (DO NOT DUPLICATE)\n"
        "Current Preventive Measures: {preventive_measures}\n"
        "Current Detective Measures: {detective_measures}\n\n"
        
        "## RETRIEVED CONTEXT FOR INSPIRATION\n"
        "{context_data}\n\n"
        
        "Generate new preventive and detective measures using multi-perspective brainstorming to reduce the occurrence and detection ratings. Focus on measures NOT currently implemented."
    )
])

    response = llm.invoke(new_measure_generation_prompt.format(
        product=element_context['Product'],
        subsystem=element_context['Subsystem'], 
        system_element=element_context['SystemElement'],
        function=element_context['Function'],
        failure_mode=element_context['FailureMode'],
        failure_cause=element_context['FailureCause'],
        failure_effect=element_context['FailureEffect'],
        preventive_measures=element_context['PreventiveMeasure'],
        detective_measures=element_context['DetectiveMeasure'],
        occurrence_rating=element_context['Occurrence'],
        detection_rating=element_context['Detection'],
        severity_rating=element_context['Severity'],
        context_data=comprehensive_context

    ))

    if debug: 
        print("Raw LLM response:", response.content)
    
    try:
        json_text = response.content.strip()
        json_text = json_text.replace('```json', '').replace('```', '').strip()
        
        parsed_response = json.loads(json_text)
        
        if isinstance(parsed_response, dict) and 'analysis_decision' in parsed_response and 'content' in parsed_response:
            analysis_decision = parsed_response['analysis_decision']
            content = parsed_response['content']

            if isinstance(content, str):
                try:
                    content = json.loads(content)
                except json.JSONDecodeError:
                    if debug:
                        print(f"Failed to parse content as JSON: {content}")
                    content = []
            
            if not isinstance(content, list):
                if debug:
                    print(f"Content is not a list, converting: {type(content)}")
                content = []
            
            result = {
                'analysis_decision': analysis_decision,
                'content': content
            }

            return result
        
        else:
            if debug:
                print("Missing required keys 'analysis_decision' or 'content'")
            return {
                'analysis_decision': 'ERROR: Invalid response structure',
                'content': []
            }
        
        
        
    except json.JSONDecodeError as e:
        if debug:
            print(f"JSON parsing failed: {e}")
            print(f"Problematic text: '{response.content}'")
        return {
            'analysis_decision': 'ERROR: JSON parsing failed',
            'content': []
        }
    except Exception as e:
        if debug:
            print(f"Error processing response: {e}")
        return {
            'analysis_decision': 'ERROR: Processing failed',
            'content': []
        }
    


