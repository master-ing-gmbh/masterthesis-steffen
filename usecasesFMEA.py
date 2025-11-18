from retriever import get_retriever, retrieve_functions_from_vector, retrieve_failures_from_vector, retrieve_existing_measures_from_vector, retrieve_risk_ratings_from_vector
from entityExtraction import extract_entities_from_question, extract_system_elements, extract_system_elements_with_functions, extract_failure_chains, extract_failure_chains_with_risk_ratings
from graphQuery import retrieve_existing_measures_from_graph, retrieve_qa_system_generation_data, retrieve_functions_from_graph, retrieve_failures_from_graph, retrieve_risk_ratings_from_graph
from outputGeneration import generate_answer_system_structure, generate_functions, generate_failures, generate_existing_measures, generate_risk_rating, generate_risk_rating_async, generate_new_measures
from misc import comprehensive_retriever, add_functions_to_table_structure, add_failure_modes_to_table_structure, add_existing_measures_to_table_structure, add_risk_rating_to_table_structure, add_new_measures_to_table_structure
import csv
import os
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor
import nest_asyncio
from datetime import datetime
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

nest_asyncio.apply()

def question_answer_system_generation(question, llm, graph, debug):
    retriever = get_retriever(amountResults=10)

    # Step 1: Extract entities
    entities = extract_entities_from_question(question, llm, debug)

    # Step 2: Execute graph query
    graph_results = retrieve_qa_system_generation_data(question, entities, llm, graph, debug=debug)

    # Step 3: Generate comprehensive context
    comprehensive_context = comprehensive_retriever(graph_results, retriever.invoke(question), debug)

    # Step 4: Generate final output
    final_output = generate_answer_system_structure(question, comprehensive_context, llm, debug)

    return final_output

def function_generation(llm, graph, current_table_structure, debug):
    retriever = get_retriever(amountResults=10)

    # Extract system elements

    system_structure_list = extract_system_elements(current_table_structure, debug)
    csv_log_data = []
    # rag retrieval for each system element
    table_structure_with_functions = []
    for element_context in system_structure_list:
        
        # graph query
        graph_context = retrieve_functions_from_graph(
            element_context, graph, debug = True
        )
        
        # vector query

        vector_context = retrieve_functions_from_vector(
            element_context, retriever, debug = False
        )
        
        # function generation
        comprehensive_context = comprehensive_retriever(graph_context, vector_context, debug)
        
        generated_functions = generate_functions(
            comprehensive_context, element_context, llm, debug
        )

        table_structure_with_functions = add_functions_to_table_structure(table_structure_with_functions,
            element_context, generated_functions, debug
        )



    if debug:
        print("Final table structure with functions: ", table_structure_with_functions)
    return table_structure_with_functions

def failure_generation(llm, graph, current_table_structure, debug):
    retriever = get_retriever(amountResults=10)

    functions_list = extract_system_elements_with_functions(current_table_structure, debug)

    csv_log_data = []

    table_structure_with_failures = []
    for function_context in functions_list:
         
        graph_context = retrieve_failures_from_graph(
            function_context, graph, debug = False
        )
        
    
        vector_context = retrieve_failures_from_vector(
            function_context, retriever, debug = False
        )
       
        comprehensive_context = comprehensive_retriever(graph_context, vector_context, debug = False)

        generated_failures = generate_failures(
            comprehensive_context, function_context, llm, debug)


        table_structure_with_failures = add_failure_modes_to_table_structure(table_structure_with_failures,
            function_context, generated_failures.get('content'), debug)
        
    
        csv_entry = {
            'function': function_context.get('Function'),
            'analysis_decision': generated_failures.get('analysis_decision', 'No analysis decision available')
        }
        csv_log_data.append(csv_entry)
    
    logs_dir = 'logs'
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    
    current_datetime = datetime.now().strftime("%Y%m%d_%H:%M:%S")
    csv_filename = f"{current_datetime}_failure_generation.csv"
    csv_filepath = os.path.join(logs_dir, csv_filename)
    
    try:
        with open(csv_filepath, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['function', 'analysis_decision']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            writer.writerows(csv_log_data)
        
        if debug:
            print(f"CSV log saved to: {csv_filepath}")
            print(f"Logged {len(csv_log_data)} failures analysis decisions")
    
    except Exception as e:
        if debug:
            print(f"Error saving CSV log: {e}")   
    
    if debug:
        print("Final table structure with failures: ", table_structure_with_failures)
    return table_structure_with_failures


async def generate_existing_measures_async(comprehensive_context, element_context, async_llm, debug: bool):
    
    from outputGeneration import generate_existing_measures_async as original_async_function
    return await original_async_function(comprehensive_context, element_context, async_llm, debug)

async def _process_single_failure_context_async(failure_context, retriever, graph, llm, async_llm, debug):
    row_id = failure_context.get('row_id', 'unknown')
    
    try:
        if debug:
            print(f"Processing failure context: {row_id}")
        
        # Use separate thread pools to avoid contention
        loop = asyncio.get_event_loop()
        
        # Create separate executor for database operations only
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="db-queries") as db_executor:
            graph_future = loop.run_in_executor(
                db_executor, 
                retrieve_existing_measures_from_graph,
                failure_context, graph, False
            )
            vector_future = loop.run_in_executor(
                db_executor,
                retrieve_existing_measures_from_vector, 
                failure_context, retriever, False
            )
            
            graph_context, vector_context = await asyncio.gather(graph_future, vector_future)
        
        comprehensive_context = comprehensive_retriever(graph_context, vector_context, debug=False)

        generated_existing_measures = await generate_existing_measures_async(
            comprehensive_context, failure_context, async_llm, debug
        )

        if 'error_details' in generated_existing_measures or \
           generated_existing_measures.get('analysis_decision', '').startswith('ERROR'):
            return {
                'success': False,
                'row_id': row_id,
                'reason': generated_existing_measures.get('analysis_decision'),
                'failure_context': failure_context
            }

        csv_entry = {
            'failure_cause': failure_context.get('FailureCause'),
            'analysis_decision': generated_existing_measures.get('analysis_decision', 'No analysis decision available')
        }

        return {
            'success': True,
            'row_id': row_id,
            'failure_context': failure_context,
            'generated_measures': generated_existing_measures.get('content'),
            'csv_entry': csv_entry
        }

    except json.JSONDecodeError as e:
        return {
            'success': False,
            'row_id': row_id,
            'reason': f'JSONDecodeError: {str(e)[:100]}',
            'failure_context': failure_context
        }
        
    except Exception as e:
        return {
            'success': False,
            'row_id': row_id,
            'reason': f'Error: {str(e)[:100]}',
            'failure_context': failure_context
        }

def existing_measure_generation(llm, graph, current_table_structure, debug):
    return asyncio.run(existing_measure_generation_async(llm, graph, current_table_structure, debug))

async def existing_measure_generation_async(llm, graph, current_table_structure, debug, batch_size=20):
    retriever = get_retriever(amountResults=10)
    
    async_llm = AzureChatOpenAI(
        api_version="2024-12-01-preview",
        azure_deployment="gpt-4.1-mini",
        model_name="gpt-4.1-mini",
        max_retries=3,
        request_timeout=30,
    )
    
    csv_log_data = []
    skipped_rows = []
    processed_count = 0
    table_structure_with_existing_measures = []
    total_contexts = len(current_table_structure)
    
    if debug:
        print(f"Starting ultimate async processing of {total_contexts} contexts with batch_size={batch_size}")
    

    all_tasks = [
        _process_single_failure_context_async(failure_context, retriever, graph, llm, async_llm, debug)
        for failure_context in current_table_structure
    ]
    
    semaphore = asyncio.Semaphore(batch_size) 
    
    async def controlled_task(task):
        async with semaphore:
            return await task
    
    if debug:
        print(f"Executing {len(all_tasks)} tasks with max {batch_size} concurrent...")
    
    results = await asyncio.gather(*[controlled_task(task) for task in all_tasks], return_exceptions=True)
    
    # Process results
    for result in results:
        if isinstance(result, Exception):
            print(f"Unexpected exception: {result}")
            continue
            
        if result['success']:
            table_structure_with_existing_measures = add_existing_measures_to_table_structure(
                table_structure_with_existing_measures,
                result['failure_context'], 
                result['generated_measures'], 
                debug
            )
            
            csv_log_data.append(result['csv_entry'])
            processed_count += 1
        else:
            if debug:
                print(f"Row {result['row_id']}: Skipping due to error")
            skipped_rows.append({'row_id': result['row_id'], 'reason': result['reason']})


    logs_dir = 'logs'
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    
    current_datetime = datetime.now().strftime("%Y%m%d_%H:%M:%S")
    csv_filename = f"{current_datetime}_existing_measures.csv"
    csv_filepath = os.path.join(logs_dir, csv_filename)
    
    # Write CSV file
    try:
        with open(csv_filepath, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['failure_cause', 'analysis_decision']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            writer.writerows(csv_log_data)
        
        if debug:
            print(f"CSV log saved to: {csv_filepath}")
            print(f"Logged {len(csv_log_data)} existing measures decisions")
    
    except Exception as e:
        if debug:
            print(f"Error saving CSV log: {e}")

    if debug:
        print("Final table structure with functions: ", table_structure_with_existing_measures)
    return table_structure_with_existing_measures

async def _process_chain_optimized_async(failure_chain_context, retriever, graph, async_llm, shared_db_executor, debug):

    failure_cause = failure_chain_context.get('FailureCause', 'unknown')
    
    try:
        if debug:
            print(f"Processing: {failure_cause[:50]}...")
        
        loop = asyncio.get_event_loop()
        
        graph_future = loop.run_in_executor(
            shared_db_executor, 
            retrieve_risk_ratings_from_graph,
            failure_chain_context, graph, False
        )
        vector_future = loop.run_in_executor(
            shared_db_executor,
            retrieve_risk_ratings_from_vector, 
            failure_chain_context, retriever, False
        )
        
        graph_context, vector_context = await asyncio.gather(graph_future, vector_future)
        
        comprehensive_context = comprehensive_retriever(graph_context, vector_context, debug=False)

        generated_risk_rating = await generate_risk_rating_async(
            comprehensive_context, failure_chain_context, async_llm, debug
        )

        if 'error_details' in generated_risk_rating or \
           generated_risk_rating.get('analysis_decision', '').startswith('ERROR'):
            return {
                'success': False,
                'failure_cause': failure_cause,
                'reason': generated_risk_rating.get('analysis_decision'),
                'failure_chain_context': failure_chain_context
            }

        csv_entry = {
            'failure_cause': failure_chain_context.get('FailureCause'),
            'failure_effect': failure_chain_context.get('FailureEffect'),
            'analysis_decision': generated_risk_rating.get('analysis_decision', 'No analysis decision available')
        }

        return {
            'success': True,
            'failure_cause': failure_cause,
            'failure_chain_context': failure_chain_context,
            'generated_risk_rating': generated_risk_rating.get('content'),
            'csv_entry': csv_entry
        }

    except json.JSONDecodeError as e:
        return {
            'success': False,
            'failure_cause': failure_cause,
            'reason': f'JSONDecodeError: {str(e)[:100]}',
            'failure_chain_context': failure_chain_context
        }
        
    except Exception as e:
        return {
            'success': False,
            'failure_cause': failure_cause,
            'reason': f'Error: {str(e)[:100]}',
            'failure_chain_context': failure_chain_context
        }

def risk_rating_generation(llm, graph, current_table_structure, debug):
    return asyncio.run(risk_rating_generation_async_optimized(llm, graph, current_table_structure, debug))

async def risk_rating_generation_async_optimized(llm, graph, current_table_structure, debug, batch_size=20):
    retriever = get_retriever(amountResults=10)

    unique_failure_chains = extract_failure_chains(current_table_structure, debug)
    
    def safe_hash_value(value):
        if value is None:
            return "None"
        elif isinstance(value, (list, tuple)):
            # Convert list/tuple to string, handling nested structures
            return str(sorted(str(item) for item in value) if value else [])
        elif isinstance(value, dict):
            # Convert dict to sorted string representation
            return str(sorted(value.items()) if value else {})
        else:
            return str(value)
    
    seen_chains = set()
    truly_unique_chains = []
    for chain in unique_failure_chains:
        try:
            key = (
                safe_hash_value(chain.get('Product', '')),
                safe_hash_value(chain.get('Subsystem', '')), 
                safe_hash_value(chain.get('SystemElement', '')),
                safe_hash_value(chain.get('Function', '')), 
                safe_hash_value(chain.get('FailureMode', '')), 
                safe_hash_value(chain.get('FailureCause', '')),
                safe_hash_value(chain.get('FailureEffect', '')), 
                safe_hash_value(chain.get('PreventiveMeasure', [])), 
                safe_hash_value(chain.get('DetectiveMeasure', []))
            )
            if key not in seen_chains:
                seen_chains.add(key)
                truly_unique_chains.append(chain)
        except Exception as e:
            if debug:
                print(f"Warning: Could not hash chain for deduplication: {e}")
            truly_unique_chains.append(chain)
    
    if debug:
        print(f"DEDUPLICATION ANALYSIS:")
        print(f"Original failure chains: {len(unique_failure_chains)}")
        print(f"Truly unique chains: {len(truly_unique_chains)}")
        if len(truly_unique_chains) > 0:
            speedup = len(unique_failure_chains) / len(truly_unique_chains)
            print(f"Theoretical speedup from deduplication: {speedup:.2f}x")
    
    async_llm = AzureChatOpenAI(
        api_version="2024-12-01-preview",
        azure_deployment="gpt-4.1-mini",
        model_name="gpt-4.1-mini",
        max_retries=3,
        request_timeout=30,
    )
    
    shared_db_executor = ThreadPoolExecutor(max_workers=6, thread_name_prefix="shared-db-risk")
    
    try:
        # Initialize tracking
        csv_log_data = []
        skipped_chains = []
        processed_count = 0
        table_structure_with_risk_rating = []
        total_chains = len(truly_unique_chains)
        
        if debug:
            print(f"Starting OPTIMIZED async risk rating processing of {total_chains} truly unique failure chains")
            print(f"   Batch size: {batch_size}")
        
        all_results = []
        for i in range(0, total_chains, batch_size):
            batch_chains = truly_unique_chains[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (total_chains + batch_size - 1) // batch_size
            
            if debug:
                print(f"Processing batch {batch_num}/{total_batches} ({len(batch_chains)} chains)")
            
            batch_tasks = [
                _process_chain_optimized_async(chain, retriever, graph, async_llm, shared_db_executor, debug)
                for chain in batch_chains
            ]
            
            # Process current batch
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            all_results.extend(batch_results)
            
            if debug:
                successful = sum(1 for r in batch_results if isinstance(r, dict) and r.get('success', False))
                print(f"   ✅ Batch {batch_num} completed: {successful}/{len(batch_results)} successful")
        
        # Process all results
        for result in all_results:
            if isinstance(result, Exception):
                print(f"❌ Unexpected exception: {result}")
                continue
                
            if result.get('success'):
                # Add to table structure
                table_structure_with_risk_rating = add_risk_rating_to_table_structure(
                    table_structure_with_risk_rating,
                    result['failure_chain_context'], 
                    result['generated_risk_rating'], 
                    debug
                )
                
                csv_log_data.append(result['csv_entry'])
                processed_count += 1
            else:
                if debug:
                    print(f"⚠️  Failure chain {result.get('failure_cause', 'unknown')}: Skipping due to error")
                skipped_chains.append({
                    'failure_cause': result.get('failure_cause', 'unknown'), 
                    'reason': result.get('reason', 'unknown error')
                })
    
    finally:
        shared_db_executor.shutdown(wait=True)
        if debug:
            print("🧹 Cleaned up shared ThreadPoolExecutor")
    
    if len(unique_failure_chains) > len(truly_unique_chains):
        actual_speedup = len(unique_failure_chains) / len(truly_unique_chains)
        print(f"   🚀 Actual speedup achieved: {actual_speedup:.2f}x (from deduplication)")
    if skipped_chains:
        print(f"   ⚠️  Skipped failure causes: {[r['failure_cause'] for r in skipped_chains[:5]]}")
        if len(skipped_chains) > 5:
            print(f"   ... and {len(skipped_chains) - 5} more")
    print("=" * 70 + "\n")
    
    logs_dir = 'logs'
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    
    current_datetime = datetime.now().strftime("%Y%m%d_%H:%M:%S")
    csv_filename = f"{current_datetime}_risk_rating_optimized.csv"
    csv_filepath = os.path.join(logs_dir, csv_filename)
    
    try:
        with open(csv_filepath, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['failure_cause', 'failure_effect', 'analysis_decision']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            writer.writerows(csv_log_data)
        
        if debug:
            print(f"📊 CSV log saved to: {csv_filepath}")
            print(f"📊 Logged {len(csv_log_data)} risk rating decisions")
    
    except Exception as e:
        if debug:
            print(f"❌ Error saving CSV log: {e}")

    if debug:
        print("🎯 Final table structure with OPTIMIZED risk ratings complete")
    return table_structure_with_risk_rating




def new_measure_generation(llm, graph, current_table_structure, debug):
    retriever = get_retriever(amountResults=10)

    unique_failure_chains_with_risk_rating = extract_failure_chains_with_risk_ratings(current_table_structure, debug)
    
    # Initialize CSV logging data
    csv_log_data = []
    table_structure_with_new_measures = []
    for failure_chain_with_ratings in unique_failure_chains_with_risk_rating:

    
        graph_context = retrieve_existing_measures_from_graph(
            failure_chain_with_ratings, graph, debug = False
        )
        
        vector_context = retrieve_existing_measures_from_vector(
            failure_chain_with_ratings, retriever, debug = False
        )
        

        comprehensive_context = comprehensive_retriever(graph_context, vector_context, debug)

        generated_new_measures = generate_new_measures(
            comprehensive_context, failure_chain_with_ratings, llm, debug)

        print("Failure chain with ratings:", failure_chain_with_ratings)
        print("Generated new measures:", generated_new_measures.get('content'))

        table_structure_with_new_measures = add_new_measures_to_table_structure(table_structure_with_new_measures,
            failure_chain_with_ratings, generated_new_measures.get('content'), debug)

        csv_entry = {
            'failure_cause': failure_chain_with_ratings.get('FailureCause'),
            'analysis_decision': generated_new_measures.get('analysis_decision', 'No analysis decision available')
        }
        csv_log_data.append(csv_entry)


    logs_dir = 'logs'
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    
    current_datetime = datetime.now().strftime("%Y%m%d_%H:%M:%S")
    csv_filename = f"{current_datetime}_new_measures.csv"
    csv_filepath = os.path.join(logs_dir, csv_filename)
    
    try:
        with open(csv_filepath, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['failure_cause', 'analysis_decision']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            writer.writerows(csv_log_data)
        
        if debug:
            print(f"CSV log saved to: {csv_filepath}")
            print(f"Logged {len(csv_log_data)} new measure decisions")
    
    except Exception as e:
        if debug:
            print(f"Error saving CSV log: {e}")
   
    if debug:
        print("Final table structure with new measures: ", table_structure_with_new_measures)
    return table_structure_with_new_measures

