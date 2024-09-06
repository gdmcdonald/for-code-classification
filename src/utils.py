import pandas as pd


def build_hierarchy_with_ids(df):
    """
    Build the Hierarchical Structure with Identifiers
    """

    hierarchy = {}
    
    for _, row in df.iterrows():
        current_level = hierarchy
        
        # Iterate over pairs of columns (category and ID)
        for i in range(0, len(df.columns), 2):  
            category_col = df.columns[i]
            id_col = df.columns[i + 1]
            
            if pd.notna(row[category_col]):
                if row[category_col] not in current_level:
                    current_level[row[category_col]] = {'id': row[id_col], 'children': {}}
                current_level = current_level[row[category_col]]['children']
    
    return hierarchy




def classify_text_with_ids(text, classifier, node, threshold=0.7, n_max=None):
    """
    Classify a given piece of text
    """
    
    labels = list(node.keys())
    predictions = classifier(text, candidate_labels=labels, multi_label=True)

    max_score = max(predictions['scores'])
    normalized_scores = [score / max_score for score in predictions['scores']]

    valid_candidates = [
        (predictions['labels'][i], predictions['scores'][i], node[predictions['labels'][i]]['id'])
        for i in range(len(labels))
        if normalized_scores[i] >= threshold
    ]

    results = []
    for label, score, id in valid_candidates:
        child_node = node[label]
        if child_node['children']:
            sub_results = classify_text_with_ids(text, classifier, child_node['children'], threshold)
            for sub_labels, sub_ids, sub_prob in sub_results:
                results.append(([label] + [sub_labels], [id] + [sub_ids], score * sub_prob))
        else:
            results.append(([label], [id], score))

    if results:
        max_combined_prob = max(result[2] for result in results)
        filtered_results = [
            result for result in results
            if result[2] >= threshold * max_combined_prob
        ]
        filtered_results.sort(key=lambda x: x[2], reverse=True)
        if n_max is not None:
            filtered_results = filtered_results[:n_max]
        results = filtered_results
        #print(results)
    # Format the results
    formatted_results = [result[0] + result[1] + [round(result[2], 3)] for result in results]
    return formatted_results



def classify_papers_with_template(papers, classifier, hierarchy_with_ids, template, threshold=0.7, n_max=3):
    """
    Classifies the combined text from specified columns in the papers DataFrame using the 
    specified classifier and hierarchy. Outputs a DataFrame with classification results joined to 
    the original papers DataFrame. The query for classification is generated using a custom template.

    Parameters:
        papers (pd.DataFrame): The input DataFrame containing columns for classification.
        classifier (Callable): The function to classify the text with IDs.
        hierarchy_with_ids (list): The hierarchical classification system with associated IDs.
        template (str): A string template to format the query (e.g., "Title: {Publication Title}, Abstract: {Abstract}").
        threshold (float, optional): The threshold for filtering classification results. Defaults to 0.7.
        n_max (int, optional): Maximum number of classifications to return. Defaults to 3.

    Returns:
        pd.DataFrame: The original papers DataFrame joined with classification results.
    """
    
    results = []

    # Process each row using the custom template to create the query
    for _, row in papers.iterrows():
        # Use the template to create a query string, replacing placeholders with column values
        query = template.format(**row)
        classification = classify_text_with_ids(query, classifier, hierarchy_with_ids, threshold=threshold, n_max=n_max)
        flattened_list = [item for sublist in classification for item in sublist]
        results.append([query] + flattened_list)
    
    # Create DataFrame with the classification results
    columns = [
        'query', 'Division_1', 'Group_1', 'Division_id_1', 'Group_id_1', 'Confidence_1', 
        'Division_2', 'Group_2', 'Division_id_2', 'Group_id_2', 'Confidence_2',  
        'Division_3', 'Group_3', 'Division_id_3', 'Group_id_3', 'Confidence_3'
    ]
    
    df_results = pd.DataFrame(results, columns=columns)
    
    # Join the original papers DataFrame with the classification results directly
    output = pd.concat([papers, df_results.drop(columns=['query'])], axis=1)
    
    return output


# Function to check for a match based on specified number of digits
def check_code_match(row, code_col, num_digits):
    forcode_list = [row['FORCODE1'], row['FORCODE2'], row['FORCODE3']]
    code_id = row[code_col]
    if pd.notna(code_id):
        code_str = str(int(code_id))[:num_digits]  # Take the specified number of digits
        for forcode in forcode_list:
            if pd.notna(forcode):
                if str(int(forcode))[:num_digits] == code_str:
                    return True
    return False


def apply_code_matches(df, match_config):
    """
    Applies the check_code_match function to multiple columns based on the match configuration.
    
    Parameters:
    df (DataFrame): The DataFrame to operate on.
    match_config (dict): A dictionary where keys are match types (e.g., 'Division', 'Group') and values are the number of digits.
    
    Returns:
    DataFrame: Modified DataFrame with match results.
    """
    for match_type, digits in match_config.items():
        for i in range(1, 4):  # Assuming there are three columns for each match type
            col_name = f'{match_type}_id_{i}'
            output_col = f'{match_type}_match_{i}'
            df[output_col] = df.apply(check_code_match, axis=1, code_col=col_name, num_digits=digits)
    
        # Use .filter to select columns based on a pattern (wildcard for "{match_type}_match.*"
        agg_cols = df.filter(like=f'{match_type}_match').columns

        # Check if any {match_type} matches are True
        df[f'Any_{match_type}_Match'] = df[agg_cols].any(axis=1)



    return df