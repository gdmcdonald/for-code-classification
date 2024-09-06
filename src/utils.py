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
