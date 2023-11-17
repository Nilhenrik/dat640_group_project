import pandas as pd

# Load the data from a file
data = pd.read_csv('./data/test.txt', sep=' ', header=None)

# Assign column names for better readability
data.columns = ['qid', 'Q0', 'docid', 'Rank', 'Score', 'Model']

# Get the unique Query_ID values in the order they appear
unique_query_ids = data['qid'].unique()

# Initialize an empty list to store the sorted groups
sorted_groups = []

# Iterate through the unique Query_ID values
for query_id in unique_query_ids:
    # Get the group corresponding to the current Query_ID
    group = data[data['qid'] == query_id]
    # Sort the group by Score in descending order
    sorted_group = group.sort_values(by='Score', ascending=False)
    # Get the top 3 results from the sorted group
    top_3_sorted_group = sorted_group.head(3)
    # Select only the Query_ID and Doc_ID columns
    top_3_sorted_group = top_3_sorted_group.loc[:, ['qid', 'docid']]
    # Append the top 3 sorted group to the list of sorted groups
    sorted_groups.append(top_3_sorted_group)

# Concatenate the sorted groups to obtain the final sorted DataFrame
sorted_data = pd.concat(sorted_groups)
sorted_data = sorted_data.rename(columns={'qid': 'qid', 'docid': 'docid'})
# Write the sorted data to a new CSV file
sorted_data.to_csv('./data/qr.csv', sep=',', index=False, header=True)
