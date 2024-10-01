import os
import pandas as pd

def process_csv_files(directory):
    total_files = 0
    files_with_queries = 0
    total_queries = 0
    all_files = 0
    max = 0
    for filename in os.listdir(directory):
        
        if filename.endswith(".csv"):
            all_files += 1
            filepath = os.path.join(directory, filename)
            df = pd.read_csv(filepath)
            if "queries" in df.columns:
               
                queries_values = df["queries"].dropna()
                score_values = df["score"].dropna()
                if not queries_values.empty and not score_values.empty and score_values.mean() >= 8:
                    print(filename)
                    files_with_queries += 1
                    total_queries += queries_values.mean()
                    if queries_values.mean() > max:
                        print(max)
                        max = queries_values.mean()
                total_files += 1

    if total_files > 0:
        query_percentage = files_with_queries / 50 * 100
        print(all_files, files_with_queries)
        average_queries = (total_queries + (all_files - 50) * 10) / files_with_queries
        print(f"Percentage of files with queries: {query_percentage:.2f}%")
        print(f"Average number of queries: {average_queries:.2f}")


directory_path = "result_test_llama"

process_csv_files(directory_path)
