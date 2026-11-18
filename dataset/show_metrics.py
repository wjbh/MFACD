# RQ1
import numpy as np
from tabulate import tabulate  # Add tabulate import for table formatting

try:
    from config import ProjectConfig, ProjectConst
except ImportError:
    from ..config import ProjectConfig, ProjectConst


def main():
    base_dir = f"{ProjectConfig.path_compiled_dataset}"
    file_prefix = "train"
    lang = "java"
    llm_types = [ProjectConst.llm_type_origin, ProjectConst.llm_type_deepseek_v3,
                 ProjectConst.llm_type_qwen3_coder_plus, ProjectConst.llm_type_chatgpt]
    param_names = ['average_word_length', 'cyclomatic_complexity', 'line_of_codes', 'max_line_length',
                   'max_nesting_depth', 'number_of_arguments', 'number_of_branches', 'number_of_expressions',
                   'number_of_loops', 'number_of_null_check', 'number_of_operands', 'number_of_operators',
                   'number_of_statements', 'number_of_unused_variables', 'number_of_variable_declarations',
                   'number_of_variable_reassignments', 'number_of_variable_references', 'total_depth_of_nesting']

    # Initialize dictionary to store statistics for each LLM type
    stats_dict = {}

    # Loop through each LLM type to process metrics
    for llm_type in llm_types:
        # Load metrics data for current LLM type
        metrics_path = f"{base_dir}/{lang}_{file_prefix}_metrics_{llm_type}.npz"
        data = np.load(metrics_path)

        # Extract normalized features and denormalization parameters
        features = data['features']  # Shape: (n_samples, n_params)
        global_min = data['global_min']  # Shape: (n_params,)
        global_max = data['global_max']  # Shape: (n_params,)

        # Denormalize features: features_denorm = features * (max - min) + min
        features_denorm = features * (global_max - global_min) + global_min

        # Compute mean statistics for each parameter (column-wise mean)
        stats = np.mean(features_denorm, axis=0)  # Shape: (n_params,)
        stats_dict[llm_type] = stats

    # Prepare table data: rows = param_names, columns = llm_types
    headers = ['Param Names'] + llm_types
    table_data = []
    for param_idx, param in enumerate(param_names):
        # For each parameter, collect stats across all LLM types
        row = [param] + [round(stats_dict[llm_type][param_idx], 4) for llm_type in llm_types]
        table_data.append(row)

    # Print formatted table
    print(tabulate(table_data, headers=headers, tablefmt='grid'))


if __name__ == "__main__":
    main()