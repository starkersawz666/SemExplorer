import os
import json
import pandas as pd
import numpy as np

_ = """
This file contains functions to generate schema for filters from configuration.
"""

# Generate statistics options from configuration
def generate_statistics(config):
    statistics = config['statistics']
    statistics_options = []
    for statistic in statistics:
        for key, value in statistic.items():
            statistics_options.append(key)
    return statistics_options

# Generate advanced statistics options from configuration
def generate_advanced_statistics(config):
    statistics = config['advanced_statistics']
    statistics_options = []
    for statistic in statistics:
        for key, value in statistic.items():
            statistics_options.append(key)
    return statistics_options

# Generate filter schema from configuration
def generate_schema(config):
    schema = []
    for item in config['basic_filters']:
        schema.append((item, 'bool'))
    for item in config['statistics']:
        for key, value in item.items():
            schema.append((key, value))
    return schema

def get_filters(schema_name, category):
    schema_path = os.path.join("./data/schema", f"{schema_name}.json")
    with open(schema_path, "r") as f:
        schema = json.load(f)
    filters_list = []
    for item in schema[category]:
        filters_list.append(item)
    return filters_list

def get_category_from_col(column):
    column = column.replace('', np.nan).replace(' ', np.nan).dropna()
    sampled_values = column.sample(n=min(8, len(column)), random_state=1)
    boolean_values = set(["true", "false", "t", "f", "yes", "no", "0", "1"])
    boolean_strict = set(["0", "1"])
    str_sampled_values = [str(val).strip().lower() for val in sampled_values]
    if all(val in boolean_values for val in str_sampled_values):
        if all(val in boolean_strict for val in str_sampled_values) and '1' in str_sampled_values:
            return "Boolean"
        elif set(str_sampled_values).intersection(["true", "false", "t", "f", "yes", "no"]):
            return "Boolean"
    if all(isinstance(val, (int, float)) or val.replace('.', '', 1).isdigit() for val in sampled_values):
        return "Statistics-basic"
    return "Text-semantic"