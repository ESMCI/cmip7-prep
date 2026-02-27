import sys
import re
from typing import List


def parse_cmor_driver_output(file_path: str):
    no_mapping_vars: List[str] = []
    no_model_vars: List[str] = []

    # Regex patterns
    mapping_pattern = re.compile(r"No mapping for (.+) in ")
    model_pattern = re.compile(
        r"Variable (.*) processed with status: ERROR \w+ input variable not found\."
    )

    with open(file_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        mapping_match = mapping_pattern.search(line)
        if mapping_match:
            no_mapping_vars.append(mapping_match.group(1))
        model_match = model_pattern.search(line)
        if model_match:
            no_model_vars.append(model_match.group(1))

    return no_mapping_vars, no_model_vars


def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <cmor_driver_output.txt>")
        sys.exit(1)
    file_path = sys.argv[1]
    no_mapping_vars, no_model_vars = parse_cmor_driver_output(file_path)

    print("Variables with no mapping:")
    for var in no_mapping_vars:
        print(var)
    print("\nVariables with no MODEL variable:")
    for var in no_model_vars:
        print(var)


if __name__ == "__main__":
    main()
