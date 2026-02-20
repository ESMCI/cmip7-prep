import sys
import re
from typing import List


def parse_cmor_driver_output(file_path: str):
    no_mapping_vars: List[str] = []
    no_cesm_vars: List[str] = []

    # Regex patterns
    mapping_pattern = re.compile(r"No mapping for (.+) in ")
    cesm_pattern = re.compile(
        r"Variable (.*) processed with status: ERROR \w+ input variable not found\."
    )

    with open(file_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        mapping_match = mapping_pattern.search(line)
        if mapping_match:
            no_mapping_vars.append(mapping_match.group(1))
        cesm_match = cesm_pattern.search(line)
        if cesm_match:
            no_cesm_vars.append(cesm_match.group(1))

    return no_mapping_vars, no_cesm_vars


def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <cmor_driver_output.txt>")
        sys.exit(1)
    file_path = sys.argv[1]
    no_mapping_vars, no_cesm_vars = parse_cmor_driver_output(file_path)

    print("Variables with no mapping:")
    for var in no_mapping_vars:
        print(var)
    print("\nVariables with no CESM variable:")
    for var in no_cesm_vars:
        print(var)


if __name__ == "__main__":
    main()
