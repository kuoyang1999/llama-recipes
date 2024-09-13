# python jsonl2json.py --input_path path_to_your_file.jsonl --output_path desired_output_file.json

import argparse
import json

def convert_jsonl_to_json(input_path, output_path):
    data = []
    with open(input_path, 'r') as file:
        for line in file:
            if line.strip():
                data.append(json.loads(line))

    with open(output_path, 'w') as file:
        json.dump(data, file, indent=4)

def main():
    parser = argparse.ArgumentParser(description="Convert JSONL to JSON")
    parser.add_argument("--input_path", type=str, required=True, help="Path to the input JSONL file")
    parser.add_argument("--output_path", type=str, required=True, help="Path to the output JSON file")
    args = parser.parse_args()

    convert_jsonl_to_json(args.input_path, args.output_path)

if __name__ == "__main__":
    main()
