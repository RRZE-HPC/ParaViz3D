#!/bin/bash

# Check arguments
if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <input_file.txt> <n>"
  exit 1
fi

input_file="$1"
n="$2"

# Check if file exists
if [ ! -f "$input_file" ]; then
  echo "Error: File '$input_file' not found."
  exit 1
fi

# Check if n is a positive integer
if ! [[ "$n" =~ ^[0-9]+$ ]]; then
  echo "Error: n must be a positive integer."
  exit 1
fi

# Create output filename
output_file="${input_file%.*}_${n}.txt"

# Empty or create output file
> "$output_file"

# Repeat contents n times
for ((i=1; i<=n; i++)); do
  cat "$input_file" >> "$output_file"
done

echo "Created '$output_file' with contents repeated $n times."
