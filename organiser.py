print("Script Started")

import ollama
import os
import sys

model = "llama3.2"

input_file = "data/grocery_list.txt"
output_file = "data/categorised_list.txt"

if not os.path.exists(input_file):
    print(f"Input file '{input_file}' doesnt exist")
    sys.exit(1)

with open(input_file, "r") as f:
    items = f.read().strip()

prompt  = f"""
You are an Ai assistant that categorizes and sorts grocery items.

Here is a list of items:

{items}

Please:

1.Categorize these items into appropriate categories such as Produce, Dairy, Bakery, Snacks, Frozen, Beverages, etc. Categorize on ur own.
2.Sort The List based on Alphabetical order within each catogery.
3.Present the organised list in a  clear and organised manner, using number and bullet points.

"""

try:
    print(f"Connecting to Ollama and generating with {model}... (this may take a moment)")
    response = ollama.generate(model = model, prompt= prompt)
    model_response = response.get("response", "No Response")

    print("======== Categorised Grocery List ========")
    print(model_response)


    with open(output_file, "w") as f:
        f.write(model_response.strip())

    print(f"The Categorised List is stored in '{output_file}'.\n")

except Exception as e:
    print(f"Error: {e}")