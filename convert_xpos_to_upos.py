import os

def convert_xpos_to_upos_in_directory(directory):
    # Scan all .conllu files in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".conllu"):  # Only process .conllu files
            input_file = os.path.join(directory, filename)
            output_file = os.path.join(directory, f"updated_{filename}")
            print(f"Processing {input_file}...")

            with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
                for line in infile:
                    if line.startswith("#"):
                        # Keep the comment lines (metadata)
                        outfile.write(line)
                    else:
                        # Process the token lines
                        columns = line.split("\t")
                        if len(columns) > 1:  # Avoid empty lines
                            upos = columns[3]  # UPOS is in the 5th column (index 4)
                            # Set XPOS (4th column) equal to UPOS (5th column)
                            columns[4] = upos  # Set XPOS to UPOS value
                            outfile.write("\t".join(columns))
                        else:
                            # Just write empty lines (e.g., between sentences)
                            outfile.write(line)

            print(f"Updated file saved as {output_file}")

languages = ['persian','indonesian','arabic','tagalog']
datasets = ['train','test']
for l in languages:
    for d in datasets:
        directory = f'{l}/{d}'
        convert_xpos_to_upos_in_directory(directory)
