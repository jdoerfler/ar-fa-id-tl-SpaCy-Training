import os

def convert_xpos_to_upos_in_directory(directory): 
    # Scan all .conllu files in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".conllu"):  # Only process .conllu files
            input_file = os.path.join(directory, filename)
            print(f"Processing {input_file}...")

            # Open the original file for reading, and for writing back the changes (overwrite)
            with open(input_file, 'r', encoding='utf-8') as infile:
                lines = infile.readlines()  # Read all lines

            # Now overwrite the same file with the changes
            with open(input_file, 'w', encoding='utf-8') as outfile:
                for line in lines:
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

            print(f"Original file {input_file} has been updated.")

languages = ['persian','indonesian','arabic','tagalog']
datasets = ['train' ,'test']
for l in languages:
    for d in datasets:
        directory = f'{l}/{d}'
        convert_xpos_to_upos_in_directory(directory)
