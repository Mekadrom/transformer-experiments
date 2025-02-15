from transformers import AutoTokenizer

import csv

def create_gpt2_metadata_tsv(output_path='gpt2_tokens.tsv'):
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    
    # Open TSV file for writing
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter='\t', quoting=csv.QUOTE_MINIMAL, quotechar='"', escapechar='\\')
        
        # Write header
        writer.writerow(['token', 'id'])
        
        # Write each token and its ID
        for token, id in sorted(tokenizer.get_vocab().items(), key=lambda x: x[1]):
            # Convert token to readable string
            decoded_token = tokenizer.convert_tokens_to_string([token])
            writer.writerow([decoded_token, id])

    print(f"Created metadata TSV file at {output_path}")
    print(f"Total tokens written: {len(tokenizer)}")

if __name__ == "__main__":
    create_gpt2_metadata_tsv()
