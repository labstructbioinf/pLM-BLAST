import os

def split_fasta(input_file, prefix, chunk_size, max_sequence_length=None):
    # Read the contents of the input file
    with open(input_file, "r") as f:
        lines = f.readlines()

    sequence_data = []
    current_sequence = ""

    for line in lines:
        if line.startswith(">"):
            if current_sequence:
                sequence_data.append(current_sequence)
            current_sequence = line
        else:
            current_sequence += line

    sequence_data.append(current_sequence)

    # Split sequences into long and short ones
    to_long = []
    to_chunk = []

    for seq in sequence_data:
        sequence_lines = seq.split("\n")[1:]
        sequence = "".join(sequence_lines)
        if max_sequence_length is not None and len(sequence) > max_sequence_length:
            to_long.append(seq)
        else:
            to_chunk.append(seq)

    # Sort sequences by length
    to_chunk.sort(key=lambda seq: len("".join(seq.split("\n")[1:])))

    num_sequences = len(to_chunk)
    input_folder = os.path.dirname(input_file)

    # Save sequences to output files
    for i in range(0, num_sequences, chunk_size):
        chunk = to_chunk[i:i + chunk_size]
        output_file = os.path.join(input_folder, f"{prefix}_{i // chunk_size + 1}.fas")
        with open(output_file, "w") as f:
            f.writelines(chunk)

    # Save long sequences to a separate file
    to_long_output = os.path.join(input_folder, f"{prefix}_CPU_.fas")
    with open(to_long_output, "w") as f:
        f.writelines(to_long)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('input', help='Fast file to split', type=str)
    parser.add_argument('prefix', help='Prefix for outputs files', type=str)
    parser.add_argument('-cs', help='chunk size', type=int, default=10, dest='CS')
    parser.add_argument('-ml', help='maximum sequence length', type=int, default=None, dest='ML')

    args = parser.parse_args()

    split_fasta(args.input, args.prefix, args.CS, args.ML)