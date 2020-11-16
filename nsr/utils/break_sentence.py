"""Break long sentences into chunks (as if they are separate ones).
"""

import os
import argparse


def rechunk_sentence(input_file: str, output_file: str, max_len: int) -> None:
    
    def _dump_buffer(buffer, fout):
        if len(buffer) == 0:
            return

        chunks = list()
        if len(buffer) <= max_len:
            chunks.append(buffer)
        else:
            start = 0
            while True:
                if start + max_len < len(buffer):
                    chunks.append(buffer[start: start + max_len])
                    start = start + max_len
                else:
                    chunks.append(buffer[start: len(buffer)])
                    break
        
        if len(chunks) > 1:
            for i in range(len(chunks) - 1):
                print("Cut-off point: {} {}".format(chunks[i][-1],
                                                    chunks[i + 1][0]))
        
        for chunk in chunks:
            for token, label in chunk:
                fout.write("{} {}\n".format(token, label))
            fout.write("\n")
    
    sentence_buffer = list()
    with open(input_file, "r") as fin:
        with open(output_file, "w") as fout:
            for line in fin:
                line = line.strip().split()
                if len(line) == 0:
                    _dump_buffer(sentence_buffer, fout)
                    sentence_buffer = list()
                else:
                    sentence_buffer.append([line[0], line[-1]])
            
            _dump_buffer(sentence_buffer, fout)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--max_len", type=int, default=200)
    args = parser.parse_args()
    
    rechunk_sentence(args.input, args.output, args.max_len)
