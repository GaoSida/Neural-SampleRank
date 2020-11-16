"""A helper script to convert IO tagged dataset to BIO
1. O tag will not be changed
2. B- tag will not be changed
3. I- tag will be changed to B if it is the beginning of an entity
    * It's the first token of a sentence
    * The previous token is not I- or B- of the same type
"""

import os
import argparse


def convert_io_to_bio(input_file: str, output_file: str) -> None:
    
    def _dump_buffer(buffer, fout):
        if len(buffer) == 0:
            return

        for i in range(len(buffer)):
            current_label = buffer[i][1]
            if current_label.startswith("I"):
                if i == 0 or buffer[i - 1][1][1:] != current_label[1:]:
                    buffer[i][1] = "B" + buffer[i][1][1:]
            
        for token, label in buffer:
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
    args = parser.parse_args()
    
    convert_io_to_bio(args.input, args.output)
