"""A helper script to convert BIO tagged dataset to BIOSE
1. O tag will not be changed
2. I- tag will be converted to E tag, if it's end of an entity
    * End of entity tag means the next tag is not the same I- tag
3. B- tag will be converted to S tag, if it's the only token in entity
    * i.e. the next tag is not the I tag of the same type
"""

import os
import argparse


def convert_bio_to_biose(input_file: str, output_file: str) -> None:
    
    def _dump_buffer(buffer, fout):
        if len(buffer) == 0:
            return

        for i in range(len(buffer)):
            current_label = buffer[i][1]
            if current_label.startswith("I"):
                if i == len(buffer) - 1 or current_label != buffer[i + 1][1]:
                    buffer[i][1] = "E" + buffer[i][1][1:]
            elif current_label.startswith("B"):
                if i == len(buffer) - 1 or \
                        buffer[i + 1][1] != "I" + current_label[1:]:
                    buffer[i][1] = "S" + buffer[i][1][1:]
            
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
    
    convert_bio_to_biose(args.input, args.output)
