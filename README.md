# Neural-SampleRank

Code for the paper "Training for Gibbs Sampling on Conditional Random Fields with Neural Scoring Factors". 
ACL Anthology [link](https://www.aclweb.org/anthology/2020.emnlp-main.406/)


## How to run

1. Setup the Python environment with conda: 
    * `conda create -n nsr_env Python=3.6.10`
    * `conda activate nsr_env`
    * `pip install -r requirements.txt`
2. Build the C++ extension `python setup.py install`
3. Run unit tests:
    * Create local cache folder for embeddings: `mkdir -p tests/model/data/cache`
        * Downloaded embeddings files can be saved there for faster unit tests
    * `sh scripts/tests.sh`
4. Run experiments:
    * `sh scripts/run.sh PATH/TO/config.yaml`
    * See `config/example.yaml` to see available fields and their usage.


## Additional tools

* `scripts/convert_to_utf8.sh`: This codebase reads data in UTF-8 encoding. Use this script to convert the original CoNLL dataset files to UTF-8.
* `scripts/join_deu_06_tagset.sh`: Use this helper to attach 2006 version of German ground truth to the text.
* `nsr/utils/break_sentence.py`: We break long sentences into 200-token chunks and treat them as separate ones.
* `nsr/utils/convert_bio.py`: Convert CoNLL data into BIO format tags (where the first token of an entity always has a B tag).
* `nsr/utils/convert_biose.py`: Convert BIO format tags into BIOSE format.


