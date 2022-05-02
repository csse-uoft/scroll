# scroll
**S**o**C**ial pu**R**pose **O**rganization **L**anguage mode**L**


The SCROLL project includes the following modules:
- NER model for extracting terms from descriptions of social purpose organizations
- Narrative building model from semantic roles



# Installation
### 1. Setup Python Environment
  - `conda env create -f PyScroll.yml python=3.9`

### 2. Install StanfordCoreNLP packages
  - `cd corenlp`
  - `curl https://downloads.cs.stanford.edu/nlp/software/stanford-corenlp-4.3.1.zip > stanford-corenlp-4.3.1.zip`
  - `open corenlp/stanford-corenlp-4.3.1.zip`
  - `curl https://downloads.cs.stanford.edu/nlp/software/stanford-parser-4.2.0.zip > stanford-parser-4.2.0.zip`
  - `open stanford-parser-4.2.0.zip`
  - `cd ..`

  - `chmod +x corenlp/nlp_server.sh`


### 3. Install Prolog
- Follow instructions here <https://www.swi-prolog.org/download/stable>



# Run program
### 1. Start nlp server
- `./corenlp/nlp_server.sh`

### 2. Run unit test code
- This will generate all files based on sentences in `output/unit_test_sentences.txt`
- `python -i run_unit_tests.py`

### 3. Load own file 
- create and place in `output/<file_prefix>/input`

       filename = 'output/%s/input/input_file.csv'%(file_prefix)
       posp = POSProcesses(col='Description', program_name=Program Name')
       posp.load_compass_paragraphs(
          filename=filename,
          file_prefix=file_prefix, random_n=None)
