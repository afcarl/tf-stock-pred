import dataset_preparation.feature_extraction as fe
import dataset_preparation.dataset_extraction as de



INPUT_DIR = "./data/stock"
OUTPUT_DIR = "./data"
COMPANY_NAME = "apple"
EXAMPLE_FN_NAME = "create_example_sequencial"
OUTPUT_NAME_SUFFIX = 'seq'
RETURN_FN = lambda x:x 

fe.run(COMPANY_NAME, path=INPUT_DIR, return_fn=RETURN_FN)
de.run(COMPANY_NAME, EXAMPLE_FN_NAME, OUTPUT_NAME_SUFFIX, in_path=INPUT_DIR, out_path=OUTPUT_DIR)