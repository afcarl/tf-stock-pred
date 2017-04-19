import dataset_preparation.feature_extraction as fe
import dataset_preparation.dataset_extraction as de



INPUT_DIR = "./data/stock"
OUTPUT_DIR = "./data"
COMPANY_NAME = "IBM"
EXAMPLE_FN_NAME = "create_example_sequencial"
OUTPUT_NAME_SUFFIX = '_seq'


# fe.run(COMPANY_NAME, path=INPUT_DIR)
de.run(COMPANY_NAME, EXAMPLE_FN_NAME, OUTPUT_NAME_SUFFIX, path=INPUT_DIR)