import dataset_preparation.feature_extraction as fe
import dataset_preparation.dataset_extraction as de
from utils.extraction_functions import compute_return


INPUT_DIR = "./data/stock"
OUTPUT_DIR = "./data"


COMPANY_NAME = "apple"
EXAMPLE_FN_NAME = "create_example_sequencial"
OUTPUT_NAME_SUFFIX = 'seq'
RETUNR_T = 'relative'

fe.run(COMPANY_NAME, return_ty=RETUNR_T, path=INPUT_DIR)
de.run(COMPANY_NAME, EXAMPLE_FN_NAME, OUTPUT_NAME_SUFFIX, in_path=INPUT_DIR, out_path=OUTPUT_DIR)