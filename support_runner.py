import dataset_preparation.feature_extraction as fe
import dataset_preparation.dataset_extraction as de
from utils.extraction_functions import compute_return


INPUT_DIR = "./data/stock"
OUTPUT_DIR = "./data"


COMPANY_NAME = "apple"
RETUNR_T = 'relative'       # relative or raw

fe.run(COMPANY_NAME, return_ty=RETUNR_T, path=INPUT_DIR)
de.run(COMPANY_NAME, in_path=INPUT_DIR, out_path=OUTPUT_DIR)