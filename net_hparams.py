from collections import namedtuple
from utils.extraction_functions import compute_return

KEYS = ['Open', 'High', 'Low', 'Close', 'Volume', 'A/D', 'Adj_Open', 'Adj_High','Adj_Low', 'Adj_Close', 'Adj_Volume',
        'MA_long', 'MA_short', 'MA_medium', 'MACD_long', 'MACD_short', 'PPO_long', 'PPO_short', 'SL']
SL = 20

FPramas = namedtuple(
    "FPramas",
    [
        "KEYS",
        "end_time",
        "start_time",
        "e_type",
        "return_type",
        "sequence_length",
        "KEYS"
    ]
)


def create_fparams():
    return FPramas(
        KEYS=KEYS,
        sequence_length=SL,
        end_time="2017-01-01",
        start_time="2002-01-01",
        e_type="reg",
        return_type={"raw": lambda x: x,
                     "relative": compute_return}
    )


HParams = namedtuple(
    "HParams",
    [
        "batch_size",
        "eval_batch_size",
        "learning_rate",
        "optimizer",
        "h_layer_size",
        "input_size",
        "num_class",
        "dropout",
        "l1_reg",
        "l2_reg",
        "sequence_length",
        "model_type",
        "one_by_one_out_filters",
        "one_by_all_out_filters",
        "KEYS",
        "hidden_layer_type"
    ])

def create_hparams():
    return HParams(
        model_type="deep_rnn",
        sequence_length=SL,
        batch_size=30,
        eval_batch_size=30,
        optimizer="Adam",
        learning_rate=0.01,
        h_layer_size=[19, 19, 47],
        input_size=19,
        num_class={"reg":1,
                   "class":2},
        dropout=0.5,
        l1_reg=0.005,
        l2_reg=0.00,
        one_by_one_out_filters=5,
        one_by_all_out_filters=3,
        KEYS=KEYS,
        hidden_layer_type="dense_layer_over_time"

    )