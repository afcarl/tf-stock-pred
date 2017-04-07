from collections import namedtuple

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
        "l2_reg"
    ])

def create_hparams():
    return HParams(
        batch_size=100,
        eval_batch_size=100,
        optimizer="Adam",
        learning_rate=0.01,
        h_layer_size=[50, 40],
        input_size=160,
        num_class=2,
        dropout=0.0,
        l2_reg=0.01
    )