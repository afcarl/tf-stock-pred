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
        "num_class"
    ])

def create_hparams():
    return HParams(
        batch_size=50,
        eval_batch_size=50,
        optimizer="Adam",
        learning_rate=0.005,
        h_layer_size=[10, 20, 10],
        input_size=200,
        num_class=2
    )