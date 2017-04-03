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
        batch_size=10,
        eval_batch_size=10,
        optimizer="SGD",
        learning_rate=0.1,
        h_layer_size=[10, 20, 10],
        input_size=4,
        num_class=3
    )