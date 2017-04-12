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
        "l2_reg",
        "sequence_length",
        "experiment_type",
        "model_type"
    ])

def create_hparams():
    return HParams(
        experiment_type="classification",
        model_type="deep_rnn",
        sequence_length=20,
        batch_size=100,
        eval_batch_size=100,
        optimizer="SGD",
        learning_rate=0.001,
        h_layer_size=[10, 64],
        input_size=17,
        num_class=2,
        dropout=0.0,
        l2_reg=0.00,
    )