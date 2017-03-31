import tensorflow as tf

FEATURE_SIZE = 120

def get_feature_columns(mode):
    feature_columns = []
    feature_columns.append(tf.contrib.layers.real_valued_column(
        column_name="features", dimension=FEATURE_SIZE, dtype=tf.float32))

    if mode == tf.contrib.learn.ModeKeys.TRAIN or mode == tf.contrib.learn.ModeKeys.EVAL:
        # During training we have a label feature
        feature_columns.append(tf.contrib.layers.real_valued_column(
            column_name="label", dimension=1, dtype=tf.int64))

    return set(feature_columns)


def create_input_fn(mode, input_files, batch_size, num_epochs):
    def input_fn():
        features = tf.contrib.layers.create_feature_spec_for_parsing(get_feature_columns(mode))

        feature_map = tf.contrib.learn.io.read_batch_features(
            file_pattern=input_files,
            batch_size=batch_size,
            features=features,
            reader=tf.TFRecordReader,
            randomize_input=False,
            num_epochs=num_epochs,
            queue_capacity=200000 + batch_size * 10,
            name="read_batch_features_{}".format(mode))

        # This is an ugly hack because of a current bug in tf.learn
        # During evaluation TF tries to restore the epoch variable which isn't defined during training
        # So we define the variable manually here
        if mode == tf.contrib.learn.ModeKeys.TRAIN:
            tf.get_variable(
                "read_batch_features_eval/file_name_queue/limit_epochs/epochs",
                initializer=tf.constant(0, dtype=tf.int64))

        if mode == tf.contrib.learn.ModeKeys.TRAIN or mode == tf.contrib.learn.ModeKeys.EVAL:
            target = feature_map.pop("label")
        else:
            # In evaluation we have 10 classes (utterances).
            # The first one (index 0) is always the correct one
            target = None
        return feature_map['features'], target
    return input_fn
