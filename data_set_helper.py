import tensorflow as tf

label_type = {"reg":tf.float32,
              "class":tf.int64}

def get_feature_columns(h_params):
    feature_columns = []

    if "rnn" in h_params.model_type:
        feature_columns.append(tf.contrib.layers.real_valued_column(column_name="length",
                                                                    dimension=1, dtype=tf.int64))
        for key in h_params.KEYS:
            feature_columns.append(tf.contrib.layers.real_valued_column(column_name=key,
                                                                        dimension=h_params.sequence_length,
                                                                        dtype=tf.float32))
        feature_columns.append(tf.contrib.layers.real_valued_column(column_name="label",
                                                                    dimension=h_params.sequence_length,
                                                                    dtype=label_type[h_params.e_type]))
    else:
        feature_columns.append(tf.contrib.layers.real_valued_column(
            column_name="features", dimension=(h_params.input_size), dtype=tf.float32))

        feature_columns.append(tf.contrib.layers.real_valued_column(column_name="label",
                                                                    dimension=1,
                                                                    dtype=label_type[h_params.e_type]))

    return set(feature_columns)


def create_input_fn(mode, input_files, batch_size, num_epochs, h_params):
    print("reading file {}".format(input_files))
    def input_fn():
        features = tf.contrib.layers.create_feature_spec_for_parsing(get_feature_columns(h_params))

        feature_map = tf.contrib.learn.io.read_batch_features(
            file_pattern=input_files,
            batch_size=batch_size,
            features=features,
            reader=tf.TFRecordReader,
            randomize_input=False,
            num_epochs=num_epochs,
            queue_capacity=100000 + batch_size * 10,
            name="read_batch_features_{}".format(mode))

        # This is an ugly hack because of a current bug in tf.learn
        # During evaluation TF tries to restore the epoch variable which isn't defined during training
        # So we define the variable manually here
        # if mode == tf.contrib.learn.ModeKeys.TRAIN:
        #     tf.get_variable(
        #         "read_batch_features_eval/file_name_queue/limit_epochs/epochs",
        #         initializer=tf.constant(0, dtype=tf.int64))

        target = feature_map.pop("label")
        if "rnn" in h_params.model_type:
            length = tf.squeeze(feature_map.pop("length"))
            features = tf.concat([tf.expand_dims(feature_map[k], 2)for k in feature_map], axis=2)
            return {'features':features, 'length':length}, target[:, -1]
        else:
            target = tf.squeeze(target, 1)
            return feature_map, target

    return input_fn


if __name__ == '__main__':

    features = tf.contrib.layers.create_feature_spec_for_parsing(get_feature_columns('seq'))

    feature_map = tf.contrib.learn.io.read_batch_features(
        file_pattern=['/home/ando/Project/tf-stock-pred/data/goldman/train_seq.tfrecords'],
        batch_size=100,
        features=features,
        reader=tf.TFRecordReader,
        randomize_input=False,
        num_epochs=1,
        # queue_capacity=10 + batch_size * 10,
        name="read_batch_features_{}".format('train'))



    #
    # filename_queue = tf.train.string_input_producer(['/home/ando/Project/tf-stock-pred/data/goldman/train_seq.tfrecords'], num_epochs=10)
    #
    # # Even when reading in multiple threads, share the filename
    # # queue.
    # context, sequence = read_and_decode(filename_queue)
    # features = tf.train.batch(dict(context, **sequence), batch_size=2)
    #
    # The op for initializing the variables.
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

    with tf.Session() as sess:
        sess.run(init_op)
        # Let's read off 3 batches just for example
        res = tf.contrib.learn.run_n(feature_map, n=2, feed_dict=None)
        for batch in res:
            for idx, key in enumerate(batch):
                print('{}\t{}'.format(key, batch[key][0][0]))
                print('{}\t{}'.format(key, batch[key][0][1]))
