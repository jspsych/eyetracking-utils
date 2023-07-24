import tensorflow as tf
import numpy as np
import os
import subprocess as sp
from sklearn.metrics.pairwise import euclidean_distances

from et_util import model_layers


def get_corrcoef_array(model, test_data, map_function):
    """Predicts embedding values and calculates Pearson product-moment 
    correlation coefficient for each subject in test data. Records
    correlation coefficient values in an array.

    :param model: embedding model to test
    :param test_data: data used to test model
    :param map_function: map function to normalize data and organize into pairs.
    :return: array of corrcoef values for each subject"""

    normed_test_data = test_data.map(map_function)
    all_test_embeddings = model.predict(normed_test_data.batch(1))

    subject_id_arr = []
    test_pairs_arr = []
    count_start = 0
    corrcoef_arr = []

    for element in test_data:
        subject_id = element[-1]
        if subject_id not in subject_id_arr:
            subject_id_arr.append(subject_id)
            count_end = count_start + sum(int(tf.equal(subject_id, element[-1]))
                                          for element in test_data)

            test_pairs = normed_test_data.skip(count_start).take(count_end - count_start)
            test_embeddings = all_test_embeddings[count_start:count_end]

            test_pairs_arr.append(np.array([e[1] for e in test_pairs.as_numpy_iterator()]))

            y_true = euclidean_distances(test_pairs_arr[-1]).flatten() * 0.025
            y_pred = euclidean_distances(test_embeddings).flatten()

            corrcoef = np.corrcoef(y_true, y_pred)[0, 1]
            corrcoef_arr.append(corrcoef)

            count_start = count_end

    return corrcoef_arr


def get_embedding_data(
        embedding_model,
        triplet_train_data,
        test_data,
        test_data_map_function,
        batched_triplet_validation_data=None,
        embedding_layer_activation="tanh",
        num_embedding_nodes=8,
        margin=0.5,
        optimizer=tf.keras.optimizers.Adam(),
        num_epochs=1,
        steps_per_epoch=100,
        batch_size=100):
    """Trains mediapipe embedding model and outputs array of correlation coefficient
    values per subject. Make sure embedding model does not have an embedding layer.
    
    :param embedding_model: model to test, excluding final embedding layer.
    :param triplet_train_data: data used to train model. must be in proper format 
    (see embedding_test_example)
    :param test_data: data used to test model
    :param test_data_map_function: map function to normalize and reformat test data 
    :param batched_triplet_validation_data: validation data used in model training
    must be in proper format (see embedding_test_example).
    must be batched.
    default None.
    :param embedding_layer_activation: activation function of embedding layer.
    default tanh
    :param num_embedding_nodes: number of nodes in the embedding layer. 
    default 8.
    :param margin: margin value to pass to siamese model
    :param optimizer: optimizer with which training model is compiled.
    default Adam.
    :param num_epochs: number of epochs to train model. 
    default 1.
    :param steps_per_epoch: number of training steps per epoch.
    default 100.
    :param batch_size: size of data batch. 
    default 100.
    :return: array of correlation coefficient values"""

    # give embedding layer desired number of nodes
    last_layer = embedding_model.layers[-1].output
    embedding_layer = tf.keras.layers.Dense(units=num_embedding_nodes,
                                            activation=embedding_layer_activation,
                                            name="embedding_layer")(last_layer)
    embedding_model = tf.keras.Model(inputs=embedding_model.input, outputs=embedding_layer)

    # define triplet loss training model
    input_shape = embedding_model.layers[0].input_shape
    for dims in input_shape:
        input_shape = (dims[1:])

    training_model_input_left_eye_1 = tf.keras.layers.Input(shape=input_shape,
                                                            name="train_input_left_1")

    training_model_input_right_eye_1 = tf.keras.layers.Input(shape=input_shape,
                                                             name="train_input_right_1")

    training_model_input_left_eye_2 = tf.keras.layers.Input(shape=input_shape,
                                                            name="train_input_left_2")
    training_model_input_right_eye_2 = tf.keras.layers.Input(shape=input_shape,
                                                             name="train_input_right_2")

    training_model_input_left_eye_3 = tf.keras.layers.Input(shape=input_shape,
                                                            name="train_input_left_3")
    training_model_input_right_eye_3 = tf.keras.layers.Input(shape=input_shape,
                                                             name="train_input_right_3")

    embedding_anchor = embedding_model(inputs=[training_model_input_left_eye_1,
                                               training_model_input_right_eye_1])
    embedding_positive = embedding_model(inputs=[training_model_input_left_eye_2,
                                                 training_model_input_right_eye_2])
    embedding_negative = embedding_model(inputs=[training_model_input_left_eye_3,
                                                 training_model_input_right_eye_3])

    output = model_layers.DistanceLayer()(embedding_anchor, embedding_positive,
                                          embedding_negative)

    embedding_model_train = tf.keras.Model(inputs=[
        training_model_input_left_eye_1, training_model_input_right_eye_1,
        training_model_input_left_eye_2, training_model_input_right_eye_2,
        training_model_input_left_eye_3, training_model_input_right_eye_3], outputs=output)

    embedding_model_siamese = model_layers.SiameseModel(embedding_model_train, margin=margin)

    # compile and train
    embedding_model_siamese.compile(
        optimizer=optimizer
    )

    embedding_model_siamese.fit(
        triplet_train_data.batch(batch_size),
        epochs=num_epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=batched_triplet_validation_data)

    corrcoef_array = get_corrcoef_array(embedding_model, test_data, test_data_map_function)

    return corrcoef_array


def mask_unused_gpus(leave_unmasked=1):
    """
    Masks all unused GPUs, except for the amount specified.
    """

    acceptable_available_memory = 1024
    command = "nvidia-smi --query-gpu=memory.free --format=csv"

    try:
        _output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]
        memory_free_info = _output_to_list(sp.check_output(command.split()))[1:]
        memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
        available_gpus = [i for i, x in enumerate(memory_free_values)
                          if x > acceptable_available_memory]

        if len(available_gpus) < leave_unmasked:
            raise ValueError(f'Found only {len(available_gpus)} usable GPUs in the system')
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, available_gpus[:leave_unmasked]))
    except Exception as exception:
        print('"nvidia-smi" is probably not installed. GPUs are not masked', exception)


def set_wandb_key(key: str):
    """
    Sets the API key to connect with the Weights and Biases server.
    """

    os.environ["WANDB_API_KEY"] = key
    os.environ["WANDB_AGENT_DISABLE_FLAPPING"] = "true"
