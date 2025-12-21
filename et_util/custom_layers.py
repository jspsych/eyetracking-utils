import keras
from keras import ops


@keras.saving.register_keras_serializable(package="et_util")
class SimpleTimeDistributed(keras.layers.Wrapper):
    """A simplified version of TimeDistributed that applies a layer to every temporal slice of an input.

    This implementation avoids for loops by using reshape operations to apply the wrapped layer
    to all time steps at once.

    Args:
        layer: a `keras.layers.Layer` instance.
    """

    def __init__(self, layer, **kwargs):
        super().__init__(layer, **kwargs)
        self.supports_masking = getattr(layer, "supports_masking", False)

    def build(self, input_shape):
        # Validate input shape has at least 3 dimensions (batch, time, ...)
        if not isinstance(input_shape, (tuple, list)) or len(input_shape) < 3:
            raise ValueError(
                "`SimpleTimeDistributed` requires input with at least 3 dimensions"
            )

        # Build the wrapped layer with shape excluding the time dimension
        super().build((input_shape[0], *input_shape[2:]))
        self.built = True

    def compute_output_shape(self, input_shape):
        # Get output shape by applying the layer to a single time slice
        child_output_shape = self.layer.compute_output_shape(
            (input_shape[0], *input_shape[2:])
        )
        # Include time dimension in the result
        return (child_output_shape[0], input_shape[1], *child_output_shape[1:])

    def call(self, inputs, training=None):
        input_shape = ops.shape(inputs)
        batch_size = input_shape[0]
        time_steps = input_shape[1]

        # Reshape inputs to combine batch and time dimensions: (batch*time, ...)
        reshaped_inputs = ops.reshape(inputs, (-1, *input_shape[2:]))

        # Apply the layer to all time steps at once
        outputs = self.layer.call(reshaped_inputs, training=training)

        # Get output dimensions
        output_shape = ops.shape(outputs)

        # Reshape back to include the separate batch and time dimensions: (batch, time, ...)
        return ops.reshape(outputs, (batch_size, time_steps, *output_shape[1:]))

    def get_config(self):
        config = super().get_config()
        # The parent Wrapper class handles serialization of self.layer
        return config

    @classmethod
    def from_config(cls, config, custom_objects=None):
        # Keras Wrapper.from_config handles deserialization of the wrapped layer
        from keras.layers import deserialize as deserialize_layer

        layer_config = config.pop("layer")
        layer = deserialize_layer(layer_config, custom_objects=custom_objects)
        return cls(layer, **config)


@keras.saving.register_keras_serializable(package="et_util")
class MaskedWeightedRidgeRegressionLayer(keras.layers.Layer):
    """
    A custom layer that performs weighted ridge regression with proper masking support.

    This layer takes embeddings, coordinates, weights, and calibration mask as explicit inputs,
    while using Keras' masking system to handle target masking. This separation allows more
    precise control over calibration points while leveraging Keras' built-in mask propagation
    for target predictions.

    Args:
        lambda_ridge (float): Regularization parameter for ridge regression
        epsilon (float): Small constant for numerical stability inside sqrt
    """

    def __init__(self, lambda_ridge, epsilon=1e-7, **kwargs):
        self.lambda_ridge = lambda_ridge
        self.epsilon = epsilon
        super(MaskedWeightedRidgeRegressionLayer, self).__init__(**kwargs)

    def call(self, inputs, mask=None):
        """
        The forward pass of the layer.

        Args:
            inputs: A list containing:
                - embeddings: Embeddings for all points (batch_size, n_points, embedding_dim)
                - coords: Coordinates for all points (batch_size, n_points, 2)
                - calibration_weights: Importance weights (batch_size, n_points, 1)
                - cal_mask: Mask for calibration points (batch_size, n_points) [EXPLICIT]

        Returns:
            Predicted coordinates for the target points (batch_size, n_points, 2)
        """
        # Unpack inputs
        embeddings, coords, calibration_weights, cal_mask = inputs

        # Ensure correct dtype, especially important for JIT
        embeddings = ops.cast(embeddings, "float32")
        coords = ops.cast(coords, "float32")
        calibration_weights = ops.cast(calibration_weights, "float32")
        cal_mask = ops.cast(cal_mask, "float32")

        # reshape weights to (batch, calibration)
        w = ops.squeeze(calibration_weights, axis=-1)

        # Pre-compute masked weights for calibration points
        w_masked = w * cal_mask
        # Add epsilon inside sqrt for numerical stability
        w_sqrt = ops.sqrt(w_masked + self.epsilon)
        w_sqrt = ops.expand_dims(w_sqrt, -1)

        # Apply calibration mask to embeddings
        cal_mask_expand = ops.expand_dims(cal_mask, -1)
        X = embeddings * cal_mask_expand

        # Weight calibration embeddings and coordinates using the masked weights
        X_weighted = X * w_sqrt
        y_weighted = coords * w_sqrt * cal_mask_expand

        # Matrix operations
        X_t = ops.transpose(X_weighted, axes=[0, 2, 1])
        X_t_X = ops.matmul(X_t, X_weighted)

        # Add regularization
        identity_matrix = ops.cast(ops.eye(ops.shape(embeddings)[-1]), "float32")
        lhs = X_t_X + self.lambda_ridge * identity_matrix

        # Compute RHS
        rhs = ops.matmul(X_t, y_weighted)

        # Solve the system
        kernel = ops.linalg.solve(lhs, rhs)

        # Apply regression using the *original* full embeddings
        output = ops.matmul(embeddings, kernel)

        return output

    def compute_output_shape(self, input_shapes):
        """
        Computes the output shape of the layer.

        Args:
            input_shapes: List of input shapes

        Returns:
            Output shape tuple
        """
        # Output shape matches coordinates: (batch_size, n_points, 2)
        return (input_shapes[0][0], input_shapes[0][1], 2)

    def get_config(self):
        """
        Returns the configuration of the layer for serialization.

        Returns:
            Dictionary containing the layer configuration
        """
        config = super(MaskedWeightedRidgeRegressionLayer, self).get_config()
        config.update({"lambda_ridge": self.lambda_ridge, "epsilon": self.epsilon})
        return config


@keras.saving.register_keras_serializable(package="et_util")
class MaskInspectorLayer(keras.layers.Layer):
    """Debug utility layer that prints mask information during execution."""

    def __init__(self, **kwargs):
        super(MaskInspectorLayer, self).__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs, mask=None):
        # Print mask information (for debugging)
        print("Layer mask:", mask)
        return inputs

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super(MaskInspectorLayer, self).get_config()
        return config
