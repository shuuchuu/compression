import tensorflow as tf


def get_lenet(
    image_size: tuple[int, int], learning_rate: float = 1e-4
) -> tf.keras.models.Model:
    def conv(filters: int, padding: str) -> tf.keras.layers.Conv2D:
        return tf.keras.layers.Conv2D(
            filters=filters, kernel_size=5, padding=padding, activation="sigmoid"
        )

    def pooling() -> tf.keras.layers.MaxPooling2D:
        return tf.keras.layers.MaxPooling2D()

    def dense(units: int, activation: str = "sigmoid") -> tf.keras.layers.Dense:
        return tf.keras.layers.Dense(units, activation=activation)

    model = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(*image_size, 3)),
            conv(6, "same"),
            pooling(),
            conv(16, "valid"),
            pooling(),
            tf.keras.layers.Flatten(),
            dense(120),
            dense(84),
            dense(6, activation="softmax"),
        ],
        name="le_net",
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model
