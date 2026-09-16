"""
======================================================================
models.py
======================================================================

Human Activity Recognition Using Smartphones
UCI HAR Dataset - Deep Learning Models

Input:
    (128, 9)

Classes:
    6

Baseline Models:
    1. CNN
    2. LSTM
    3. BiLSTM
    4. CNN-LSTM

Proposed Model:
    5. Residual CNN + SE + BiLSTM + Multi-Head Attention
       + Temporal Attention

Author : Pranesh
======================================================================
"""

import os
import tensorflow as tf

from tensorflow.keras import Model, regularizers

from tensorflow.keras.layers import (
    Input,
    Conv1D,
    MaxPooling1D,
    GlobalAveragePooling1D,
    GlobalMaxPooling1D,
    BatchNormalization,
    LayerNormalization,
    Activation,
    Dropout,
    Dense,
    LSTM,
    Bidirectional,
    SpatialDropout1D,
    MultiHeadAttention,
    Concatenate,
    Add,
    Multiply,
    Reshape,
    Layer
)

from tensorflow.keras.optimizers import Adam


# =====================================================================
# CONFIGURATION
# =====================================================================

INPUT_SHAPE = (128, 9)

NUM_CLASSES = 6

LEARNING_RATE = 0.001

L2_WEIGHT = 1e-4


ACTIVITY_NAMES = [

    "LAYING",

    "SITTING",

    "STANDING",

    "WALKING",

    "WALKING_DOWNSTAIRS",

    "WALKING_UPSTAIRS"

]


# =====================================================================
# TEMPORAL ATTENTION POOLING
# =====================================================================

class TemporalAttentionPooling(Layer):

    """
    Weighted temporal pooling.

    Given:

        features:
            (batch, time, features)

        attention_weights:
            (batch, time, 1)

    Computes:

        context = sum(
            attention_weight * feature
        )

    across the time dimension.
    """

    def __init__(
        self,
        **kwargs
    ):

        super().__init__(
            **kwargs
        )

    def call(
        self,
        inputs
    ):

        features, attention_weights = inputs

        return tf.reduce_sum(
            features * attention_weights,
            axis=1
        )

    def get_config(self):

        config = super().get_config()

        return config


# =====================================================================
# CONVOLUTION BLOCK
# =====================================================================

def conv_block(
    x,
    filters,
    kernel_size,
    dropout_rate=0.15
):

    """
    Standard CNN block.

    Conv1D
       ↓
    BatchNorm
       ↓
    ReLU
       ↓
    SpatialDropout
    """

    x = Conv1D(

        filters=filters,

        kernel_size=kernel_size,

        padding="same",

        kernel_regularizer=regularizers.l2(
            L2_WEIGHT
        )

    )(x)

    x = BatchNormalization()(x)

    x = Activation(
        "relu"
    )(x)

    x = SpatialDropout1D(
        dropout_rate
    )(x)

    return x


# =====================================================================
# SQUEEZE AND EXCITATION
# =====================================================================

def squeeze_excitation(
    x,
    ratio=8
):

    """
    Squeeze-and-Excitation channel attention.

    Learns the importance of different
    feature channels.
    """

    channels = x.shape[-1]

    if channels is None:

        raise ValueError(
            "Channel dimension must be known."
        )

    reduced_channels = max(
        channels // ratio,
        8
    )

    # --------------------------------------------------------------
    # Squeeze
    # --------------------------------------------------------------

    se = GlobalAveragePooling1D()(
        x
    )

    # --------------------------------------------------------------
    # Excitation
    # --------------------------------------------------------------

    se = Dense(

        reduced_channels,

        activation="relu",

        kernel_regularizer=regularizers.l2(
            L2_WEIGHT
        )

    )(se)

    se = Dense(

        channels,

        activation="sigmoid",

        kernel_regularizer=regularizers.l2(
            L2_WEIGHT
        )

    )(se)

    # --------------------------------------------------------------
    # Reshape
    # --------------------------------------------------------------

    se = Reshape(
        (1, channels)
    )(se)

    # --------------------------------------------------------------
    # Reweight
    # --------------------------------------------------------------

    output = Multiply()(
        [
            x,
            se
        ]
    )

    return output


# =====================================================================
# RESIDUAL CNN BLOCK
# =====================================================================

def residual_block(
    x,
    filters,
    kernel_size=5,
    dropout_rate=0.15
):

    """
    Residual CNN block.

    Main path:
        Conv
        ↓
        BN
        ↓
        ReLU
        ↓
        Dropout
        ↓
        Conv
        ↓
        BN
        ↓
        SE Attention

    Shortcut:
        Identity / 1x1 Conv

    Final:
        Add
        ↓
        ReLU
    """

    shortcut = x

    # ==============================================================
    # FIRST CONVOLUTION
    # ==============================================================

    x = Conv1D(

        filters=filters,

        kernel_size=kernel_size,

        padding="same",

        kernel_regularizer=regularizers.l2(
            L2_WEIGHT
        )

    )(x)

    x = BatchNormalization()(x)

    x = Activation(
        "relu"
    )(x)

    x = SpatialDropout1D(
        dropout_rate
    )(x)

    # ==============================================================
    # SECOND CONVOLUTION
    # ==============================================================

    x = Conv1D(

        filters=filters,

        kernel_size=kernel_size,

        padding="same",

        kernel_regularizer=regularizers.l2(
            L2_WEIGHT
        )

    )(x)

    x = BatchNormalization()(x)

    # ==============================================================
    # SE ATTENTION
    # ==============================================================

    x = squeeze_excitation(
        x
    )

    # ==============================================================
    # MATCH SHORTCUT CHANNELS
    # ==============================================================

    if shortcut.shape[-1] != filters:

        shortcut = Conv1D(

            filters=filters,

            kernel_size=1,

            padding="same",

            kernel_regularizer=regularizers.l2(
                L2_WEIGHT
            )

        )(shortcut)

        shortcut = BatchNormalization()(
            shortcut
        )

    # ==============================================================
    # RESIDUAL ADD
    # ==============================================================

    x = Add()(
        [
            x,
            shortcut
        ]
    )

    x = Activation(
        "relu"
    )(x)

    return x


# =====================================================================
# DEEP MODELS CLASS
# =====================================================================

class DeepModels:

    """
    Deep learning architecture builder
    for UCI HAR raw inertial signals.
    """

    def __init__(
        self,
        input_shape=INPUT_SHAPE,
        num_classes=NUM_CLASSES,
        learning_rate=LEARNING_RATE
    ):

        self.input_shape = input_shape

        self.num_classes = num_classes

        self.learning_rate = learning_rate

    # =================================================================
    # COMPILE MODEL
    # =================================================================

    def compile_model(
        self,
        model
    ):

        """
        Compile model using Adam.

        clipnorm:
            Helps control exploding gradients.
        """

        optimizer = Adam(

            learning_rate=self.learning_rate,

            clipnorm=1.0

        )

        model.compile(

            optimizer=optimizer,

            loss="sparse_categorical_crossentropy",

            metrics=[
                "accuracy"
            ]

        )

        return model

    # =================================================================
    # BASELINE 1 - CNN
    # =================================================================

    def build_cnn(
        self
    ):

        """
        Baseline CNN.

        Input:
            128 x 9

        Learns local temporal motion patterns.
        """

        inputs = Input(

            shape=self.input_shape,

            name="sensor_input"

        )

        # --------------------------------------------------------------
        # CNN Block 1
        # --------------------------------------------------------------

        x = conv_block(

            inputs,

            filters=64,

            kernel_size=7,

            dropout_rate=0.15

        )

        x = MaxPooling1D(
            pool_size=2
        )(x)

        # --------------------------------------------------------------
        # CNN Block 2
        # --------------------------------------------------------------

        x = conv_block(

            x,

            filters=128,

            kernel_size=5,

            dropout_rate=0.20

        )

        x = MaxPooling1D(
            pool_size=2
        )(x)

        # --------------------------------------------------------------
        # CNN Block 3
        # --------------------------------------------------------------

        x = conv_block(

            x,

            filters=256,

            kernel_size=3,

            dropout_rate=0.20

        )

        x = MaxPooling1D(
            pool_size=2
        )(x)

        # --------------------------------------------------------------
        # Global pooling
        # --------------------------------------------------------------

        average_features = GlobalAveragePooling1D()(
            x
        )

        maximum_features = GlobalMaxPooling1D()(
            x
        )

        x = Concatenate()(
            [
                average_features,
                maximum_features
            ]
        )

        # --------------------------------------------------------------
        # Dense classifier
        # --------------------------------------------------------------

        x = Dense(

            256,

            activation="relu",

            kernel_regularizer=regularizers.l2(
                L2_WEIGHT
            )

        )(x)

        x = BatchNormalization()(x)

        x = Dropout(
            0.40
        )(x)

        x = Dense(

            128,

            activation="relu"

        )(x)

        x = Dropout(
            0.30
        )(x)

        outputs = Dense(

            self.num_classes,

            activation="softmax",

            name="activity_output"

        )(x)

        model = Model(

            inputs,

            outputs,

            name="Baseline_CNN"

        )

        return self.compile_model(
            model
        )

    # =================================================================
    # BASELINE 2 - LSTM
    # =================================================================

    def build_lstm(
        self
    ):

        """
        Baseline LSTM.

        Learns temporal dependencies
        from sensor sequences.
        """

        inputs = Input(

            shape=self.input_shape,

            name="sensor_input"

        )

        # --------------------------------------------------------------
        # LSTM Layer 1
        # --------------------------------------------------------------

        x = LSTM(

            128,

            return_sequences=True,

            dropout=0.20

        )(inputs)

        # --------------------------------------------------------------
        # LSTM Layer 2
        # --------------------------------------------------------------

        x = LSTM(

            64,

            dropout=0.20

        )(x)

        # --------------------------------------------------------------
        # Dense classifier
        # --------------------------------------------------------------

        x = Dense(

            128,

            activation="relu",

            kernel_regularizer=regularizers.l2(
                L2_WEIGHT
            )

        )(x)

        x = Dropout(
            0.40
        )(x)

        x = Dense(

            64,

            activation="relu"

        )(x)

        x = Dropout(
            0.30
        )(x)

        outputs = Dense(

            self.num_classes,

            activation="softmax",

            name="activity_output"

        )(x)

        model = Model(

            inputs,

            outputs,

            name="Baseline_LSTM"

        )

        return self.compile_model(
            model
        )

    # =================================================================
    # BASELINE 3 - BiLSTM
    # =================================================================

    def build_bilstm(
        self
    ):

        """
        Baseline Bidirectional LSTM.
        """

        inputs = Input(

            shape=self.input_shape,

            name="sensor_input"

        )

        # --------------------------------------------------------------
        # BiLSTM Layer 1
        # --------------------------------------------------------------

        x = Bidirectional(

            LSTM(

                128,

                return_sequences=True,

                dropout=0.20

            )

        )(inputs)

        # --------------------------------------------------------------
        # BiLSTM Layer 2
        # --------------------------------------------------------------

        x = Bidirectional(

            LSTM(

                64,

                dropout=0.20

            )

        )(x)

        # --------------------------------------------------------------
        # Classifier
        # --------------------------------------------------------------

        x = Dense(

            128,

            activation="relu",

            kernel_regularizer=regularizers.l2(
                L2_WEIGHT
            )

        )(x)

        x = Dropout(
            0.40
        )(x)

        x = Dense(

            64,

            activation="relu"

        )(x)

        x = Dropout(
            0.30
        )(x)

        outputs = Dense(

            self.num_classes,

            activation="softmax",

            name="activity_output"

        )(x)

        model = Model(

            inputs,

            outputs,

            name="Baseline_BiLSTM"

        )

        return self.compile_model(
            model
        )

    # =================================================================
    # BASELINE 4 - CNN-LSTM
    # =================================================================

    def build_cnn_lstm(
        self
    ):

        """
        Baseline CNN-LSTM.

        CNN:
            Local sensor patterns

        LSTM:
            Temporal dependencies
        """

        inputs = Input(

            shape=self.input_shape,

            name="sensor_input"

        )

        # --------------------------------------------------------------
        # CNN Block 1
        # --------------------------------------------------------------

        x = conv_block(

            inputs,

            filters=64,

            kernel_size=7,

            dropout_rate=0.15

        )

        x = MaxPooling1D(
            2
        )(x)

        # --------------------------------------------------------------
        # CNN Block 2
        # --------------------------------------------------------------

        x = conv_block(

            x,

            filters=128,

            kernel_size=5,

            dropout_rate=0.20

        )

        x = MaxPooling1D(
            2
        )(x)

        # --------------------------------------------------------------
        # LSTM
        # --------------------------------------------------------------

        x = LSTM(

            128,

            return_sequences=True,

            dropout=0.20

        )(x)

        x = LSTM(

            64,

            dropout=0.20

        )(x)

        # --------------------------------------------------------------
        # Classifier
        # --------------------------------------------------------------

        x = Dense(

            128,

            activation="relu",

            kernel_regularizer=regularizers.l2(
                L2_WEIGHT
            )

        )(x)

        x = Dropout(
            0.40
        )(x)

        outputs = Dense(

            self.num_classes,

            activation="softmax",

            name="activity_output"

        )(x)

        model = Model(

            inputs,

            outputs,

            name="Baseline_CNN_LSTM"

        )

        return self.compile_model(
            model
        )

    # =================================================================
    # PROPOSED MODEL
    # =================================================================

    def build_proposed_model(
        self
    ):

        """
        ==============================================================
        PROPOSED ARCHITECTURE
        ==============================================================

        Raw Sensor Signals
                |
                v
        Initial Conv1D
                |
                v
        Residual CNN Block
                |
                v
        SE Channel Attention
                |
                v
        Residual CNN Block
                |
                v
        SE Channel Attention
                |
                v
        Residual CNN Block
                |
                v
              BiLSTM
                |
                v
        Multi-Head Self Attention
                |
                v
          Residual Connection
                |
                v
              BiLSTM
                |
                v
        Temporal Attention
                |
                v
          Dense Classifier
                |
                v
             Softmax
                |
                v
           6 Activities

        ==============================================================
        """

        inputs = Input(

            shape=self.input_shape,

            name="sensor_input"

        )

        # =============================================================
        # INITIAL CONVOLUTION
        # =============================================================

        x = Conv1D(

            filters=64,

            kernel_size=7,

            padding="same",

            kernel_regularizer=regularizers.l2(
                L2_WEIGHT
            ),

            name="initial_conv"

        )(inputs)

        x = BatchNormalization(
            name="initial_bn"
        )(x)

        x = Activation(
            "relu",
            name="initial_relu"
        )(x)

        # =============================================================
        # RESIDUAL BLOCK 1
        # =============================================================

        x = residual_block(

            x,

            filters=64,

            kernel_size=5,

            dropout_rate=0.10

        )

        x = MaxPooling1D(

            pool_size=2,

            name="pool_1"

        )(x)

        # =============================================================
        # RESIDUAL BLOCK 2
        # =============================================================

        x = residual_block(

            x,

            filters=128,

            kernel_size=5,

            dropout_rate=0.15

        )

        x = MaxPooling1D(

            pool_size=2,

            name="pool_2"

        )(x)

        # =============================================================
        # RESIDUAL BLOCK 3
        # =============================================================

        x = residual_block(

            x,

            filters=256,

            kernel_size=3,

            dropout_rate=0.15

        )

        # =============================================================
        # FIRST BiLSTM
        # =============================================================

        x = Bidirectional(

            LSTM(

                128,

                return_sequences=True,

                dropout=0.20

            ),

            name="bilstm_1"

        )(x)

        # =============================================================
        # MULTI-HEAD SELF ATTENTION
        # =============================================================

        attention_output = MultiHeadAttention(

            num_heads=4,

            key_dim=32,

            dropout=0.10,

            name="multi_head_attention"

        )(

            query=x,

            value=x,

            key=x

        )

        # =============================================================
        # RESIDUAL ATTENTION CONNECTION
        # =============================================================

        x = Add(

            name="attention_residual"

        )(

            [
                x,
                attention_output
            ]

        )

        x = LayerNormalization(

            name="attention_layer_norm"

        )(x)

        # =============================================================
        # SECOND BiLSTM
        # =============================================================

        x = Bidirectional(

            LSTM(

                64,

                return_sequences=True,

                dropout=0.15

            ),

            name="bilstm_2"

        )(x)

        # =============================================================
        # TEMPORAL ATTENTION
        # =============================================================

        attention_scores = Dense(

            64,

            activation="tanh",

            name="temporal_attention_projection"

        )(x)

        attention_scores = Dense(

            1,

            activation=None,

            name="temporal_attention_score"

        )(attention_scores)

        # Softmax over time
        attention_weights = tf.keras.layers.Softmax(

            axis=1,

            name="temporal_attention_weights"

        )(attention_scores)

        # =============================================================
        # TEMPORAL ATTENTION POOLING
        # =============================================================

        x = TemporalAttentionPooling(

            name="temporal_context"

        )(

            [
                x,
                attention_weights
            ]

        )

        # =============================================================
        # CLASSIFICATION HEAD
        # =============================================================

        x = Dense(

            256,

            activation="relu",

            kernel_regularizer=regularizers.l2(
                L2_WEIGHT
            ),

            name="dense_256"

        )(x)

        x = BatchNormalization(
            name="dense_bn_256"
        )(x)

        x = Dropout(

            0.40,

            name="dropout_256"

        )(x)

        x = Dense(

            128,

            activation="relu",

            kernel_regularizer=regularizers.l2(
                L2_WEIGHT
            ),

            name="dense_128"

        )(x)

        x = BatchNormalization(
            name="dense_bn_128"
        )(x)

        x = Dropout(

            0.30,

            name="dropout_128"

        )(x)

        x = Dense(

            64,

            activation="relu",

            kernel_regularizer=regularizers.l2(
                L2_WEIGHT
            ),

            name="dense_64"

        )(x)

        x = Dropout(

            0.20,

            name="dropout_64"

        )(x)

        # =============================================================
        # OUTPUT
        # =============================================================

        outputs = Dense(

            self.num_classes,

            activation="softmax",

            name="activity_output"

        )(x)

        # =============================================================
        # CREATE MODEL
        # =============================================================

        model = Model(

            inputs=inputs,

            outputs=outputs,

            name="Proposed_CNN_BiLSTM_Attention"

        )

        return self.compile_model(
            model
        )

    # =================================================================
    # BUILD ALL MODELS
    # =================================================================

    def build_all_models(
        self
    ):

        """
        Returns five models:

            cnn
            lstm
            bilstm
            cnn_lstm
            proposed
        """

        models = {}

        # =============================================================
        # BASELINES
        # =============================================================

        print()
        print("=" * 75)
        print("BUILDING BASELINE MODELS")
        print("=" * 75)

        print()
        print("[1/5] CNN")

        models["cnn"] = self.build_cnn()

        print(
            "[2/5] LSTM"
        )

        models["lstm"] = self.build_lstm()

        print(
            "[3/5] BiLSTM"
        )

        models["bilstm"] = self.build_bilstm()

        print(
            "[4/5] CNN-LSTM"
        )

        models["cnn_lstm"] = self.build_cnn_lstm()

        # =============================================================
        # PROPOSED
        # =============================================================

        print()
        print("=" * 75)
        print("BUILDING PROPOSED MODEL")
        print("=" * 75)

        print()
        print(
            "[5/5] CNN-BiLSTM-Attention"
        )

        models["proposed"] = (
            self.build_proposed_model()
        )

        print()
        print("=" * 75)
        print("ALL MODELS BUILT SUCCESSFULLY")
        print("=" * 75)

        return models

    # =================================================================
    # MODEL INFORMATION
    # =================================================================

    @staticmethod
    def get_model_info(
        model
    ):

        total_parameters = (
            model.count_params()
        )

        trainable_parameters = sum(

            tf.keras.backend.count_params(
                weight
            )

            for weight
            in model.trainable_weights

        )

        non_trainable_parameters = sum(

            tf.keras.backend.count_params(
                weight
            )

            for weight
            in model.non_trainable_weights

        )

        return {

            "name":
                model.name,

            "total_parameters":
                int(total_parameters),

            "trainable_parameters":
                int(trainable_parameters),

            "non_trainable_parameters":
                int(non_trainable_parameters)

        }

    # =================================================================
    # PRINT MODEL INFORMATION
    # =================================================================

    @staticmethod
    def print_model_info(
        model
    ):

        info = DeepModels.get_model_info(
            model
        )

        print()
        print("-" * 70)

        print(
            "Model:",
            info["name"]
        )

        print(
            "Total Parameters:",
            f"{info['total_parameters']:,}"
        )

        print(
            "Trainable Parameters:",
            f"{info['trainable_parameters']:,}"
        )

        print(
            "Non-Trainable Parameters:",
            f"{info['non_trainable_parameters']:,}"
        )

        print("-" * 70)

    # =================================================================
    # PRINT SUMMARY
    # =================================================================

    @staticmethod
    def print_summary(
        model
    ):

        print()
        print("=" * 80)

        print(
            "MODEL SUMMARY"
        )

        print(
            model.name
        )

        print("=" * 80)

        model.summary()

        print("=" * 80)

    # =================================================================
    # SAVE ARCHITECTURE
    # =================================================================

    @staticmethod
    def save_architecture(
        model,
        output_path
    ):

        directory = os.path.dirname(
            output_path
        )

        if directory:

            os.makedirs(
                directory,
                exist_ok=True
            )

        with open(

            output_path,

            "w",

            encoding="utf-8"

        ) as file:

            file.write(
                model.to_json()
            )

        print()
        print(
            "Architecture saved:"
        )

        print(
            output_path
        )


# =====================================================================
# MAIN
# =====================================================================

def main():

    print()
    print("=" * 80)

    print(
        "HUMAN ACTIVITY RECOGNITION"
    )

    print(
        "DEEP LEARNING MODEL ARCHITECTURES"
    )

    print("=" * 80)

    print()

    print(
        "Input Shape   :",
        INPUT_SHAPE
    )

    print(
        "Classes       :",
        NUM_CLASSES
    )

    print(
        "Learning Rate :",
        LEARNING_RATE
    )

    print(
        "L2 Weight     :",
        L2_WEIGHT
    )

    # ==============================================================
    # ACTIVITIES
    # ==============================================================

    print()

    print(
        "Activity Classes:"
    )

    for index, activity in enumerate(
        ACTIVITY_NAMES
    ):

        print(
            f"{index} -> {activity}"
        )

    # ==============================================================
    # CREATE BUILDER
    # ==============================================================

    builder = DeepModels(

        input_shape=INPUT_SHAPE,

        num_classes=NUM_CLASSES,

        learning_rate=LEARNING_RATE

    )

    # ==============================================================
    # BUILD MODELS
    # ==============================================================

    models = builder.build_all_models()

    # ==============================================================
    # PARAMETER COMPARISON
    # ==============================================================

    print()
    print("=" * 80)

    print(
        "MODEL PARAMETER COMPARISON"
    )

    print("=" * 80)

    for name, model in models.items():

        print()

        print(
            f"{name.upper()}"
        )

        builder.print_model_info(
            model
        )

    # ==============================================================
    # PROPOSED MODEL SUMMARY
    # ==============================================================

    proposed_model = models[
        "proposed"
    ]

    print()
    print("=" * 80)

    print(
        "PROPOSED MODEL SUMMARY"
    )

    print("=" * 80)

    builder.print_summary(
        proposed_model
    )

    # ==============================================================
    # SAVE ARCHITECTURE
    # ==============================================================

    base_dir = os.path.dirname(
        os.path.abspath(__file__)
    )

    architecture_dir = os.path.join(

        base_dir,

        "models",

        "architectures"

    )

    os.makedirs(

        architecture_dir,

        exist_ok=True

    )

    architecture_path = os.path.join(

        architecture_dir,

        "proposed_cnn_bilstm_attention.json"

    )

    builder.save_architecture(

        proposed_model,

        architecture_path

    )

    # ==============================================================
    # FINAL MESSAGE
    # ==============================================================

    print()
    print("=" * 80)

    print(
        "MODEL SETUP COMPLETED SUCCESSFULLY"
    )

    print("=" * 80)

    print()

    print(
        "BASELINE MODELS"
    )

    print(
        "1. CNN"
    )

    print(
        "2. LSTM"
    )

    print(
        "3. BiLSTM"
    )

    print(
        "4. CNN-LSTM"
    )

    print()

    print(
        "PROPOSED MODEL"
    )

    print(
        "5. Residual CNN + SE + BiLSTM"
    )

    print(
        "   + Multi-Head Self Attention"
    )

    print(
        "   + Temporal Attention"
    )

    print()

    print(
        "Input:",
        INPUT_SHAPE
    )

    print(
        "Output Classes:",
        NUM_CLASSES
    )

    print()

    print(
        "Ready for training."
    )


# =====================================================================
# EXECUTION
# =====================================================================

if __name__ == "__main__":

    main()