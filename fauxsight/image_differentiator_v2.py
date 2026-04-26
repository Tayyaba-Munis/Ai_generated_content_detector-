import os
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, 
    GlobalMaxPooling2D, Concatenate, Dense, Dropout, 
    BatchNormalization, Add, Activation
)
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
)
from sklearn.model_selection import train_test_split
from tensorflow.keras import mixed_precision

# ── 0. Hardware Optimization & Multi-GPU Strategy ─────────────────────────────
mixed_precision.set_global_policy('mixed_float16')
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

# Initialize the dual-GPU distribution strategy
strategy = tf.distribute.MirroredStrategy()
print(f"Number of GPUs in sync: {strategy.num_replicas_in_sync}")

# ── 1. Configuration ──────────────────────────────────────────────────────────
# Ensure this matches the exact folder name Kaggle generated for the dataset
DATASET_DIR        = '/home/hassaan/Project/Dataset'
MODEL_OUT          = 'fauxsight_v4_final_best.keras'
CHECKPOINT_PATH    = 'fauxsight_v4_checkpoint.keras'
LOG_PATH           = 'training_log_v4_final.csv'

IMG_SIZE           = 224  

# Scale batch size based on the number of GPUs. 
# 64 per T4 GPU = 128 Global Batch Size.
PER_REPLICA_BATCH_SIZE = 64
GLOBAL_BATCH_SIZE      = PER_REPLICA_BATCH_SIZE * strategy.num_replicas_in_sync

LR_INITIAL         = 1e-3 
SAMPLING_FRACTION  = 0.35
TOTAL_EPOCHS       = 40

MAX_SAMPLES_PER_CLASS = 50_000
REAL_SOURCES = ['afhq', 'celebahq', 'coco', 'ffhq', 'imagenet', 'landscape', 'metfaces']

# ── 2. Dataset Scanning ───────────────────────────────────────────────────────
def collect_image_paths(root_dir):
    file_paths, labels = [], []
    print("Scanning dataset folders...")
    for folder in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder)
        if not os.path.isdir(folder_path):
            continue
        base_label = 0 if folder.lower() in REAL_SOURCES else 1
        for subdir, _, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                    if '0_real' in subdir.lower():
                        final_label = 0
                    elif '1_fake' in subdir.lower() or 'fake' in subdir.lower():
                        final_label = 1
                    else:
                        final_label = base_label
                    file_paths.append(os.path.join(subdir, file))
                    labels.append(final_label)

    df = pd.DataFrame({'path': file_paths, 'label': labels})

    if MAX_SAMPLES_PER_CLASS is not None:
        real_df = df[df['label'] == 0].sample(
            n=min(MAX_SAMPLES_PER_CLASS, (df['label'] == 0).sum()),
            random_state=42
        )
        fake_df = df[df['label'] == 1].sample(
            n=min(MAX_SAMPLES_PER_CLASS, (df['label'] == 1).sum()),
            random_state=42
        )
        df = pd.concat([real_df, fake_df]).sample(frac=1, random_state=42).reset_index(drop=True)

    n_real = (df['label'] == 0).sum()
    n_fake = (df['label'] == 1).sum()
    print(f"  Real: {n_real:,}  |  Fake: {n_fake:,}  |  Imbalance: {n_fake/n_real:.2f}x")
    return df

# ── 3. Data Split ─────────────────────────────────────────────────────────────
full_df = collect_image_paths(DATASET_DIR)

train_df, val_df = train_test_split(
    full_df, test_size=0.2,
    stratify=full_df['label'],
    random_state=42
)
print(f"Train: {len(train_df):,}  |  Val: {len(val_df):,}")

# ── 4. Data Pipeline ──────────────────────────────────────────────────────────
def load_train(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = tf.cast(img, tf.float32) / 255.0

    # Lighter augmentations to preserve intrinsic AI artifacts
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, 0.1)
    
    return tf.clip_by_value(img, 0.0, 1.0), label

def load_val(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = tf.cast(img, tf.float32) / 255.0
    return img, label

real_df_train = train_df[train_df['label'] == 0]
fake_df_train = train_df[train_df['label'] == 1]

real_ds = (
    tf.data.Dataset.from_tensor_slices((real_df_train['path'].values, real_df_train['label'].values))
    .shuffle(10000).repeat()
)
fake_ds = (
    tf.data.Dataset.from_tensor_slices((fake_df_train['path'].values, fake_df_train['label'].values))
    .shuffle(10000).repeat()
)

train_ds = tf.data.Dataset.sample_from_datasets([real_ds, fake_ds], weights=[0.5, 0.5])
train_ds = (
    train_ds
    .map(load_train, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(GLOBAL_BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

val_ds = (
    tf.data.Dataset.from_tensor_slices((val_df['path'].values, val_df['label'].values))
    .map(load_val, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(GLOBAL_BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

VAL_STEPS       = len(val_df) // GLOBAL_BATCH_SIZE
STEPS_PER_EPOCH = int(len(train_df) * SAMPLING_FRACTION) // GLOBAL_BATCH_SIZE

# ── 5. Model Architecture ─────────────────────────────────────────────────────
HE = 'he_normal'

def residual_block(x, filters):
    shortcut  = x

    x = Conv2D(filters, (3, 3), padding='same', use_bias=False,
               kernel_initializer=HE)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters, (3, 3), padding='same', use_bias=False,
               kernel_initializer=HE)(x)
    x = BatchNormalization()(x)

    if shortcut.shape[-1] != filters:
        shortcut = Conv2D(filters, (1, 1), padding='same', use_bias=False,
                          kernel_initializer=HE)(shortcut)
        shortcut = BatchNormalization()(shortcut)

    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    return x

def build_model(img_size, lr, loss_fn):
    inputs = Input(shape=(img_size, img_size, 3))

    x = Conv2D(32, (3, 3), padding='same', use_bias=False,
               kernel_initializer=HE, name='stem')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = residual_block(x, 32)
    x = MaxPooling2D(name='pool1')(x)
    x = Dropout(0.15)(x)

    x = residual_block(x, 64)
    x = residual_block(x, 64)
    x = MaxPooling2D(name='pool2')(x)
    x = Dropout(0.2)(x)

    x = residual_block(x, 128)
    x = residual_block(x, 128)
    x = MaxPooling2D(name='pool3')(x)
    x = Dropout(0.25)(x)

    x = residual_block(x, 256)
    x = residual_block(x, 256)
    x = MaxPooling2D(name='pool4')(x)
    x = Dropout(0.3)(x)

    avg_pool = GlobalAveragePooling2D()(x)
    max_pool = GlobalMaxPooling2D()(x)
    x = Concatenate()([avg_pool, max_pool])
    
    x = Dense(512, use_bias=False, kernel_initializer=HE)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(128, use_bias=False, kernel_initializer=HE)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.3)(x)

    outputs = Dense(1, activation='sigmoid', dtype='float32',
                    kernel_initializer='glorot_uniform')(x)

    model = Model(inputs, outputs)

    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss=loss_fn,
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc')
        ]
    )
    return model

# ── 6. Unified Training ───────────────────────────────────────────────────────
print(f"\n--- FauxSight v4 | Full Training ---")
print(f"Steps/epoch: {STEPS_PER_EPOCH}  |  Val steps: {VAL_STEPS}")

# Model creation and compilation MUST be wrapped inside the strategy scope
with strategy.scope():
    loss_fn = tf.keras.losses.BinaryCrossentropy()
    
    if os.path.exists(CHECKPOINT_PATH):
        print(f"Resuming training from checkpoint: {CHECKPOINT_PATH}")
        model = load_model(CHECKPOINT_PATH)
    else:
        model = build_model(IMG_SIZE, LR_INITIAL, loss_fn=loss_fn)
        
model.summary()

callbacks = [
    ModelCheckpoint(MODEL_OUT, monitor='val_auc', save_best_only=True, mode='max', verbose=1),
    ModelCheckpoint(CHECKPOINT_PATH, save_best_only=False),
    # Increased patience to 12 to give the learning rate scheduler time to drop the LR and recover
    EarlyStopping(monitor='val_auc', patience=12, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6, verbose=1),
    CSVLogger(LOG_PATH),
]

history = model.fit(
    train_ds,
    steps_per_epoch=STEPS_PER_EPOCH,
    validation_data=val_ds,
    validation_steps=VAL_STEPS,
    epochs=TOTAL_EPOCHS,
    callbacks=callbacks
)

# ── 7. Results Plot ──────────────────────────────────────────────────────────
def plot_results(h):
    def get(k):
        return h.history.get(k, [])

    metrics = [
        ('accuracy',    'val_accuracy',  'Accuracy'),
        ('loss',        'val_loss',       'Loss'),
        ('auc',         'val_auc',        'AUC'),
        ('recall',      'val_recall',     'Recall (Fake detection)'),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    for ax, (t_key, v_key, title) in zip(axes.flatten(), metrics):
        train = get(t_key)
        val   = get(v_key)
        ax.plot(train, label='Train')
        ax.plot(val,   label='Val')
        ax.set_title(title)
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_results_final.png', dpi=150)
    print("Plot saved: training_results_final.png")

plot_results(history)

print(f"\nFinal best model: {MODEL_OUT}  ← use this for inference")