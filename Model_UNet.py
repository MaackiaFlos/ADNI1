import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
import tensorflow as tf
from tensorflow.keras import layers, models, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

def load_and_preprocess_image(image_path, target_size=(128, 128)):
    img = cv2.imread(f"{image_path}.png", cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, target_size)
    img = (img.astype(np.float32) - img.mean()) / img.std()
    img = cv2.GaussianBlur(img, (5, 5), 0)

    median_intensity = np.median(img)
    lower_threshold = int(max(0, 0.7 * median_intensity * 255))
    upper_threshold = int(min(255, 1.3 * median_intensity * 255))

    img = cv2.Canny((img * 255).astype(np.uint8), lower_threshold, upper_threshold)
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=-1)

    return img

def load_dataset():
    df = pd.read_csv(r"ADNI1_Complete_1Yr_1.5T.csv")
    images = []
    labels = []
    for _, row in df.iterrows():
        image_path = os.path.join("Middle Slice", str(row["Image Data ID"]))
        image = load_and_preprocess_image(image_path)
        images.append(image)
        if row["Group"] == "AD":
            label = 0
        elif row["Group"] == "MCI":
            label = 1
        elif row["Group"] == "CN":
            label = 2
        labels.append(label)
    images = np.array(images)
    labels = np.array(labels)

    labels = to_categorical(labels, num_classes=3)

    X_train, X_test, Y_train, Y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

    return X_train, X_test, Y_train, Y_test

def build_model(input_shape, num_classes):
    inputs = Input(shape=input_shape)

    # Contracting Path
    def conv_block(x, filters):
        x = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        return x

    c1 = conv_block(inputs, 32)  # Reduced filters
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = conv_block(p1, 64)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    c3 = conv_block(p2, 128)
    p3 = layers.MaxPooling2D((2, 2))(c3)

    # Bottleneck
    c4 = conv_block(p3, 256)

    # Expansive Path
    def upconv_block(x, filters, skip_connection):
        x = layers.Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(x)
        x = layers.concatenate([x, skip_connection])
        x = conv_block(x, filters)
        return x

    u5 = upconv_block(c4, 128, c3)
    u6 = upconv_block(u5, 64, c2)
    u7 = upconv_block(u6, 32, c1)

    # Global Average Pooling and Output
    gap = layers.GlobalAveragePooling2D()(u7)
    outputs = layers.Dense(num_classes, activation='softmax', dtype='float32')(gap)  # Ensure output is float32

    model = models.Model(inputs, outputs)

    return model

def train_model(X_train, X_test, Y_train, Y_test, input_shape, num_classes):
    class_weights = class_weight.compute_class_weight(
        'balanced',
        classes=np.unique(np.argmax(Y_train, axis=1)),
        y=np.argmax(Y_train, axis=1)
    )
    class_weights = dict(enumerate(class_weights))

    model = build_model(input_shape, num_classes)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    datagen = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest'
    )
    datagen.fit(X_train)

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)
    checkpoint = ModelCheckpoint("Model UNet/best_model_UNet.keras", monitor='val_accuracy', save_best_only=True, mode='max')

    history = model.fit(datagen.flow(X_train, Y_train, batch_size=32),
                        epochs=50,
                        validation_data=(X_test, Y_test),
                        callbacks=[early_stopping, reduce_lr, checkpoint],
                        class_weight=class_weights
                        )

    test_loss, test_acc = model.evaluate(X_test, Y_test, verbose=2)
    print(f"\nTest accuracy: {test_acc:.4f}")

    return model, history

def plot_training_history(history):
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training Accuracy History', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.legend(loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('Model UNet/Training Accuracy History.png')
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training Loss History', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('Model UNet/Training Loss History.png')
    plt.show()

def evaluate_model(model, X_test, Y_test):
    # Predict the labels for the test set
    Y_pred = model.predict(X_test)
    Y_pred_classes = np.argmax(Y_pred, axis=1)
    Y_true_classes = np.argmax(Y_test, axis=1)

    # Confusion Matrix
    cm = confusion_matrix(Y_true_classes, Y_pred_classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['AD', 'MCI', 'CN'],
                yticklabels=['AD', 'MCI', 'CN'],
                annot_kws={"size": 16})  # Increase annotation font size
    plt.title('Confusion Matrix', fontsize=18, pad=20)
    plt.xlabel('Predicted Label', fontsize=16, labelpad=15)
    plt.ylabel('True Label', fontsize=16, labelpad=15)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.savefig('Model UNet/Confusion_Matrix.png', dpi=300, bbox_inches='tight')  # Save with high resolution
    plt.show()

    # Classification Report
    print("\nClassification Report:")
    report = classification_report(Y_true_classes, Y_pred_classes, target_names=['AD', 'MCI', 'CN'], output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df = report_df.round(2)  # Round values to 2 decimal places
    print(report_df)

    # Plot classification report as a table
    plt.figure(figsize=(8, 4))
    plt.axis('off')  # Hide axes
    plt.table(cellText=report_df.values,
              colLabels=report_df.columns,
              rowLabels=report_df.index,
              loc='center',
              cellLoc='center',
              colColours=['#f7f7f7'] * len(report_df.columns))  # Add light gray background to headers
    plt.title('Classification Report', fontsize=18, pad=20)
    plt.tight_layout()
    plt.savefig('Model UNet/Classification_Report.png', dpi=300, bbox_inches='tight')  # Save with high resolution
    plt.show()

def main():
    input_shape = (128, 128, 1)

    X_train, X_test, Y_train, Y_test = load_dataset()

    folder_path = 'Model UNet'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    model, history = train_model(X_train, X_test, Y_train, Y_test, input_shape, num_classes=3)

    model.save("Model UNet/Model_UNet.keras")
    model.save("Model UNet/Model_UNet.h5")

    plot_training_history(history)

    evaluate_model(model, X_test, Y_test)

if __name__ == '__main__':
    main()