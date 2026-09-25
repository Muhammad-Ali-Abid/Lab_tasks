import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
from glob import glob

documents = [
    "The sky is clear today.",
    "The weather is pleasant and cool.",
    "A beautiful day brings fresh opportunities.",
    "The sun rose over the hills, and as the sun climbed higher, we watched the sun, photographed the sun, and enjoyed the warmth of the sun.",
    "The children are playing in the park.",
    "My brother enjoys reading adventure stories.",
    "The teacher explained the lesson clearly.",
    "A small bird landed on the window.",
    "We visited the museum last weekend.",
    "The garden is filled with colorful flowers.",
    "She drinks tea every morning.",
    "The train arrived at the station early.",
    "Learning Python requires regular practice.",
    "The students completed their assignments.",
    "A gentle breeze moved through the trees.",
    "My friend bought a new bicycle.",
    "The library provides a quiet place to study.",
    "Fresh fruit is available at the market.",
    "The dog waited patiently near the door.",
    "We cooked rice and vegetables for dinner.",
    "Music helps me relax after a busy day.",
    "The river flows through the quiet village.",
    "He repaired the broken chair yesterday.",
    "Our team worked together on the project.",
    "The baby smiled at her mother.",
    "Heavy rain delayed the football match.",
    "The mountain road was narrow and steep.",
    "I keep my notes in a blue notebook.",
    "The chef prepared a delicious meal.",
    "They planted a tree beside the school.",
    "The computer processes numerical data quickly.",
    "A good explanation includes practical examples.",
    "My sister enjoys painting landscapes.",
    "The shop closes at eight every evening.",
    "We watched a documentary about marine animals.",
    "Regular exercise can improve physical strength.",
    "The bus stopped near the university.",
    "He placed the keys on the wooden table.",
    "The classroom has large windows.",
    "Practice helps students understand difficult concepts."
]

print("--- Raw Text Document ---")
for i , doc in enumerate(documents):
    print(f"Doc {i + 1}: {doc}")

vectorizer = CountVectorizer()

# vectorizer = CountVectorizer(stop_words = "english")
# vectorizer = CountVectorizer(ngram_range = (1,2))
# vectorizer = CountVectorizer(token_pattern=r"(?u)\b\w+\b")
# vectorizer = CountVectorizer(ngram_range = (1,2) , token_pattern=r"(?u)\b\w+\b")

x_text = vectorizer.fit_transform(documents)
feature_names = vectorizer.get_feature_names_out()
text_matrix = x_text.toarray()

print("--- Bag-of-words (BOW) Transformation ---")
print(f"Vocabulary (features): {feature_names}")
print(f"Shape of feature matrix: {text_matrix.shape} (documents x unique terms)")

df_text =pd.DataFrame(text_matrix , columns = feature_names , index = [f"Doc{i + 1}" for i in range(len(documents))])

print("BoW numerical feature matrix")
print(df_text)

print("Is 'a' in the vocabulary? ", 'a' in feature_names)
print("Is 'the' in the vocabulary?", 'the' in feature_names)
print()
print("Default token pattern :", vectorizer.token_pattern)
print("Lowercase enabled     :", vectorizer.lowercase)
print()
print("The token pattern requires 2+ word characters, which is why 'a' is dropped.")

total_cells = text_matrix.size
zero_cells = (text_matrix == 0).sum()

print(f"Matrix Cells: {total_cells}")
print(f"Zero Cells: {zero_cells}")
print(f"Sparsity: {zero_cells / total_cells:.1%}")
print(f"Type returned by fit_transform: {type(x_text)}")

#Answer 1: Each column represents a unique word in the vocabulary built from all three documents. Each cell shows how many times that word appears in the corresponding document.

#Answer 2: It is a numerical document–term matrix of shape (3, 11). Each document is represented by a fixed-length vector of 11 word counts, with each position corresponding to a specific vocabulary word.

#Answer 3: Bag-of-Words keeps word counts but ignores word order, grammar, and context. For example, “dog bites man” and “man bites dog” produce identical vectors despite having different meanings. In this implementation, lowercasing also removes capitalization information, and the default token pattern excludes punctuation and single-character words such as “a.”


def image_features(path):
    image_file_path = path

    original_image = None
    if image_file_path and os.path.exists(image_file_path):
        try:
            original_image = Image.open(image_file_path)
            print(f"--- Loaded Image from File: {image_file_path} ---")
        except Exception as e:
            print(f"Error loading image from {image_file_path}: {e}")
            dummy = np.random.randint(0, 256, size=(100, 100, 3), dtype=np.uint8)
            original_image = Image.fromarray(dummy, 'RGB')
            print("--- Created Dummy Image (100x100 RGB) ---")
    else:
        print("No valid image file path provided or file not found.")
        dummy = np.random.randint(0, 256, size=(100, 100, 3), dtype=np.uint8)
        original_image = Image.fromarray(dummy, 'RGB')
        print("--- Created Dummy Image (100x100 RGB) ---")

    original_image = original_image.convert('RGB') 
    # original_image = original_image.resize((64 , 64)) 
    original_image_array = np.array(original_image)

    h, w, c = original_image_array.shape

    print(f"Original Image Size (H, W, Channels): {original_image_array.shape}")
    print(f"Total Features (Pixels) in RGB: {h * w * c}")
    print(f"Total Features (Pixels) in RGB: {original_image_array.size}")

    grayscale_image = original_image.convert('L')
    grayscale_array = np.array(grayscale_image)

    print("--- Grayscale Conversion (Feature Reduction) ---")
    print(f"Grayscale Image Size (H, W): {grayscale_array.shape}")
    print(f"Total Features (Pixels) after Grayscale: {grayscale_array.size}")
    print()
    print(f"Reduction factor: {(h * w * c) / grayscale_array.size:.0f}x fewer features")

    flattened_features = grayscale_array.flatten()

    print("--- Pixel Flattening (Vectorization) ---")
    print(f"Flattened Feature Vector Shape: {flattened_features.shape}")
    print()
    print("First 10 Features (Pixel Values) in Machine Understandable Format:")
    print(flattened_features[:10])
    print()
    print(f"Data Type of Final Features: {flattened_features.dtype}")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].imshow(original_image)
    axes[0].set_title(f"1. Original RGB\n{h}x{w}x{c} = {h*w*c:,} features")
    axes[0].axis('off')

    axes[1].imshow(grayscale_image, cmap='gray')
    axes[1].set_title(f"2. Grayscale\n{h}x{w} = {h*w:,} features")
    axes[1].axis('off')

    axes[2].imshow(flattened_features[:2000].reshape(1, -1), cmap='gray', aspect='auto')
    axes[2].set_title(f"3. Flattened vector\n(first 2,000 of {flattened_features.size:,})")
    axes[2].set_yticks([])

    plt.tight_layout()
    plt.show()

# for i in range(20):
#     image_features(f"images\img{i}.jpg")

image_features("images\img0.jpg")

def image_to_features(path, size):
    with Image.open(path) as image:
        image = image.convert("RGB")
        image = image.resize(size)
        image = image.convert("L")

        return np.array(image).flatten()

CLASS_DIRS = {0: "data/class_a", 1: "data/class_b"}
SIZE = (64, 64)

X_list, y_list = [], []
for label, folder in CLASS_DIRS.items():
    paths = sorted(glob(os.path.join(folder, "*")))
    for p in paths:
        try:
            X_list.append(image_to_features(p, size=SIZE))
            y_list.append(label)
        except Exception as e:
            print(f"[SKIP] {p} -> {e}")

if X_list:
    X = np.vstack(X_list)
    y = np.array(y_list)
    np.savez("dataset.npz", X=X, y=y)

    print(f"X shape: {X.shape}   y shape: {y.shape}")
    print(f"Samples: {X.shape[0]}   Features per sample: {X.shape[1]}")
    print(f"Samples per feature: {X.shape[0] / X.shape[1]:.4f}")
    print()
    print("Saved dataset.npz")
else:
    print("No images found. Create data/class_a and data/class_b and add images.")

#Answer 1 : Grayscale conversion replaces the three RGB values of each pixel with one brightness value. This reduces the number of features from H * W * 3 to H * W, simplifying the representation and reducing storage and processing requirements. It preserves useful brightness patterns, such as edges and shapes, but discards color information.

#Answer 2 : The vector length is H * W because each pixel contributes one grayscale value. For example, a 64 * 64 grayscale image produces a vector containing 4,096 values.

#Answer 3 : These models typically expect a two-dimensional input matrix of shape (number of samples, number of features). Flattening converts each image’s pixel matrix into a one-dimensional feature vector that becomes one row of this input matrix. The pixel values remain unchanged.

#Answer 4 : Resizing ensures that every image produces a feature vector of the same length, allowing the vectors to be stacked into one numerical matrix. For example, resizing every image to 64 * 64 produces 4,096 features per grayscale image, regardless of its original dimensions. Reducing image size can also lower memory use and computation, although some visual detail may be lost.