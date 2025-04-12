import os
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from nltk.tokenize import word_tokenize
from collections import Counter
import nltk

nltk.download('punkt')

# Image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Vocabulary class
class Vocabulary:
    def __init__(self, threshold=5):
        self.word2idx = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2, '<UNK>': 3}
        self.idx2word = {0: '<PAD>', 1: '<SOS>', 2: '<EOS>', 3: '<UNK>'}
        self.threshold = threshold
        self.word_count = Counter()

    def build_vocab(self, captions):
        for caption in captions:
            tokens = word_tokenize(caption.lower())
            self.word_count.update(tokens)

        words = [word for word, count in self.word_count.items() if count >= self.threshold]
        for idx, word in enumerate(words, len(self.word2idx)):
            self.word2idx[word] = idx
            self.idx2word[idx] = word

    def encode_caption(self, caption):
        tokens = word_tokenize(caption.lower())
        return [self.word2idx.get('<SOS>')] + [self.word2idx.get(token, self.word2idx['<UNK>']) for token in tokens] + [self.word2idx.get('<EOS>')]

    def decode_caption(self, indices):
        return ' '.join([self.idx2word.get(idx, '<UNK>') for idx in indices])

# Custom Dataset
class ImageCaptionDataset(Dataset):
    def __init__(self, image_dir, caption_file, vocab):
        self.image_dir = image_dir
        self.captions = pd.read_csv(caption_file)
        self.vocab = vocab

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, index):
        caption = str(self.captions.iloc[index, 1])
        image_path = os.path.join(self.image_dir, str(self.captions.iloc[index, 0]))

        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        try:
            image = Image.open(image_path).convert("RGB")
            image = transform(image)
        except Exception as e:
            raise RuntimeError(f"Error loading image: {image_path} - {e}")

        caption_encoded = self.vocab.encode_caption(caption)
        return image, torch.tensor(caption_encoded)

# Padding and DataLoader
from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    images, captions = zip(*batch)
    
    # Ensure image sizes are consistent
    images = torch.stack(images, 0)
    
    # Pad captions to same length
    captions = [torch.tensor(cap, dtype=torch.long) for cap in captions]
    captions_padded = pad_sequence(captions, batch_first=True, padding_value=0)

    return images, captions_padded

# Initialize Vocabulary
vocab = Vocabulary(threshold=5)
dataset = pd.read_csv("data/captions/captions.csv")
vocab.build_vocab(dataset['caption'].tolist())

# Load Data
dataset = ImageCaptionDataset(image_dir="data/images", caption_file="data/captions/captions.csv", vocab=vocab)
dataloader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn)

print("Data loading complete. Ready for training!")
