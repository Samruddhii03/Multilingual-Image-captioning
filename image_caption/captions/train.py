import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from models.cnn_model import CNNFeatureExtractor
from models.caption_generator import CaptionGenerator
from utils.data_loader import ImageCaptionDataset, Vocabulary


# Hyperparameters
vocab_size = 5000
embed_size = 256
hidden_size = 512
num_layers = 1
batch_size = 32
learning_rate = 0.001
num_epochs = 10

# Load dataset
vocab = Vocabulary()
captions_path = 'data/captions/captions.csv'
image_dir = 'data/images'
dataset = ImageCaptionDataset(image_dir, captions_path, vocab)
vocab.build_vocab(dataset.captions['caption'].tolist())

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize models
feature_extractor = CNNFeatureExtractor()
caption_model = CaptionGenerator(vocab_size=len(vocab.word2idx), embed_size=embed_size, hidden_size=hidden_size, num_layers=num_layers)

# Loss and optimizer
criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(caption_model.parameters(), lr=learning_rate)

# Training loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
caption_model.to(device)

for epoch in range(num_epochs):
    for images, captions in dataloader:
        images, captions = images.to(device), captions.to(device)

        # Extract features
        features = feature_extractor.extract_features(images)

        # Forward pass
        outputs = caption_model(features, captions[:, :-1])
        loss = criterion(outputs.reshape(-1, vocab_size), captions[:, 1:].reshape(-1))

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')

# Save model
torch.save(caption_model.state_dict(), 'checkpoints/caption_generator.h5')
print("Model saved!")
