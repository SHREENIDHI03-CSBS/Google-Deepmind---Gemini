import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2Tokenizer
from gemini_model import GeminiModel
from torchvision import transforms
from PIL import Image
import os
import json

class ImageTextDataset(Dataset):
    def __init__(self, data_json, image_folder):
        with open(data_json, "r") as f:
            self.data = json.load(f)
        self.image_folder = image_folder
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = os.path.join(self.image_folder, item["image"])
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        text = item["prompt"] + " " + item["answer"]
        encoding = self.tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=32)
        return {
            "image": image,
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze()
        }

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = ImageTextDataset("data/train.json", "data/images")
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    model = GeminiModel().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=5e-5)
    loss_fn = nn.CrossEntropyLoss(ignore_index=model.language_model.config.pad_token_id)

    model.train()
    for epoch in range(3):
        total_loss = 0
        for batch in dataloader:
            images = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(
                input_ids=input_ids[:, :-1],
                attention_mask=attention_mask[:, :-1],
                images=images
            )
            logits = outputs.logits
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), input_ids[:, 1:].reshape(-1))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()

        print(f"Epoch {epoch+1} Loss: {total_loss / len(dataloader)}")

    torch.save(model.state_dict(), "gemini_model_trained.pth")

if __name__ == "__main__":
    train()
