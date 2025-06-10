import torch
from PIL import Image
from gemini_model import GeminiModel
from transformers import GPT2Tokenizer
from torchvision import transforms

def load_image(img_path):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    return preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0)

def predict(image_path, prompt):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    model = GeminiModel().to(device)
    model.eval()

    image = load_image(image_path).to(device)
    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(device)

    with torch.no_grad():
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            images=image
        )
        logits = outputs.logits[:, -1, :]
        next_token = torch.argmax(logits, dim=-1).unsqueeze(1)
        output_text = tokenizer.decode(next_token[0])
    
    return prompt + output_text

if __name__ == "__main__":
    result = predict("samples/cat.jpg", "What is in the image? ")
    print("Prediction:", result)
