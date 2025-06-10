import gradio as gr
import torch
from transformers import GPT2Tokenizer
from PIL import Image
from gemini_model import GeminiModel  
from torchvision import transforms

 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GeminiModel().to(device)
model.load_state_dict(torch.load("gemini_model_trained.pth", map_location=device))
model.eval()

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def predict(img, prompt):
    image_tensor = preprocess(img).unsqueeze(0).to(device)
    input = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(
            input_ids=input["input_ids"],
            attention_mask=input["attention_mask"],
            images=image_tensor
        )
        logits = outputs.logits[:, -1, :]
        next_token = torch.argmax(logits, dim=-1)
        output = tokenizer.decode(next_token)

    return prompt + output

 
demo = gr.Interface(
    fn=predict,
    inputs=[gr.Image(type="pil"), gr.Textbox(label="Prompt", placeholder="What is in this image?")],
    outputs="text",
    title="Gemini Multimodal Demo",
    description="Upload an image and ask a question about it. The model will generate a response."
)

if __name__ == "__main__":
    demo.launch()
