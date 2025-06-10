# Google-Deepmind---Gemini
## Overview :

Gemini is a family of multimodal models designed to process and reason over both language and vision inputs in a tightly integrated manner. The goal of this repo is to:
- Provide a flexible baseline implementation of Gemini-like multi-modal transformers.
- Support inference and fine-tuning across text, images, or mixed input streams.
- Allow research exploration into cross-modal fusion, attention routing, and token alignment.

##  Key Features

- Multi-stream input processing (text and vision embeddings)
- Dual encoder-decoder transformer blocks
- Support for BPE (Byte Pair Encoding) and Patch embeddings
- Image adapter modules for various vision backbones (e.g. CLIP ViT-B, ResNet)
- Cross-attention fusion between modalities
- LoRA + QLoRA support for lightweight fine-tuning
- Training recipes for common datasets (e.g., VQAv2, COCO Captions, OK-VQA)

## Model Architecture


                    +-----------------+
                    |  Vision Encoder |
                    +-----------------+
                            |
                    +-----------------+
                    |  Image Adapter  |
                    +-----------------+
                            |
      Text Input  ---> [Tokenizer] ---> [Text Embeddings]
                            |
                            v
                    +----------------------+
                    |  Multi-Modal Encoder |
                    +----------------------+
                            |
                    +----------------------+
                    |  Language Decoder    |
                    +----------------------+
                            |
                    +----------------------+
                    |     Output Tokens    |
                    +----------------------+

## Requirements :
Python 3.10+

PyTorch ≥ 2.0

torchvision, transformers, accelerate

CUDA-enabled GPU
 
## Supported Tasks:
Image Captioning

Visual Question Answering

Text-Image Matching

Multimodal Reasoning

## Technical Stack:

| Layer/Component        | Technology/Service                        |  
|------------------------|-------------------------------------------| 
| **Frontend**           | React (Vite)                              |  
| **Backend**            | Python (LangGraph, FastAPI)               |  
| **AI Model API**       | Google Gemini API                         |  
| **Agent Framework**    | LangGraph                                 | 
| **Database**           | PostgreSQL                                |  
| **Pub/Sub & Caching**  | Redis                                     | 
| **Containerization**   | Docker, Docker Compose                    |  
| **Environment Config** | .env files                                |  
| **Cloud Platform**     | Google Cloud Vertex AI (optional)         | 
| **Authentication**     | API Keys (Google Gemini API)              |  

## License :
This project is licensed under the Apache-2.0 License.  
