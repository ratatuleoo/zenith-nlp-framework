import torch
import typer
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from typing import List

from ..core.bpe_tokenizer import BPETokenizer
from ..models.bert_model import BERT
from ..tasks.text_classification import TextClassifier

app = FastAPI(title="My NLP Framework API")
model_artifacts = {}

class ClassifyRequest(BaseModel):
    text: str
    max_len: int = 128

class ClassifyResponse(BaseModel):
    prediction: int
    probabilities: List[float]

@app.post("/classify", response_model=ClassifyResponse)
async def classify_text(request: ClassifyRequest):

    tokenizer = model_artifacts.get("tokenizer")
    model = model_artifacts.get("model")
    device = model_artifacts.get("device")

    if not tokenizer or not model:
        raise HTTPException(status_code=503, detail="Model or tokenizer not loaded. The API is not ready.")

    token_ids = tokenizer.tokenize(request.text)
    if len(token_ids) < request.max_len:
        token_ids += [0] * (request.max_len - len(token_ids))
    else:
        token_ids = token_ids[:request.max_len]
    
    ids_tensor = torch.tensor([token_ids], dtype=torch.long).to(device)
    segment_tensor = torch.ones_like(ids_tensor).to(device)

    with torch.no_grad():
        outputs = model(ids=ids_tensor, segment_info=segment_tensor)
        probabilities = torch.softmax(outputs, dim=1).squeeze().tolist()
        prediction = torch.argmax(outputs, dim=1).item()

    return ClassifyResponse(prediction=prediction, probabilities=probabilities)

def serve(
    model_path: str = typer.Option(..., "--model-path", help="Path to the trained .pth model file."),
    tokenizer_path: str = typer.Option(..., "--tokenizer-path", help="Path to the trained bpe_tokenizer.json file."),
    vocab_size: int = typer.Option(10000, help="Vocabulary size of the tokenizer."),
    embed_size: int = typer.Option(768, help="Embedding size of the model."),
    num_classes: int = typer.Option(2, help="Number of output classes."),
    host: str = "0.0.0.0",
    port: int = 8000,
):
    """
    Launches the FastAPI server for the NLP framework.
    
    This command-line tool loads a user-trained model and its corresponding tokenizer,
    then starts a web server to serve predictions.
    """
    print("--- Starting NLP Framework API Server ---")
    
    try:
        print(f"Loading tokenizer from: {tokenizer_path}")
        tokenizer = BPETokenizer()
        tokenizer.load(tokenizer_path)
        model_artifacts["tokenizer"] = tokenizer
        
        print(f"Loading model from: {model_path}")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        

        backbone = BERT(vocab_size=vocab_size, embed_size=embed_size)
        model = TextClassifier(backbone=backbone, embed_size=embed_size, num_classes=num_classes)
        
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()

        model_artifacts["model"] = model
        model_artifacts["device"] = device

        print(f"Model and tokenizer loaded successfully. Running on device: {device}")

    except Exception as e:
        print(f"!!! ERROR: Failed to load model or tokenizer: {e}")
        return

    print(f"Starting server at http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    typer.run(serve)
