# Train Decoder only (GPT-2) Text Generation Model & host quantized version on Gradio

This project demonstrates how to train a simplified GPT-2-like model, quantize it to reduce its size for efficient inference, and then deploy it as a user-friendly web application using Gradio.

## Features

*   **Custom GPT-like Model**: Implementation of a basic transformer-based language model.
*   **Training Loop**: Includes a training loop with a learning rate schedule (warmup and cosine decay) and gradient clipping.
*   **Post-training Dynamic Quantization**: Reduces model size significantly (e.g., from ~1.4GB to ~300MB) without requiring a calibration dataset, making it suitable for deployment.
*   **Text Generation**: Implements a `generate_text` function using top-k sampling for diverse outputs.
*   **Gradio Web Interface**: A simple and interactive UI to experiment with the text generation model.
*   **Hugging Face Spaces Deployment Ready**: The generated `app.py` and `requirements.txt` are designed for easy deployment to Hugging Face Spaces.

### 1. Prepare Data

This project expects a text file named `input.txt` in the root directory. This file will be used to train the GPT model.

### 2. Run the Notebook

Execute all cells in the Jupyter/Colab notebook sequentially. The notebook performs the following steps:

*   **Model Definition**: Defines the `GPT`, `Block`, `MLP`, and `CausalSelfAttention` classes.
*   **Training**: Trains the GPT model using the `DataLoaderLite` and saves `model_checkpoint.pt`.
    *   *(Note: Training can be computationally intensive and may take a long time. A pre-trained `model_checkpoint.pt` is assumed for the quantization and inference steps.)*
*   **Quantization**: Applies post-training dynamic quantization to the saved model and saves a smaller `model_checkpoint_quantized.pt`.
*   **Model Loading & Tokenizer Initialization**: Loads the quantized model and initializes the `tiktoken` encoder.
*   **Text Generation Function**: Defines the `generate_text` function to interact with the loaded model.
*   **Gradio Interface**: Creates and launches the Gradio web application.

### 3. Training Loss Achieved:
Below screenshot illustrates the training loss achieved:

<img width="300" height="500" alt="image" src="https://github.com/user-attachments/assets/21ca63e2-8d1f-4f66-bb48-0e65083836b2" />


## Gradio app created for this model can be accessed at the following location:

https://huggingface.co/spaces/saneshashank/gpt2.toy.model.shakespeare.text

<img width="750" height="500" alt="image" src="https://github.com/user-attachments/assets/c4209878-109f-4178-8a93-71dfbc2b108c" />


## Model Details

The core of this project is a GPT-2 inspired decoder-only transformer model. The model was trained from scratch and then dynamically quantized using `torch.quantization.quantize_dynamic` to convert its `nn.Linear` layers to `int8` for reduced memory footprint and faster inference on CPU.

## Example Generated Output

*(When running locally or on Spaces with prompt "Hello, my name is", Max Length: 50, Number of Return Sequences: 3)*

```
Generated Text 1:
Hello, my name is alter'd, they.

JOHN OF GAUNT:
Returnare thy knave: I have patience to have.

Roman:
No, fellow, for foul captain, good coward was made end I

Generated Text 2:
Hello, my name is children's my very.

GREMIO:
I be to some myself are in leisure,
Yet that we'll leave thy leisure for the cause to your state, man
That, for I'll marvell

Generated Text 3:
Hello, my name is alter'd, heADied her, man,
That do us leisure, I'll beShall, son of my true, my gentle what thou art to thy gross, myer.

Second Citizen:


```

