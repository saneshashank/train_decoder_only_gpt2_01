# GPT-2 Text Generation with Quantization and Gradio

This project demonstrates how to train a simplified GPT-2-like model, quantize it to reduce its size for efficient inference, and then deploy it as a user-friendly web application using Gradio.

## Features

*   **Custom GPT-like Model**: Implementation of a basic transformer-based language model.
*   **Training Loop**: Includes a training loop with a learning rate schedule (warmup and cosine decay) and gradient clipping.
*   **Post-training Dynamic Quantization**: Reduces model size significantly (e.g., from ~1.4GB to ~300MB) without requiring a calibration dataset, making it suitable for deployment.
*   **Text Generation**: Implements a `generate_text` function using top-k sampling for diverse outputs.
*   **Gradio Web Interface**: A simple and interactive UI to experiment with the text generation model.
*   **Hugging Face Spaces Deployment Ready**: The generated `app.py` and `requirements.txt` are designed for easy deployment to Hugging Face Spaces.

## Setup Instructions

### 1. Clone the Repository (or download notebook)

If you're using this as a standalone project:

```bash
git clone <your_repo_url>
cd <your_repo_directory>
```

### 2. Install Dependencies

It is highly recommended to use a virtual environment.

```bash
pip install -r requirements.txt
```

**`requirements.txt` contents:**
```
torch
tiktoken
gradio
```

### 3. Prepare Data

This project expects a text file named `input.txt` in the root directory. This file will be used to train the GPT model.

### 4. Run the Notebook

Execute all cells in the Jupyter/Colab notebook sequentially. The notebook performs the following steps:

*   **Model Definition**: Defines the `GPT`, `Block`, `MLP`, and `CausalSelfAttention` classes.
*   **Training**: Trains the GPT model using the `DataLoaderLite` and saves `model_checkpoint.pt`.
    *   *(Note: Training can be computationally intensive and may take a long time. A pre-trained `model_checkpoint.pt` is assumed for the quantization and inference steps.)*
*   **Quantization**: Applies post-training dynamic quantization to the saved model and saves a smaller `model_checkpoint_quantized.pt`.
*   **Model Loading & Tokenizer Initialization**: Loads the quantized model and initializes the `tiktoken` encoder.
*   **Text Generation Function**: Defines the `generate_text` function to interact with the loaded model.
*   **Gradio Interface**: Creates and launches the Gradio web application.

## How to Use the Gradio App

### Local Usage

Once you run the Gradio interface cell in the notebook, it will provide a local URL (e.g., `http://127.0.0.1:7860`) and a public share link. Open either of these in your browser to interact with the text generation model.

### Deployment to Hugging Face Spaces

To deploy this application to Hugging Face Spaces:

1.  **Save `app.py`**: Create a file named `app.py` in your project's root directory with the following content:

    ```python
    import torch
    import torch.nn as nn
    from torch.nn import functional as F
    import math
    from dataclasses import dataclass
    import tiktoken
    import gradio as gr
    import os # Required for os.path.getsize in some contexts

    # === Model Architecture (Copy from notebook) ===
    # Copy the GPTConfig, CausalSelfAttention, MLP, Block, and GPT classes here.
    # Ensure GPTConfig has vocab_size=50304 or matches your trained model.

    @dataclass
    class GPTConfig:
        block_size: int = 1024
        vocab_size: int = 50257
        n_layer: int = 12
        n_head: int = 12
        n_embd: int = 768

    class CausalSelfAttention(nn.Module):
        def __init__(self, config):
            super().__init__()
            assert config.n_embd % config.n_head == 0
            self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
            self.c_proj = nn.Linear(config.n_embd, config.n_embd)
            self.c_proj.NANGPT_SCALE_INIT = 1
            self.n_head = config.n_head
            self.n_embd = config.n_embd
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))

        def forward(self, x):
            B, T, C = x.size()
            qkv = self.c_attn(x)
            q, k, v = qkv.split(self.n_embd, dim=2)
            k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
            q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
            v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            y = att @ v

            y = y.transpose(1, 2).contiguous().view(B, T, C)
            y = self.c_proj(y)
            return y

    class MLP(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd)
            self.gelu    = nn.GELU(approximate='tanh')
            self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd)
            self.c_proj.NANOGPT_SCALE_INIT = 1

        def forward(self, x):
            x = self.c_fc(x)
            x = self.gelu(x)
            x = self.c_proj(x)
            return x

    class Block(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.ln_1 = nn.LayerNorm(config.n_embd)
            self.attn = CausalSelfAttention(config)
            self.ln_2 = nn.LayerNorm(config.n_embd)
            self.mlp = MLP(config)

        def forward(self, x):
            x = x + self.attn(self.ln_1(x))
            x = x + self.mlp(self.ln_2(x))
            return x

    class GPT(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config

            self.transformer = nn.ModuleDict(dict(
                wte = nn.Embedding(config.vocab_size, config.n_embd),
                wpe = nn.Embedding(config.block_size, config.n_embd),
                h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f = nn.LayerNorm(config.n_embd),
            ))
            self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
            self.transformer.wte.weight = self.lm_head.weight
            self.apply(self._init_weights)

        def _init_weights(self, module):
            if isinstance(module, nn.Linear):
                std = 0.02
                if hasattr(module, 'NANGPT_SCALE_INIT'):
                    std *= (2 * self.config.n_layer) ** -0.5
                torch.nn.init.normal_(module.weight, mean = 0.0, std = std)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std = 0.02)

        def forward(self, idx, targets=None):
            B, T = idx.size()
            assert T <= self.config.block_size
            pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
            pos_emb = self.transformer.wpe(pos)
            tok_emb = self.transformer.wte(idx)
            x = tok_emb + pos_emb
            for block in self.transformer.h:
                x = block(x)
            x = self.transformer.ln_f(x)
            logits = self.lm_head(x)
            loss = None
            if targets is not None:
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            return logits, loss

    # === Model Loading and Quantization ===
    model = GPT(GPTConfig(vocab_size=50304)) # Ensure config matches your trained model
    model.eval() # Set to eval mode before quantization

    # Apply dynamic quantization to the model
    model = torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear},
        dtype=torch.qint8
    )

    # Load the state_dict into the now quantized model
    checkpoint = torch.load('model_checkpoint_quantized.pt', map_location='cpu') # Load to CPU
    model.load_state_dict(checkpoint)
    model.to('cpu') # Ensure quantized model runs on CPU

    # === Tokenizer Initialization ===
    enc = tiktoken.get_encoding('gpt2')

    # === Text Generation Function ===
    def generate_text(prompt, max_length, num_return_sequences):
        start_ids = enc.encode(prompt)
        x = (torch.tensor(start_ids, dtype=torch.long, device='cpu')[None, ...])

        generated_texts = []
        for _ in range(num_return_sequences):
            current_x = x.clone()
            while current_x.size(1) < max_length:
                with torch.no_grad():
                    logits = model(current_x)[0]
                    logits = logits[:, -1, :]
                    probs = F.softmax(logits, dim=-1)
                    topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                    ix = torch.multinomial(topk_probs, 1)
                    xcol = torch.gather(topk_indices, -1, ix)
                    current_x = torch.cat((current_x, xcol), dim=1)

            tokens = current_x[0, :max_length].tolist()
            decoded = enc.decode(tokens)
            generated_texts.append(decoded)

        return generated_texts

    # === Gradio Interface ===
    interface = gr.Interface(
        fn=generate_text,
        inputs=[
            gr.Textbox(label='Enter your prompt here:', value='Hello, my name is'),
            gr.Slider(minimum=10, maximum=200, value=50, step=1, label='Max Length'),
            gr.Slider(minimum=1, maximum=5, value=3, step=1, label='Number of Return Sequences')
        ],
        outputs=gr.Textbox(label='Generated Text:', lines=10),
        title='GPT-2 Text Generation'
    )

    # Launch the interface
    interface.launch()
    ```

2.  **Upload `model_checkpoint_quantized.pt`**: Ensure the quantized model file (`model_checkpoint_quantized.pt`) is in the same directory as your `app.py` file.

3.  **Create a New Space**: Go to [Hugging Face Spaces](https://huggingface.co/spaces/new) and create a new Space. Choose "Gradio" as the SDK. You can choose a small CPU-based instance for this quantized model.

4.  **Upload Files**: Upload `app.py`, `requirements.txt`, and `model_checkpoint_quantized.pt` to your Hugging Face Space.

Your app should now be accessible and running on Hugging Face Spaces!

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

