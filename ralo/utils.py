
import io, json
import torch

def tensor_to_bytes(t):
    buffer = io.BytesIO()
    torch.save(t, buffer)
    return buffer.getvalue()
def bytes_to_tensor(b):
    buffer = io.BytesIO(b)
    try:
        return torch.load(buffer, weights_only=True, map_location="cpu")
    except TypeError:
        buffer.seek(0)
        return torch.load(buffer, map_location="cpu")
def make_bytes_list(blist):
    buffer = io.BytesIO()
    buffer.write(len(blist).to_bytes(4, 'big'))
    for b in blist:
        buffer.write(len(b).to_bytes(4, 'big'))
        buffer.write(b)
    return buffer.getvalue()
def bytes_list_to_list(b):
    buffer = io.BytesIO(b)
    num = int.from_bytes(buffer.read(4), 'big')
    blist = []
    for _ in range(num):
        l = int.from_bytes(buffer.read(4), 'big')
        blist.append(buffer.read(l))
    return blist

def json_to_bytes_list(data):
    tensors = [(k,v) for k, v in data.items() if isinstance(v, torch.Tensor)]
    others = {k:v for k, v in data.items() if not isinstance(v, torch.Tensor)}
    others['#tensors'] = [k for k, v in tensors]
    blist = [json.dumps(others).encode()]
    for _, v in tensors: blist.append(tensor_to_bytes(v))
    return make_bytes_list(blist)

def bytes_list_to_json(b):
    blist = bytes_list_to_list(b)
    if len(blist) < 1: return {}
    others = json.loads(blist[0])
    tkeys = others.pop('#tensors', [])
    tensors = {k:bytes_to_tensor(v) for k, v in zip(tkeys, blist[1:])}
    return {**others, **tensors}

def save_model(name, model, tokenizer=None):
    state_dict = model.state_dict()
    state_dict = type(state_dict)({k: v.cpu() for k, v in state_dict.items()})
    model.save_pretrained(name, state_dict=state_dict)
    if tokenizer is not None: tokenizer.save_pretrained(name)

def enable_gradient_checkpointing(model, ratio=1):
    model.train()
    model.gradient_checkpointing_enable()
    if ratio >= 1: return
    # Try to find layers in common model structures
    layers = None
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        layers = model.model.layers
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        layers = model.transformer.h  # GPT-2 style
    elif hasattr(model, 'layers'):
        layers = model.layers
    elif hasattr(model, 'model') and hasattr(model.model, 'encoder') and hasattr(model.model.encoder, 'layer'):
        layers = model.model.encoder.layer  # BERT style
    
    if layers is None:
        # If we can't find layers, just enable full gradient checkpointing
        print("[WARNING] Could not find model layers for partial gradient checkpointing, using full checkpointing")
        return
    
    total_layers = len(layers)
    start_idx = total_layers - int(total_layers * ratio)
    for i, layer in enumerate(layers):
        if hasattr(layer, 'gradient_checkpointing'):
            layer.gradient_checkpointing = i >= start_idx

# -------- Gradient (de)serialization helpers for cross-component use --------
GRAD_PREFIX = "grad::"

def encode_gradients(grad_dict):
    """Prefix gradient tensor keys so they survive json+tensor serialization."""
    return {f"{GRAD_PREFIX}{name}": tensor for name, tensor in grad_dict.items()}

def extract_gradient_tensors(payload):
    """Pop gradient tensors out of a payload produced by encode_gradients/json_to_bytes_list.
    Returns {param_name: tensor}.
    """
    grads = {}
    for key in list(payload.keys()):
        if isinstance(key, str) and key.startswith(GRAD_PREFIX):
            param_name = key[len(GRAD_PREFIX):]
            grads[param_name] = payload.pop(key)
    return grads

def _fp32_forward_pre_hook(module, input):
    """Pre-hook to convert input to fp32 before forward pass"""
    if isinstance(input, (tuple, list)) and len(input) > 0:
        # Convert first input (hidden states) to fp32
        return (input[0].to(torch.float32),) + input[1:]
    elif isinstance(input, torch.Tensor):
        return (input.to(torch.float32),)
    else:
        return input

def pad_lists(list_of_lists, pad_value):
    """
    Padding different token lengths in log probabilities.
    """
    import torch
    max_len = max(len(xs) for xs in list_of_lists)
    out = torch.full((len(list_of_lists), max_len), pad_value)
    for i, xs in enumerate(list_of_lists):
        out[i, :len(xs)] = torch.tensor(xs)
    return out

def convert_lm_head_to_fp32(model):
    """
    Convert only the lm_head (output layer) to fp32 for better numerical stability.
    This is a common practice in fine-tuning to maintain higher precision in the output layer.
    Uses forward hook to automatically convert bfloat16 inputs to fp32.
    """
    if hasattr(model, 'lm_head'):
        lm_head = model.lm_head
        # Convert lm_head to fp32
        lm_head = lm_head.to(torch.float32)
        # Register forward pre-hook to convert input to fp32
        lm_head.register_forward_pre_hook(_fp32_forward_pre_hook)
        model.lm_head = lm_head
        print("[MODEL] Converted lm_head to fp32 (with automatic input casting via hook)")
    elif hasattr(model, 'get_output_embeddings'):
        output_emb = model.get_output_embeddings()
        if output_emb is not None:
            # Convert to fp32
            output_emb = output_emb.to(torch.float32)
            # Register forward pre-hook
            output_emb.register_forward_pre_hook(_fp32_forward_pre_hook)
            model.set_output_embeddings(output_emb)
            print("[MODEL] Converted output embeddings to fp32 (with automatic input casting via hook)")
    else:
        # Try to find common output layer names
        for name, module in model.named_modules():
            if 'lm_head' in name.lower() or ('output' in name.lower() and 'embedding' in name.lower()):
                # Convert to fp32
                module = module.to(torch.float32)
                # Register forward pre-hook
                module.register_forward_pre_hook(_fp32_forward_pre_hook)
                # Get parent module to replace the child
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]
                if parent_name:
                    parent = model
                    for part in parent_name.split('.'):
                        parent = getattr(parent, part)
                    setattr(parent, child_name, module)
                else:
                    setattr(model, child_name, module)
                print(f"[MODEL] Converted {name} to fp32 (with automatic input casting via hook)")
                break
        else:
            print("[MODEL] Warning: Could not find lm_head or output embeddings to convert to fp32")

