from megatransformer import megatransformer
from criteria import labelsmooth

def trace_tensor_devices(tensor, visited=None, depth=0):
    """
    Traces through the computational graph of a tensor and prints device information
    for all tensors in the graph.
    
    Args:
        tensor: The PyTorch tensor to start tracing from
        visited: Set of visited tensor IDs (used internally for recursion)
        depth: Current depth in the graph (used for indentation)
    """
    if visited is None:
        visited = set()
    
    # Get unique ID for the tensor
    tensor_id = id(tensor)
    if tensor_id in visited:
        return
    visited.add(tensor_id)
    
    # Print information about current tensor
    indent = "  " * depth
    print(f"{indent}Tensor(shape={tensor.shape}, dtype={tensor.dtype}, "
          f"device={tensor.device}, requires_grad={tensor.requires_grad})")
    
    # If tensor has grad_fn, traverse its next_functions
    if tensor.grad_fn is not None:
        print(f"{indent}grad_fn: {type(tensor.grad_fn).__name__}")
        for i, next_fn in enumerate(tensor.grad_fn.next_functions):
            if next_fn[0] is not None:
                # Some next_functions might be None
                print(f"{indent}  Input {i}:")
                # Try to get the tensor from the grad_fn
                if hasattr(next_fn[0], 'variable'):
                    next_tensor = next_fn[0].variable
                    trace_tensor_devices(next_tensor, visited, depth + 2)
                else:
                    print(f"{indent}    (No direct tensor access)")

def find_device_mismatches(model):
    """
    Analyzes a PyTorch model and prints information about tensors on different devices.
    
    Args:
        model: The PyTorch model to analyze
    """
    devices = {}
    
    def register_tensor(name, tensor):
        device = tensor.device
        if device not in devices:
            devices[device] = []
        devices[device].append(name)
    
    # Collect device information for all parameters
    for name, param in model.named_parameters():
        register_tensor(f"Parameter: {name}", param)
    
    # Collect device information for all buffers
    for name, buffer in model.named_buffers():
        register_tensor(f"Buffer: {name}", buffer)
    
    # Print summary of devices
    print("\nDevice Distribution Summary:")
    for device, tensors in devices.items():
        print(f"\nDevice: {device}")
        print("Tensors:")
        for tensor_name in tensors:
            print(f"  - {tensor_name}")

# Example usage for your encoder-decoder case:
def analyze_encoder_decoder_devices(args, encoder: megatransformer.Encoder, decoder: megatransformer.Decoder, encoder_sequences, decoder_sequences, src_key_padding_mask, tgt_key_padding_mask, lengths):
    """
    Specifically analyzes an encoder-decoder transformer setup for device mismatches.
    
    Args:
        encoder: The encoder model
        decoder: The decoder model
        sample_input: A sample input tensor
        sample_target: A sample target tensor
    """
    print("=== Analyzing Encoder ===")
    find_device_mismatches(encoder)
    
    print("\n=== Analyzing Decoder ===")
    find_device_mismatches(decoder)
    
    print("\n=== Analyzing Sample Input ===")
    print(f"Input device: {encoder_sequences.device}")
    
    print("\n=== Analyzing Sample Target ===")
    print(f"Target device: {decoder_sequences.device}")

    loss_fn = labelsmooth.LabelSmoothedCE(args)
    
    # Try a forward pass and trace the computation graph
    print("\n=== Tracing Computation Graph ===")
    
    encoder_sequences = encoder_sequences.to(args.encoder_device)
    decoder_sequences = decoder_sequences.to(args.decoder_device)
    src_key_padding_mask = src_key_padding_mask.to(args.encoder_device)
    tgt_key_padding_mask = tgt_key_padding_mask.to(args.decoder_device)
    encoder.embed_tokens = encoder.embed_tokens.to(args.encoder_device)

    encoder_sequences, _ = encoder(encoder_sequences, src_key_padding_mask)

    encoder_sequences = encoder_sequences.to(args.decoder_device)
    src_key_padding_mask = src_key_padding_mask.to(args.decoder_device)
    decoder.embed_tokens = decoder.embed_tokens.to(args.decoder_device)
    decoder.lm_head = decoder.lm_head.to(args.decoder_device)

    decoder_output, _ = decoder(decoder_sequences, encoder_sequences, src_key_padding_mask, tgt_key_padding_mask)
    loss = loss_fn(decoder_output, decoder_sequences, lengths=lengths)  # Replace with your loss function
    print("\nTracing backward graph from loss:")
    trace_tensor_devices(loss)

# Usage example:
"""
# Initialize your models
encoder = Encoder().to('cuda:0')
decoder = Decoder().to('cuda:1')

# Create sample data
sample_input = torch.randn(1, 32, 512).to('cuda:0')
sample_target = torch.randn(1, 32, 512).to('cuda:1')

# Analyze the setup
analyze_encoder_decoder_devices(encoder, decoder, sample_input, sample_target)
"""