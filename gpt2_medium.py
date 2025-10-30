import torch
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer

class GPT2Medium:
    """
    Medium wrapper for GPT-2 to work with official iMEC implementation.
    Mimics the interface expected by IMECEncoder/IMECDecoder.
    """
    
    def __init__(self, model_name='gpt2', device='cpu', temp=1.0, probs_top_k=None):
        print(f"Loading GPT-2 model: {model_name}")
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.model.eval()
        
        self.device = torch.device(device)
        self.model.to(self.device)
        
        self.temp = temp
        self.probs_top_k = probs_top_k
        
        self.input_ids = None
        self.action_labels = torch.arange(len(self.tokenizer), device=self.device)
        self.tokens_generated = []
        
        self.MAX_CONTEXT = 1024  # GPT-2's max context
        
        print(f"✓ GPT2Medium initialized")
        print(f"  Vocab size: {len(self.tokenizer)}")
        print(f"  Device: {self.device}")
        print(f"  Temperature: {self.temp}")
    
    def reset(self, context=None):
        """Initialize with context."""
        if context is None:
            context = ""
        
        self.input_ids = self.tokenizer.encode(
            context, 
            return_tensors='pt'
        ).to(self.device)
        
        self.tokens_generated = []
        
        return self._get_probs()
    
    def step(self, action):
        """
        Take a step by appending the action (token) to context.
        
        Args:
            action: Token ID (int or tensor)
        """
        # Ensure action is an integer
        if isinstance(action, torch.Tensor):
            action = int(action.item())
        else:
            action = int(action)
        
        self.tokens_generated.append(action)
        
        # Append token to input
        new_token = torch.tensor([[action]], device=self.device, dtype=torch.long)
        self.input_ids = torch.cat([self.input_ids, new_token], dim=1)
        
        # Truncate context if needed
        if self.input_ids.shape[1] > self.MAX_CONTEXT:
            self.input_ids = self.input_ids[:, -self.MAX_CONTEXT:]
        
        return self._get_probs()
    
    def _get_probs(self):
        """
        Get probability distribution over next token.
        
        Returns:
            probs: numpy array of probabilities
            info: dict with metadata
        """
        with torch.no_grad():
            outputs = self.model(self.input_ids)
            logits = outputs.logits[0, -1, :]
            
            # Apply temperature
            if self.temp != 1.0:
                logits = logits / self.temp
            
            # Apply top-k filtering if requested
            if self.probs_top_k is not None and self.probs_top_k > 0:
                top_k = min(self.probs_top_k, logits.shape[0])
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = -float('Inf')
            
            # Convert to probabilities
            probs = torch.softmax(logits, dim=0)
            
            # Calculate entropy
            probs_np = probs.cpu().numpy()
            probs_clean = probs_np[probs_np > 1e-10]
            if len(probs_clean) > 0:
                entropy_raw = -np.sum(probs_clean * np.log2(probs_clean))
            else:
                entropy_raw = 0.0
            
            info = {
                'medium_entropy': float(entropy_raw),
                'medium_entropy_raw': float(entropy_raw),
                'end_of_line': False  # GPT-2 doesn't have natural line endings
            }
            
            return probs_np, info
    
    def get_output(self, clean_up=False):
        """
        Get generated tokens.
        
        Args:
            clean_up: Whether to remove trailing tokens (not used for GPT-2)
        
        Returns:
            List of token IDs
        """
        return self.tokens_generated
    
    def humanify(self, tokens):
        """
        Convert token IDs to human-readable text.
        
        Args:
            tokens: List of token IDs
        
        Returns:
            String of decoded text
        """
        return self.tokenizer.decode(tokens)
