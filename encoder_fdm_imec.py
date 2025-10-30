"""
DIRECT FDM+iMEC ENCODER (Official iMEC Implementation)

Pipeline:
1. Generate ASK-modulated carriers for each agent (ALICE, BOB, CHARLIE)
2. Superpose signals (Frequency Division Multiplexing)
3. Quantize continuous signal to bits
4. Encrypt with one-time pad (makes distribution uniform - CRITICAL for iMEC!)
5. Apply OFFICIAL iMEC encoding for perfect security

Output: Saves all data needed for decoding to 'encoded_data.pkl'
"""

import numpy as np
import pickle
import sys
import os
import bitarray
import math

# Import official iMEC
from imec import IMECEncoder, apply_random_mask
from gpt2_medium import GPT2Medium


def ask_modulate(bits, freq, length, sample_rate=100.0):
    """
    ASK (Amplitude Shift Keying) modulation with proper sampling.
    """
    samples_per_bit = length // len(bits)
    signal = np.zeros(length)
    
    for i, bit in enumerate(bits):
        start = i * samples_per_bit
        end = min((i + 1) * samples_per_bit, length)
        amplitude = 1.0 if bit == 1 else 0.1
        duration = (end - start) / sample_rate
        t = np.linspace(0, duration, end - start)
        carrier = np.sin(2 * np.pi * freq * t)
        signal[start:end] = amplitude * carrier
    
    return signal


def quantize_to_bits(signal, bits_per_sample=8):
    """
    Quantize continuous signal to binary representation.
    """
    signal_min = float(signal.min())
    signal_max = float(signal.max())
    signal_range = signal_max - signal_min
    
    if signal_range < 1e-10:
        signal_range = 1.0
        print("WARNING: Signal has near-zero range!")
    
    normalized = (signal - signal_min) / signal_range
    max_level = (2 ** bits_per_sample) - 1
    quantized = np.round(normalized * max_level).astype(np.uint16)
    
    signal_bits = ''.join(format(int(val), f'0{bits_per_sample}b') 
                          for val in quantized)
    
    metadata = {
        'num_samples': len(signal),
        'bits_per_sample': bits_per_sample,
        'signal_min': signal_min,
        'signal_max': signal_max,
        'quantization_levels': max_level + 1
    }
    
    return signal_bits, metadata


def verify_uniformity(bit_string, name="Bit string"):
    """Check if bit string has uniform distribution (should be ~50% ones)."""
    ones_count = sum(int(b) for b in bit_string)
    ones_ratio = ones_count / len(bit_string)
    
    print(f"\n{name} uniformity check:")
    print(f"  Length: {len(bit_string)} bits")
    print(f"  Ones ratio: {ones_ratio:.4f} (target: 0.5000)")
    
    if 0.48 <= ones_ratio <= 0.52:
        print(f"  ✓ Distribution is uniform")
    else:
        print(f"  ⚠️  WARNING: Distribution may not be uniform!")
    
    return ones_ratio


def main():
    print("="*80)
    print("DIRECT FDM+iMEC ENCODER (Official iMEC - 12-bit blocks)")
    print("="*80)
    
    # ========================================================================
    # CONFIGURATION
    # ========================================================================
    
    messages = {
        'ALICE':   [0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0],
        'BOB':     [1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0],
        'CHARLIE': [1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0]
    }
    
    agent_frequencies = {
        'ALICE': 1.0,
        'BOB': 2.0,
        'CHARLIE': 3.0
    }
    
    # Signal parameters
    num_samples = 5000
    sample_rate = 100.0
    bits_per_sample = 8
    
    # iMEC parameters
    block_size_bits = 12  # 12-bit blocks (4,096 values)
    context = "The future of artificial intelligence"
    
    print(f"\nConfiguration:")
    print(f"  Agents: {list(messages.keys())}")
    print(f"  Message length: {len(messages['ALICE'])} bits per agent")
    print(f"  Signal length: {num_samples} samples")
    print(f"  iMEC block size: {block_size_bits} bits")
    
    # ========================================================================
    # STEP 1: FREQUENCY MULTIPLEXING
    # ========================================================================
    
    print("\n" + "="*80)
    print("STEP 1: FREQUENCY MULTIPLEXING (ASK + SUPERPOSITION)")
    print("="*80)
    
    alice_signal = ask_modulate(messages['ALICE'], agent_frequencies['ALICE'], 
                                num_samples, sample_rate)
    bob_signal = ask_modulate(messages['BOB'], agent_frequencies['BOB'], 
                              num_samples, sample_rate)
    charlie_signal = ask_modulate(messages['CHARLIE'], agent_frequencies['CHARLIE'], 
                                  num_samples, sample_rate)
    
    combined_signal = alice_signal + bob_signal + charlie_signal
    
    print(f"\nCombined signal:")
    print(f"  Samples: {len(combined_signal)}")
    print(f"  Range: [{combined_signal.min():.3f}, {combined_signal.max():.3f}]")
    
    # ========================================================================
    # STEP 2: QUANTIZATION
    # ========================================================================
    
    print("\n" + "="*80)
    print("STEP 2: QUANTIZATION (SIGNAL → BITS)")
    print("="*80)
    
    signal_bits, quant_metadata = quantize_to_bits(combined_signal, bits_per_sample)
    
    print(f"\n✓ Quantization complete:")
    print(f"  Output: {len(signal_bits)} bits")
    
    # ========================================================================
    # STEP 3: ENCRYPTION (ONE-TIME PAD)
    # ========================================================================
    
    print("\n" + "="*80)
    print("STEP 3: ENCRYPTION (ONE-TIME PAD)")
    print("="*80)
    
    # Convert to bitarray for official iMEC
    signal_bitarray = bitarray.bitarray(signal_bits)
    
    # Apply one-time pad using official iMEC function
    mask_cfg = {
        "input_key": b'\x03' * 64,
        "sample_seed_prefix": b'sample',
        "input_nonce": b'\x01' * 16
    }
    
    encrypted_bitarray = apply_random_mask(signal_bitarray, **mask_cfg)
    encrypted_bits = encrypted_bitarray.to01()  # Convert back to string
    
    print(f"\n✓ Encryption complete:")
    print(f"  Ciphertext: {len(encrypted_bits)} bits")
    
    verify_uniformity(encrypted_bits, "Ciphertext")
    
    # ========================================================================
    # STEP 4: OFFICIAL iMEC ENCODING
    # ========================================================================
    
    print("\n" + "="*80)
    print("STEP 4: OFFICIAL iMEC ENCODING (12-BIT BLOCKS)")
    print("="*80)
    
    # Initialize GPT-2 medium
    print("\nInitializing GPT-2 medium...")
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    medium = GPT2Medium(
        model_name='gpt2',
        device=device,
        temp=0.95,
        probs_top_k=50
    )
    
    # Initialize official iMEC encoder
    print(f"\nInitializing official iMEC encoder (block_size={block_size_bits})...")
    encoder = IMECEncoder(
        block_size=block_size_bits,
        medium=medium,
        clean_up_output=False,
        belief_entropy_threshold=1e-10,
        mec_mode='dense'
    )
    
    # Encode
    print(f"\nEncoding {len(encrypted_bitarray)} bits with official iMEC...")
    stegotext, imec_tokens, enc_stats = encoder.encode(
        private_message_bit=encrypted_bitarray,
        context=context,
        verbose=True
    )
    
    print(f"\n✓ Official iMEC encoding complete:")
    print(f"  Input: {len(encrypted_bitarray)} bits")
    print(f"  Output: {len(imec_tokens)} tokens")
    print(f"  Efficiency: {enc_stats.get('bits_per_step', 0):.3f} bits/token")
    print(f"  Steps: {enc_stats.get('n_steps', 0)}")
    
    print(f"\nStegotext preview (first 200 chars):")
    print(f"  {stegotext[:200]}...")
    
    # ========================================================================
    # SAVE ENCODED DATA
    # ========================================================================
    
    print("\n" + "="*80)
    print("SAVING ENCODED DATA")
    print("="*80)
    
    encoded_data = {
        # Original messages
        'messages': messages,
        'agent_frequencies': agent_frequencies,
        
        # Signal processing
        'num_samples': num_samples,
        'sample_rate': sample_rate,
        'quant_metadata': quant_metadata,
        
        # Encoded data
        'context': context,
        'imec_tokens': imec_tokens,
        'stegotext': stegotext,
        
        # Decryption (for official iMEC decoder)
        'mask_cfg': mask_cfg,
        
        # iMEC parameters
        'block_size_bits': block_size_bits,
        'n_chunks': int(math.ceil(len(encrypted_bitarray) / block_size_bits)),
        
        # Statistics
        'enc_stats': enc_stats,
        
        # Verification
        'combined_signal': combined_signal,
        'individual_signals': {
            'ALICE': alice_signal,
            'BOB': bob_signal,
            'CHARLIE': charlie_signal
        }
    }
    
    output_file = 'encoded_data.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump(encoded_data, f)
    
    print(f"\n✓ Saved to: {output_file}")
    print(f"\nData saved:")
    print(f"  - Stegotext: {len(imec_tokens)} tokens")
    print(f"  - Encryption key saved in mask_cfg")
    print(f"  - Block size: {block_size_bits} bits")
    
    print("\n" + "="*80)
    print("✓ ENCODING COMPLETE!")
    print("="*80)
    
    print(f"\nNext step: Run decoder with official iMEC")


if __name__ == "__main__":
    main()

