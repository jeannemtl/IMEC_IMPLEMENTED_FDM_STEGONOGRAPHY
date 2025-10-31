
import numpy as np
import pickle
import bitarray
import torch
from imec import IMECEncoder, apply_random_mask
from gpt2_medium import GPT2Medium


def ask_modulate(bits, freq, length, sample_rate=100.0):
    samples_per_bit = length // len(bits)
    signal = np.zeros(length)
    for i, bit in enumerate(bits):
        start, end = i * samples_per_bit, min((i + 1) * samples_per_bit, length)
        amplitude = 1.0 if bit == 1 else 0.1
        t = np.linspace(0, (end - start) / sample_rate, end - start)
        signal[start:end] = amplitude * np.sin(2 * np.pi * freq * t)
    return signal


def quantize_to_bits(signal, bits_per_sample=2):
    signal_min, signal_max = float(signal.min()), float(signal.max())
    signal_range = signal_max - signal_min if (signal_max - signal_min) > 1e-10 else 1.0
    normalized = (signal - signal_min) / signal_range
    quantized = np.round(normalized * ((2 ** bits_per_sample) - 1)).astype(np.uint16)
    return ''.join(format(int(v), f'0{bits_per_sample}b') for v in quantized), {
        'num_samples': len(signal), 'bits_per_sample': bits_per_sample,
        'signal_min': signal_min, 'signal_max': signal_max,
        'quantization_levels': 2 ** bits_per_sample
    }


print("="*80)
print("WORKING ENCODER - 4-bit blocks (full vocabulary)")
print("="*80)

messages = {'ALICE': [0,1,0,0], 'BOB': [1,1,1,0], 'CHARLIE': [1,1,1,0]}
freqs = {'ALICE': 1.0, 'BOB': 2.0, 'CHARLIE': 3.0}

# Small test
num_samples = 20  # Very small for testing
bits_per_sample = 2  
block_size = 4  # 4-bit blocks!

total_bits = num_samples * bits_per_sample
n_blocks = (total_bits + block_size - 1) // block_size

print(f"\n⚡ CONFIGURATION:")
print(f"  Samples: {num_samples}")
print(f"  Bits/sample: {bits_per_sample}")
print(f"  Total bits: {total_bits}")
print(f"  Block size: {block_size} bits ({2**block_size} values)")
print(f"  Blocks: {n_blocks}")
print(f"\n  MEC size: 16 × 50,257 = 805,120 elements")
print(f"  Expected time per MEC: ~2-5 minutes")
print(f"  Total time: ~{n_blocks * 3:.0f}-{n_blocks * 5:.0f} minutes")
print(f"\n  This WILL complete (unlike 6-bit blocks)!")

# FDM
print("\n" + "="*80)
print("STEP 1: FDM + QUANTIZATION + ENCRYPTION")
print("="*80)

alice = ask_modulate(messages['ALICE'], freqs['ALICE'], num_samples)
bob = ask_modulate(messages['BOB'], freqs['BOB'], num_samples)
charlie = ask_modulate(messages['CHARLIE'], freqs['CHARLIE'], num_samples)
combined = alice + bob + charlie

signal_bits, metadata = quantize_to_bits(combined, bits_per_sample)
signal_ba = bitarray.bitarray(signal_bits)

mask_cfg = {"input_key": b'\x03' * 64, "sample_seed_prefix": b'sample', "input_nonce": b'\x01' * 16}
encrypted = apply_random_mask(signal_ba, **mask_cfg)

print(f"✓ Ready to encode: {len(encrypted)} bits in {n_blocks} blocks")

# Encode
print("\n" + "="*80)
print("STEP 2: iMEC ENCODING (4-bit blocks)")
print("="*80)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
medium = GPT2Medium(model_name='gpt2', device=device, temp=0.95, probs_top_k=50)

print(f"\nInitializing iMEC encoder (block_size={block_size})...")
encoder = IMECEncoder(
    block_size=block_size,
    medium=medium,
    clean_up_output=False,
    belief_entropy_threshold=1e-10,
    mec_mode='dense'
)

print(f"\n🚀 Starting encoding...")
print(f"   This will take approximately {n_blocks * 3:.0f}-{n_blocks * 5:.0f} minutes")
print(f"   ({n_blocks} blocks × ~3-5 min/block)")
print(f"\n   Be patient! Each MEC computation is working through")
print(f"   805,120 probability couplings on CPU.\n")

import time
start_time = time.time()
block_times = []

try:
    # Monkey-patch to show progress
    original_mec = encoder.medium.mec
    def timed_mec(*args, **kwargs):
        block_start = time.time()
        result = original_mec(*args, **kwargs)
        block_elapsed = time.time() - block_start
        block_times.append(block_elapsed)
        
        total_elapsed = time.time() - start_time
        blocks_done = len(block_times)
        avg_time = np.mean(block_times)
        remaining = (n_blocks - blocks_done) * avg_time
        
        print(f"  Block {blocks_done}/{n_blocks}: {block_elapsed:.1f}s "
              f"(avg: {avg_time:.1f}s/block, "
              f"~{remaining/60:.1f} min remaining)")
        
        return result
    
    encoder.medium.mec = timed_mec
    
    stegotext, tokens, stats = encoder.encode(
        private_message_bit=encrypted,
        context="The future of AI",
        verbose=True
    )
    
    elapsed = time.time() - start_time
    
    print(f"\n{'='*80}")
    print(f"✓ ENCODING COMPLETE!")
    print(f"{'='*80}")
    print(f"\nTime: {elapsed/60:.1f} minutes ({elapsed:.0f}s)")
    print(f"Tokens: {len(tokens)}")
    print(f"Efficiency: {len(encrypted) / len(tokens):.2f} bits/token")
    
    if 'avg_KL/mean' in stats:
        print(f"\n📊 KL DIVERGENCE: {stats['avg_KL/mean']:.6f}")
        if stats['avg_KL/mean'] < 0.01:
            print(f"  ✅ EXCELLENT! Near-perfect security")
    
    # Save
    with open('encoded_4bit.pkl', 'wb') as f:
        pickle.dump({
            'tokens': tokens, 'stats': stats, 'mask_cfg': mask_cfg,
            'signal': combined, 'metadata': metadata, 'messages': messages,
            'block_size': block_size, 'n_blocks': n_blocks,
            'encoding_time': elapsed
        }, f)
    
    print(f"\n✓ Saved to: encoded_4bit.pkl")
    print(f"\n{'='*80}")
    print(f"SUCCESS! 4-bit blocks make official iMEC practical!")
    print(f"{'='*80}")
    
except KeyboardInterrupt:
    print(f"\n\n⚠️  Interrupted after {(time.time() - start_time)/60:.1f} minutes")
    if block_times:
        print(f"   Completed {len(block_times)}/{n_blocks} blocks")
        print(f"   Average: {np.mean(block_times):.1f}s per block")
except Exception as e:
    print(f"\n\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
