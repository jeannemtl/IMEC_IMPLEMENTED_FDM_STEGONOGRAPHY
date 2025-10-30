"""
DIRECT FDM+iMEC DECODER (Official iMEC Implementation)

Pipeline:
1. Load encoded data (stegotext tokens, encryption key, metadata)
2. Decode iMEC tokens → encrypted bits (using official iMEC)
3. Decrypt with one-time pad → signal bits
4. Reconstruct continuous signal from bits
5. Apply FFT + bandpass filtering to extract each agent's message
"""

import numpy as np
import pickle
import sys
import bitarray
import matplotlib.pyplot as plt
from scipy import signal as scipy_signal
from scipy.fft import fft, fftfreq

# Import official iMEC
from imec import IMECDecoder, remove_random_mask
from gpt2_medium import GPT2Medium


def reconstruct_from_bits(signal_bits, metadata):
    """
    Reconstruct continuous signal from binary representation.
    """
    bits_per_sample = metadata['bits_per_sample']
    num_samples = metadata['num_samples']
    
    # Parse bits back to quantized integer values
    quantized = []
    for i in range(num_samples):
        start = i * bits_per_sample
        end = start + bits_per_sample
        
        if end <= len(signal_bits):
            sample_bits = signal_bits[start:end]
            quantized.append(int(sample_bits, 2))
        else:
            print(f"WARNING: Insufficient bits for sample {i}")
            break
    
    quantized = np.array(quantized, dtype=np.float64)
    
    # De-normalize from [0, max_level] to [signal_min, signal_max]
    max_level = metadata['quantization_levels'] - 1
    normalized = quantized / max_level
    
    signal_min = metadata['signal_min']
    signal_max = metadata['signal_max']
    reconstructed = normalized * (signal_max - signal_min) + signal_min
    
    return reconstructed


def bandpass_filter(signal, freq_low, freq_high, sample_rate=100.0, order=4):
    """
    Apply bandpass filter to extract specific frequency band.
    """
    nyquist = sample_rate * 0.5
    
    # Normalize frequencies to Nyquist frequency
    low = freq_low / nyquist
    high = freq_high / nyquist
    
    # Clamp to valid range (0, 1)
    low = max(min(low, 0.99), 0.01)
    high = max(min(high, 0.99), 0.01)
    
    if low >= high:
        print(f"WARNING: Invalid frequency range [{low}, {high}]")
        return signal
    
    try:
        # Design Butterworth bandpass filter
        b, a = scipy_signal.butter(order, [low, high], btype='band')
        
        # Apply filter (zero-phase filtering)
        filtered = scipy_signal.filtfilt(b, a, signal)
        
        return filtered
    except Exception as e:
        print(f"WARNING: Bandpass filter failed: {e}")
        return signal


def decode_ask(filtered_signal, original_bits_length, sample_rate=100.0):
    """
    Decode ASK-modulated signal to recover binary message.
    """
    # Envelope detection using Hilbert transform
    analytic_signal = scipy_signal.hilbert(filtered_signal)
    envelope = np.abs(analytic_signal)
    
    # Decode bits from envelope
    samples_per_bit = len(filtered_signal) // original_bits_length
    recovered_bits = []
    
    # Adaptive threshold (between min and max envelope)
    threshold = (envelope.max() + envelope.min()) / 2
    
    for i in range(original_bits_length):
        start = i * samples_per_bit
        end = min((i + 1) * samples_per_bit, len(envelope))
        
        # Average envelope in this bit period
        avg_amplitude = envelope[start:end].mean()
        
        # Threshold decision
        bit = 1 if avg_amplitude > threshold else 0
        recovered_bits.append(bit)
    
    return recovered_bits


def calculate_accuracy(original_bits, recovered_bits):
    """Calculate bit error rate."""
    if len(original_bits) != len(recovered_bits):
        print(f"WARNING: Length mismatch: {len(original_bits)} vs {len(recovered_bits)}")
        min_len = min(len(original_bits), len(recovered_bits))
        original_bits = original_bits[:min_len]
        recovered_bits = recovered_bits[:min_len]
    
    correct = sum(1 for orig, recv in zip(original_bits, recovered_bits) 
                  if orig == recv)
    accuracy = correct / len(original_bits)
    
    return accuracy


def plot_fft_analysis(signal, sample_rate, title="FFT Analysis"):
    """
    Plot FFT to visualize frequency content.
    """
    # Compute FFT
    N = len(signal)
    yf = fft(signal)
    xf = fftfreq(N, 1/sample_rate)
    
    # Plot only positive frequencies
    plt.figure(figsize=(12, 4))
    plt.plot(xf[:N//2], 2.0/N * np.abs(yf[:N//2]))
    plt.grid(True)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.title(title)
    plt.xlim([0, 5])
    
    # Mark expected carrier frequencies
    for freq, label in [(1.0, 'ALICE'), (2.0, 'BOB'), (3.0, 'CHARLIE')]:
        plt.axvline(x=freq, color='r', linestyle='--', alpha=0.5, label=label)
    
    plt.legend()
    plt.tight_layout()
    plt.savefig('fft_analysis.png', dpi=150)
    print(f"✓ Saved FFT plot to: fft_analysis.png")


def main():
    print("="*80)
    print("DIRECT FDM+iMEC DECODER (Official iMEC - 12-bit blocks)")
    print("="*80)
    
    # ========================================================================
    # LOAD ENCODED DATA
    # ========================================================================
    
    print("\n" + "="*80)
    print("LOADING ENCODED DATA")
    print("="*80)
    
    input_file = 'encoded_data.pkl'
    try:
        with open(input_file, 'rb') as f:
            encoded_data = pickle.load(f)
        print(f"✓ Loaded: {input_file}")
    except FileNotFoundError:
        print(f"ERROR: {input_file} not found!")
        print("Please run encoder_fdm_imec.py first.")
        sys.exit(1)
    
    # Extract data
    context = encoded_data['context']
    imec_tokens = encoded_data['imec_tokens']
    mask_cfg = encoded_data['mask_cfg']
    quant_metadata = encoded_data['quant_metadata']
    messages = encoded_data['messages']
    agent_frequencies = encoded_data['agent_frequencies']
    sample_rate = encoded_data.get('sample_rate', 100.0)
    block_size_bits = encoded_data['block_size_bits']
    n_chunks = encoded_data['n_chunks']
    
    print(f"\nLoaded data:")
    print(f"  Stegotext: {len(imec_tokens)} tokens")
    print(f"  Block size: {block_size_bits} bits")
    print(f"  Number of blocks: {n_chunks}")
    print(f"  Agents: {list(messages.keys())}")
    
    # ========================================================================
    # STEP 1: OFFICIAL iMEC DECODING
    # ========================================================================
    
    print("\n" + "="*80)
    print("STEP 1: OFFICIAL iMEC DECODING")
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
    
    # Initialize official iMEC decoder
    print(f"\nInitializing official iMEC decoder...")
    decoder = IMECDecoder(
        block_size=block_size_bits,
        n_chunks=n_chunks,
        medium=medium,
        clean_up_output=False,
        belief_entropy_threshold=1e-10,
        mec_mode='dense'
    )
    
    # Decode
    print(f"\nDecoding {len(imec_tokens)} tokens with official iMEC...")
    encrypted_bitarray, dec_stats = decoder.decode(
        public_message_str=None,  # Not used
        public_message_token=imec_tokens,
        private_message_bitlen=n_chunks * block_size_bits,
        context=context,
        verbose=True
    )
    
    print(f"✓ iMEC decoding complete:")
    print(f"  Output: {len(encrypted_bitarray)} bits")
    
    # ========================================================================
    # STEP 2: DECRYPTION
    # ========================================================================
    
    print("\n" + "="*80)
    print("STEP 2: DECRYPTION (ONE-TIME PAD)")
    print("="*80)
    
    # Decrypt using official iMEC function
    decrypted_bitarray = remove_random_mask(encrypted_bitarray, **mask_cfg)
    decrypted_bits = decrypted_bitarray.to01()  # Convert to string
    
    print(f"✓ Decryption complete:")
    print(f"  Decrypted: {len(decrypted_bits)} bits")
    
    # ========================================================================
    # STEP 3: SIGNAL RECONSTRUCTION
    # ========================================================================
    
    print("\n" + "="*80)
    print("STEP 3: SIGNAL RECONSTRUCTION (BITS → SIGNAL)")
    print("="*80)
    
    recovered_signal = reconstruct_from_bits(decrypted_bits, quant_metadata)
    
    print(f"✓ Signal reconstructed:")
    print(f"  Samples: {len(recovered_signal)}")
    print(f"  Expected: {quant_metadata['num_samples']} samples")
    print(f"  Range: [{recovered_signal.min():.3f}, {recovered_signal.max():.3f}]")
    
    # Compare with original signal (if available)
    if 'combined_signal' in encoded_data:
        original_signal = encoded_data['combined_signal']
        
        # Truncate to same length
        min_len = min(len(original_signal), len(recovered_signal))
        orig_trunc = original_signal[:min_len]
        recv_trunc = recovered_signal[:min_len]
        
        mse = np.mean((orig_trunc - recv_trunc) ** 2)
        print(f"\nReconstruction quality:")
        print(f"  MSE: {mse:.6f}")
        print(f"  RMSE: {np.sqrt(mse):.6f}")
    
    # ========================================================================
    # STEP 4: FFT ANALYSIS & MESSAGE EXTRACTION
    # ========================================================================
    
    print("\n" + "="*80)
    print("STEP 4: MESSAGE EXTRACTION (FFT + BANDPASS + ASK DECODE)")
    print("="*80)
    
    # Plot FFT
    plot_fft_analysis(recovered_signal, sample_rate, 
                     "FFT Analysis of Recovered Signal")
    
    # Extract each agent's message
    recovered_messages = {}
    accuracies = {}
    
    for agent, freq in agent_frequencies.items():
        print(f"\n{agent} (carrier: {freq} Hz):")
        
        # Define bandpass filter range
        bandwidth = 0.4  # Hz
        freq_low = freq - bandwidth
        freq_high = freq + bandwidth
        
        print(f"  Bandpass filter: [{freq_low:.2f}, {freq_high:.2f}] Hz")
        
        # Apply bandpass filter
        filtered = bandpass_filter(recovered_signal, freq_low, freq_high, sample_rate)
        
        # Decode ASK
        original_bits = messages[agent]
        recovered_bits = decode_ask(filtered, len(original_bits), sample_rate)
        
        # Calculate accuracy
        accuracy = calculate_accuracy(original_bits, recovered_bits)
        
        recovered_messages[agent] = recovered_bits
        accuracies[agent] = accuracy
        
        # Display results
        print(f"  Original:  {original_bits}")
        print(f"  Recovered: {recovered_bits}")
        print(f"  Accuracy:  {accuracy*100:.1f}% ({int(accuracy*len(original_bits))}/{len(original_bits)} bits correct)")
        
        if accuracy >= 0.9:
            print(f"  ✓ Excellent recovery!")
        elif accuracy >= 0.75:
            print(f"  ⚠️  Good recovery (some errors)")
        else:
            print(f"  ❌ Poor recovery")
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    
    print("\n" + "="*80)
    print("✓ DECODING COMPLETE!")
    print("="*80)
    
    print(f"\nPipeline summary:")
    print(f"  1. iMEC decode:    {len(imec_tokens)} tokens → {len(encrypted_bitarray)} bits")
    print(f"  2. Decrypt:        {len(decrypted_bits)} bits recovered")
    print(f"  3. Reconstruct:    {len(recovered_signal)} samples")
    print(f"  4. FFT extract:    3 agents recovered")
    
    print(f"\nMessage recovery results:")
    overall_accuracy = np.mean(list(accuracies.values()))
    
    for agent in ['ALICE', 'BOB', 'CHARLIE']:
        acc = accuracies[agent]
        status = "✓" if acc >= 0.9 else "⚠️" if acc >= 0.75 else "❌"
        print(f"  {status} {agent:8s}: {acc*100:.1f}%")
    
    print(f"\n  Overall: {overall_accuracy*100:.1f}%")
    
    if overall_accuracy >= 0.9:
        print("\n✓ SUCCESS: All messages recovered with high accuracy!")
    elif overall_accuracy >= 0.75:
        print("\n⚠️  PARTIAL SUCCESS: Most messages recovered")
    else:
        print("\n❌ FAILURE: Poor message recovery")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
