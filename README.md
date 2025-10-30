# FDM Multiplexing Steganography with Official iMEC Implementation

This project combines **Frequency Division Multiplexing (FDM)** with the **official iMEC (Information-theoretic Minimum Entropy Coupling)** steganography implementation to enable secure multi-agent communication hidden in GPT-2 generated text.

## Overview

**Pipeline:**
1. **FDM Encoding**: Multiple agents' messages → ASK-modulated carriers at different frequencies → Superposed signal
2. **Quantization**: Continuous signal → Binary representation
3. **Encryption**: One-time pad ensures uniform distribution (required for iMEC)
4. **iMEC Steganography**: Encrypted bits → Natural-looking GPT-2 text (provably secure)
5. **Decoding**: Reverse process with FFT analysis to extract individual messages

## Features

- ✅ **Provably Secure**: Uses official iMEC with minimum entropy coupling
- ✅ **Multi-Agent**: Supports multiple simultaneous hidden channels via FDM
- ✅ **Natural Covertext**: GPT-2 generated stegotext
- ✅ **Perfect Secrecy**: One-time pad encryption before steganography

## Installation
```bash
# Install dependencies
pip install bitarray torch transformers scipy matplotlib numpy

# Clone this repo
git clone https://github.com/jeannemtl/IMEC_IMPLEMENTED_FDM_STEGONOGRAPHY.git
cd IMEC_IMPLEMENTED_FDM_STEGONOGRAPHY
```

## Usage

### Encoding (3 agents: ALICE, BOB, CHARLIE)
```bash
python encoder_fdm_imec.py
```

This will:
- Generate FDM signals for 3 agents at 1Hz, 2Hz, 3Hz
- Quantize and encrypt the combined signal
- Encode using official iMEC into GPT-2 stegotext
- Save to `encoded_data.pkl`

### Decoding
```bash
python decoder_fdm_imec.py
```

This will:
- Load stegotext and decrypt using iMEC
- Reconstruct the FDM signal
- Apply FFT + bandpass filtering
- Extract and verify each agent's message
- Generate `fft_analysis.png` showing frequency peaks

## File Structure
```
.
├── encoder_fdm_imec.py    # Main encoding pipeline
├── decoder_fdm_imec.py    # Main decoding pipeline
├── gpt2_medium.py         # GPT-2 wrapper for iMEC
├── imec.py                # Official iMEC encoder/decoder
├── mec.py                 # Minimum entropy coupling implementation
├── utils.py               # Utility functions
└── README.md              # This file
```

## How It Works

### 1. Frequency Division Multiplexing (FDM)
Each agent's binary message is modulated onto a unique carrier frequency using ASK (Amplitude Shift Keying):
- **ALICE**: 1.0 Hz carrier
- **BOB**: 2.0 Hz carrier  
- **CHARLIE**: 3.0 Hz carrier

Signals are superposed (added together) to create a composite signal.

### 2. Steganography with iMEC
The official iMEC implementation provides:
- **Minimum Entropy Coupling**: Optimal mapping between ciphertext and covertext distributions
- **Information-Theoretic Security**: Provably secure against statistical attacks
- **Adaptive Encoding**: Uses GPT-2's probability distributions for natural text

### 3. Signal Recovery
Decoder uses:
- **FFT Analysis**: Identifies frequency peaks corresponding to each agent
- **Bandpass Filtering**: Isolates individual carrier frequencies
- **Envelope Detection**: Demodulates ASK to recover bits

## Example Output
```
Message recovery results:
  ✓ ALICE    : 100.0% (16/16 bits correct)
  ✓ BOB      : 93.8% (15/16 bits correct)
  ✓ CHARLIE  : 100.0% (16/16 bits correct)

  Overall: 97.9%
```

## Parameters

Key configurable parameters in `encoder_fdm_imec.py`:
- `block_size_bits`: iMEC block size (12 bits = 4,096 values)
- `num_samples`: Signal length (5,000 samples)
- `bits_per_sample`: Quantization precision (8 bits = 256 levels)
- `agent_frequencies`: Carrier frequencies for each agent

## Attribution

This project uses the **official iMEC implementation** with modifications:

**Original iMEC Files** (modified for compatibility):
- `imec.py` - iMEC encoder/decoder
- `mec.py` - Minimum entropy coupling algorithms  
- `utils.py` - Utility functions (fixed imports for transformers)

**Original Authors**: [iMEC paper authors]  
**Original Repository**: [Link if public]  
**License**: [Original license]

**My Contributions**:
- `gpt2_medium.py` - GPT-2 medium wrapper for iMEC
- `encoder_fdm_imec.py` - Complete FDM+iMEC encoding pipeline
- `decoder_fdm_imec.py` - FFT-based decoding and signal recovery
- Integration of FDM with official iMEC

## Technical Details

**Block Size**: 12 bits (4,096 possible values per block)  
**Context Window**: 1,024 tokens (GPT-2 max)  
**Entropy Threshold**: 1e-10 (stopping criterion)  
**MEC Mode**: Dense (full coupling matrix)

## References

- Original iMEC paper: [Citation]
- Minimum Entropy Coupling: [Theory paper]
- GPT-2: Radford et al., "Language Models are Unsupervised Multitask Learners"

## License

See original iMEC license for `imec.py`, `mec.py`, `utils.py`.  
My contributions (FDM integration) are [your license choice].

## Requirements

- Python 3.8+
- PyTorch
- Transformers (HuggingFace)
- NumPy, SciPy, Matplotlib
- bitarray

## Notes

- Encoding 40,000 bits takes several minutes with official iMEC
- GPU recommended for faster encoding (CUDA support included)
- Generated stegotext is natural GPT-2 text indistinguishable from normal output

## Future Work

- Support for more agents (4+ frequencies)
- Variable block sizes
- Alternative language models (GPT-3, LLaMA)
- Real-time encoding/decoding
