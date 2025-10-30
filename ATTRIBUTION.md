# Third-Party Code Attribution

## iMEC Implementation

The following files are derived from the official iMEC (Information-theoretic Minimum Entropy Coupling) steganography implementation:

### Files:
- `imec.py` - iMEC encoder and decoder classes
- `mec.py` - Minimum entropy coupling algorithm implementation
- `utils.py` - Utility functions (modified to use `transformers` instead of `pytorch_transformers`)

### Original Authors:
[List authors from original paper/repo]

### Original Repository:
[Link to repo if publicly available]

### License:
[Original license - MIT/Apache/BSD/etc]

### Modifications Made:
1. **utils.py**: Changed `from pytorch_transformers import` to `from transformers import`
2. **utils.py**: Changed `from src.image_transformer import` to `from image_transformer import`
3. Added `try-except` blocks for optional imports

All modifications maintain compatibility with the original iMEC algorithm while updating for modern Python/PyTorch versions.

## My Original Work

The following files are original contributions:
- `gpt2_medium.py` - Medium wrapper for GPT-2
- `encoder_fdm_imec.py` - FDM encoding pipeline
- `decoder_fdm_imec.py` - FDM decoding with FFT analysis
- `README.md` - Project documentation
- This file (`ATTRIBUTION.md`)

These files integrate Frequency Division Multiplexing with the official iMEC implementation to enable multi-agent steganographic communication.
