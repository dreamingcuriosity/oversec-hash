# OverSec - Enhanced 512-bit Hash Function

A cryptographic hash function implementation featuring dynamic prime generation, multi-lane processing, and extensive nonlinear transformations.

## Overview

OverSec is a 512-bit hash function that generates unique fingerprints for text input. It uses dynamic prime numbers seeded from the input text itself, making each hash computation unique to its input's characteristics.

## Features

- **512-bit output**: Produces 128-character hexadecimal hashes
- **Dynamic prime generation**: Generates cryptographically strong primes based on input text using Miller-Rabin primality testing
- **Multi-lane architecture**: Processes data through 16 independent 512-bit state lanes
- **Avalanche effect**: Single bit changes cascade throughout the entire output
- **Nonlinear operations**: Multiple layers of mixing, diffusion, and transformation
- **Adaptive rounds**: Number of processing rounds scales with input characteristics

## Requirements

```bash
python 3.6+
```

No external dependencies required - uses only Python standard library.

## Installation

```bash
git clone <repository-url>
cd oversec
```

## Usage

### Basic Hashing

```python
python test9.py
```

Then enter a string when prompted (minimum 2 characters).

### Command-Line Options

When running the program, you'll be presented with 4 options:

1. **Avalanche Measurement** - Tests how a single bit change affects the output
2. **Hash Comparison** - Compare hashes of two different strings
3. **Collision Detection** - Attempt to find hash collisions (for testing)
4. **File Hashing** - Hash the contents of a file

### File Hashing Example

```bash
python test9.py
# Choose option 4
# Enter input file path: document.txt
# Enter output file path: document.hash
```

### Programmatic Usage

```python
from test9 import process_text_512_bit

result = process_text_512_bit("Hello, World!")
print(f"Hash: {result['hexadecimal_result']}")
print(f"Rounds used: {result['num_rounds']}")
```

## Architecture

### Processing Phases

1. **Initialization**: Dynamic prime generation from input text
2. **Absorption**: Input bytes processed through multi-lane architecture
3. **Extended Mixing**: Cross-lane mixing with nonlinear operations
4. **Additional Processing**: Modular arithmetic and state injection
5. **Compression**: 16 lanes compressed to single 512-bit state
6. **Correlation**: Mathematical correlation operations
7. **Final Diffusion**: Multiple rounds of mixing and rotation

### Key Components

- **Miller-Rabin Primality Testing**: Generates probable primes for cryptographic operations
- **S-boxes**: Dynamically generated substitution boxes for byte-level nonlinearity
- **Multi-lane States**: 16 independent 512-bit processing lanes
- **Cross-mixing**: Inter-lane operations for increased diffusion
- **Prime-based Operations**: Eight different prime-number-based transformation functions
- **Rotation Schedules**: Dynamic bit rotation patterns

## Security Considerations

**⚠️ IMPORTANT: This is an experimental/educational implementation.**

This hash function is NOT:
- Formally analyzed for cryptographic security
- Peer-reviewed by cryptographic experts
- Recommended for production cryptographic applications
- Suitable for password hashing (use Argon2, bcrypt, or scrypt instead)

This implementation is suitable for:
- Educational purposes
- Understanding hash function design
- Non-critical checksumming
- Experimental projects

For production systems, use established standards like SHA-256, SHA-3, or BLAKE3.

### ⚠️ Denial of Service (DoS) Warning

**CRITICAL: This hash function is computationally expensive and vulnerable to DoS attacks.**

The design prioritizes security properties over performance, making it susceptible to resource exhaustion:

- **CPU intensive**: 64-128 rounds of nonlinear operations per hash
- **Memory intensive**: 16 × 512-bit state lanes = 1KB+ per hash operation
- **Linear scaling**: Processing time increases with input length
- **No rate limiting**: Built-in protection against abuse

**DoS Attack Vectors:**
1. **Large input attacks**: Attacker sends extremely long strings
2. **Rapid request flooding**: Multiple concurrent hash requests
3. **Algorithmic complexity**: Exploiting O(n) time complexity
4. **Memory exhaustion**: Parallel requests consuming system RAM

**Mitigation Strategies (if used in any system):**
```python
# Input length limits
MAX_INPUT_LENGTH = 10000  # bytes

# Rate limiting (example)
from functools import lru_cache
@lru_cache(maxsize=1000)
def cached_hash(text):
    return process_text_512_bit(text)

# Timeout protection
import signal
def timeout_handler(signum, frame):
    raise TimeoutError("Hash operation exceeded time limit")

signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(5)  # 5 second timeout
```

**Never expose this hash function directly to untrusted input without:**
- Strict input length validation
- Rate limiting per IP/user
- Request timeouts
- Resource monitoring and alerts
- Load balancing and request queuing

## Algorithm Details

### Dynamic Prime Generation

Primes are generated using the Miller-Rabin primality test with 20 rounds of testing, seeded from SHA-512 hash of input text. This ensures:
- Each input generates unique prime constants
- Primes are cryptographically strong
- Deterministic behavior for the same input

### State Representation

The 512-bit state is represented as 8 × 64-bit parts:
```
State512 = [part0, part1, part2, part3, part4, part5, part6, part7]
Total bits = 8 × 64 = 512 bits
```

### Round Calculation

Number of rounds is dynamically determined by:
```
base_rounds = 64 + (text_seed_hash % 32)
additional = length_factor + checksum_factor + xor_factor
total_rounds = clamp(base_rounds + additional, 64, 128)
```

## Performance

Approximate performance on modern hardware:
- Short strings (<100 chars): ~50-100ms
- Medium strings (100-1000 chars): ~100-500ms
- Large files (>1KB): ~1-5s per KB

Performance is intentionally sacrificed for security properties.

## Testing

The implementation includes built-in testing functions:

```python
# Avalanche effect test
measure_avalanche_512("test string")

# Collision search (limited)
find_collision_512bit()

# Hash comparison
compare_hashes_512()
```

## Output Format

Hash output is a 128-character hexadecimal string:
```
Example: a3f2c1d4e5b6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0f1a2...
```

## Known Limitations

1. **Speed**: Multiple diffusion rounds make it relatively slow
2. **Memory**: Uses significant memory for multi-lane processing
3. **Analysis**: Not formally proven secure against all attack vectors
4. **Integer handling**: Limited to Python's integer precision

## Future Improvements

Potential enhancements:
- Hardware acceleration (GPU/SIMD)
- Parallel lane processing
- Formal security proofs
- Additional test vectors
- Constant-time implementation

## Contributing

This is an educational/experimental project. Contributions for:
- Security analysis
- Performance optimization
- Additional test cases
- Documentation improvements

are welcome.

## License

No license... *yet*..

## Author

**dreamingcuriosity**

## References

- Miller-Rabin Primality Test
- Cryptographic Hash Functions
- Sponge Construction
- S-box Design Principles

## Disclaimer

This software is provided "as is" without warranty of any kind. Use at your own risk. Not suitable for cryptographic applications requiring formal security guarantees.
