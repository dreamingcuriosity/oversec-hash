import math
import random
import time
import hashlib
import sys

# Fix 1: Add proper primality testing for cryptographic security
def is_probable_prime(n, k=20):
    """Miller-Rabin primality test"""
    if n < 2:
        return False
    if n in (2, 3):
        return True
    if n % 2 == 0:
        return False
    
    # Write n as 2^r * d + 1
    r, d = 0, n - 1
    while d % 2 == 0:
        r += 1
        d //= 2
    
    def check_composite(a):
        x = pow(a, d, n)
        if x in (1, n - 1):
            return False
        for _ in range(r - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                return False
        return True
    
    for _ in range(k):
        a = random.randrange(2, n - 1)
        if check_composite(a):
            return False
    return True

# Fix 2: Proper dynamic prime generation with actual primality testing
def generate_dynamic_primes(seed_text, count=8, bits=512):
    """Generate actual primes dynamically from text"""
    primes = []
    seed_hash = hashlib.sha512(seed_text.encode()).digest()
    
    for i in range(count):
        # Use different portions of hash for each prime
        prime_seed = int.from_bytes(seed_hash[i*8:(i+1)*8], 'big')
        
        # Ensure we get a large odd number
        candidate = (prime_seed | (1 << (bits-1)) | 1) & ((1 << bits) - 1)
        
        # Find the next actual prime
        while not is_probable_prime(candidate):
            candidate += 2
            candidate &= ((1 << bits) - 1)  # Keep within bit limit
            if candidate < (1 << (bits-1)):  # Wrap around if needed
                candidate = (1 << (bits-1)) | 1
        
        primes.append(candidate)
    
    return primes

def make_prime_like(n):
    """Make a number prime-like (for demonstration purposes)"""
    if n % 2 == 0:
        n += 1
    
    # Ensure it's not divisible by small primes
    small_primes = [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
    for p in small_primes:
        while n % p == 0:
            n += 2
    
    return n

# Fix 3: Define constants before use and make them properly sized
MASK64 = 0xFFFFFFFFFFFFFFFF
MASK512 = (1 << 512) - 1

# These will be set dynamically
PRIME1 = PRIME2 = PRIME3 = PRIME4 = PRIME5 = PRIME6 = PRIME7 = None

class DynamicPrimeManager:
    """Manages dynamic prime generation based on input text"""
    def __init__(self):
        self.prime_cache = {}
        self.prime_functions = [
            self._prime_func1, self._prime_func2, self._prime_func3,
            self._prime_func4, self._prime_func5, self._prime_func6,
            self._prime_func7, self._prime_func8
        ]
    
    def _prime_func1(self, x, round_key):
        return ((x * 0x9E3779B97F4A7C15) ^ self.rotate_left_64(x, 31) ^ round_key) & MASK64
    
    def _prime_func2(self, x, round_key):
        return ((x * 0xBF58476D1CE4E5B9) | self.rotate_right_64(x, 17) ^ round_key) & MASK64
    
    def _prime_func3(self, x, round_key):
        return ((x * 0x94D049BB133111EB) + self.rotate_left_64(x, 13) ^ round_key) & MASK64
    
    def _prime_func4(self, x, round_key):
        return ((x ^ 0xC6A4A7935BD1E995) * self.rotate_right_64(x, 29) ^ round_key) & MASK64
    
    def _prime_func5(self, x, round_key):
        return ((x | 0x85EBCA77C2B2AE63) ^ (x * self.rotate_left_64(x, 19)) ^ round_key) & MASK64
    
    def _prime_func6(self, x, round_key):
        return ((x + 0x27D4EB2F165667C5) & self.rotate_right_64(x, 23) ^ round_key) & MASK64
    
    def _prime_func7(self, x, round_key):
        return ((x * 0x9E3779B97F4A7C15) ^ (x | self.rotate_left_64(x, 37)) ^ round_key) & MASK64
    
    def _prime_func8(self, x, round_key):
        return ((x ^ 0x5A8279996ED9EBA1) + self.rotate_right_64(x, 41) ^ round_key) & MASK64
    
    def rotate_left_64(self, val, n):
        n = n % 64
        return ((val << n) | (val >> (64 - n))) & MASK64
    
    def rotate_right_64(self, val, n):
        n = n % 64
        return ((val >> n) | (val << (64 - n))) & MASK64
    
    def get_prime_operation(self, index, round_key):
        """Get a prime-based operation with the given round key"""
        func = self.prime_functions[index % len(self.prime_functions)]
        return lambda x: func(x, round_key.parts[index % 8])

class State512:
    def __init__(self, value=0):
        if isinstance(value, int):
            self.parts = [
                value & MASK64,
                (value >> 64) & MASK64,
                (value >> 128) & MASK64,
                (value >> 192) & MASK64,
                (value >> 256) & MASK64,
                (value >> 320) & MASK64,
                (value >> 384) & MASK64,
                (value >> 448) & MASK64
            ]
        else:
            self.parts = [p & MASK64 for p in value]

    def to_int(self):
        return (self.parts[0] |
                (self.parts[1] << 64) |
                (self.parts[2] << 128) |
                (self.parts[3] << 192) |
                (self.parts[4] << 256) |
                (self.parts[5] << 320) |
                (self.parts[6] << 384) |
                (self.parts[7] << 448))

    def __getitem__(self, index):
        return self.parts[index]

    def __setitem__(self, index, value):
        self.parts[index] = value & MASK64

    def __str__(self):
        return format(self.to_int(), '0128x')

    def copy(self):
        return State512(self.parts.copy())

def rotate_left_64(val, n, bits=64):
    n = n % bits
    return ((val << n) | (val >> (bits - n))) & MASK64

def rotate_right_64(val, n, bits=64):
    n = n % bits
    return ((val >> n) | (val << (bits - n))) & MASK64

def rotate_left_512(state, n):
    """Rotate 512-bit state left by n bits"""
    total_bits = 512
    n = n % total_bits

    if n == 0:
        return state

    big_val = state.to_int()
    result_val = ((big_val << n) | (big_val >> (512 - n))) & MASK512
    return State512(result_val)

def rotate_right_512(state, n):
    """Rotate 512-bit state right by n bits"""
    total_bits = 512
    n = n % total_bits

    if n == 0:
        return state

    big_val = state.to_int()
    result_val = ((big_val >> n) | (big_val << (512 - n))) & MASK512
    return State512(result_val)

# Fix 4: Enhanced operation classes with proper implementation
class CryptoOperation:
    """Base class for cryptographic operations"""
    def apply(self, state, round_key):
        raise NotImplementedError

class NonlinearMixOperation(CryptoOperation):
    def apply(self, state, round_key):
        return nonlinear_mix_512(state)

class DiffusionOperation(CryptoOperation):
    def apply(self, state, round_key):
        return full_nonlinear_diffusion_512(state)

class PrimeBasedOperation(CryptoOperation):
    def __init__(self, prime_manager, operation_index):
        self.prime_manager = prime_manager
        self.operation_index = operation_index
    
    def apply(self, state, round_key):
        result = state.copy()
        prime_op = self.prime_manager.get_prime_operation(self.operation_index, round_key)
        for i in range(8):
            result.parts[i] = prime_op(result.parts[i])
        return result

class CrossPartOperation(CryptoOperation):
    def apply(self, state, round_key):
        result = state.copy()
        for i in range(8):
            left = result.parts[(i - 1) % 8]
            right = result.parts[(i + 1) % 8]
            diagonal = result.parts[(i + 4) % 8]
            result.parts[i] = ((result.parts[i] * left) ^ 
                              rotate_left_64(result.parts[i], 19) ^ 
                              right ^ rotate_right_64(diagonal, 7)) & MASK64
        return result

class SBoxOperation(CryptoOperation):
    def __init__(self, sbox, inv_sbox):
        self.sbox = sbox
        self.inv_sbox = inv_sbox
    
    def apply(self, state, round_key):
        result = state.copy()
        for i in range(8):
            x = result.parts[i]
            bytes_out = []
            for j in range(8):
                b = (x >> (j * 8)) & 0xFF
                prev_byte = (x >> (((j-1) % 8) * 8)) & 0xFF
                b = self.sbox[(b ^ prev_byte) & 0xFF]
                b = self.inv_sbox[((b * 127) ^ (round_key.parts[i] >> (j * 8))) & 0xFF]
                bytes_out.append(b)
            result.parts[i] = sum(b << (j * 8) for j, b in enumerate(bytes_out)) & MASK64
        return result

def nonlinear_mix_64(x):
    """64-bit non-linear mixing"""
    global PRIME1, PRIME2
    if PRIME1 is None or PRIME2 is None:
        raise ValueError("Primes not initialized")
    
    x = (x * PRIME1) & MASK64
    x ^= (x & 0xAAAAAAAAAAAAAAAA) >> 1
    x ^= (x | 0x5555555555555555) << 1
    x &= MASK64

    result = 0
    for i in range(8):
        byte_val = (x >> (i * 8)) & 0xFF
        sboxed = ((byte_val * 251) ^ (byte_val >> 4)) & 0xFF
        result |= sboxed << (i * 8)
    x = result

    x = (x * PRIME2) & MASK64

    left = x & 0xFFFFFFFF
    right = (x >> 32) & 0xFFFFFFFF
    temp = ((left * 0x9E3779B1) ^ rotate_left_64(right, 17, 32)) & 0xFFFFFFFF
    left ^= temp
    right ^= ((temp * 0x85EBCA77) & 0xFFFFFFFF)
    x = (right << 32) | left

    return x & MASK64

def nonlinear_mix_512(state):
    """Non-linear mixing for 512-bit state with dynamic operations"""
    global PRIME1, PRIME2, PRIME3, PRIME4, PRIME5, PRIME6, PRIME7
    result = State512()

    primes = [PRIME1, PRIME2, PRIME3, PRIME4, PRIME5, PRIME6, PRIME7, PRIME1]
    
    # Apply different non-linear operations to each 64-bit part
    for i in range(8):
        x = state.parts[i]
        x = (x * primes[i]) & MASK64

        x ^= (x & 0xAAAAAAAAAAAAAAAA) >> 1
        x ^= (x | 0x5555555555555555) << 1
        x &= MASK64

        byte_result = 0
        for j in range(8):
            byte_val = (x >> (j * 8)) & 0xFF
            sboxed = ((byte_val * 251) ^ (byte_val >> 4)) & 0xFF
            byte_result |= sboxed << (j * 8)
        x = byte_result

        result.parts[i] = x & MASK64

    # Enhanced cross-part mixing for 512-bit
    for i in range(8):
        left_part = result.parts[(i - 1) % 8]
        right_part = result.parts[(i + 1) % 8]
        diagonal_part = result.parts[(i + 4) % 8]

        result.parts[i] = ((result.parts[i] * left_part) ^
                          rotate_left_64(result.parts[i], 17) ^
                          right_part ^ diagonal_part) & MASK64

    return result

def full_nonlinear_diffusion_512(state):
    """Cascade of non-linear operations ensuring full bit diffusion for 512-bit"""
    global PRIME3, PRIME4, PRIME5
    result = State512()

    operations = [
        lambda x: ((x * 251) ^ (x << 1)) & 0xFF,
        lambda x: ((x * 127) ^ (x >> 3)) & 0xFF,
        lambda x: ((x * 193) ^ rotate_left_64(x, 5)) & 0xFF,
        lambda x: ((x ^ 0xAA) * 167) & 0xFF
    ]

    # Process each 64-bit part with enhanced diffusion
    for i in range(8):
        x = state.parts[i]

        # Apply operation sequence to each byte
        byte_result = 0
        for j in range(8):
            byte_val = (x >> (j * 8)) & 0xFF
            for op in operations:
                byte_val = op(byte_val)
            byte_result |= byte_val << (j * 8)
        x = byte_result

        # Enhanced mixing with dynamic rotations
        rotation_pattern = [8, 16, 24, 32, 40, 48]
        for shift in rotation_pattern:
            rotated = rotate_left_64(x, shift)
            x = ((x * PRIME3) ^ (x & rotated)) & MASK64

        for shift in rotation_pattern:
            rotated = rotate_right_64(x, shift)
            x = ((x * PRIME4) | (x & rotated)) & MASK64

        x = (x * PRIME5) & MASK64
        result.parts[i] = x

    # Enhanced cross-part diffusion for 512-bit
    for i in range(8):
        prev_part = result.parts[(i - 1) % 8]
        next_part = result.parts[(i + 1) % 8]
        far_part = result.parts[(i + 4) % 8]

        result.parts[i] = ((result.parts[i] * prev_part) ^
                          rotate_left_64(result.parts[i], 23) ^
                          next_part ^ rotate_right_64(far_part, 11)) & MASK64

    return result

def generate_dynamic_sbox(seed):
    """Generate a non-linear S-box with enhanced randomness"""
    sbox = list(range(256))
    state = nonlinear_mix_64(seed)
    
    # Multiple passes for better randomness
    for pass_num in range(3):
        for i in range(255, 0, -1):
            state = nonlinear_mix_64(state + i + pass_num * PRIME1)
            j = state % (i + 1)
            sbox[i], sbox[j] = sbox[j], sbox[i]
    
    return sbox

def generate_inverse_sbox(sbox):
    inv_sbox = [0] * 256
    for i, v in enumerate(sbox):
        inv_sbox[v] = i
    return inv_sbox

def calculate_round_params_512(data, text_seed=None):
    """Enhanced round calculation for 512-bit security with text-based seeding"""
    length = len(data)
    checksum = sum(data) & MASK64
    xor_all = 0
    for b in data:
        xor_all ^= b

    # Use text seed if provided for dynamic behavior
    if text_seed:
        seed_hash = sum(ord(c) for c in text_seed) & MASK64
        base_rounds = 64 + (seed_hash % 32)
    else:
        base_rounds = 64

    length_factor = (length % 16) + 1
    checksum_factor = (checksum % 17)
    xor_factor = (xor_all % 17)

    num_rounds = base_rounds + length_factor + checksum_factor + xor_factor
    num_rounds = max(64, min(128, num_rounds))

    # Enhanced rotation schedule with text influence
    rotation_base = nonlinear_mix_64(checksum ^ (length * PRIME1))
    if text_seed:
        text_bytes = text_seed.encode()
        rotation_base ^= sum(text_bytes[:8])  # Use first 8 bytes
    
    rotation_schedule = []
    for i in range(num_rounds):
        rotation_base = nonlinear_mix_64(rotation_base + i)
        rotation_schedule.append((rotation_base % 127) + 1)

    # Enhanced key generation with text influence
    key_base = nonlinear_mix_64(xor_all ^ (checksum * PRIME2))
    if text_seed:
        text_bytes = text_seed.encode()
        key_base ^= sum(text_bytes[:8])  # Use first 8 bytes
    
    round_keys = []
    for i in range(num_rounds):
        key_base = nonlinear_mix_64(key_base ^ (i * PRIME3))
        key_parts = []
        for j in range(8):
            key_base = nonlinear_mix_64(key_base + j)
            key_parts.append(key_base)
        round_keys.append(State512(key_parts))

    return num_rounds, rotation_schedule, round_keys

def super_nonlinear_round_512(state, input_byte, round_key, rotation, sbox, inv_sbox, prime_manager):
    """Fully non-linear round function for 512-bit with enhanced operations"""
    global PRIME1, PRIME2, PRIME3, PRIME5
    
    # Convert input byte to 512-bit influence with more operations
    byte_spread = State512()
    for i in range(8):
        # Multiple operations for better diffusion
        spread_val = (input_byte * PRIME1 * (i + 1)) & MASK64
        spread_val ^= rotate_left_64(input_byte, i * 13)
        spread_val = (spread_val * PRIME2) & MASK64
        spread_val ^= rotate_right_64(input_byte, i * 7)
        byte_spread.parts[i] = spread_val

    # Layer 1: Non-linear mixing with input
    temp_state = State512()
    for i in range(8):
        temp_state.parts[i] = ((state.parts[i] * PRIME3) ^ byte_spread.parts[i]) & MASK64

    # Layer 2: Full non-linear diffusion
    temp_state = full_nonlinear_diffusion_512(temp_state)

    # Additional operation layer
    cross_op = CrossPartOperation()
    temp_state = cross_op.apply(temp_state, round_key)

    # Layer 3: Key mixing with non-linear operations
    for i in range(8):
        temp_state.parts[i] = ((temp_state.parts[i] * round_key.parts[i]) ^
                              rotate_left_64(temp_state.parts[i], rotation)) & MASK64

    # Layer 4: S-box on each byte with cross-byte dependencies
    sbox_op = SBoxOperation(sbox, inv_sbox)
    temp_state = sbox_op.apply(temp_state, round_key)

    # Prime-based operations
    prime_op = PrimeBasedOperation(prime_manager, rotation % 8)
    temp_state = prime_op.apply(temp_state, round_key)

    # Layer 5: Final non-linear mixing
    temp_state = full_nonlinear_diffusion_512(temp_state)
    for i in range(8):
        temp_state.parts[i] = (temp_state.parts[i] * PRIME5) & MASK64

    return temp_state

def cross_mix_nonlinear_512(states, prime_manager):
    """Non-linear cross-mixing of multiple 512-bit states with enhanced operations"""
    n = len(states)
    if n == 1:
        return states

    new_states = []
    for i in range(n):
        mixed = states[i].copy()
        for j in range(n):
            if i != j:
                for k in range(8):
                    # Enhanced mixing with prime operations
                    prime_op = prime_manager.get_prime_operation((i + j + k) % 8, State512())
                    temp = prime_op(states[j].parts[k])
                    mixed.parts[k] = ((mixed.parts[k] & temp) ^
                                     rotate_left_64(temp, (i * 7 + j * 11 + k * 3) % 64)) & MASK64
                    mixed.parts[k] = (mixed.parts[k] * PRIME1) & MASK64
        mixed = full_nonlinear_diffusion_512(mixed)
        new_states.append(mixed)

    return new_states

def final_compression_nonlinear_512(states, round_keys, prime_manager):
    """Compress multiple 512-bit states to single 512-bit output with enhanced operations"""
    result = State512()

    # Non-linear combination of all states
    for i, s in enumerate(states):
        rotated = rotate_left_512(s, i * 13)
        for j in range(8):
            result.parts[j] = ((result.parts[j] * PRIME1) ^ (s.parts[j] & rotated.parts[j])) & MASK64

    # Enhanced compression rounds for 512-bit
    for i, key in enumerate(round_keys[:16]):
        # Alternate between different operations
        if i % 3 == 0:
            # Prime-based operation
            prime_op = PrimeBasedOperation(prime_manager, i % 8)
            result = prime_op.apply(result, key)
        elif i % 3 == 1:
            # Cross-part operation
            cross_op = CrossPartOperation()
            result = cross_op.apply(result, key)
        else:
            # Standard diffusion
            result = full_nonlinear_diffusion_512(result)
            
        result = rotate_left_512(result, (i * 11) % 128)

    # Final passes with varied operations
    for i in range(12):
        if i % 2 == 0:
            result = full_nonlinear_diffusion_512(result)
        else:
            prime_op = PrimeBasedOperation(prime_manager, i % 8)
            result = prime_op.apply(result, round_keys[i % len(round_keys)])
        
        for j in range(8):
            result.parts[j] = (result.parts[j] * PRIME5) & MASK64

    return result

def correlation_operation_dynamic_512(x, y, round_key, prime_manager):
    """Non-linear correlation operation for 512-bit with enhanced operations"""
    try:
        # Fix 5: Handle math domain errors properly
        y_abs = abs(y) if y != -1 else 1
        x_abs = abs(x)
        
        # More robust mathematical operations
        base = (x_abs**2 * math.log(y_abs + 1) +
                math.sin(x_abs) * math.sqrt(y_abs + 1) +
                math.cos(y_abs) * math.exp(-x_abs/1000)) / (1 + x_abs + y_abs)
        int_base = int(abs(base * 1000000)) & MASK64

        base_512 = State512()
        for i in range(8):
            base_512.parts[i] = nonlinear_mix_64(int_base + i * PRIME1) ^ round_key.parts[i]

        # Apply prime operations
        prime_op = PrimeBasedOperation(prime_manager, int_base % 8)
        mixed = prime_op.apply(base_512, round_key)
        mixed = nonlinear_mix_512(mixed)
        return full_nonlinear_diffusion_512(mixed)
    except (ValueError, ZeroDivisionError, OverflowError):
        # Fallback operation
        base_512 = State512()
        for i in range(8):
            base_512.parts[i] = ((x & MASK64) * (y & MASK64)) ^ round_key.parts[i]
        return full_nonlinear_diffusion_512(base_512)

def process_text_512_bit(text):
    """Complete 512-bit hash function with full avalanche effect and dynamic primes"""
    if len(text) < 2:
        raise ValueError("Text must have at least 2 characters")

    # Fix 6: Generate dynamic primes based on input text
    dynamic_primes = generate_dynamic_primes(text)
    global PRIME1, PRIME2, PRIME3, PRIME4, PRIME5, PRIME6, PRIME7
    PRIME1, PRIME2, PRIME3, PRIME4, PRIME5, PRIME6, PRIME7 = dynamic_primes[:7]

    # Initialize prime manager
    prime_manager = DynamicPrimeManager()

    ascii_numbers = [ord(char) for char in text]
    print(f"1. ASCII: {ascii_numbers}")

    num_rounds, rotation_schedule, round_keys = calculate_round_params_512(ascii_numbers, text)
    print(f"2. Rounds: {num_rounds}")

    # Generate S-boxes with unique seed
    sbox_seed = sum((b * (i + 1) * PRIME1) for i, b in enumerate(ascii_numbers)) & MASK64
    sbox = generate_dynamic_sbox(sbox_seed)
    inv_sbox = generate_inverse_sbox(sbox)

    # Initialize 16 independent 512-bit states
    num_lanes = 16
    states = []
    for i in range(num_lanes):
        seed_base = (PRIME1 * (i + 1) + sbox_seed + len(text) + PRIME2 * i) & MASK64
        lane_parts = []
        for j in range(8):
            part_seed = nonlinear_mix_64(seed_base + j * PRIME3)
            lane_parts.append(part_seed)
        states.append(State512(lane_parts))

    print(f"3. Initial states (512-bit):")
    for i, s in enumerate(states[:4]):
        print(f"   Lane {i}: {s}")

    # PHASE 1: Absorption with enhanced operations
    for byte_idx, byte_val in enumerate(ascii_numbers):
        lane = byte_idx % num_lanes
        rotation = rotation_schedule[byte_idx % num_rounds]
        key = round_keys[byte_idx % num_rounds]

        states[lane] = super_nonlinear_round_512(
            states[lane], byte_val, key, rotation, sbox, inv_sbox, prime_manager
        )

        if (byte_idx + 1) % 4 == 0:
            states = cross_mix_nonlinear_512(states, prime_manager)

    print(f"4. After absorption: {[str(s) for s in states[:2]]}...")

    # PHASE 2: Extended mixing with enhanced operations
    for r in range(num_rounds):
        for lane in range(num_lanes):
            states[lane] = full_nonlinear_diffusion_512(states[lane])
            states[lane] = rotate_left_512(states[lane], rotation_schedule[r])
            
            # Alternate between operation types
            if r % 4 == 0:
                prime_op = PrimeBasedOperation(prime_manager, r % 8)
                states[lane] = prime_op.apply(states[lane], round_keys[r])
            else:
                for i in range(8):
                    states[lane].parts[i] = ((states[lane].parts[i] * round_keys[r].parts[i]) ^
                                           rotate_left_64(states[lane].parts[i], rotation_schedule[r])) & MASK64

        states = cross_mix_nonlinear_512(states, prime_manager)

    print(f"5. After mixing: {[str(s) for s in states[:2]]}...")

    # PHASE 3: Additional processing with enhanced operations
    processed_numbers = ascii_numbers.copy()
    for i in range(len(processed_numbers) - 1):
        # Fix 7: Handle division by zero
        divisor = processed_numbers[i + 1] if processed_numbers[i + 1] != 0 else 1
        modulo_result = processed_numbers[i] % divisor
        decimal_value = modulo_result / 100.0
        multiplied = processed_numbers[i] * decimal_value
        processed_numbers[i] = int(processed_numbers[i]) | int(multiplied)

        for lane in range(num_lanes):
            injection_val = (processed_numbers[i] * round_keys[i % num_rounds].parts[0]) & MASK64
            injection = State512([injection_val] * 8)
            
            # Use prime operations for injection
            prime_op = PrimeBasedOperation(prime_manager, i % 8)
            injection = prime_op.apply(injection, round_keys[i % num_rounds])
            
            for j in range(8):
                states[lane].parts[j] = ((states[lane].parts[j] * injection.parts[j]) ^
                                       injection.parts[j]) & MASK64
            states[lane] = full_nonlinear_diffusion_512(states[lane])

    # PHASE 4: Compression with enhanced operations
    state = final_compression_nonlinear_512(states, round_keys, prime_manager)
    print(f"6. After compression: {state}")

    # PHASE 5: Correlation with enhanced operations
    division_result = state.to_int() // 16
    remainder = state.to_int() % 16
    correlation_key = State512()
    for i in range(8):
        correlation_key.parts[i] = (round_keys[0].parts[i] * round_keys[-1].parts[i]) & MASK64

    correlation_output = correlation_operation_dynamic_512(
        division_result % 10000, remainder, correlation_key, prime_manager
    )

    for i in range(8):
        state.parts[i] = ((state.parts[i] * correlation_output.parts[i]) ^
                         correlation_output.parts[i]) & MASK64

    # PHASE 6: Final diffusion with enhanced operations
    for i in range(16):
        # Vary operations in final rounds
        if i % 3 == 0:
            state = full_nonlinear_diffusion_512(state)
        elif i % 3 == 1:
            prime_op = PrimeBasedOperation(prime_manager, i % 8)
            state = prime_op.apply(state, round_keys[i % len(round_keys)])
        else:
            state = nonlinear_mix_512(state)
            
        for j in range(8):
            state.parts[j] = (state.parts[j] * PRIME1) & MASK64
        state = rotate_left_512(state, 17 + i)

    hex_result = str(state)
    print(f"7. Final 512-bit hash: {hex_result}")

    # Verify 512-bit usage
    print(f"8. Bit distribution check:")
    for i in range(8):
        part_hex = format(state.parts[i], '016x')
        print(f"   Part {i}: {part_hex}")

    return {
        'ascii_numbers': ascii_numbers,
        'processed_numbers': processed_numbers,
        'num_rounds': num_rounds,
        'hexadecimal_result': hex_result,
        'hash_int': state.to_int()
    }

# Fix 8: Add missing utility functions
def measure_avalanche_512(text):
    """Measure avalanche effect by changing one bit"""
    original_result = process_text_512_bit(text)
    original_hash = original_result['hexadecimal_result']
    
    # Create a modified version (change last character)
    modified_text = text[:-1] + chr(ord(text[-1]) ^ 1)
    modified_result = process_text_512_bit(modified_text)
    modified_hash = modified_result['hexadecimal_result']
    
    # Calculate bit difference
    original_bits = bin(int(original_hash, 16))[2:].zfill(512)
    modified_bits = bin(int(modified_hash, 16))[2:].zfill(512)
    
    diff_count = sum(1 for a, b in zip(original_bits, modified_bits) if a != b)
    diff_percentage = (diff_count / 512) * 100
    
    print(f"\nAvalanche Effect Measurement:")
    print(f"Original:  {original_hash}")
    print(f"Modified:  {modified_hash}")
    print(f"Bits changed: {diff_count}/512 ({diff_percentage:.2f}%)")
    
    return diff_percentage

def compare_hashes_512():
    """Compare hashes of two different strings"""
    text1 = input("Enter first string: ")
    text2 = input("Enter second string: ")
    
    if len(text1) < 2 or len(text2) < 2:
        print("Both strings must be at least 2 characters long")
        return
    
    hash1 = process_text_512_bit(text1)['hexadecimal_result']
    hash2 = process_text_512_bit(text2)['hexadecimal_result']
    
    print(f"\nHash Comparison:")
    print(f"'{text1}' -> {hash1}")
    print(f"'{text2}' -> {hash2}")
    print(f"Same hash: {hash1 == hash2}")

def find_collision_512bit():
    """Attempt to find hash collisions (for testing)"""
    print("Searching for collisions...")
    start_time = time.time()
    hashes = {}
    
    for i in range(1000):  # Limit for demonstration
        test_str = f"test{i}"
        hash_val = process_text_512_bit(test_str)['hexadecimal_result']
        
        if hash_val in hashes:
            print(f"Collision found!")
            print(f"'{hashes[hash_val]}' and '{test_str}' both hash to: {hash_val}")
            return
        
        hashes[hash_val] = test_str
    
    end_time = time.time()
    print(f"No collisions found in {len(hashes)} hashes (time: {end_time - start_time:.2f}s)")

def main():
    print(f"\n{'='*80}")
    print("ENHANCED 512-BIT HASH FUNCTION WITH DYNAMIC PRIMES")
    print(f"{'='*80}")
    
    choice = input("Choose an option:\n 1. Avalanche Measurement\n 2. Hash Comparison\n 3. Collision Detection\n 4. File Hashing\nEnter choice (1/2/3/4): ")
    
    if choice == '1':
        user_input = input("\nEnter a string to measure avalanche effect (at least 2 characters): ")
        if len(user_input) >= 2:
            measure_avalanche_512(user_input)
        else:
            print("Input must be at least 2 characters long.")
    elif choice == '2':
        compare_hashes_512()
    elif choice == '3':
        find_collision_512bit()
    elif choice == '4':
        file_path = input("Enter file path to hash: ")
        output_path = input("Enter output file path: ")
        try:
            with open(file_path, "r", encoding='utf-8') as f1:
                content = f1.read().strip()
            if len(content) >= 2:
                hash_result = process_text_512_bit(content)
                with open(output_path, "w") as f2:
                    f2.write(hash_result['hexadecimal_result'] + "\n")
                print(f"File hashed successfully. 512-bit hash saved to {output_path}")
            else:
                print("File content must be at least 2 characters long.")
        except Exception as e:
            print(f"Error processing file: {e}")
    else:
        user_input = input("\nEnter a string to hash (at least 2 characters): ")
        if len(user_input) >= 2:
            try:
                result = process_text_512_bit(user_input)
                print(f"'{user_input}' -> {result['hexadecimal_result']}")
            except Exception as e:
                print(f"Error: {e}")
        else:
            print("Input must be at least 2 characters long.")

if __name__ == "__main__":
    main()
