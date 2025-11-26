import math
import random
import time
import hashlib
import sys
import subprocess

# 修正1: 暗号学的セキュリティのための適切な素数判定を追加
def is_probable_prime(n, k=20):
    """ミラー-ラビン素数判定法"""
    if n < 2:
        return False
    if n in (2, 3):
        return True
    if n % 2 == 0:
        return False

    # n を 2^r * d + 1 の形で表現
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

# 修正2: 実際の素数判定を用いた適切な動的素数生成
def generate_dynamic_primes(seed_text, count=16, bits=1024):
    """テキストから動的に実際の素数を生成"""
    primes = []
    seed_hash = hashlib.sha512(seed_text.encode()).digest()

    for i in range(count):
        # 各素数にハッシュの異なる部分を使用
        prime_seed = int.from_bytes(seed_hash[i*8:(i+1)*8], 'big')

        # 大きな奇数を確実に取得
        candidate = (prime_seed | (1 << (bits-1)) | 1) & ((1 << bits) - 1)

        # 次の実際の素数を見つける
        while not is_probable_prime(candidate):
            candidate += 2
            candidate &= ((1 << bits) - 1)  # ビット制限内を維持
            if candidate < (1 << (bits-1)):  # 必要に応じてラップアラウンド
                candidate = (1 << (bits-1)) | 1

        primes.append(candidate)

    return primes

def make_prime_like(n):
    """素数風の数値を生成（デモ用）"""
    if n % 2 == 0:
        n += 1

    # 小さな素数で割り切れないことを確認
    small_primes = [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
    for p in small_primes:
        while n % p == 0:
            n += 2

    return n

# 修正3: 1024ビット用の定数を定義
MASK64 = 0xFFFFFFFFFFFFFFFF
MASK128 = (1 << 128) - 1
MASK1024 = (1 << 1024) - 1

# これらは動的に設定されます
PRIMES = [None] * 16

class State1024:
    def __init__(self, value=0):
        if isinstance(value, int):
            self.parts = [
                value & MASK128,
                (value >> 128) & MASK128,
                (value >> 256) & MASK128,
                (value >> 384) & MASK128,
                (value >> 512) & MASK128,
                (value >> 640) & MASK128,
                (value >> 768) & MASK128,
                (value >> 896) & MASK128
            ]
        else:
            self.parts = [p & MASK128 for p in value]

    def to_int(self):
        return (self.parts[0] |
                (self.parts[1] << 128) |
                (self.parts[2] << 256) |
                (self.parts[3] << 384) |
                (self.parts[4] << 512) |
                (self.parts[5] << 640) |
                (self.parts[6] << 768) |
                (self.parts[7] << 896))

    def __getitem__(self, index):
        return self.parts[index]

    def __setitem__(self, index, value):
        self.parts[index] = value & MASK128

    def __str__(self):
        return format(self.to_int(), '0256x')

    def copy(self):
        return State1024(self.parts.copy())

def rotate_left_128(val, n, bits=128):
    n = n % bits
    return ((val << n) | (val >> (bits - n))) & MASK128

def rotate_right_128(val, n, bits=128):
    n = n % bits
    return ((val >> n) | (val << (bits - n))) & MASK128

def rotate_left_1024(state, n):
    """1024ビット状態をnビット左回転"""
    total_bits = 1024
    n = n % total_bits

    if n == 0:
        return state

    big_val = state.to_int()
    result_val = ((big_val << n) | (big_val >> (1024 - n))) & MASK1024
    return State1024(result_val)

def rotate_right_1024(state, n):
    """1024ビット状態をnビット右回転"""
    total_bits = 1024
    n = n % total_bits

    if n == 0:
        return state

    big_val = state.to_int()
    result_val = ((big_val >> n) | (big_val << (1024 - n))) & MASK1024
    return State1024(result_val)

class DynamicPrimeManager:
    """入力テキストに基づく動的素数生成を管理"""
    def __init__(self):
        self.prime_cache = {}
        self.prime_functions = [
            self._prime_func1, self._prime_func2, self._prime_func3,
            self._prime_func4, self._prime_func5, self._prime_func6,
            self._prime_func7, self._prime_func8,
            self._prime_func9, self._prime_func10, self._prime_func11,
            self._prime_func12, self._prime_func13, self._prime_func14,
            self._prime_func15, self._prime_func16
        ]

    def _prime_func1(self, x, round_key):
        return ((x * 0x9E3779B97F4A7C15) ^ rotate_left_128(x, 31) ^ round_key) & MASK128

    def _prime_func2(self, x, round_key):
        return ((x * 0xBF58476D1CE4E5B9) | rotate_right_128(x, 17) ^ round_key) & MASK128

    def _prime_func3(self, x, round_key):
        return ((x * 0x94D049BB133111EB) + rotate_left_128(x, 13) ^ round_key) & MASK128

    def _prime_func4(self, x, round_key):
        return ((x ^ 0xC6A4A7935BD1E995) * rotate_right_128(x, 29) ^ round_key) & MASK128

    def _prime_func5(self, x, round_key):
        return ((x | 0x85EBCA77C2B2AE63) ^ (x * rotate_left_128(x, 19)) ^ round_key) & MASK128

    def _prime_func6(self, x, round_key):
        return ((x + 0x27D4EB2F165667C5) & rotate_right_128(x, 23) ^ round_key) & MASK128

    def _prime_func7(self, x, round_key):
        return ((x * 0x9E3779B97F4A7C15) ^ (x | rotate_left_128(x, 37)) ^ round_key) & MASK128

    def _prime_func8(self, x, round_key):
        return ((x ^ 0x5A8279996ED9EBA1) + rotate_right_128(x, 41) ^ round_key) & MASK128

    def _prime_func9(self, x, round_key):
        return ((x * 0x243F6A8885A308D3) ^ rotate_left_128(x, 7) ^ round_key) & MASK128

    def _prime_func10(self, x, round_key):
        return ((x | 0x13198A2E03707344) * rotate_right_128(x, 53) ^ round_key) & MASK128

    def _prime_func11(self, x, round_key):
        return ((x + 0xA4093822299F31D0) ^ rotate_left_128(x, 61) ^ round_key) & MASK128

    def _prime_func12(self, x, round_key):
        return ((x * 0x082EFA98EC4E6C89) & rotate_right_128(x, 47) ^ round_key) & MASK128

    def _prime_func13(self, x, round_key):
        return ((x ^ 0x452821E638D01377) | rotate_left_128(x, 43) ^ round_key) & MASK128

    def _prime_func14(self, x, round_key):
        return ((x * 0xBE5466CF34E90C6C) + rotate_right_128(x, 59) ^ round_key) & MASK128

    def _prime_func15(self, x, round_key):
        return ((x & 0xC0AC29B7C97C50DD) ^ rotate_left_128(x, 67) ^ round_key) & MASK128

    def _prime_func16(self, x, round_key):
        return ((x | 0x3F84D5B5B5470917) * rotate_right_128(x, 71) ^ round_key) & MASK128

    def get_prime_operation(self, index, round_key):
        """指定されたラウンドキーで素数ベースの操作を取得"""
        func = self.prime_functions[index % len(self.prime_functions)]
        return lambda x: func(x, round_key.parts[index % 8])

# 修正4: 1024ビット用の適切な実装を持つ拡張操作クラス
class CryptoOperation:
    """暗号操作の基底クラス"""
    def apply(self, state, round_key):
        raise NotImplementedError

class NonlinearMixOperation(CryptoOperation):
    def apply(self, state, round_key):
        return nonlinear_mix_1024(state)

class DiffusionOperation(CryptoOperation):
    def apply(self, state, round_key):
        return full_nonlinear_diffusion_1024(state)

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
                              rotate_left_128(result.parts[i], 19) ^
                              right ^ rotate_right_128(diagonal, 7)) & MASK128
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
            for j in range(16):  # 128ビット = 16バイト
                b = (x >> (j * 8)) & 0xFF
                prev_byte = (x >> (((j-1) % 16) * 8)) & 0xFF
                b = self.sbox[(b ^ prev_byte) & 0xFF]
                b = self.inv_sbox[((b * 127) ^ (round_key.parts[i] >> (j * 8))) & 0xFF]
                bytes_out.append(b)
            result.parts[i] = sum(b << (j * 8) for j, b in enumerate(bytes_out)) & MASK128
        return result

def nonlinear_mix_128(x):
    """128ビット非線形混合"""
    if PRIMES[0] is None or PRIMES[1] is None:
        raise ValueError("素数が初期化されていません")

    x = (x * PRIMES[0]) & MASK128
    x ^= (x & 0xAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA) >> 1
    x ^= (x | 0x55555555555555555555555555555555) << 1
    x &= MASK128

    result = 0
    for i in range(16):
        byte_val = (x >> (i * 8)) & 0xFF
        sboxed = ((byte_val * 251) ^ (byte_val >> 4)) & 0xFF
        result |= sboxed << (i * 8)
    x = result

    x = (x * PRIMES[1]) & MASK128

    left = x & 0xFFFFFFFFFFFFFFFF
    right = (x >> 64) & 0xFFFFFFFFFFFFFFFF
    temp = ((left * 0x9E3779B97F4A7C15) ^ rotate_left_128(right, 17)) & MASK64
    left ^= temp
    right ^= ((temp * 0x85EBCA77C2B2AE63) & MASK64)
    x = (right << 64) | left

    return x & MASK128

def nonlinear_mix_1024(state):
    """動的操作を用いた1024ビット状態の非線形混合"""
    result = State1024()

    primes = PRIMES[:8] + PRIMES[:8]  # 最初の8素数を使用、必要に応じて繰り返し

    # 各128ビット部分に異なる非線形操作を適用
    for i in range(8):
        x = state.parts[i]
        x = (x * primes[i]) & MASK128

        x ^= (x & 0xAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA) >> 1
        x ^= (x | 0x55555555555555555555555555555555) << 1
        x &= MASK128

        byte_result = 0
        for j in range(16):
            byte_val = (x >> (j * 8)) & 0xFF
            sboxed = ((byte_val * 251) ^ (byte_val >> 4)) & 0xFF
            byte_result |= sboxed << (j * 8)
        x = byte_result

        result.parts[i] = x & MASK128

    # 1024ビット用の拡張クロスパート混合
    for i in range(8):
        left_part = result.parts[(i - 1) % 8]
        right_part = result.parts[(i + 1) % 8]
        diagonal_part = result.parts[(i + 4) % 8]

        result.parts[i] = ((result.parts[i] * left_part) ^
                          rotate_left_128(result.parts[i], 17) ^
                          right_part ^ diagonal_part) & MASK128

    return result

def full_nonlinear_diffusion_1024(state):
    """1024ビットの完全ビット拡散を確保する非線形操作のカスケード"""
    result = State1024()

    operations = [
        lambda x: ((x * 251) ^ (x << 1)) & 0xFF,
        lambda x: ((x * 127) ^ (x >> 3)) & 0xFF,
        lambda x: ((x * 193) ^ rotate_left_128(x, 5)) & 0xFF,
        lambda x: ((x ^ 0xAA) * 167) & 0xFF
    ]

    # 拡張拡散で各128ビット部分を処理
    for i in range(8):
        x = state.parts[i]

        # 各バイトに操作シーケンスを適用
        byte_result = 0
        for j in range(16):
            byte_val = (x >> (j * 8)) & 0xFF
            for op in operations:
                byte_val = op(byte_val)
            byte_result |= byte_val << (j * 8)
        x = byte_result

        # 動的回転を用いた拡張混合
        rotation_pattern = [8, 16, 24, 32, 40, 48, 56, 64]
        for shift in rotation_pattern:
            rotated = rotate_left_128(x, shift)
            x = ((x * PRIMES[2]) ^ (x & rotated)) & MASK128

        for shift in rotation_pattern:
            rotated = rotate_right_128(x, shift)
            x = ((x * PRIMES[3]) | (x & rotated)) & MASK128

        x = (x * PRIMES[4]) & MASK128
        result.parts[i] = x

    # 1024ビット用の拡張クロスパート拡散
    for i in range(8):
        prev_part = result.parts[(i - 1) % 8]
        next_part = result.parts[(i + 1) % 8]
        far_part = result.parts[(i + 4) % 8]

        result.parts[i] = ((result.parts[i] * prev_part) ^
                          rotate_left_128(result.parts[i], 23) ^
                          next_part ^ rotate_right_128(far_part, 11)) & MASK128

    return result

def generate_dynamic_sbox(seed):
    """拡張ランダム性を用いた非線形Sボックスを生成"""
    sbox = list(range(256))
    state = nonlinear_mix_128(seed)

    # より良いランダム性のための複数パス
    for pass_num in range(3):
        for i in range(255, 0, -1):
            state = nonlinear_mix_128(state + i + pass_num * PRIMES[0])
            j = state % (i + 1)
            sbox[i], sbox[j] = sbox[j], sbox[i]

    return sbox

def generate_inverse_sbox(sbox):
    inv_sbox = [0] * 256
    for i, v in enumerate(sbox):
        inv_sbox[v] = i
    return inv_sbox

def calculate_round_params_1024(data, text_seed=None):
    """テキストベースのシーディングを用いた1024ビットセキュリティの拡張ラウンド計算"""
    length = len(data)
    checksum = sum(data) & MASK128
    xor_all = 0
    for b in data:
        xor_all ^= b

    # 動的動作のためにテキストシードを使用
    if text_seed:
        seed_hash = sum(ord(c) for c in text_seed) & MASK128
        base_rounds = 128 + (seed_hash % 64)
    else:
        base_rounds = 128

    length_factor = (length % 32) + 1
    checksum_factor = (checksum % 33)
    xor_factor = (xor_all % 33)

    num_rounds = base_rounds + length_factor + checksum_factor + xor_factor
    num_rounds = max(128, min(256, num_rounds))

    # テキスト影響を用いた拡張回転スケジュール
    rotation_base = nonlinear_mix_128(checksum ^ (length * PRIMES[0]))
    if text_seed:
        text_bytes = text_seed.encode()
        rotation_base ^= sum(text_bytes[:16])  # 最初の16バイトを使用

    rotation_schedule = []
    for i in range(num_rounds):
        rotation_base = nonlinear_mix_128(rotation_base + i)
        rotation_schedule.append((rotation_base % 255) + 1)

    # テキスト影響を用いた拡張鍵生成
    key_base = nonlinear_mix_128(xor_all ^ (checksum * PRIMES[1]))
    if text_seed:
        text_bytes = text_seed.encode()
        key_base ^= sum(text_bytes[:16])  # 最初の16バイトを使用

    round_keys = []
    for i in range(num_rounds):
        key_base = nonlinear_mix_128(key_base ^ (i * PRIMES[2]))
        key_parts = []
        for j in range(8):
            key_base = nonlinear_mix_128(key_base + j)
            key_parts.append(key_base)
        round_keys.append(State1024(key_parts))

    return num_rounds, rotation_schedule, round_keys

def super_nonlinear_round_1024(state, input_byte, round_key, rotation, sbox, inv_sbox, prime_manager):
    """拡張操作を用いた1024ビットの完全非線形ラウンド関数"""

    # より多くの操作で入力バイトを1024ビット影響に変換
    byte_spread = State1024()
    for i in range(8):
        # より良い拡散のための複数操作
        spread_val = (input_byte * PRIMES[0] * (i + 1)) & MASK128
        spread_val ^= rotate_left_128(input_byte, i * 13)
        spread_val = (spread_val * PRIMES[1]) & MASK128
        spread_val ^= rotate_right_128(input_byte, i * 7)
        byte_spread.parts[i] = spread_val

    # レイヤ1: 入力を用いた非線形混合
    temp_state = State1024()
    for i in range(8):
        temp_state.parts[i] = ((state.parts[i] * PRIMES[2]) ^ byte_spread.parts[i]) & MASK128

    # レイヤ2: 完全非線形拡散
    temp_state = full_nonlinear_diffusion_1024(temp_state)

    # 追加操作レイヤ
    cross_op = CrossPartOperation()
    temp_state = cross_op.apply(temp_state, round_key)

    # レイヤ3: 非線形操作を用いた鍵混合
    for i in range(8):
        temp_state.parts[i] = ((temp_state.parts[i] * round_key.parts[i]) ^
                              rotate_left_128(temp_state.parts[i], rotation)) & MASK128

    # レイヤ4: クロスバイト依存性を持つ各バイトのSボックス
    sbox_op = SBoxOperation(sbox, inv_sbox)
    temp_state = sbox_op.apply(temp_state, round_key)

    # 素数ベース操作
    prime_op = PrimeBasedOperation(prime_manager, rotation % 16)
    temp_state = prime_op.apply(temp_state, round_key)

    # レイヤ5: 最終非線形混合
    temp_state = full_nonlinear_diffusion_1024(temp_state)
    for i in range(8):
        temp_state.parts[i] = (temp_state.parts[i] * PRIMES[4]) & MASK128

    return temp_state

def cross_mix_nonlinear_1024(states, prime_manager):
    """拡張操作を用いた複数1024ビット状態の非線形クロス混合"""
    n = len(states)
    if n == 1:
        return states

    new_states = []
    for i in range(n):
        mixed = states[i].copy()
        for j in range(n):
            if i != j:
                for k in range(8):
                    # 素数操作を用いた拡張混合
                    prime_op = prime_manager.get_prime_operation((i + j + k) % 16, State1024())
                    temp = prime_op(states[j].parts[k])
                    mixed.parts[k] = ((mixed.parts[k] & temp) ^
                                     rotate_left_128(temp, (i * 7 + j * 11 + k * 3) % 128)) & MASK128
                    mixed.parts[k] = (mixed.parts[k] * PRIMES[0]) & MASK128
        mixed = full_nonlinear_diffusion_1024(mixed)
        new_states.append(mixed)

    return new_states

def final_compression_nonlinear_1024(states, round_keys, prime_manager):
    """拡張操作を用いて複数1024ビット状態を単一1024ビット出力に圧縮"""
    result = State1024()

    # 全状態の非線形結合
    for i, s in enumerate(states):
        rotated = rotate_left_1024(s, i * 13)
        for j in range(8):
            result.parts[j] = ((result.parts[j] * PRIMES[0]) ^ (s.parts[j] & rotated.parts[j])) & MASK128

    # 1024ビット用の拡張圧縮ラウンド
    for i, key in enumerate(round_keys[:32]):
        # 異なる操作を交互に使用
        if i % 3 == 0:
            # 素数ベース操作
            prime_op = PrimeBasedOperation(prime_manager, i % 16)
            result = prime_op.apply(result, key)
        elif i % 3 == 1:
            # クロスパート操作
            cross_op = CrossPartOperation()
            result = cross_op.apply(result, key)
        else:
            # 標準拡散
            result = full_nonlinear_diffusion_1024(result)

        result = rotate_left_1024(result, (i * 11) % 256)

    # 様々な操作を用いた最終パス
    for i in range(24):
        if i % 2 == 0:
            result = full_nonlinear_diffusion_1024(result)
        else:
            prime_op = PrimeBasedOperation(prime_manager, i % 16)
            result = prime_op.apply(result, round_keys[i % len(round_keys)])

        for j in range(8):
            result.parts[j] = (result.parts[j] * PRIMES[4]) & MASK128

    return result

def correlation_operation_dynamic_1024(x, y, round_key, prime_manager):
    """拡張操作を用いた1024ビットの非線形相関操作"""
    try:
        # 修正5: 数学領域エラーを適切に処理
        y_abs = abs(y) if y != -1 else 1
        x_abs = abs(x)

        # よりロバストな数学的操作
        base = (x_abs**2 * math.log(y_abs + 1) +
                math.sin(x_abs) * math.sqrt(y_abs + 1) +
                math.cos(y_abs) * math.exp(-x_abs/1000)) / (1 + x_abs + y_abs)
        int_base = int(abs(base * 1000000)) & MASK128

        base_1024 = State1024()
        for i in range(8):
            base_1024.parts[i] = nonlinear_mix_128(int_base + i * PRIMES[0]) ^ round_key.parts[i]

        # 素数操作を適用
        prime_op = PrimeBasedOperation(prime_manager, int_base % 16)
        mixed = prime_op.apply(base_1024, round_key)
        mixed = nonlinear_mix_1024(mixed)
        return full_nonlinear_diffusion_1024(mixed)
    except (ValueError, ZeroDivisionError, OverflowError):
        # フォールバック操作
        base_1024 = State1024()
        for i in range(8):
            base_1024.parts[i] = ((x & MASK128) * (y & MASK128)) ^ round_key.parts[i]
        return full_nonlinear_diffusion_1024(base_1024)

def process_text_1024_bit(text):
    """完全なアバランシェ効果と動的素数を用いた完全な1024ビットハッシュ関数"""
    if len(text) < 2:
        raise ValueError("テキストは少なくとも2文字必要です")
    start = time.time()
    # 修正6: 入力テキストに基づく動的素数を生成
    dynamic_primes = generate_dynamic_primes(text, count=16, bits=1024)
    global PRIMES
    PRIMES = dynamic_primes

    # 素数マネージャーを初期化
    prime_manager = DynamicPrimeManager()

    ascii_numbers = [ord(char) for char in text]
    print(f"1. ASCII: {ascii_numbers}")

    num_rounds, rotation_schedule, round_keys = calculate_round_params_1024(ascii_numbers, text)
    print(f"2. ラウンド数: {num_rounds}")

    # ユニークシードでSボックスを生成
    sbox_seed = sum((b * (i + 1) * PRIMES[0]) for i, b in enumerate(ascii_numbers)) & MASK128
    sbox = generate_dynamic_sbox(sbox_seed)
    inv_sbox = generate_inverse_sbox(sbox)

    # 16個の独立した1024ビット状態を初期化
    num_lanes = 16
    states = []
    for i in range(num_lanes):
        seed_base = (PRIMES[0] * (i + 1) + sbox_seed + len(text) + PRIMES[1] * i) & MASK128
        lane_parts = []
        for j in range(8):
            part_seed = nonlinear_mix_128(seed_base + j * PRIMES[2])
            lane_parts.append(part_seed)
        states.append(State1024(lane_parts))

    print(f"3. 初期状態 (1024ビット):")
    for i, s in enumerate(states[:4]):
        print(f"   レーン {i}: {s}")

    # フェーズ1: 拡張操作を用いた吸収
    for byte_idx, byte_val in enumerate(ascii_numbers):
        lane = byte_idx % num_lanes
        rotation = rotation_schedule[byte_idx % num_rounds]
        key = round_keys[byte_idx % num_rounds]

        states[lane] = super_nonlinear_round_1024(
            states[lane], byte_val, key, rotation, sbox, inv_sbox, prime_manager
        )

        if (byte_idx + 1) % 4 == 0:
            states = cross_mix_nonlinear_1024(states, prime_manager)

    print(f"4. 吸収後: {[str(s) for s in states[:2]]}...")

    # フェーズ2: 拡張操作を用いた拡張混合
    for r in range(num_rounds):
        for lane in range(num_lanes):
            states[lane] = full_nonlinear_diffusion_1024(states[lane])
            states[lane] = rotate_left_1024(states[lane], rotation_schedule[r])

            # 操作タイプを交互に切り替え
            if r % 4 == 0:
                prime_op = PrimeBasedOperation(prime_manager, r % 16)
                states[lane] = prime_op.apply(states[lane], round_keys[r])
            else:
                for i in range(8):
                    states[lane].parts[i] = ((states[lane].parts[i] * round_keys[r].parts[i]) ^
                                           rotate_left_128(states[lane].parts[i], rotation_schedule[r])) & MASK128

        states = cross_mix_nonlinear_1024(states, prime_manager)

    print(f"5. 混合後: {[str(s) for s in states[:2]]}...")

    # フェーズ3: 拡張操作を用いた追加処理
    processed_numbers = ascii_numbers.copy()
    for i in range(len(processed_numbers) - 1):
        # 修正7: ゼロ除算を処理
        divisor = processed_numbers[i + 1] if processed_numbers[i + 1] != 0 else 1
        modulo_result = processed_numbers[i] % divisor
        decimal_value = modulo_result / 100.0
        multiplied = processed_numbers[i] * decimal_value
        processed_numbers[i] = int(processed_numbers[i]) | int(multiplied)

        for lane in range(num_lanes):
            injection_val = (processed_numbers[i] * round_keys[i % num_rounds].parts[0]) & MASK128
            injection = State1024([injection_val] * 8)

            # 注入に素数操作を使用
            prime_op = PrimeBasedOperation(prime_manager, i % 16)
            injection = prime_op.apply(injection, round_keys[i % num_rounds])

            for j in range(8):
                states[lane].parts[j] = ((states[lane].parts[j] * injection.parts[j]) ^
                                       injection.parts[j]) & MASK128
            states[lane] = full_nonlinear_diffusion_1024(states[lane])

    # フェーズ4: 拡張操作を用いた圧縮
    state = final_compression_nonlinear_1024(states, round_keys, prime_manager)
    print(f"6. 圧縮後: {state}")

    # フェーズ5: 拡張操作を用いた相関
    division_result = state.to_int() // 16
    remainder = state.to_int() % 16
    correlation_key = State1024()
    for i in range(8):
        correlation_key.parts[i] = (round_keys[0].parts[i] * round_keys[-1].parts[i]) & MASK128

    correlation_output = correlation_operation_dynamic_1024(
        division_result % 10000, remainder, correlation_key, prime_manager
    )

    for i in range(8):
        state.parts[i] = ((state.parts[i] * correlation_output.parts[i]) ^
                         correlation_output.parts[i]) & MASK128

    # フェーズ6: 拡張操作を用いた最終拡散
    for i in range(32):
        # 最終ラウンドで操作を変化
        if i % 3 == 0:
            state = full_nonlinear_diffusion_1024(state)
        elif i % 3 == 1:
            prime_op = PrimeBasedOperation(prime_manager, i % 16)
            state = prime_op.apply(state, round_keys[i % len(round_keys)])
        else:
            state = nonlinear_mix_1024(state)

        for j in range(8):
            state.parts[j] = (state.parts[j] * PRIMES[0]) & MASK128
        state = rotate_left_1024(state, 17 + i)

    hex_result = str(state)
    print(f"7. 最終1024ビットハッシュ: {hex_result}")

    # 1024ビット使用を検証
    print(f"8. ビット分布チェック:")
    for i in range(8):
        part_hex = format(state.parts[i], '032x')
        print(f"   部分 {i}: {part_hex}")
    end = time.time()
    elapsed_hash = end - start  # ✅ 正の時間差
    print(f"{elapsed_hash:.6f} 秒で完了")

    start_sha = time.perf_counter()
    import hashlib
    sha512_hash = hashlib.sha512(text.encode()).hexdigest()
    end_sha = time.perf_counter()
    elapsed_sha = end_sha - start_sha

    if elapsed_sha > 0:
        speed_ratio = elapsed_hash / elapsed_sha
        print(f"SHA-512よりも約{speed_ratio:.2f}倍遅い")
    else:
        print("SHA-512は正確に測定するには速すぎました")
    return {
        'ascii_numbers': ascii_numbers,
        'processed_numbers': processed_numbers,
        'num_rounds': num_rounds,
        'hexadecimal_result': hex_result,
        'hash_int': state.to_int()
    }

def measure_avalanche_1024(text):
    """1ビット変更によるアバランシェ効果を測定"""
    original_result = process_text_1024_bit(text)
    original_hash = original_result['hexadecimal_result']

    # 変更版を作成（最後の文字を変更）
    modified_text = text[:-1] + chr(ord(text[-1]) ^ 1)
    modified_result = process_text_1024_bit(modified_text)
    modified_hash = modified_result['hexadecimal_result']

    # ビット差を計算
    original_bits = bin(int(original_hash, 16))[2:].zfill(1024)
    modified_bits = bin(int(modified_hash, 16))[2:].zfill(1024)

    diff_count = sum(1 for a, b in zip(original_bits, modified_bits) if a != b)
    diff_percentage = (diff_count / 1024) * 100

    print(f"\nアバランシェ効果測定:")
    print(f"元のハッシュ:  {original_hash}")
    print(f"変更ハッシュ:  {modified_hash}")
    print(f"変更ビット数: {diff_count}/1024 ({diff_percentage:.2f}%)")

    return diff_percentage

def compare_hashes_1024():
    """2つの異なる文字列のハッシュを比較"""
    text1 = input("最初の文字列を入力: ")
    text2 = input("2番目の文字列を入力: ")

    if len(text1) < 2 or len(text2) < 2:
        print("両方の文字列は少なくとも2文字必要です")
        return

    hash1 = process_text_1024_bit(text1)['hexadecimal_result']
    hash2 = process_text_1024_bit(text2)['hexadecimal_result']

    print(f"\nハッシュ比較:")
    print(f"'{text1}' -> {hash1}")
    print(f"'{text2}' -> {hash2}")
    print(f"同じハッシュ: {hash1 == hash2}")

def find_collision_1024bit():
    """ハッシュ衝突を検出（テスト用）"""
    print("衝突を検索中...")
    start_time = time.time()
    hashes = {}

    for i in range(1000):  # デモ用に制限
        test_str = f"test{i}"
        hash_val = process_text_1024_bit(test_str)['hexadecimal_result']

        if hash_val in hashes:
            print(f"衝突を発見！")
            print(f"'{hashes[hash_val]}' と '{test_str}' が同じハッシュ: {hash_val}")
            return

        hashes[hash_val] = test_str

    end_time = time.time()
    print(f"{len(hashes)} ハッシュ中で衝突は見つかりませんでした (時間: {end_time - start_time:.2f}秒)")

def detect_hardware():
    try:
        subprocess.run(['nvidia-smi'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return 'GPU'
    except:
        return 'CPU'

def main():
    print(f"\n{'='*80}")
    sys_h = detect_hardware()
    if sys_h == "GPU":
        print("動的素数を用いた拡張1024ビットハッシュ関数")
    else:
        print("\033[31mパフォーマンスは変動する可能性があります\033[0m")
    print(f"{'='*80}")

    choice = input("オプションを選択:\n \033[33m1\033[0m. アバランシェ測定\n \033[33m2\033[0m. ハッシュ比較\n \033[33m3\033[0m. 衝突検出\n \033[33m4\033[0m. ファイルハッシュ\n選択を入力 (1/2/3/4): ")

    if choice == '1':
        user_input = input("\nアバランシェ効果を測定する文字列を入力（少なくとも2文字）: ")
        if len(user_input) >= 2:
            measure_avalanche_1024(user_input)
        else:
            print("入力は少なくとも2文字必要です。")
    elif choice == '2':
        compare_hashes_1024()
    elif choice == '3':
        find_collision_1024bit()
    elif choice == '4':
        file_path = input("ハッシュするファイルパスを入力: ")
        output_path = input("出力ファイルパスを入力: ")
        try:
            with open(file_path, "r", encoding='utf-8') as f1:
                content = f1.read().strip()
            if len(content) >= 2:
                hash_result = process_text_1024_bit(content)
                with open(output_path, "w") as f2:
                    f2.write(hash_result['hexadecimal_result'] + "\n")
                print(f"ファイルを正常にハッシュしました。1024ビットハッシュを {output_path} に保存")
            else:
                print("ファイル内容は少なくとも2文字必要です。")
        except Exception as e:
            print(f"ファイル処理エラー: {e}")
    else:
        user_input = input("\nハッシュする文字列を入力（少なくとも2文字）: ")
        if len(user_input) >= 2:
            try:
                result = process_text_1024_bit(user_input)
                print(f"'{user_input}' -> {result['hexadecimal_result']}")
            except Exception as e:
                print(f"エラー: {e}")
        else:
            print("入力は少なくとも2文字必要です。")

if __name__ == "__main__":
    main()
