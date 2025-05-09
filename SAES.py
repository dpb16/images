# Simplified AES (S-AES) Implementation

# S-AES constants
# S-box for substitution
SBOX = [
    [0x9, 0x4, 0xA, 0xB],
    [0xD, 0x1, 0x8, 0x5],
    [0x6, 0x2, 0x0, 0x3],
    [0xC, 0xE, 0xF, 0x7]
]

# Inverse S-box for decryption
INV_SBOX = [
    [0xA, 0x5, 0x9, 0xB],
    [0x1, 0x7, 0x8, 0xF],
    [0x6, 0x0, 0x2, 0x3],
    [0xC, 0x4, 0xD, 0xE]
]

# MixColumns constant matrix
MIX_COL_MATRIX = [
    [1, 4],
    [4, 1]
]

# Inverse MixColumns constant matrix
INV_MIX_COL_MATRIX = [
    [9, 2],
    [2, 9]
]

# Galois Field multiplication for S-AES
def gf_mult(a, b):
    """Galois Field Multiplication in GF(2^4) with irreducible polynomial x^4 + x + 1."""
    product = 0
    for i in range(4):
        if (b & 1) == 1:
            product ^= a
        high_bit = a & 0x8
        a <<= 1
        if high_bit == 0x8:
            a ^= 0x13  # x^4 + x + 1 = 0b10011
        b >>= 1
    return product & 0xF

# Key Expansion
def key_expansion(key):
    """Expand the 16-bit key into two 16-bit round keys."""
    w = [0] * 6
    
    # Split the key into two 8-bit words
    w[0] = (key >> 8) & 0xFF
    w[1] = key & 0xFF
    
    # Round constants
    rcon1 = 0x80  # 10000000
    rcon2 = 0x30  # 00110000
    
    # Generate w[2] and w[3] (first round key)
    rot_w1 = ((w[1] << 4) & 0xF0) | ((w[1] >> 4) & 0x0F)
    sub_rot_w1 = substitute_word(rot_w1)
    w[2] = w[0] ^ rcon1 ^ sub_rot_w1
    w[3] = w[2] ^ w[1]
    
    # Generate w[4] and w[5] (second round key)
    rot_w3 = ((w[3] << 4) & 0xF0) | ((w[3] >> 4) & 0x0F)
    sub_rot_w3 = substitute_word(rot_w3)
    w[4] = w[2] ^ rcon2 ^ sub_rot_w3
    w[5] = w[4] ^ w[3]
    
    # Combine to form round keys
    round_key1 = (w[2] << 8) | w[3]
    round_key2 = (w[4] << 8) | w[5]
    
    return round_key1, round_key2

def substitute_word(word):
    """Apply S-box substitution to a byte."""
    high_nibble = (word >> 4) & 0xF
    low_nibble = word & 0xF
    
    row_h, col_h = high_nibble >> 2, high_nibble & 0x3
    row_l, col_l = low_nibble >> 2, low_nibble & 0x3
    
    sub_high = SBOX[row_h][col_h]
    sub_low = SBOX[row_l][col_l]
    
    return (sub_high << 4) | sub_low

# Encryption functions
def add_round_key(state, round_key):
    """XOR the state with the round key."""
    return state ^ round_key

def substitute_bytes(state):
    """Apply S-box substitution to each byte in the state."""
    result = 0
    for i in range(4):  # Process 4 nibbles (16 bits)
        nibble = (state >> (12 - 4*i)) & 0xF
        row, col = nibble >> 2, nibble & 0x3
        sub_nibble = SBOX[row][col]
        result |= (sub_nibble << (12 - 4*i))
    return result

def inverse_substitute_bytes(state):
    """Apply inverse S-box substitution to each byte in the state."""
    result = 0
    for i in range(4):  # Process 4 nibbles (16 bits)
        nibble = (state >> (12 - 4*i)) & 0xF
        row, col = nibble >> 2, nibble & 0x3
        sub_nibble = INV_SBOX[row][col]
        result |= (sub_nibble << (12 - 4*i))
    return result

def shift_rows(state):
    """Perform the ShiftRows step."""
    # Extract the 4 nibbles
    n0 = (state >> 12) & 0xF
    n1 = (state >> 8) & 0xF
    n2 = (state >> 4) & 0xF
    n3 = state & 0xF
    
    # Shift the rows (swap n1 and n3)
    return (n0 << 12) | (n3 << 8) | (n2 << 4) | n1

def mix_columns(state):
    """Perform the MixColumns step."""
    # Extract the two columns (each 8 bits)
    c0 = (state >> 8) & 0xFF
    c1 = state & 0xFF
    
    # Extract the 4 elements (each 4 bits)
    s00 = (c0 >> 4) & 0xF
    s10 = c0 & 0xF
    s01 = (c1 >> 4) & 0xF
    s11 = c1 & 0xF
    
    # Apply the MixColumns transformation
    s00_new = gf_mult(MIX_COL_MATRIX[0][0], s00) ^ gf_mult(MIX_COL_MATRIX[0][1], s10)
    s10_new = gf_mult(MIX_COL_MATRIX[1][0], s00) ^ gf_mult(MIX_COL_MATRIX[1][1], s10)
    s01_new = gf_mult(MIX_COL_MATRIX[0][0], s01) ^ gf_mult(MIX_COL_MATRIX[0][1], s11)
    s11_new = gf_mult(MIX_COL_MATRIX[1][0], s01) ^ gf_mult(MIX_COL_MATRIX[1][1], s11)
    
    # Combine the new columns
    c0_new = (s00_new << 4) | s10_new
    c1_new = (s01_new << 4) | s11_new
    
    return (c0_new << 8) | c1_new

def inverse_mix_columns(state):
    """Perform the Inverse MixColumns step."""
    # Extract the two columns (each 8 bits)
    c0 = (state >> 8) & 0xFF
    c1 = state & 0xFF
    
    # Extract the 4 elements (each 4 bits)
    s00 = (c0 >> 4) & 0xF
    s10 = c0 & 0xF
    s01 = (c1 >> 4) & 0xF
    s11 = c1 & 0xF
    
    # Apply the Inverse MixColumns transformation
    s00_new = gf_mult(INV_MIX_COL_MATRIX[0][0], s00) ^ gf_mult(INV_MIX_COL_MATRIX[0][1], s10)
    s10_new = gf_mult(INV_MIX_COL_MATRIX[1][0], s00) ^ gf_mult(INV_MIX_COL_MATRIX[1][1], s10)
    s01_new = gf_mult(INV_MIX_COL_MATRIX[0][0], s01) ^ gf_mult(INV_MIX_COL_MATRIX[0][1], s11)
    s11_new = gf_mult(INV_MIX_COL_MATRIX[1][0], s01) ^ gf_mult(INV_MIX_COL_MATRIX[1][1], s11)
    
    # Combine the new columns
    c0_new = (s00_new << 4) | s10_new
    c1_new = (s01_new << 4) | s11_new
    
    return (c0_new << 8) | c1_new

def encrypt(plaintext, key):
    """Encrypt 16-bit plaintext using 16-bit key."""
    # Generate round keys
    round_key1, round_key2 = key_expansion(key)
    
    # Initial round
    state = add_round_key(plaintext, key)
    
    # Round 1
    state = substitute_bytes(state)
    state = shift_rows(state)
    state = mix_columns(state)
    state = add_round_key(state, round_key1)
    
    # Round 2 (final)
    state = substitute_bytes(state)
    state = shift_rows(state)
    state = add_round_key(state, round_key2)
    
    return state

def decrypt(ciphertext, key):
    """Decrypt 16-bit ciphertext using 16-bit key."""
    # Generate round keys
    round_key1, round_key2 = key_expansion(key)
    
    # Initial round
    state = add_round_key(ciphertext, round_key2)
    state = shift_rows(state)
    state = inverse_substitute_bytes(state)
    
    # Round 1
    state = add_round_key(state, round_key1)
    state = inverse_mix_columns(state)
    state = shift_rows(state)
    state = inverse_substitute_bytes(state)
    
    # Final round
    state = add_round_key(state, key)
    
    return state

# Test cases
def run_test_cases():
    print("S-AES Test Cases:")
    
    # Test case 1
    key = 0xABCD
    plaintext = 0x1234
    
    ciphertext = encrypt(plaintext, key)
    decrypted = decrypt(ciphertext, key)
    
    print(f"Key: 0x{key:04X}")
    print(f"Plaintext: 0x{plaintext:04X}")
    print(f"Ciphertext: 0x{ciphertext:04X}")
    print(f"Decrypted: 0x{decrypted:04X}")
    print(f"Decryption successful: {plaintext == decrypted}")
    
    # Test case 2
    key = 0x4AF5
    plaintext = 0xB1D3
    
    ciphertext = encrypt(plaintext, key)
    decrypted = decrypt(ciphertext, key)
    
    print("\nTest case 2:")
    print(f"Key: 0x{key:04X}")
    print(f"Plaintext: 0x{plaintext:04X}")
    print(f"Ciphertext: 0x{ciphertext:04X}")
    print(f"Decrypted: 0x{decrypted:04X}")
    print(f"Decryption successful: {plaintext == decrypted}")

if __name__ == "__main__":
    run_test_cases()