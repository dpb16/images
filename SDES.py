def permute(input_bits, permutation_table):
    """Permute the input bits according to the permutation table."""
    return [input_bits[i - 1] for i in permutation_table]

def left_shift(bits, n_shifts):
    """Perform a circular left shift on the bits."""
    return bits[n_shifts:] + bits[:n_shifts]

def xor(bits1, bits2):
    """Perform XOR operation on two bit arrays."""
    return [b1 ^ b2 for b1, b2 in zip(bits1, bits2)]

def s_box_substitution(bits, s_box):
    """Perform S-box substitution."""
    row = bits[0] * 2 + bits[3]
    col = bits[1] * 2 + bits[2]
    val = s_box[row][col]
    return [val // 2, val % 2]

def generate_subkeys(key):
    """Generate two subkeys from the 10-bit master key."""
    # Initial permutation (P10)
    p10_table = [3, 5, 2, 7, 4, 10, 1, 9, 8, 6]
    key = permute(key, p10_table)
    
    # Split into two halves
    left = key[:5]
    right = key[5:]
    
    # Generate first subkey (K1)
    left = left_shift(left, 1)
    right = left_shift(right, 1)
    
    # Permutation P8 to get K1
    p8_table = [6, 3, 7, 4, 8, 5, 10, 9]
    k1 = permute(left + right, p8_table)
    
    # Generate second subkey (K2)
    left = left_shift(left, 2)
    right = left_shift(right, 2)
    
    # Permutation P8 to get K2
    k2 = permute(left + right, p8_table)
    
    return k1, k2

def function_F(right_half, subkey):
    """Apply the Feistel function F to the right half."""
    # Expansion permutation (EP)
    ep_table = [4, 1, 2, 3, 2, 3, 4, 1]
    expanded = permute(right_half, ep_table)
    
    # XOR with subkey
    xored = xor(expanded, subkey)
    
    # Split the 8 bits into two 4-bit blocks
    left = xored[:4]
    right = xored[4:]
    
    # S-box substitution
    s0 = [
        [1, 0, 3, 2],
        [3, 2, 1, 0],
        [0, 2, 1, 3],
        [3, 1, 3, 2]
    ]
    s1 = [
        [0, 1, 2, 3],
        [2, 0, 1, 3],
        [3, 0, 1, 0],
        [2, 1, 0, 3]
    ]
    
    left_result = s_box_substitution(left, s0)
    right_result = s_box_substitution(right, s1)
    
    # Combine the results
    combined = left_result + right_result
    
    # Permutation P4
    p4_table = [2, 4, 3, 1]
    return permute(combined, p4_table)

def encrypt_block(plaintext, key):
    """Encrypt an 8-bit block using S-DES."""
    # Generate subkeys
    k1, k2 = generate_subkeys(key)
    
    # Initial permutation (IP)
    ip_table = [2, 6, 3, 1, 4, 8, 5, 7]
    plaintext = permute(plaintext, ip_table)
    
    # Split into left and right halves
    left = plaintext[:4]
    right = plaintext[4:]
    
    # First round
    new_right = xor(left, function_F(right, k1))
    left = right
    right = new_right
    
    # Second round
    new_right = xor(left, function_F(right, k2))
    left = right
    right = new_right
    
    # Switch left and right
    combined = right + left
    
    # Inverse initial permutation (IP^-1)
    ip_inverse_table = [4, 1, 3, 5, 7, 2, 8, 6]
    return permute(combined, ip_inverse_table)

def decrypt_block(ciphertext, key):
    """Decrypt an 8-bit block using S-DES."""
    # Generate subkeys
    k1, k2 = generate_subkeys(key)
    
    # Initial permutation (IP)
    ip_table = [2, 6, 3, 1, 4, 8, 5, 7]
    ciphertext = permute(ciphertext, ip_table)
    
    # Split into left and right halves
    left = ciphertext[:4]
    right = ciphertext[4:]
    
    # First round (using K2 for decryption)
    new_right = xor(left, function_F(right, k2))
    left = right
    right = new_right
    
    # Second round (using K1 for decryption)
    new_right = xor(left, function_F(right, k1))
    left = right
    right = new_right
    
    # Switch left and right
    combined = right + left
    
    # Inverse initial permutation (IP^-1)
    ip_inverse_table = [4, 1, 3, 5, 7, 2, 8, 6]
    return permute(combined, ip_inverse_table)

# Convert between bit arrays and integers
def bits_to_int(bits):
    """Convert a bit array to an integer."""
    return sum(bit << i for i, bit in enumerate(reversed(bits)))

def int_to_bits(n, length):
    """Convert an integer to a bit array of specified length."""
    return [(n >> i) & 1 for i in range(length-1, -1, -1)]

# Test cases
def run_test_cases():
    print("S-DES Test Cases:")
    # Test case 1
    key = [1, 0, 1, 0, 0, 0, 0, 0, 1, 0]  # 10-bit key
    plaintext = [1, 0, 0, 1, 0, 1, 1, 1]  # 8-bit plaintext
    
    ciphertext = encrypt_block(plaintext, key)
    decrypted = decrypt_block(ciphertext, key)
    
    print(f"Key (binary): {key}")
    print(f"Key (decimal): {bits_to_int(key)}")
    print(f"Plaintext (binary): {plaintext}")
    print(f"Plaintext (decimal): {bits_to_int(plaintext)}")
    print(f"Ciphertext (binary): {ciphertext}")
    print(f"Ciphertext (decimal): {bits_to_int(ciphertext)}")
    print(f"Decrypted (binary): {decrypted}")
    print(f"Decrypted (decimal): {bits_to_int(decrypted)}")
    print(f"Decryption successful: {plaintext == decrypted}")
    
    # Test case 2
    key = [0, 1, 1, 1, 1, 0, 0, 1, 1, 0]
    plaintext = [0, 0, 0, 0, 1, 1, 1, 1]
    
    ciphertext = encrypt_block(plaintext, key)
    decrypted = decrypt_block(ciphertext, key)
    
    print("\nTest case 2:")
    print(f"Key (binary): {key}")
    print(f"Key (decimal): {bits_to_int(key)}")
    print(f"Plaintext (binary): {plaintext}")
    print(f"Plaintext (decimal): {bits_to_int(plaintext)}")
    print(f"Ciphertext (binary): {ciphertext}")
    print(f"Ciphertext (decimal): {bits_to_int(ciphertext)}")
    print(f"Decrypted (binary): {decrypted}")
    print(f"Decrypted (decimal): {bits_to_int(decrypted)}")
    print(f"Decryption successful: {plaintext == decrypted}")

if __name__ == "__main__":
    run_test_cases()