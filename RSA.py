import random

def gcd(a, b):
    while b != 0:
        a, b = b, a % b
    return a

def modinv(a, m):
    for x in range(1, m):
        if (a * x) % m == 1:
            return x
    return None

def is_prime(num):
    if num <= 1:
        return False
    for i in range(2, int(num**0.5)+1):
        if num % i == 0:
            return False
    return True

def generate_random_prime(start=100, end=300):
    while True:
        num = random.randint(start, end)
        if is_prime(num):
            return num

def generate_keys():
    p = generate_random_prime()
    q = generate_random_prime()

    while q == p:
        q = generate_random_prime()

    print(f"Randomly chosen primes:\np = {p}, q = {q}")

    n = p * q
    phi = (p - 1) * (q - 1)

    e = random.randint(2, phi - 1)
    while gcd(e, phi) != 1:
        e = random.randint(2, phi - 1)

    d = modinv(e, phi)

    return ((e, n), (d, n))

def encrypt(plaintext, public_key):
    e, n = public_key
    cipher = [(ord(char) ** e) % n for char in plaintext]
    return cipher

def decrypt(ciphertext, private_key):
    d, n = private_key
    plain = [chr((char ** d) % n) for char in ciphertext]
    return ''.join(plain)

def main():
    print("RSA Encryption/Decryption with Random Keys")

    public_key, private_key = generate_keys()
    print("\nPublic Key:", public_key)
    print("Private Key:", private_key)

    message = input("\nEnter message to encrypt: ")
    encrypted = encrypt(message, public_key)
    print("\nEncrypted:", encrypted)

    decrypted = decrypt(encrypted, private_key)
    print("Decrypted:", decrypted)

if __name__ == "__main__":
    main()