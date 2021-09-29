encode_table = {
    r'0':  0, r'1':  1, r'2':  2, r'3':  3, r'4':  4, r'5':  5, r'6':  6, r'7':  7, r'8':  8,
    r'9':  9, r'A': 10, r'B': 11, r'C': 12, r'D': 13, r'E': 14, r'F': 15, r'G': 16, r'H': 17,
    r'I': 18, r'J': 19, r'K': 20, r'L': 21, r'M': 22, r'N': 23, r'O': 24, r'P': 25, r'Q': 26,
    r'R': 27, r'S': 28, r'T': 29, r'U': 30, r'V': 31, r'W': 32, r'X': 33, r'Y': 34, r'Z': 35,
    r' ': 36, r'$': 37, r'%': 38, r'*': 39, r'+': 40, r'-': 41, r'.': 42, r'/': 43, r':': 44
}

decode_table = { v: k for k, v in zip(encode_table.keys(), encode_table.values()) }

def encode(data:str) -> str:
    groups = list()
    result = ''
    left_pad = lambda x, n: '0'*(n - len(x)) + x
    
    for i in range(0, len(data), 2):
        groups.append(data[i:i+2])
    
    for group in groups:
        if len(group) == 2:
            char1, char2 = encode_table[group[0]], encode_table[group[1]]
            result += left_pad(bin(char1*45 + char2)[2:], 11)
        elif len(group) == 1:
            char1 = encode_table[group]
            result += left_pad(bin(char1)[2:], 6)
    return result

def decode(data:str) -> str:
    groups = list()
    result = ''
    
    for i in range(0, len(data), 11):
        groups.append(data[i:i+11])
    
    for group in groups:
        num = int(group, 2)
        if len(group) == 11:
            char1, char2 = num // 45, num % 45
            result += decode_table[char1] + decode_table[char2]
        elif len(group) == 6:
            result += decode_table[num]
    return result

def is_encodable(data:str) -> bool:
    for x in data:
        if x not in encode_table.keys():
            return False
    else:
        return True if data != '' else False

def binary_character_length(length:int) -> int:
    return length//2*11 + length%2*6
