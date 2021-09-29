def encode(data:str) -> str:
    data = data.encode('latin1')
    result = ''
    left_pad = lambda x, n: '0'*(n - len(x)) + x
    
    for char in data:
        result += left_pad(bin(char)[2:], 8)
    return result

def decode(data:str) -> str:
    groups = list()
    
    for i in range(0, len(data), 8):
        groups.append(data[i:i+8])
    
    result = bytes([ int(group, 2) for group in groups ]).decode('latin1')
    return result

def is_encodable(data:str) -> bool:
    try:
        data.encode('latin1')
    except:
        return False
    else:
        return True

def binary_character_length(length:int) -> int:
    return length * 8
