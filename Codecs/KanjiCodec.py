def encode(data:str) -> str:
    result = ''
    dec2hex = lambda x: hex(x)[2:]
    hex2dec = lambda x: int(x, 16)
    left_pad = lambda x, n: '0'*(n - len(x)) + x
    
    for i in range(len(data)):
        char = hex2dec(data[i].encode('cp932').hex())
        if 0x8140 <= char <= 0x9FFC:
            diff = left_pad(dec2hex(char - 0x8140), 4)
            result += left_pad(bin(hex2dec(diff[:2])*0xC0 + hex2dec(diff[2:]))[2:], 13)
        elif 0xE040 <= char <= 0xEBBF:
            diff = left_pad(dec2hex(char - 0xC140), 4)
            result += left_pad(bin(hex2dec(diff[:2])*0xC0 + hex2dec(diff[2:]))[2:], 13)
    return result

def decode(data:str) -> str:
    groups = list()
    result = ''
    dec2hex = lambda x: hex(x)[2:]
    hex2dec = lambda x: int(x, 16)
    left_pad = lambda x, n: '0'*(n - len(x)) + x
    
    for i in range(0, len(data), 13):
        groups.append(data[i:i+13])
    
    for group in groups:
        char = int(group, 2)
        msb = left_pad(dec2hex(char // 0xC0), 2)
        lsb = left_pad(dec2hex(char % 0xC0), 2)
        diff = msb + lsb
        if group[0] == '0':
            result += bytes.fromhex(dec2hex(hex2dec(diff) + 0x8140)).decode('cp932')
        elif group[0] == '1':
            result += bytes.fromhex(dec2hex(hex2dec(diff) + 0xC140)).decode('cp932')
    return result

def is_encodable(data:str) -> bool:
    dec2hex = lambda x: hex(x)[2:]
    hex2dec = lambda x: int(x, 16)
    
    for i in range(len(data)):
        char = hex2dec(data[i].encode('cp932').hex())
        if 0x8140 <= char <= 0x9FFC or 0xE040 <= char <= 0xEBBF:
            pass
        else:
            return False
    else:
        return True

def binary_character_length(length:int) -> int:
    return length * 13
