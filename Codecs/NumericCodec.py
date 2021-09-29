def encode(data:str) -> str:
    groups = list()
    result = ''
    pad_num = { 3: 10, 2: 7, 1: 4 }
    left_pad = lambda x, n: '0'*(n - len(x)) + x
    
    for i in range(0, len(data), 3):
        groups.append(data[i:i+3])
    
    for group in groups:
        result += left_pad(bin(int(group))[2:], pad_num[len(group)])
    return result

def decode(data:str) -> str:
    groups = list()
    result = ''
    pad_num = { 10: 3, 7: 2, 4: 1 }
    left_pad = lambda x, n: '0'*(n - len(x)) + x
    
    for i in range(0, len(data), 10):
        groups.append(data[i:i+10])
    
    for group in groups:
        result += left_pad(str(int(group, 2)), pad_num[len(group)])
    return result

def is_encodable(data:str) -> bool:
    try:
        int(data)
    except:
        return False
    else:
        return True

def binary_character_length(length:int) -> int:
    return length//3*10 + (length%3==2)*7 + (length%3==1)*4
