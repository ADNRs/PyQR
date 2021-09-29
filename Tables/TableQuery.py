import numpy as np

class __TableQuery:
    def __init__(self):
        read_file = lambda file_name: np.genfromtxt(file_name, dtype=str, delimiter=',', skip_header=1)
        self.character_capacity = read_file('./tables/CharacterCapacity.txt')
        self.codeword_info = read_file('./tables/CodewordInfo.txt')
        self.format_info_string = read_file('./tables/FormatInfoString.txt')
        self.version_info_string = read_file('./tables/VersionInfoString.txt')
        self.remainder_bits = read_file('./tables/RemainderBits.txt')
        self.position_of_alignment_pattern = read_file('./tables/PositionOfAlignmentPattern.txt')
    
    def get_character_capacity(self, version:int, ec_level:str, mode:str) -> int:
        ec_lvl_dict = { 'L': 0, 'M': 1, 'Q': 2, 'H': 3 }
        mode_dict = { 'Numeric': 2, 'Alphanumeric': 3, 'Byte': 4, 'Kanji': 5 }
        return int(self.character_capacity[(version - 1)*4 + ec_lvl_dict[ec_level]][mode_dict[mode]])
    
    def get_codeword_info(self, version:int, ec_level:str) -> tuple:
        ec_lvl_dict = { 'L': 0, 'M': 1, 'Q': 2, 'H': 3 }
        str2int = lambda x: int(x) if x else 0
        return tuple(str2int(x) for x in self.codeword_info[(version - 1)*4 + ec_lvl_dict[ec_level]][2:])
    
    def get_num_of_data_codewords(self, version:int, ec_level:str) -> int:
        return self.get_codeword_info(version, ec_level)[0]
    
    def get_num_of_ec_codewords_per_block(self, version:int, ec_level:str) -> int:
        return self.get_codeword_info(version, ec_level)[1]
    
    def get_num_of_block_in_group_1(self, version:int, ec_level:str) -> int:
        return self.get_codeword_info(version, ec_level)[2]
    
    def get_num_of_data_codewords_in_block_of_qroup_1(self, version:int, ec_level:str) -> int:
        return self.get_codeword_info(version, ec_level)[3]
    
    def get_num_of_block_in_group_2(self, version:int, ec_level:str) -> int:
        return self.get_codeword_info(version, ec_level)[4]
    
    def get_num_of_data_codewords_in_block_of_qroup_2(self, version:int, ec_level:str) -> int:
        return self.get_codeword_info(version, ec_level)[5]
    
    def get_format_info_string(self, ec_level:str, mask_string:str) -> str:
        ec_lvl_dict = { 'L': 0, 'M': 1, 'Q': 2, 'H': 3 }
        return self.format_info_string[ec_lvl_dict[ec_level]*8 + int(mask_string, 2)][2]
    
    def get_version_info_string(self, version:int) -> str:
        return self.version_info_string[version-7][1]
    
    def get_position_of_alignment_patterns(self, version:int) -> tuple:
        return tuple(int(x) for x in self.position_of_alignment_pattern[version - 2][1:] if x)
    
    def get_remainder_bits(self, version:int) -> int:
        return int(self.remainder_bits[version-1][1])
    
    def get_mode_indicator(self, mode:str) -> str:
        return {
            'Numeric'          : '0001',
            'Alphanumeric'     : '0010',
            'Byte'             : '0100',
            'Kanji'            : '1000',
            'Structured Append': '0011',
            'ECI'              : '0111',
            'FNC1 1st'         : '0101',
            'FNC1 2nd'         : '1001',
            'End'              : '0000'
        }[mode]
    
    def get_mode_name(self, mode_indicator:str) -> str:
        return {
            '0001': 'Numeric',
            '0010': 'Alphanumeric',
            '0100': 'Byte',
            '1000': 'Kanji',
            '0011': 'Structured Append',
            '0111': 'ECI',
            '0101': 'FNC1 1st',
            '1001': 'FNC1 2nd',
            '0000': 'End'
        }[mode_indicator]
    
    def get_character_count_indicator(self, version:int, mode:str) -> int:
        return {
            'Numeric'     : lambda v: 10 if v < 10 else 12 if v < 27 else 14,
            'Alphanumeric': lambda v:  9 if v < 10 else 11 if v < 27 else 13,
            'Byte'        : lambda v:  8 if v < 10 else 16,
            'Kanji'       : lambda v:  8 if v < 10 else 10 if v < 27 else 12
        }[mode](version)
    
    def get_encode_ec_level_bit(self, ec_level:str) -> str:
        return {
            'L': '01',
            'M': '00',
            'Q': '11',
            'H': '10'
        }[ec_level]
    
    def get_decode_ec_level(self, ec_level_bit:str) -> str:
        return {
            '11': 'L',
            '10': 'M',
            '01': 'Q',
            '00': 'H'
        }[ec_level_bit]
    
    def get_mask_strings(self) -> tuple:
        return ( '000', '001', '010', '011', '100', '101', '110', '111' )
    
    def get_encode_mask(self, mask_string:str) -> object:
        return {
            '000': lambda i, j:       (i + j) % 2       == 0,
            '001': lambda i, j:          i % 2          == 0,
            '010': lambda i, j:          j % 3          == 0,
            '011': lambda i, j:       (i + j) % 3       == 0,
            '100': lambda i, j:    (j//3 + i//2) % 2    == 0,
            '101': lambda i, j:  (i * j)%2 + (i * j)%3  == 0,
            '110': lambda i, j:  ((i * j)%3 + i*j) % 2  == 0,
            '111': lambda i, j: ((i * j)%3 + i + j) % 2 == 0
        }[mask_string]
    
    def get_decode_mask(self, mask_string:str) -> object:
        return {
            '111': lambda i, j:          j % 3          == 0,
            '110': lambda i, j:       (i + j) % 3       == 0,
            '101': lambda i, j:       (i + j) % 2       == 0,
            '100': lambda i, j:          i % 2          == 0,
            '011': lambda i, j:  ((i * j)%3 + i*j) % 2  == 0,
            '010': lambda i, j: ((i * j)%3 + i + j) % 2 == 0,
            '001': lambda i, j:    (j//3 + i//2) % 2    == 0,
            '000': lambda i, j:  (i * j)%2 + (i * j)%3  == 0
        }[mask_string]

query = __TableQuery()
