from Tables import query
from Codecs import NumericCodec      as Num
from Codecs import AlphanumericCodec as Alp
from Codecs import Latin1Codec       as Lat
from Codecs import KanjiCodec        as Kan
from Codecs import RSCodec           as RS
from Utils.ImageHelper  import *
from Utils.QRCodeHelper import *

codecs = { 'Numeric': Num, 'Alphanumeric': Alp, 'Byte': Lat, 'Kanji': Kan }

class QRCodeEncodeError(Exception):
    pass

class QRCodeDecodeError(Exception):
    pass

class QRCodeEncoder:
    def __init__(self):
        self.message = None
        self.ec_level = None
        self.path = None
        self.mode = None
        self.version = None
        self.data = None
        self.data_codewords = None
        self.ec_codewords = None
        self.final_message = None
        self.canvas = None
    
    def encode(self, message:str, ec_level:str, path:str) -> None:
        self.message = message
        self.ec_level = ec_level
        self.path = path
        self.__find_mode()
        self.__find_version()
        self.__construct_data()
        self.__generate_data_codewords()
        self.__generate_ec_codewords()
        self.__construct_final_message()
        self.__construct_final_image()
    
    def info(self):
        print('Type:', str(self.version) + '-' + self.ec_level)
        print('Mode:', self.mode)
        print('Message:')
        print(self.message)
        print('Final message:')
        print(self.final_message)
        self.canvas.show()
    
    def __find_mode(self):
        for mode, codec in zip(codecs.keys(), codecs.values()):
            if codec.is_encodable(self.message):
                self.mode = mode
                return
        else:
            raise QRCodeEncodeError('input message can\'t be encoded')
    
    def __find_version(self):
        for ver in range(1, 41):
            if query.get_character_capacity(ver, self.ec_level, self.mode) >= len(self.message):
                self.version = ver
                return
        else:
            raise QRCodeEncodeError('input message is too long')
    
    def __construct_data(self):
        character_count = query.get_character_count_indicator(self.version, self.mode)
        num_of_total_bits = query.get_num_of_data_codewords(self.version, self.ec_level) * 8
        left_pad  = lambda x, n: '0'*(n - len(x)) + x
        
        self.data  = query.get_mode_indicator(self.mode)                    # add mode indicator
        self.data += left_pad(bin(len(self.message))[2:], character_count)  # add character count indicator
        self.data += codecs[self.mode].encode(self.message)                 # add encoded message
        self.data += '0' * min(4, num_of_total_bits - len(self.data))       # add terminator
        self.data += '0' * (8 - len(self.data)&7)                           # pad the length to a multiple of 8
        for i in range((num_of_total_bits - len(self.data)) // 8):
            if i % 2 == 0:                                                  # pad 11101100 and 00010001 until
                self.data += '11101100'                                     # reaching num_of_total_bits
            else:
                self.data += '00010001'
    
    def __generate_data_codewords(self):
        num_of_block_in_group_1 = query.get_num_of_block_in_group_1(self.version, self.ec_level)
        num_of_dc_of_group_1 = query.get_num_of_data_codewords_in_block_of_qroup_1(self.version, self.ec_level)
        num_of_block_in_group_2 = query.get_num_of_block_in_group_2(self.version, self.ec_level)
        num_of_dc_of_group_2 = query.get_num_of_data_codewords_in_block_of_qroup_2(self.version, self.ec_level)
        
        data_codewords = list()
        for i in range(0, len(self.data), 8):
            data_codewords.append(self.data[i:i+8])
        
        group_1 = [ list() for x in range(num_of_block_in_group_1) ]
        group_2 = [ list() for x in range(num_of_block_in_group_2) ] if num_of_block_in_group_2 else []
        
        for data_codeword in data_codewords:
            num_of_dc = [num_of_dc_of_group_1]*len(group_1) + [num_of_dc_of_group_2]*len(group_2)
            for block, len_limit in zip(group_1 + group_2, num_of_dc):
                if len(block) < len_limit:
                    block.append(data_codeword)
                    break
        
        if group_2:
            self.data_codewords = [ group_1, group_2 ]
        else:
            self.data_codewords = [ group_1 ]
    
    def __generate_ec_codewords(self):
        num_of_block_in_group_1 = query.get_num_of_block_in_group_1(self.version, self.ec_level)
        num_of_block_in_group_2 = query.get_num_of_block_in_group_2(self.version, self.ec_level)
        num_of_ec_codewords = query.get_num_of_ec_codewords_per_block(self.version, self.ec_level)
        left_pad = lambda x, n: '0'*(n - len(x)) + x
        map2int = lambda x: [ int(elem, 2) for elem in x ]
        map2bin = lambda x: [ left_pad(bin(elem)[2:], 8) for elem in x ]
        
        group_1, group_2 = list(), list()
        
        RS.init_tables(0b100011101)
        for idx, group in enumerate(self.data_codewords):
            for block in group:
                ec_codeword = map2bin(RS.rs_encode_msg(map2int(block), num_of_ec_codewords))[len(block):]
                if idx == 0:
                    group_1.append(ec_codeword)
                elif idx == 1:
                    group_2.append(ec_codeword)
        
        if group_2:
            self.ec_codewords = [ group_1, group_2 ]
        else:
            self.ec_codewords = [ group_1 ]
    
    def __construct_final_message(self):
        num_of_dc_of_group_1 = query.get_num_of_data_codewords_in_block_of_qroup_1(self.version, self.ec_level)
        num_of_dc_of_group_2 = query.get_num_of_data_codewords_in_block_of_qroup_2(self.version, self.ec_level)
        num_of_ec_codewords = query.get_num_of_ec_codewords_per_block(self.version, self.ec_level)
        num_of_remainder_bits = query.get_remainder_bits(self.version)
        
        self.final_message = ''
        
        for i in range(max(num_of_dc_of_group_1, num_of_dc_of_group_2)):
            for group in self.data_codewords:
                for block in group:
                    try:
                        self.final_message += block[i]
                    except IndexError:
                        pass
        
        for i in range(num_of_ec_codewords):
            for group in self.ec_codewords:
                for block in group:
                    self.final_message += block[i]
        
        self.final_message += '0' * num_of_remainder_bits
    
    def __construct_final_image(self):
        self.canvas = QRCodeCanvas(self.version)
        width = 21 + 4*(self.version - 1)
        info = list()
        
        info += get_position_info(self.version)
        info += get_separator_info(self.version)
        info += get_alignment_info(self.version)
        info += get_timing_info(self.version)
        info += get_dark_module_info(self.version)
        info += get_version_info(self.version)
        info += get_message_info(self.version, self.final_message)
        self.canvas.set_by_list(info)

        # choose mask pattern
        lowest_penalty = float('inf')
        best_canvas = None
        for mask_str in query.get_mask_strings():
            canvas = self.canvas.copy()
            canvas.set_by_list(get_format_info(self.version, self.ec_level, mask_str))
            canvas.do_masking(query.get_encode_mask(mask_str))
            penalty = canvas.evaluate_mask_penalty()
            if penalty < lowest_penalty:
                lowest_penalty = penalty
                best_canvas = canvas
        self.canvas = best_canvas
        self.canvas.save(self.path, mul=9)
    
class QRCodeDecoder:
    def __init__(self):
        self.array = None
        self.version = None
        # attr. for format info
        self.ec_level = None
        self.mask = None
        # attr. for decoding
        self.final_message = None
        self.data_codewords = None
        self.ec_codewords = None
        # attr. for result
        self.data = None
        self.mode = None
        self.length = None
        self.message = None
    
    def decode(self, path):
        self.array = QRCodeExtractor().extract(path)
        show_array(self.array)
        self.__decode_format_info()
        self.__decode_version_info()
        self.__demask()
        show_array(self.array)
        self.__read_final_message()
        self.__regenerate_data_codewords()
        self.__regenerate_ec_codewords()
        self.__restore_data_codewords()
        self.__decode_data()
    
    def info(self):
        print('Type:', str(self.version) + '-' + self.ec_level)
        print('Mode:', self.mode)
        print('Message:')
        print(self.message)
    
    def __restore_format_info(self, code):
        def hamming_weight(x):
            weight = 0
            while x > 0:
                weight += x & 1
                x >>= 1
            return weight

        bin2int = lambda x: int(x, 2)
        int2bin = lambda x: bin(x)[2:]
        left_pad = lambda x, n: (n-len(x))*'0' + x

        best_dist = 15
        best_code = -1
        code = bin2int(code)
        for ec_lvl in ['L', 'M', 'Q', 'H']:
            for mask_str in query.get_mask_strings():
                test_code = bin2int(query.get_format_info_string(ec_lvl, mask_str))
                test_dist = hamming_weight(code ^ test_code)
                if test_dist < best_dist:
                    best_dist = test_dist
                    best_code = test_code
                elif test_dist == best_dist:
                    best_code = -1
        if best_code == -1:
            return ''
        best_code = int2bin(best_code)
        return left_pad(best_code, 15)
    
    def __decode_format_info(self):
        # set up format1 indices, exclude the timing patterns
        f1_idx = reversed([
            *[(x, 8) for x in range(9) if x != 6],
            *reversed([(8, x) for x in range(8) if x != 6])
        ])
        # set up format2 indices, exclude the timing patterns
        f2_idx = reversed([
            *reversed([(8, x) for x in range(len(self.array)-8, len(self.array))]),
            *[(x, 8) for x in range(len(self.array)-7, len(self.array))]
        ])
        # extract format info
        format1 = ''.join([str(self.array[r][c]) for r, c in f1_idx])
        format2 = ''.join([str(self.array[r][c]) for r, c in f2_idx])
        format1 = self.__restore_format_info(format1)
        format2 = self.__restore_format_info(format2)
        correct_format = format1 if format1 == format2 else format1 if format1 else format2 if format2 else ''
        assert correct_format, 'Format information recovery failed'
        # set necessary info
        self.ec_level = query.get_decode_ec_level(correct_format[:2])
        self.mask = query.get_decode_mask(correct_format[2:5])
    
    def __restore_version_info(self, code):
        def hamming_weight(x):
            weight = 0
            while x > 0:
                weight += x & 1
                x >>= 1
            return weight

        bin2int = lambda x: int(x, 2)
        int2bin = lambda x: bin(x)[2:]
        left_pad = lambda x, n: (n-len(x))*'0' + x

        best_dist = 18
        best_code = -1
        code = bin2int(code)
        for ver in range(7, 41):
            test_code = bin2int(query.get_version_info_string(ver))
            test_dist = hamming_weight(code ^ test_code)
            if test_dist < best_dist:
                best_dist = test_dist
                best_code = test_code
            elif test_dist == best_dist:
                best_code = -1
        if best_code == -1:
            return ''
        best_code = int2bin(best_code)
        return left_pad(best_code, 18)
    
    def __decode_version_info(self):
        # obtain version by calculating with the size of the matrix
        self.version = ((self.array.shape[0] - 21) // 4) + 1
        # double check with version info
        if self.version >= 7:
            width = self.array.shape[0]
            v1_idx = list(reversed([(x, y) for x in range(6) for y in range(width-11, width-8)]))
            v2_idx = list(reversed([(x, y) for y in range(6) for x in range(width-11, width-8)]))
            version1 = ''.join([str(self.array[r][c]) for r, c in v1_idx])
            version2 = ''.join([str(self.array[r][c]) for r, c in v2_idx])
            version1 = self.__restore_version_info(version1)
            version2 = self.__restore_version_info(version2)
            assert version1 == version2, 'version restore failed'
        assert (self.array.shape[0] - 21) % 4 == 0, 'This QR Code is not valid.'
        assert 1 <= self.version <= 40, 'This QR Code is not valid.'
    
    def __demask(self):
        # get indices that excludes content region (position, alignment and timing)
        non_info  = list()
        non_info += get_position_info(self.version)
        non_info += get_separator_info(self.version)
        non_info += get_alignment_info(self.version)
        non_info += get_timing_info(self.version)
        non_info += get_dark_module_info(self.version)
        non_info += get_version_info(self.version)
        non_info += get_format_info(self.version)

        forbid_pos = [ (x[0], x[1]) for x in non_info ]
        # start demasking
        for r in range(self.array.shape[0]):
            for c in range(self.array.shape[1]):
                if (r, c) not in forbid_pos:
                    self.array[r][c] = self.array[r][c] ^ self.mask(r, c)
    
    def __read_final_message(self):
        num_of_remainder_bits = query.get_remainder_bits(self.version)
        
        non_info  = list()
        non_info += get_position_info(self.version)
        non_info += get_separator_info(self.version)
        non_info += get_alignment_info(self.version)
        non_info += get_timing_info(self.version)
        non_info += get_dark_module_info(self.version)
        non_info += get_version_info(self.version)
        non_info += get_format_info(self.version)

        forbid_pos = [ (x[0], x[1]) for x in non_info ]
        width = self.array.shape[0]
        self.final_message = str()
        is_upward, is_first, r, c = True, True, width - 1, width - 1
        for x in range(width*width - width):
            if (r, c) not in forbid_pos:
                self.final_message += str(self.array[r][c])
            if is_first:
                r, c = r, c - 1
                if c < 0:
                    r, c = r - (1 if is_upward else -1), c + 1
                else:
                    is_first = False
            else:
                r, c = r - (1 if is_upward else -1), c + 1
                if (r < 0) if is_upward else (r >= width):
                    r, c = r + (1 if is_upward else -1), c - 2
                    is_upward = not is_upward
                    if c == 6:
                        c = c - 1
                is_first = True
        self.final_message = self.final_message[:len(self.final_message)-num_of_remainder_bits]
        
    def __regenerate_data_codewords(self):
        num_of_data_codewords = query.get_num_of_data_codewords(self.version, self.ec_level)
        num_of_block_in_group_1 = query.get_num_of_block_in_group_1(self.version, self.ec_level)
        num_of_dc_of_group_1 = query.get_num_of_data_codewords_in_block_of_qroup_1(self.version, self.ec_level)
        num_of_block_in_group_2 = query.get_num_of_block_in_group_2(self.version, self.ec_level)
        num_of_dc_of_group_2 = query.get_num_of_data_codewords_in_block_of_qroup_2(self.version, self.ec_level)
        
        data_codewords = list()
        i = 0
        for x in range(num_of_data_codewords):
            data_codewords.append(self.final_message[i:i+8])
            i += 8
        
        group_1 = [ list() for x in range(num_of_block_in_group_1) ]
        group_2 = [ list() for x in range(num_of_block_in_group_2) ] if num_of_block_in_group_2 else []
        
        for data_codeword in data_codewords:
            limits = [ num_of_dc_of_group_1 ]*len(group_1) + [ num_of_dc_of_group_2 ]*len(group_2)
            for block, limit in zip(group_1 + group_2, limits):
                max_len = max([ len(b) for b in group_1 + group_2 ])
                min_len = min([ len(b) for b in group_1 + group_2 ])
                if len(block) < limit and (min_len == max_len or len(block) != max_len):
                    block.append(data_codeword)
                    break
        self.data_codewords = [ group_1, group_2 ]
    
    
    
    def __regenerate_ec_codewords(self):
        num_of_data_codewords = query.get_num_of_data_codewords(self.version, self.ec_level)
        num_of_block_in_group_1 = query.get_num_of_block_in_group_1(self.version, self.ec_level)
        num_of_block_in_group_2 = query.get_num_of_block_in_group_2(self.version, self.ec_level)
        num_of_block = num_of_block_in_group_1 + num_of_block_in_group_2
        num_of_ec_codewords = query.get_num_of_ec_codewords_per_block(self.version, self.ec_level)
        
        ec_codewords = list()
        i = 0
        bias = num_of_data_codewords * 8
        for x in range(num_of_ec_codewords*num_of_block):
            ec_codewords.append(self.final_message[bias+i:bias+i+8])
            i += 8
        
        group_1 = [ list() for x in range(num_of_block_in_group_1) ]
        group_2 = [ list() for x in range(num_of_block_in_group_2) ] if num_of_block_in_group_2 else []
        
        for ec_codeword in ec_codewords:
            limits = [ num_of_ec_codewords ] * num_of_block
            for block, limit in zip(group_1 + group_2, limits):
                max_len = max([ len(b) for b in group_1 + group_2 ])
                min_len = min([ len(b) for b in group_1 + group_2 ])
                if len(block) < limit and (min_len == max_len or len(block) != max_len):
                    block.append(ec_codeword)
                    break
        self.ec_codewords = [ group_1, group_2 ]
    
    def __restore_data_codewords(self):
        num_of_ec_codewords = query.get_num_of_ec_codewords_per_block(self.version, self.ec_level)
        self.data = ''
        left_pad  = lambda x, n: '0'*(n - len(x)) + x
        map2int = lambda x: [ int(elem, 2) for elem in x ]
        map2bin = lambda x: [ left_pad(bin(elem)[2:], 8) for elem in x ]
        
        RS.init_tables(0b100011101)
        for data_group, ec_group in zip(self.data_codewords, self.ec_codewords):
            for data_block, ec_block in zip(data_group, ec_group):
                msg = map2int(data_block) + map2int(ec_block)
                restore_msg, _ = RS.rs_correct_msg(msg, num_of_ec_codewords)
                for msg in map2bin(restore_msg):
                    self.data += msg
                
    def __decode_data(self):
        bin2int = lambda x: int(x, 2)
        self.mode = query.get_mode_name(self.data[:4])
        
        character_count_indicator = query.get_character_count_indicator(self.version, self.mode)
        character_len = bin2int(self.data[4:character_count_indicator+4])
        decoder = codecs[self.mode].decode
        binary_message_len = codecs[self.mode].binary_character_length(character_len)
        
        start = character_count_indicator + 4
        end = start + binary_message_len
        self.message = decoder(self.data[start:end])

class PyQR:
    def __init__(self):
        pass
    