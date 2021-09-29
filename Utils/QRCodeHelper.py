from itertools import product
from Tables import query
from .ImageHelper import *

get_width = lambda v: int(21 + 4*(v - 1))

def get_position_info(version:int) -> list:
    width = get_width(version)
    info = list()
    color = [
        [1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 1],
        [1, 0, 1, 1, 1, 0, 1],
        [1, 0, 1, 1, 1, 0, 1],
        [1, 0, 1, 1, 1, 0, 1],
        [1, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1]
    ]
    position = [(r, c, color[r][c]) for r in range(7) for c in range(7)]
    
    info += [x for x in position]                             # upper-left
    info += [(x[0], x[1]+width-7, x[2]) for x in position]    # upper-right
    info += [(x[0]+width-7, x[1], x[2]) for x in position]    # lower-left
    return info

def get_separator_info(version:int) -> list:
    width = get_width(version)
    info = list()
    
    info += [*[(x, 7, 0) for x in range(8)], *[(7, x, 0) for x in range(7)]]                # upper-left
    info += [*[(width-x-1, 7, 0) for x in range(8)], *[(width-8, x, 0) for x in range(7)]]  # upper-right
    info += [*[(x, width-8, 0) for x in range(8)], *[(7, width-x-1, 0) for x in range(7)]]  # lower-left
    return info

def get_alignment_info(version:int) -> list:
    if version == 1:
        return list()
    
    alignment_pos = query.get_position_of_alignment_patterns(version)
    width = get_width(version)
    info = list()
    color = [
        [1, 1, 1, 1, 1],
        [1, 0, 0, 0, 1],
        [1, 0, 1, 0, 1],
        [1, 0, 0, 0, 1],
        [1, 1, 1, 1, 1]
    ]
    alignment = [(r, c, color[r+2][c+2]) for r in range(-2, 3) for c in range(-2, 3)]
    
    possible_center_pos = list(product(alignment_pos, repeat=2))
    occupied_info = get_position_info(version) + get_separator_info(version)
    occupied_pos = [(x[0], x[1]) for x in occupied_info]
    for r, c in possible_center_pos:
        possible_pos = [(r+x[0], c+x[1], x[2]) for x in alignment]
        for pr, pc, _ in possible_pos:
            if (pr, pc) in occupied_pos:
                break
        else:
            info += possible_pos
    return info

def get_timing_info(version:int) -> list:
    width = get_width(version)
    info = list()
    
    info += [(6, x, not(x%2)) for x in range(8, width-8)]   # horizontal timing pattern
    info += [(x, 6, not(x%2)) for x in range(8, width-8)]   # vertical timing pattern
    return info

def get_dark_module_info(version:int) -> list:
    return [(4*version + 9, 8, 1)]

def get_format_info(version:int, ec_level:str='', mask_string:str='') -> list:
    width = get_width(version)
    info = list()
    mask_string = mask_string or '000'
    ec_level = ec_level or 'L'
    format_string = query.get_format_info_string(ec_level, mask_string)
    pos = [
            *[(x, 8) for x in range(9) if x != 6],
            *reversed([(8, x) for x in range(8) if x != 6]),
            *reversed([(8, x) for x in range(width-8, width)]),
            *[(x, 8) for x in range(width-7, width)]
    ]
    
    for char, (r, c) in zip(format_string[::-1]*2, pos):
        info += [(r, c, int(char))]
    return info

def get_version_info(version:int) -> list:
    if version < 7:
        return list()
    
    width = get_width(version)
    info = list()
    version_string = query.get_version_info_string(version)
    pos = [
        *[(x, y) for x in range(6) for y in range(width-11, width-8)],
        *[(x, y) for y in range(6) for x in range(width-11, width-8)]
    ]
    
    for char, (r, c) in zip(version_string[::-1]*2, pos):
        info += [(r, c, int(char))]

    return info

def get_message_info(version:int, final_message:str) -> list:
    width = get_width(version)
    info = list()
    non_info  = list()
    non_info += get_position_info(version)
    non_info += get_separator_info(version)
    non_info += get_alignment_info(version)
    non_info += get_timing_info(version)
    non_info += get_dark_module_info(version)
    non_info += get_version_info(version)
    non_info += get_format_info(version)
    
    forbid_pos = [ (x[0], x[1]) for x in non_info ]
    i = 0
    is_upward, is_first, r, c = True, True, width - 1, width - 1
    while True:
        if (r, c) not in forbid_pos:
            info += [(r, c, int(final_message[i]))]
            i += 1
            if i == len(final_message):
                break
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
    return info

class QRCodeCanvas:
    WHITE = 0
    GRAY  = 130
    BLACK = 255
    
    def __init__(self, version:int) -> None:
        width = get_width(version)
        self.version = version
        self.data = np.zeros((width, width), dtype=int) + QRCodeCanvas.GRAY
    
    def copy(self) -> object:
        copied = QRCodeCanvas(1)
        copied.version = self.version
        copied.data = self.data.copy()
        return copied
    
    def set_val(self, row:int, col:int, val:int) -> None:
        self.data[row][col] = val
    
    def set_white(self, row:int, col:int) -> None:
        self.set_val(row, col, QRCodeCanvas.WHITE)
    
    def set_gray(self, row:int, col:int) -> None:
        self.set_val(row, col, QRCodeCanvas.GRAY)
    
    def set_black(self, row:int, col:int) -> None:
        self.set_val(row, col, QRCodeCanvas.BLACK)
    
    def set_by_list(self, info) -> None:
        for r, c, color in info:
            if color == 1:
                self.set_black(r, c)
            else:
                self.set_white(r, c)
    
    def do_masking(self, mask_func:object) -> None:
        info  = list()
        info += get_position_info(self.version)
        info += get_separator_info(self.version)
        info += get_alignment_info(self.version)
        info += get_timing_info(self.version)
        info += get_dark_module_info(self.version)
        info += get_version_info(self.version)
        info += get_format_info(self.version)
        
        forbid_pos = [ (x[0], x[1]) for x in info ]
        for r in range(self.data.shape[0]):
            for c in range(self.data.shape[1]):
                if (r, c) not in forbid_pos:
                    self.data[r][c] = (self.data[r][c]//255 ^ mask_func(r, c)) * QRCodeCanvas.BLACK

    def evaluate_mask_penalty(self) -> int:
        width = self.data.shape[0]
        penalty = [0, 0, 0, 0, 0]
        n = [0, 3, 3, 40, 10]
        
        def evaluate_feature_1():
            ## test row
            for r in range(width):
                curr_color = self.data[r][0]
                count = 1
                for c in range(1, width):
                    if self.data[r][c] == curr_color:
                        count += 1
                    else:
                        curr_color = self.data[r][c]
                        if count >= 5:
                            penalty[1] += n[1] + count - 5
                        count = 1
                if count >= 5:
                    penalty[1] += n[1] + count - 5
            ## test column
            for c in range(width):
                curr_color = self.data[0][c]
                count = 1
                for r in range(1, width):
                    if self.data[r][c] == curr_color:
                        count += 1
                    else:
                        curr_color = self.data[r][c]
                        if count >= 5:
                            penalty[1] += n[1] + count - 5
                        count = 1
                if count >= 5:
                    penalty[1] += n[1] + count - 5
        
        def evaluate_feature_2():
            block_sum = [ QRCodeCanvas.BLACK * 4, QRCodeCanvas.WHITE * 4 ]

            for r in range(width-1):
                for c in range(width-1):
                    if np.sum(self.data[r:r+2, c:c+2]) in block_sum:
                        penalty[2] += n[2]
        
        def evaluate_feature_3():
            ratio = [
                QRCodeCanvas.BLACK,
                QRCodeCanvas.WHITE,
                QRCodeCanvas.BLACK,
                QRCodeCanvas.BLACK,
                QRCodeCanvas.BLACK,
                QRCodeCanvas.WHITE,
                QRCodeCanvas.BLACK
            ]
            patterns = [
                ratio + [QRCodeCanvas.WHITE]*4,
                [QRCodeCanvas.WHITE]*4 + ratio
            ]
            
            for r in range(width):
                for c in range(width-10):
                    # evaluate horizontally
                    if list(self.data[r][c:c+11].reshape(-1)) in patterns:
                        penalty[3] += n[3]
                    # evaluate vertivally
                    if list(self.data.T[r][c:c+11].reshape(-1)) in patterns:
                        penalty[3] += n[3]
        
        def evaluate_feature_4():
            total_module = self.data.size
            total_black_module = np.sum(self.data == QRCodeCanvas.WHITE)
            proportion = total_black_module / total_module * 100
            k = int(abs(proportion - 50) // 5)
            penalty[4] += k * n[4]
        
        evaluate_feature_1()
        evaluate_feature_2()
        evaluate_feature_3()
        evaluate_feature_4()
        return sum(penalty)
    
    def show(self) -> None:
        show_array(self.data)
    
    def show_with_line(self) -> None:
        #XXX
        def add_quiet_zone(data:np.ndarray) -> np.ndarray:
            width = data.shape[0]
            new_width = width + 2
            new_data = np.zeros((new_width, new_width), dtype=int) + QRCodeCanvas.WHITE
            
            new_data[1:width+1, 1:width+1] = data
            return new_data
        
        def resize(data:np.ndarray) -> np.ndarray:
            width = data.shape[0]
            new_width = width * 9
            new_data = np.zeros((new_width, new_width), dtype=int)
            for r in range(width):
                for c in range(width):
                    new_data[9*r:9*(r+1), 9*c:9*(c+1)] = data[r][c]
            return new_data
        
        new_data = np.array(Image.fromarray(resize(255 - self.data)).convert('RGB'))
        for r in range(0, new_data.shape[0], 9):
            for c in range(new_data.shape[1]):
                if r == 0:
                    continue
                new_data[r][c] = np.asarray([255, 115, 0])
        for c in range(0, new_data.shape[0], 9):
            for r in range(new_data.shape[1]):
                if c == 0:
                    continue
                new_data[r][c] = np.asarray([255, 115, 0])
        
        show_array(new_data, cmap='viridis')
    
    def save(self, path:str, mul:int=9) -> None:
        def add_quiet_zone(data:np.ndarray) -> np.ndarray:
            width = data.shape[0]
            new_width = width + 8
            new_data = np.zeros((new_width, new_width), dtype=int) + QRCodeCanvas.WHITE
            
            new_data[4:width+4, 4:width+4] = data
            return new_data
        
        def resize(data:np.ndarray) -> np.ndarray:
            width = data.shape[0]
            new_width = width * mul
            new_data = np.zeros((new_width, new_width), dtype=int)
            for r in range(width):
                for c in range(width):
                    new_data[mul*r:mul*(r+1), mul*c:mul*(c+1)] = data[r][c]
            return new_data
        
        save_array_as_image(resize(add_quiet_zone(self.data)) / 255, path)

class QRCodeExtractor:
    def __init__(self):
        self.a = 0
        return
    
    def __remove_quiet_zone(self, array):
        upper_left, upper_right, lower_left = None, None, None
        # search the upper-left black pixel
        for r in range(array.shape[0]):
            for c in range(array.shape[1]):
                if array[r][c] == 1:
                    upper_left = (r, c)
                    break
            if upper_left is not None:
                break
        # search the upper-right black pixel
        for r in range(array.shape[0]):
            for c in reversed(range(array.shape[1])):
                if array[r][c] == 1:
                    upper_right = (r, c)
                    break
            if upper_right is not None:
                break
        # search the lower-left black pixel
        for r in reversed(range(array.shape[0])):
            for c in range(array.shape[1]):
                if array[r][c] == 1:
                    lower_left = (r, c)
                    break
            if lower_left is not None:
                break
        # determine the boundaries of the area that excludes the quiet zone
        upper_boundary = min(upper_left[0], upper_right[0])
        lower_boundary = lower_left[0]
        left_boundary  = min(upper_right[1], lower_left[1])
        right_boundary = upper_right[1]
        # allocate the new matrix without the quiet zone
        new_array = list()
        for r in range(upper_boundary, lower_boundary+1):
            row = list()
            for c in range(left_boundary, right_boundary+1):
                row.append(array[r][c])
            new_array.append(row)
        new_array = np.asarray(new_array)
        assert new_array.shape[0] == new_array.shape[1], 'This QR Code is not valid.'
        return new_array
    
    def __normalize_matrix(self, array):
        # check the size for representing a module
        for size in range(array.shape[0]+1):
            if array[size][size] == 0:
                break
        # allocate the normalized matrix
        new_array = list()
        for r in range(0, array.shape[0], size):
            row = list()
            for c in range(0, array.shape[1], size):
                row.append(array[r][c])
            new_array.append(row)
        return np.asarray(new_array)
    
    def extract(self, path):
        try:
            array = self.extract_from_original(path)
        except:
            array = self.extract_from_image(path)
        assert array.shape[0] == array.shape[1], 'extraction fail'
        assert (array.shape[0] - 21) % 4 == 0, 'extraction fail'
        return array
    
    def extract_from_original(self, path):
        array = load_image_as_array(path)
        array = self.__remove_quiet_zone(array)
        array = self.__normalize_matrix(array)
        return array
    
    def extract_from_image(self, path):
        print('-'*20, path, '-'*20)
        print('Process: Reading image')
        show_array(np.asarray(Image.open(path).convert('RGB')))
        img = np.asarray(Image.open(path).convert('L'))
        h, w = img.shape
        if img.size > 500000:
            img = bilinear_interpolation(img, 0.5)
        show_array(img)

        print('Process: Gray level slicing')
        img_g = gray_level_slicing(img, 100)
        show_array(img_g)

        print('Process: Convoluting with Sobel operator in x and y direction')
        img_conv = sobel_operation(img_g)
        img_g_conv = gray_level_slicing(scale(img_conv), 200)
        show_array(img_g_conv)

        print('Process: Finding patterns')
        import time
        starttime = time.time()
        group = pattern_detector(img_g_conv)
        interval = time.time() - starttime
        print('Running Time: %dm %ds' % (interval // 60, interval % 60))

        print('\n\nProcess: Finding the locations of the three position patterns')
        position = find_position(group)
        tmat = img_g_conv.copy()
        for r, c in position[0]:
            highlight_pos = [(r + x, c + y) for x in range(-10, 11) for y in range(-10, 11)]
            for r, c in highlight_pos:
                tmat[r][c] = 255
        show_array(tmat)

        tmat = np.zeros_like(img_g_conv)
        for r, c in position[1][0] + position[1][1] + position[1][2]:
            tmat[r][c] = 255
        for r, c in position[0]:
            highlight_pos = [(r + x, c + y) for x in range(-3, 4) for y in range(-3, 4)]
            for r, c in highlight_pos:
                tmat[r][c] = 255
        show_array(tmat)

        print('Process: Find the four vertices of QR Code')
        from_points = get_qrcode_vertex(img_g_conv, *position)
        tmat = img_g_conv.copy()
        for r, c in from_points:
            highlight_pos = [(r + x, c + y) for x in range(-3, 4) for y in range(-3, 4)]
            for r, c in highlight_pos:
                tmat[r][c] = 255
        show_array(tmat)
        img_fp = img.copy()
        for r, c in from_points:
            highlight_pos = [(r + x, c + y) for x in range(-3, 4) for y in range(-3, 4)]
            for r, c in highlight_pos:
                img_fp[r][c] = 255
        show_array(img_fp)

        print('Process: Perspective transform')
        from_points = [(y, x) for x, y in from_points]
        img_p = perspective_transform(img, from_points)
        img_p = gray_level_slicing(img_p, 100)
        show_array(np.logical_not((img_p / 255)).astype(int), cmap='binary')

        print('Process: Resampling')
        img_r = resampling(img_p)

        print('Process: Result')
        img_r = np.logical_not((img_r / 255)).astype(int)
        show_array(img_r, cmap='binary')
        return img_r
