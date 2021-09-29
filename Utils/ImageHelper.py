import warnings
warnings.filterwarnings('ignore')
from math import floor, ceil
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from array2gif import write_gif
from numba import jit
plt.style.use('seaborn-dark')

# cnt = 0
imgs = list()

def load_image_as_array(path:str) -> np.ndarray:
    return (1 - (np.asarray(Image.open(path).convert('L')) / 255)).astype(int)

def save_array_as_image(array:np.ndarray, path:str) -> None:
    Image.fromarray((1 - array) * 255).convert('L').save(path)

def show_array(array:np.ndarray, cmap:str='binary') -> None:
    plt.imshow(array, cmap=cmap)
    plt.axis('off')
    
    imgs.append(array.copy())
#     global cnt
#     plt.savefig('./Images/__' + str(cnt) + '.png', dpi=100, bbox_inches='tight')
#     cnt += 1
    
    plt.show()

def clean_imgs():
    imgs = list()

def do_animation(fps=10):
    for x in range(len(imgs)):
        imgs[x] = imgs[x].transpose((1, 0, 2))
    
    for x in range(10):
        imgs.append(imgs[-1])
    
    print('num of imgs:', len(imgs))
    write_gif(imgs, './Images/__out.gif', fps=fps)
    print('done')

def gray_level_slicing(mat, thres):
    mat = mat.copy()
    lt_indices = mat < thres
    mat[lt_indices] = 0
    mat[np.logical_not(lt_indices)] = 255
    return mat

@jit
def conv2d(mat, kernel):
    conved_mat = np.zeros(mat.shape)
    # assume the kernel size is 2-dimensional
    kh = kernel.shape[0] # kernel height
    kw = kernel.shape[1] # kernel width
    h = mat.shape[0] # mat height
    w = mat.shape[1] # mat width
    # construct padded mat
    padded_mat = np.zeros((h + kh - 1, w + kw - 1))
    ## fill original pixel
    padded_mat[kh//2 : h + kh//2, kw//2 : w + kw//2] = mat
    ## pad top edge and bottom edge
    for x in range(kh//2):
        for y in range(w):
            padded_mat[x][y+kw//2] = mat[0][y] # pad top
            padded_mat[h+kh//2+x][y+kw//2] = mat[h-1][y] # pad bottom
    ## pad left edge and right edge
    for x in range(h):
        for y in range(kw//2):
            padded_mat[x+kh//2][y] = mat[x][0] # pad left
            padded_mat[x+kh//2][w+kw//2+y] = mat[x][w-1] # pad right
    ## pad four vertices
    for x in range(kh//2):
        for y in range(kw//2):
            padded_mat[x][y] = mat[0][0] # pad upper left
            padded_mat[h+kh//2+x][y] = mat[h-1][0] # pad upper right
            padded_mat[x][w+kw//2+y] = mat[0][w-1] # pad lower left
            padded_mat[h+kh//2+x][w+kw//2+y] = mat[h-1][w-1] # pad lower right
    # start calculation
    for x in range(h):
        for y in range(w):
            for i in range(-1*(kh//2), kh//2 + 1):
                for j in range(-1*(kw//2), kw//2 + 1):
                    conved_mat[x][y] += \
                        padded_mat[x+i+kh//2][y+j+kw//2] * kernel[i + kh//2][j + kw//2]
    return conved_mat


@jit
def bilinear_interpolation(mat, zoom_rate):
    height = mat.shape[0] # height for original image
    width = mat.shape[1] # width for original image
    zoom_height = round(height * zoom_rate) # height for new image
    zoom_width = round(width * zoom_rate) # width for new image
    zoom_mat = np.zeros((zoom_height, zoom_width)) # creating a numpy mat for new image
    # start image processing
    for x in range(zoom_height):
        for y in range(zoom_width):
            # set up necessary variables
            floor_x, floor_y = floor(x / zoom_rate), floor(y / zoom_rate)
            ceil_x, ceil_y = ceil(x / zoom_rate), ceil(y / zoom_rate)
            x1, x2 = round(floor_x * zoom_rate), round(ceil_x * zoom_rate)
            y1, y2 = round(floor_y * zoom_rate), round(ceil_y * zoom_rate)
            idx_x1 = floor_x if floor_x >= 0 else 0
            idx_x2 = ceil_x if ceil_x < height - 1 else height - 1
            idx_y1 = floor_y if floor_y >= 0 else 0
            idx_y2 = ceil_y if ceil_y < width - 1 else width - 1
            fq11 = mat[idx_x1][idx_y1]
            fq12 = mat[idx_x1][idx_y2]
            fq21 = mat[idx_x2][idx_y1]
            fq22 = mat[idx_x2][idx_y2]
            # computation starts
            if (x2 - x1)*(y2 - y1):
                zoom_mat[x][y] = \
                    (fq11*(x2 - x)*(y2 - y) + \
                     fq21*(x - x1)*(y2 - y) + \
                     fq12*(x2 - x)*(y - y1) + \
                     fq22*(x - x1)*(y - y1)) / ((x2 - x1)*(y2 - y1))
            else:
                if (x2 - x1):
                    zoom_mat[x][y] = fq11/2 + fq21/2
                elif (y2 - y1):
                    zoom_mat[x][y] = fq11/2 + fq12/2
                else:
                    zoom_mat[x][y] = fq11
    return zoom_mat

def sobel_operation(mat):
    sobelx = np.asarray([
        [1, 0, -1],
        [2, 0, -2],
        [1, 0, -1]
    ])
    sobely = np.asarray([
        [ 1,  2,  1],
        [ 0,  0,  0],
        [-1, -2, -1]
    ])
    sobeled_mat = np.sqrt(conv2d(mat, sobelx)**2 + conv2d(mat, sobely)**2)
    return sobeled_mat

def scale(mat):
    mat = mat.copy()
    # scale the value from [min_val, max_val] into [0, 255] for displaying
    min_val = np.min(mat)
    max_val = np.max(mat)
    multiplier = 255 / (max_val - min_val)
    mat -= min_val
    mat *= multiplier
    return np.clip(mat, 0, 255)

def pattern_detector(mat):
    edge_indices = np.argwhere(mat == 255).tolist()
    edge_indices = [tuple(x) for x in edge_indices]
    direction = [
        (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1)
    ]
    group = list()
    current_group = list()
    while edge_indices:
        print('\r%6d' % len(edge_indices), end='')
        p = edge_indices[0]
        current_group.append(p)
        edge_indices.remove(p)
        possible_loc = [(p[0] + x[0], p[1] + x[1]) for x in direction]
        candidate_loc = list()
        for loc in possible_loc:
            if loc in edge_indices:
                candidate_loc.append(loc)
        while candidate_loc:
            print('\r%6d' % len(edge_indices), end='')
            p = candidate_loc[0]
            current_group.append(p)
            edge_indices.remove(p)
            candidate_loc.remove(p)
            possible_loc = [(p[0] + x[0], p[1] + x[1]) for x in direction]
            for loc in possible_loc:
                if loc in edge_indices and loc not in candidate_loc:
                    candidate_loc.append(loc)
        group.append(current_group)
        current_group = list()
    print('\r        ')
    return group

def find_position(group):
    centroids = list()
    for g in group:
        if len(g) < 4:
            c = (-1, -1)
            continue
        max_r, min_r, max_c, min_c = -np.inf, np.inf, -np.inf, np.inf
        for p in g:
            if p[0] > max_r: max_r = p[0]
            if p[0] < min_r: min_r = p[0]
            if p[1] > max_c: max_c = p[1]
            if p[1] < min_c: min_c = p[1]
        c = ((max_r + min_r) / 2, (max_c + min_c) / 2)
        centroids.append(c)
    count = dict()
    dist = lambda a, b: ((a[0] - b[0])**2 + (a[1] - b[1])**2) ** (1/2)
    for c in centroids:
        if c == (-1, -1):
            continue
        if not count:
            count[c] = 1
        else:
            for key in count.keys():
                if dist(key, c) < 2:
                    count[key] += 1
                    break
            else:
                count[c] = 1
    position = list()
    for k, v in zip(count.keys(), count.values()):
        if v == 3:
            r, c = centroids[centroids.index(k)]
            position.append((int(round(r)), int(round(c))))
    position_pattern = list()
    for pos in position:
        candidate_pattern = list()
        for g in group:
            if len(g) < 4:
                continue
            max_r, min_r, max_c, min_c = -np.inf, np.inf, -np.inf, np.inf
            for p in g:
                if p[0] > max_r: max_r = p[0]
                if p[0] < min_r: min_r = p[0]
                if p[1] > max_c: max_c = p[1]
                if p[1] < min_c: min_c = p[1]
            c = ((max_r + min_r) / 2, (max_c + min_c) / 2)
            if dist(c,  pos) < 2:
                candidate_pattern.append(g)
        max_cand = list()
        for cand in candidate_pattern:
            if len(cand) > len(max_cand):
                max_cand = cand
        position_pattern.append(max_cand)
    return position, position_pattern

def get_qrcode_vertex(mat, position, position_pattern):
    assert len(position) == 3, 'Error: Len = %d not fit.' % len(position)
    dist = lambda a, b: ((a[0] - b[0])**2 + (a[1] - b[1])**2) ** (1/2)
    len01 = dist(position[0], position[1])
    len02 = dist(position[0], position[2])
    len12 = dist(position[1], position[2])
    
    if max(len01, len02, len12) == len01:
        p1, p2, pa = position
    elif max(len01, len02, len12) == len02:
        p1, pa, p2 = position
    else:
        pa, p1, p2 = position
    
    if pa[0] < (p1[0] + p2[0]) // 2 and p1[1] < p2[1]:
        p1, p2 = p2, p1
    elif pa[0] > (p1[0] + p2[0]) // 2 and p2[1] < p1[1]:
        p1, p2 = p2, p1
    else:
        assert True, 'what the fuck?'
    
    midp = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
    ratio = 2
    
    vec = lambda p, mid: (p[0] - mid[0], p[1] - mid[1])
    add = lambda p, q: (p[0] + q[0], p[1] + q[1])
    mul = lambda v, c: (v[0] * c, v[1] * c)
    rnd = lambda v: (int(round(v[0])), int(round(v[1])))
    adjust = lambda p, mid, r: rnd(add(p, mul(vec(p, mid), r)))
    
    ppa, pp1, pp2 = adjust(pa, midp, 1), adjust(p1, midp, 1), adjust(p2, midp, 1)
    dppapa, dpp1p1, dpp2p2 = dist(ppa, pa), dist(pp1, p1), dist(pp2, p2)
    
    all_position_pattern = list()
    for x in position_pattern:
        all_position_pattern.extend(x)
    
    for p in all_position_pattern:
        if dist(ppa, p) < dppapa:
            pa = p
            dppapa = dist(ppa, pa)
        if dist(pp1, p) < dpp1p1:
            p1 = p
            dpp1p1 = dist(pp1, p1)
        if dist(pp2, p) < dpp2p2:
            p2 = p
            dpp2p2 = dist(pp2, p2)
    
    dpap1p, dpap2p = -np.inf, -np.inf
    p1p, p2p = p1, p2
    for p in all_position_pattern:
        if dist(pa, p) > dpap1p and dist(p, p1) < dist(p, p2):
            p1p = p
            dpap1p = dist(pa, p1p)
        if dist(pa, p) > dpap2p and dist(p, p2) < dist(p, p1):
            p2p = p
            dpap2p = dist(pa, p2p)
    
    x1, y1, x2, y2 = p1[1], p1[0], p1p[1], p1p[0]
    x3, y3, x4, y4 = p2[1], p2[0], p2p[1], p2p[0]
    
    a = (y2 - y1) / (x2 - x1 + 1e-15)
    b = (y1*x2 - x1*y2) / (x2 - x1 + 1e-15)
    c = (y4 - y3) / (x4 - x3 + 1e-15)
    d = (y3*x4 - x3*y4) / (x4 - x3 + 1e-15)
    
    pb = rnd(((a*d - b*c) / (a - c), (d - b) / (a - c)))
    return pa, p1, pb, p2
    
def perspective_transform(mat, from_points):
    def solve_coef(to_points, from_points):
        A = np.zeros((8, 8))
        B = np.array(from_points).reshape(-1)
        for i, p in enumerate(zip(to_points, from_points)):
            tp, fp = p
            A[i*2]   = [tp[0], tp[1], 1, 0, 0, 0, -fp[0]*tp[0], -fp[0]*tp[1]]
            A[i*2+1] = [0, 0, 0, tp[0], tp[1], 1, -fp[1]*tp[0], -fp[1]*tp[1]]
        coef = (np.linalg.inv(A.T @ A) @ A.T @ B).reshape(-1)
        return coef
    
    dist = lambda a, b: ((a[0] - b[0])**2 + (a[1] - b[1])**2) ** (1/2)
    size = int(round(dist(from_points[0], from_points[1])))
    to_points = [(0, 0), (size, 0), (size, size), (0, size)]
    coef = solve_coef(to_points, from_points)
    
    mat = Image.fromarray(mat)
    mat = mat.transform(
        (size, size),
        Image.PERSPECTIVE,
        coef,
        Image.BILINEAR
    )
    return np.asarray(mat.convert('L'))

def resampling(mat):
    block_cnt = list()
    for r in range(mat.shape[0]):
        b_cnt = 0
        color = mat[r][0]
        for c in range(mat.shape[1]):
            b_cnt += 1
            if mat[r][c] != color:
                color = mat[r][c]
                block_cnt.append(b_cnt)
                b_cnt = 0
    for c in range(mat.shape[1]):
        b_cnt = 0
        color = mat[0][c]
        for r in range(mat.shape[0]):
            b_cnt += 1
            if mat[r][c] != color:
                color = mat[r][c]
                block_cnt.append(b_cnt)
                b_cnt = 0
    
    block_hist = np.bincount(block_cnt)
    sorted_arg = np.argsort(block_hist)
    e_block_size = np.average(sorted_arg[-1:], weights=block_hist[sorted_arg[-1:]])
    e_module_num = int(round(mat.shape[0] / e_block_size))
    e_version = int(round((e_module_num - 21) / 4)) + 1
    module_num = (e_version - 1)*4 + 21
    block_size = int(round(e_block_size))
    
    bs = int(round(e_block_size))
    half_bs = int(round(e_block_size / 2))
    re_mat = np.zeros((module_num, module_num), dtype=np.int16)
    
    def print_line(img, row_idx, col_idx):
        img_rgb = np.array(Image.fromarray(img).convert('RGB'))
        for r in row_idx:
            for c in range(img.shape[1]):
                img_rgb[r][c] = np.asarray([255, 0, 0])
        for c in col_idx:
            for r in range(img.shape[0]):
                img_rgb[r][c] = np.asarray([255, 0, 0])
        show_array(img_rgb)
        return
    
    sobeled_img = gray_level_slicing(scale(sobel_operation(mat)), 150)
    ratio = 0.6
    while True:
        row_line_idx = list()
        prev_num = -np.inf
        for r in range(sobeled_img.shape[0]):
            num = np.sum(sobeled_img[r, ...] == 255)
            if num > prev_num and (not len(row_line_idx) or row_line_idx[-1] + block_size*ratio < r):
                row_line_idx.append(r + 1 if r + 1 < mat.shape[0] else r)
            prev_num = num
        if len(row_line_idx) == module_num:
            row_line_idx.append(mat.shape[0] - 1)
        if len(row_line_idx) == module_num + 1 or ratio > 1:
            break
        ratio += 0.1
    
    ratio = 0.6
    while True:
        col_line_idx = list()
        prev_num = -np.inf
        for c in range(sobeled_img.shape[1]):
            num = np.sum(sobeled_img[..., c] == 255)
            if num > prev_num and (not len(col_line_idx) or col_line_idx[-1] + block_size*ratio < c):
                col_line_idx.append(c + 1 if c + 1 < mat.shape[1] else c)
            prev_num = num
        if len(col_line_idx) == module_num:
            col_line_idx.append(mat.shape[1] - 1)
        if len(col_line_idx) == module_num + 1 or ratio > 1:
            break
        ratio += 0.1
    
    print_line(mat, row_line_idx, col_line_idx)
    
    for r in range(len(row_line_idx) - 1):
        for c in range(len(col_line_idx) - 1):
            hist = \
            np.bincount(
                np.reshape(
                    mat[row_line_idx[r]+1:row_line_idx[r+1], col_line_idx[c]+1:col_line_idx[c+1]],
                    -1
                )
            )
            re_mat[r][c] = np.argmax(hist)
    
    return re_mat
