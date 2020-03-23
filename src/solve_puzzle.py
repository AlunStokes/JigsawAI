import numpy as np
from skimage.io import imread, imsave
import matplotlib.pyplot as plt

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def disp_image(im_arr):
    dim = np.sqrt(im_arr.shape[0]).astype(int)
    k = 1
    i = 0
    while i < dim:
        j = 0
        while j < dim:
            plt.subplot(dim, dim, k)
            plt.imshow(im_arr[k - 1])
            j += 1
            k += 1
        i += 1
    plt.show()

def arr_to_im(im_arr):
    dim = np.sqrt(im_arr.shape[0]).astype(int)
    piece_size = im_arr.shape[1]
    im = np.zeros((piece_size * dim, piece_size * dim, 3)).astype(np.uint8)
    k = 0
    i = 0
    while i < dim:
        j = 0
        while j < dim:
            im[j * piece_size : (j + 1) * piece_size, i * piece_size : (i + 1) * piece_size] = im_arr[k]
            j += 1
            k += 1
        i += 1
    return im

def segment_image(im, dim):
    if im.shape[0] / dim != im.shape[0] // dim:
        raise Exception("Image dimension must be divisible by dim")
    elif im.shape[0] != im.shape[1]:
        raise Exception("Image must be square")
    piece_size = im.shape[0] // dim
    im_arr = []
    i = 0
    while i < dim:
        j = 0
        while j < dim:
            im_arr.append(im[i*piece_size:(i+1)*piece_size, j*piece_size:(j+1)*piece_size])
            j += 1
        i += 1
    return np.array(im_arr)

def find_adjacent(board, x, y):
    l = []
    shifts = [(0,-1), (1,0), (0,1), (-1,0)]
    for shift in shifts:
        if 0 <= x + shift[0] < board.shape[0] and  0 <= y + shift[1] < board.shape[1]:
            if board[x + shift[0], y + shift[1]] != -1:
                l.append((board[x + shift[0], y + shift[1]], shift))
        else:
            continue
    return l

def check_pieces(board, indices, x, y):
    pieces_to_check = find_adjacent(board, x, y)
    if len(pieces_to_check) == 0:
        return -1
    piece = pieces_to_check[-1]

    i = 0
    while i < indices.shape[0]:
        j = 0
        while j < indices.shape[0]:
            if board[x + int(piece[1][0]), y + int(piece[1][1])] == indices[i, j]:
                if (not 0 <= i - int(piece[1][0]) < indices.shape[0]) or (not 0 <= j - int(piece[1][1]) < indices.shape[1]):
                    return -1
                return  indices[i - int(piece[1][0]), j - int(piece[1][1])]
            j += 1
        i += 1
    return -1

def num_placed(board):
    n = 0
    i = 0
    while i < board.shape[0]:
        j = 0
        while j < board.shape[1]:
            if board[i,j] != -1:
                n += 1
            j += 1
        i += 1
    return n

def reduce_board(board):
    #From top
    top_row = 0
    i = 0
    while i < board.shape[0]:
        j = 0
        non_neg = False
        while j < board.shape[1]:
            if board[i,j] != -1:
                non_neg = True
                break
            j += 1
        if non_neg:
            break
        top_row += 1
        i += 1

    #From right
    right_col = 0
    i = board.shape[0] - 1
    while i >= 0:
        j = board.shape[1] - 1
        non_neg = False
        while j >= 0:
            if board[j,i] != -1:
                non_neg = True
                break
            j -= 1
        if non_neg:
            break
        right_col += 1
        i -= 1

    #From bottom
    bottom_row = 0
    i = board.shape[0] - 1
    while i >= 0:
        j = board.shape[1] - 1
        non_neg = False
        while j >= 0:
            if board[i,j] != -1:
                non_neg = True
                break
            j -= 1
        if non_neg:
            break
        bottom_row += 1
        i -= 1

    #From left
    left_col = 0
    i = 0
    while i < board.shape[0]:
        j = 0
        non_neg = False
        while j < board.shape[1]:
            if board[j,i] != -1:
                non_neg = True
                break
            j += 1
        if non_neg:
            break
        left_col += 1
        i += 1

    return board[top_row:board.shape[1] - bottom_row, left_col:board.shape[0] - right_col]

def reconstruct_from_pos(im_arr, board, indices_shuffled):
    dim = np.sqrt(len(im_arr)).astype(int)
    piece_size = im_arr.shape[1]

    im = np.zeros((piece_size * board.shape[0], piece_size * board.shape[1], 3))

    i = 0
    while i < board.shape[0]:
        j = 0
        while j < board.shape[1]:
            im[i * piece_size : (i + 1) * piece_size, j * piece_size : (j + 1) * piece_size] = im_arr[indices_shuffled[i,j]]
            j += 1
        i += 1

    return im.astype(np.uint8)


def solve(im_arr):
    NORTH, S, W, E = (0, -1), (0, 1), (-1, 0), (1, 0) # directions
    turn_right = {NORTH: E, E: S, S: W, W: NORTH} # old -> new direction

    dim = np.sqrt(len(im_arr)).astype(int)
    indices = np.arange(dim**2)

    im_arr, indices_shuffled = unison_shuffled_copies(im_arr, indices)

    im_shuffled = arr_to_im(im_arr)

    indices = np.reshape(indices, (dim,dim))
    indices_shuffled = np.reshape(indices_shuffled, (dim,dim))

    board = np.ones((2*dim + 1, 2*dim + 1)) - 2
    x = dim
    y = dim

    dx, dy = W

    board[x,y] = 0

    i = 1
    while num_placed(board) < dim**2:
        n_dx, n_dy = turn_right[dx,dy]
        n_x, n_y = x + n_dx, y + n_dy
        if (0 <= n_x < board.shape[0] and 0 <= n_y < board.shape[1] and board[n_x,n_y] == -1):
            x, y = n_x, n_y
            dx, dy = n_dx, n_dy
            if len(find_adjacent(board,x,y)) == 0:
                continue
            p = check_pieces(board, indices, x, y)
            #exit()
            board[x,y] = p

        else:
            x, y = x + dx, y + dy
            if not (0 <= x < board.shape[0] and 0 <= y < board.shape[1]):
                break
            else:
                if board[x,y] == -1:
                    p = check_pieces(board, indices, x, y)
                    board[x,y] = p

    board = reduce_board(board).astype(int)
    print(indices)
    print(board)
    im = reconstruct_from_pos(im_arr, board, indices_shuffled)

    return im, im_shuffled


if __name__ == '__main__':
    im = imread('./images/1.png')
    im_arr = segment_image(im, 3)
    im_r, im_shuffled = solve(im_arr)
    plt.subplot(1,2,1)
    plt.imshow(im_shuffled)
    plt.subplot(1,2,2)
    plt.imshow(im_r)
    plt.show()
