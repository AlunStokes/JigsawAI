import numpy as np

def is_ordered(p):
    n = p[0]
    i = 1
    while i < len(p):
        if n > p[i]:
            return False
        n = p[i]
        i += 1
    return True

def split_image(im, dim):
    sq_dim = im.shape[0]//dim
    images = np.ndarray((dim * dim, sq_dim, sq_dim, 3))
    sq_num = 0
    i = 0
    while i < dim:
        j = 0
        while j < dim:
            images[sq_num] = im[i * sq_dim: (i + 1) * sq_dim, j * sq_dim: (j + 1) * sq_dim, :]
            sq_num += 1
            j += 1
        i += 1
    return images

def stitch_image(ims, order='default'):
    sq_dim = int(ims.shape[0] ** (1/2))
    if isinstance(order, str):
        order = np.array(list(range(0, sq_dim * sq_dim)))
    else:
        if len(order) != sq_dim * sq_dim:
            raise Exception("Length of permutation must be equal to number of images")

    im = np.ndarray((sq_dim * ims.shape[1], sq_dim * ims.shape[1], 3))

    x_count = 0
    y_count = 0

    for i in order:
        im[x_count * ims.shape[1]:(x_count + 1) * ims.shape[1], y_count * ims.shape[1]:(y_count + 1) * ims.shape[1], :] = ims[i]
        if (y_count + 1) % sq_dim == 0:
            y_count = 0
            x_count += 1
        else:
            y_count += 1

    return im

def generate_matches_north(dim, perm_mat):
    n_mat = np.zeros((dim, dim, 1))
    l = 0
    while l < dim:
        m = 0
        while m < dim:
            if l == 0:
                if perm_mat[l,m] in np.linspace(0, dim - 1, dim):
                    n_mat[l,m] = 1
            else:
                if (perm_mat[l,m] - perm_mat[l - 1, m]) == dim:
                     n_mat[l,m] = 1
            m += 1
        l += 1
    return n_mat

def generate_matches_east(dim, perm_mat):
    e_mat = np.zeros((dim, dim, 1))
    l = 0
    while l < dim:
        m = 0
        while m < dim:
            if m == dim - 1:
                if perm_mat[l,m][0] in np.linspace(dim - 1, dim**2 - 1, dim):
                    e_mat[l,m] = 1
            else:
                if (perm_mat[l,m + 1] - perm_mat[l, m]) == 1 and perm_mat[l,m] not in np.linspace(dim - 1, dim**2 - 1, dim):
                    e_mat[l,m] = 1
            m += 1
        l += 1
    return e_mat

def generate_matches_south(dim, perm_mat):
    s_mat = np.zeros((dim, dim, 1))
    l = 0
    while l < dim:
        m = 0
        while m < dim:
            if l == dim - 1:
                if perm_mat[l,m] in np.linspace(dim**2 - dim, dim**2 - 1, dim):
                    s_mat[l,m] = 1
            else:
                if (perm_mat[l+1,m] - perm_mat[l, m]) == dim:
                     s_mat[l,m] = 1
            m += 1
        l += 1
    return s_mat

def generate_matches_west(dim, perm_mat):
    w_mat = np.zeros((dim, dim, 1))
    l = 0
    while l < dim:
        m = 0
        while m < dim:
            if m == 0:
                if perm_mat[l,m][0] in np.linspace(0, dim**2 - dim, dim):
                    w_mat[l,m] = 1
            else:
                if (perm_mat[l,m] - perm_mat[l, m - 1]) == 1 and perm_mat[l,m] not in np.linspace(0, dim**2 - dim, dim):
                    w_mat[l,m] = 1
            m += 1
        l += 1
    return w_mat

def stitch_image_masked(ims, order):
    sq_dim = int(ims.shape[0] ** (1/2))

    im = np.ndarray((sq_dim * ims.shape[1], sq_dim * ims.shape[1], 3))

    x_count = 0
    y_count = 0

    for i in order:
        if i == -1:
            im[x_count * ims.shape[1]:(x_count + 1) * ims.shape[1], y_count * ims.shape[1]:(y_count + 1) * ims.shape[1], :] = np.zeros(ims[i].shape)
        else:
            im[x_count * ims.shape[1]:(x_count + 1) * ims.shape[1], y_count * ims.shape[1]:(y_count + 1) * ims.shape[1], :] = ims[i]
        if (y_count + 1) % sq_dim == 0:
            y_count = 0
            x_count += 1
        else:
            y_count += 1

    return im

def permutation_to_mask(n):
    j = 0
    while j < len(n):
        if n[j] == 0:
            n[j] = -1
        else:
            n[j] = j
        j += 1

    return n
