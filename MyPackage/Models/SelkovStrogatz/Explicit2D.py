import numpy as np
from numba import njit

@njit(fastmath=True)
def calc_next(U, V, D_u, D_v, coeff_h_t, d_t, a, b):
    Utop = U[0:-2, 1:-1]
    Uleft = U[1:-1, 0:-2]
    Ubottom = U[2:, 1:-1]
    Uright = U[1:-1, 2:]
    Ucenter = U[1:-1, 1:-1]
    U_laplacian = (Utop + Uleft + Ubottom + Uright - 4 * Ucenter)

    Vtop = V[0:-2, 1:-1]
    Vleft = V[1:-1, 0:-2]
    Vbottom = V[2:, 1:-1]
    Vright = V[1:-1, 2:]
    Vcenter = V[1:-1, 1:-1]
    V_laplacian = (Vtop + Vleft + Vbottom + Vright - 4 * Vcenter)
    
    U[1:-1, 1:-1], V[1:-1, 1:-1] = \
        Ucenter - d_t * Ucenter + d_t * a * Vcenter + d_t * Ucenter ** 2 * Vcenter + coeff_h_t * D_u * U_laplacian, \
        Vcenter + d_t * b - d_t * a * Vcenter -  d_t * Ucenter ** 2 * Vcenter + coeff_h_t * D_v * V_laplacian

    for Z in (U, V):
        Z[0, :] = Z[1, :]
        Z[-1, :] = Z[-2, :]
        Z[:, 0] = Z[:, 1]
        Z[:, -1] = Z[:, -2]

    return U, V

@njit(fastmath=True)
def calc_n_iters(U, V, D_u, D_v, coeff_h_t, d_t, a, b, n_iters, steps_to_save):
    res_u = np.empty((len(steps_to_save) + 2, *U.shape),dtype=np.float32)
    res_v = np.empty((len(steps_to_save)+ 2, *V.shape),dtype=np.float32)
    res_u[0] = U
    res_v[0] = V
    i = 1
    for index in range(n_iters):        
        if index in steps_to_save:
            res_u[i] = U
            res_v[i] = V
            i+=1
        U,V = calc_next(U, V, D_u, D_v, coeff_h_t, d_t, a, b)
    res_u[-1] = U
    res_v[-1] = V
    return res_u, res_v