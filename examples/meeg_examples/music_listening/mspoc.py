import numpy as np
import scipy.linalg as la
import scipy.signal as signal

def mspoc(X, Y, **kwargs):
    """
    Multimodal Source Power Co-modulation Analysis (mSPoC)
    """
    # Параметры по умолчанию
    opt = {
        'n_component_sets': 1,
        'tau_vector': [0],
        'use_log': True,
        'n_random_initializations': 10,
        'max_optimization_iterations': 20,
        'kappa_tau': 10**-3,
        'kappa_y': 10**-3,
        'pca_Y_var_expl': 0.95,
        'eps': 10**-3,
        'mask': None,
        'Cxx': None,
        'Cxxe': None,
        'Cyy': None,
        'verbose': 1,
        'component_idx': 1,
        'is_white_y': 0
    }
    opt.update(kwargs)

    if opt['verbose'] > 0:
        print('\n--- Begin mSPoC analysis ---')

    tau = np.array(opt['tau_vector']).flatten()
    Nt = len(tau)
    
    if X is not None and len(X) > 0:
        Nex = X.shape[2]
    elif opt['Cxxe'] is not None and len(opt['Cxxe']) > 0:
        Nex = opt['Cxxe'].shape[2]
    else:
        raise ValueError('X or Cxxe must be given!')
        
    Ney = Y.shape[1]

    if Nex != Ney:
        raise ValueError('X and Y must have the same number of epochs!!!')
    Ne = Nex

    if opt['mask'] is None:
        opt['mask'] = np.ones(Ne, dtype=bool)
    else:
        opt['mask'] = np.array(opt['mask'], dtype=bool)
        
    # В Python индексация с 0
    if len(tau) - 1 > 0:
        opt['mask'][:len(tau)-1] = False

    # Отбеливание (Whitening)
    if opt['verbose'] > 0:
        print('   start whitening')

    Cxxe_w, Cxxe, Cxx, Mx = util_mspoc__prepare_X_signal(X, opt)

    # Вычитание временного среднего
    Y = Y - np.mean(Y, axis=1, keepdims=True)
    if 'My' in opt:
        My = opt['My']
        Y_w = My.T @ Y
    else:
        Y_w, My, _ = util_mspoc__prepare_Y_signal(Y, opt)

    if opt['verbose'] > 0:
        print(f'   using {Mx.shape[1]} X-components')
        print(f'   using {My.shape[1]} Y-components')

    n_component_sets = opt['n_component_sets']
    if n_component_sets > Mx.shape[1]:
        n_component_sets = Mx.shape[1]
        
    # Удаление выбросов
    x_rm_idx = detect_max_eigenvalue_epochs(Cxxe_w, None, False, 2)
    thr = np.mean(Y_w) + 2.5 * np.std(np.mean(Y_w, axis=0))
    y_rm_idx = detect_high_variance_epochs(Y_w, thr, False)

    rm_idx = np.union1d(x_rm_idx, y_rm_idx).astype(int)
    if opt['verbose'] > 1:
        print(f'   removing {len(rm_idx)} outlier trials.')
        
    Cxxe_w = np.delete(Cxxe_w, rm_idx, axis=2)
    Y_w = np.delete(Y_w, rm_idx, axis=1)
    
    # Обновляем маску
    mask_list = list(opt['mask'])
    for idx in sorted(rm_idx, reverse=True):
        del mask_list[idx]
    opt['mask'] = np.array(mask_list)

    # Оптимизация наборов фильтров
    Wx = np.zeros((Mx.shape[1], n_component_sets))
    Wy = np.zeros((My.shape[1], n_component_sets))
    Wtau = np.zeros((Nt, n_component_sets))
    r_values = np.zeros(n_component_sets)

    for k in range(n_component_sets):
        if opt['verbose'] > 0:
            print(f'  optimizing component set {k+1}/{n_component_sets}')
            
        if k > 0:
            Bx = la.null_space(Wx[:, :k].T)
            # ИЗМЕНЕНИЕ: Отменяем ортогональную проекцию для Y (дефляцию).
            # Теперь Y может свободно коррелировать с новыми компонентами X.
            By = np.eye(Wy.shape[0]) 
        else:
            Bx = np.eye(Wx.shape[0])
            By = np.eye(Wy.shape[0])
            
        Cxxe_dfl = np.zeros((Bx.shape[1], Bx.shape[1], Cxxe_w.shape[2]))
        for n in range(Cxxe_w.shape[2]):
            Cxxe_dfl[:, :, n] = Bx.T @ Cxxe_w[:, :, n] @ Bx
            
        Y_dfl = By.T @ Y_w
        
        wx, wy, wtau, r, _ = optimize_filters(Cxxe_dfl, Y_dfl, Mx @ Bx, My @ By, opt)
        
        Wx[:, k] = Bx @ wx
        Wy[:, k] = By @ wy
        Wtau[:, k] = wtau
        r_values[k] = r

    # Восстановление паттернов в исходном пространстве сенсоров
    Wx = Mx @ Wx
    Wy = My @ Wy

    for k in range(Wx.shape[1]):
        Wx[:, k] = Wx[:, k] / np.sqrt(Wx[:, k].T @ Cxx @ Wx[:, k])
        Wy[:, k] = Wy[:, k] / np.std(Wy[:, k].T @ Y)

    Ax = Cxx @ Wx @ la.inv(Wx.T @ Cxx @ Wx)
    Sy = Wy.T @ Y
    Ay = Y @ (Y.T @ Wy) @ la.inv(Sy @ Sy.T)

    Nx = Cxxe.shape[0]
    Cxxe_vec = Cxxe.reshape((Nx * Nx, Ne), order='F')
    corr_values = np.zeros(Ax.shape[1])
    Atau = np.zeros_like(Wtau)
    
    for k in range(Ay.shape[1]):
        Wtau[:, k] = np.sign(np.sum(Wtau[:, k])) * Wtau[:, k]
        
        mm_idx = np.argmax(np.abs(Ax[:, k]))
        sgn = np.sign(Ax[mm_idx, k])
        Ax[:, k] = sgn * Ax[:, k]
        Wx[:, k] = sgn * Wx[:, k]
            
        sy = Wy[:, k].T @ Y
        mm_idx = np.argmax(np.abs(Ay[:, k]))
        c = np.sign(Y[mm_idx, :] @ sy.T)
        Ay[:, k] = c * Ay[:, k]
        Wy[:, k] = c * Wy[:, k]
        
        wx_vec = np.outer(Wx[:, k], Wx[:, k]).reshape((Nx * Nx,), order='F')
        px = wx_vec.T @ Cxxe_vec
        px = px - np.mean(px)
        
        pxf = signal.lfilter(Wtau[:, k], [1.0], px)
        tmp = np.corrcoef(pxf, sy)
        corr_values[k] = tmp[0, 1]
        
        Pxe = np.zeros((Nt, Ne))
        for kk in range(Nt):
            Pxe[kk, :] = np.roll(px, tau[kk])
            
        Cpp = Pxe @ Pxe.T
        Atau[:, k] = Cpp @ Wtau[:, k]

    out = {
        'corr_values': corr_values,
        'Atau': Atau,
        'Wx': Wx,
        'Wy': Wy,
        'Wtau': Wtau,
        'Ax': Ax,
        'Ay': Ay
    }
    return Wx, Wy, Wtau, Ax, Ay, out


def optimize_filters(Cxxe, Y, BMx, BMy, opt):
    tau = np.array(opt['tau_vector']).flatten()
    Nx = Cxxe.shape[0]
    Ny = Y.shape[0]
    Ne = Cxxe.shape[2]
    Nt = len(tau)

    r_tmp = np.zeros(opt['n_random_initializations'])
    Wx_tmp = np.zeros((Nx, opt['n_random_initializations']))
    Wy_tmp = np.zeros((Ny, opt['n_random_initializations']))
    Wt_tmp = np.zeros((Nt, opt['n_random_initializations']))

    Cxxe_vec = Cxxe.reshape((Nx * Nx, Ne), order='F')
    mask = opt['mask']
    Cxx = np.mean(Cxxe[:, :, mask], axis=2)

    kappa_y = opt['kappa_y']
    kappa_tau = opt['kappa_tau']

    Dyy = BMy.T @ BMy
    Dyy = Dyy / np.trace(Dyy)
    Dpp = np.eye(Nt)
    
    Y_mask = Y[:, mask]
    Cyy = np.cov(Y_mask)
    if Cyy.ndim == 0:
        Cyy = np.array([[Cyy]])
    Cyy = Cyy / np.trace(Cyy)
    Cyy_inv = la.inv(Cyy + kappa_y * Dyy)

    for n in range(opt['n_random_initializations']):
        wx = np.random.randn(Nx)
        wx_vec = np.outer(wx, wx).reshape((Nx * Nx,), order='F')
        px = wx_vec.T @ Cxxe_vec
        
        if opt['use_log']:
            px = np.log(px)
            
        ii = 0
        converged = False
        r_last_iter = -np.inf
        r_iter = np.zeros(opt['max_optimization_iterations'])
        Pxe = np.zeros((Nt, Ne))
        
        while ii < opt['max_optimization_iterations'] and not converged:
            px = px - np.mean(px[mask])
            for k in range(Nt):
                Pxe[k, :] = np.roll(px, tau[k])
                
            wt, wy = my_reg_CCA2(Pxe[:, mask], Y[:, mask], kappa_tau, Dpp, Cyy_inv)
            
            Cxxe_vec_flt = signal.lfilter(wt, [1.0], Cxxe_vec[:, mask], axis=1)
            sy = wy.T @ Y
            sy = (sy - np.mean(sy[mask])) / np.std(sy[mask])
            
            Cxxz = (sy[mask] @ Cxxe_vec_flt.T).reshape((Nx, Nx), order='F') / Ne
            
            # Решение обобщенной задачи на собственные значения
            D, W = la.eig(Cxxz, Cxx)
            sorted_idx = np.argsort(np.abs(D))[::-1]
            W = W[:, sorted_idx]
            wx = np.real(W[:, 0])
            
            wx_vec = np.outer(wx, wx).reshape((Nx * Nx,), order='F')
            px = wx_vec.T @ Cxxe_vec
            if opt['use_log']:
                px = np.log(px)
                
            px_flt = signal.lfilter(wt, [1.0], px[mask])
            
            tmp_r = np.corrcoef(sy[mask], px_flt)
            r = tmp_r[0, 1]
            r_iter[ii] = r
            
            if opt['verbose'] > 2:
                print(f'   iter {ii+1:02d} --> abs(r) = {abs(r):.3f}')
                
            if abs(abs(r) - abs(r_last_iter)) < opt['eps']:
                converged = True
                
            r_last_iter = r
            ii += 1
            
        Wx_tmp[:, n] = wx
        Wy_tmp[:, n] = wy
        Wt_tmp[:, n] = wt
        r_tmp[n] = r_last_iter
        
        if opt['verbose'] > 1:
            conv_str = f"Converged after {ii} iterations!" if converged else f"Not converged after {ii} iterations!"
            print(f'   run {n+1} done with corr = {r_tmp[n]:.3f} -> {conv_str}')

    max_idx = np.argmax(np.abs(r_tmp))
    max_corr = r_tmp[max_idx]
    wx = Wx_tmp[:, max_idx]
    wy = Wy_tmp[:, max_idx]
    wt = Wt_tmp[:, max_idx]

    if opt['verbose'] > 0:
        print(f' Choosing result from run {max_idx+1}, with corr = {max_corr:g}')

    return wx, wy, wt, max_corr, None


def my_reg_CCA2(Pxe, Ye, kappa_tau, Dp, Cyy_inv):
    Ne = Ye.shape[1]
    
    Pxe = Pxe - np.mean(Pxe, axis=1, keepdims=True)
    Pxe = Pxe / np.mean(np.std(Pxe, axis=1, ddof=0))
    
    Cpp = (Pxe @ Pxe.T) / Ne
    Cpy = (Pxe @ Ye.T) / Ne
    Cyp = (Ye @ Pxe.T) / Ne
    
    Cpp = Cpp / np.trace(Cpp) + kappa_tau * Dp / np.trace(Dp)
    Cyy_inv_Cyp = Cyy_inv @ Cyp
    
    if Pxe.shape[0] > 1:
        M = la.solve(Cpp, Cpy) @ Cyy_inv_Cyp
        # Находим ведущий собственный вектор
        D, V = la.eig(M)
        max_idx = np.argmax(np.real(D))
        wt = np.real(V[:, max_idx])
    else:
        wt = np.array([1.0])
        
    wy = Cyy_inv_Cyp @ wt
    return wt, wy

def util_mspoc__prepare_X_signal(X, opt):
    """
    Пространственное отбеливание сигнала X и вычисление матриц ковариации.
    """
    Cxx = opt.get('Cxx', None)
    Cxxe = opt.get('Cxxe', None)
    mask = opt.get('mask', None)

    # Вычисление покадровых ковариаций, если они не заданы
    if Cxxe is not None and len(Cxxe) > 0:
        pass
    else:
        Nx = X.shape[1]
        Ne = X.shape[2]
        Cxxe = np.zeros((Nx, Nx, Ne))
        for e in range(Ne):
            # В MATLAB cov(X) считает переменными столбцы. 
            # rowvar=False в numpy делает то же самое.
            Cxxe[:, :, e] = np.cov(X[:, :, e], rowvar=False)

    # Усредненная ковариация
    if Cxx is not None and len(Cxx) > 0:
        pass
    else:
        if mask is not None:
            Cxx = np.mean(Cxxe[:, :, mask], axis=2)
        else:
            Cxx = np.mean(Cxxe, axis=2)

    # Декомпозиция (eigh предпочтительнее для симметричных матриц)
    D, V = la.eigh(Cxx)
    
    # Сортировка по убыванию
    sort_idx = np.argsort(D)[::-1]
    ev_sorted = D[sort_idx]
    V = V[:, sort_idx]

    # Оценка ранга данных
    tol = ev_sorted[0] * 1e-5
    r = np.sum(ev_sorted > tol)
    n_components = r

    # Фильтры отбеливания
    Mx = V @ np.diag(ev_sorted ** -0.5)
    Mx = Mx[:, :n_components]

    # Отбеливание покадровых матриц ковариации
    Cxxe_w = np.zeros((n_components, n_components, Cxxe.shape[2]))
    for e in range(Cxxe.shape[2]):
        Cxxe_w[:, :, e] = Mx.T @ Cxxe[:, :, e] @ Mx

    return Cxxe_w, Cxxe, Cxx, Mx


def util_mspoc__prepare_Y_signal(Y, opt):
    """
    Отбеливание сигнала Y и опциональное снижение размерности (PCA).
    """
    pca_Y_var_expl = opt.get('pca_Y_var_expl', 1)
    Cyy = opt.get('Cyy', None)

    Ny, Ne = Y.shape
    
    if Cyy is None or len(Cyy) == 0:
        if Ne > Ny:
            # Пространственная ковариация
            Cyy = (Y @ Y.T) / Ne
        else:
            # Временная ковариация (dual problem)
            Cyy = (Y.T @ Y)
            
    D, V = la.eigh(Cyy)
    sort_idx = np.argsort(D)[::-1]
    ev_sorted = D[sort_idx]
    V = V[:, sort_idx]

    My = V @ np.diag(ev_sorted ** -0.5)

    # PCA dim-reduction
    var_expl = np.cumsum(ev_sorted) / np.sum(ev_sorted)
    # Находим первый индекс, где объясненная дисперсия >= порога
    idx = np.where(var_expl >= pca_Y_var_expl)[0]
    n = idx[0] + 1 if len(idx) > 0 else len(var_expl)
    
    My = My[:, :n]

    if Ne > Ny:
        Y_w = My.T @ Y
    else:
        # Решение через матричное деление diag(std(...)) \ V'
        std_V = np.std(V[:, :n], axis=0, ddof=1)
        Y_w = V[:, :n].T / std_V[:, np.newaxis]
        
        My = np.sqrt(Ne) * Y @ (V[:, :n] @ np.diag(1.0 / ev_sorted[:n]))

    return Y_w, My, Cyy


def detect_max_eigenvalue_epochs(X_epo, threshold=None, show_plot=True, perc_factor=2.5):
    """
    Находит эпохи, чьи максимальные собственные значения превышают порог.
    """
    if X_epo.ndim == 3:
        if X_epo.shape[0] == X_epo.shape[1]:
            Cxxe = X_epo
        else:
            Nx = X_epo.shape[1]
            Ne = X_epo.shape[2]
            Cxxe = np.zeros((Nx, Nx, Ne))
            for e in range(Ne):
                Cxxe[:, :, e] = np.cov(X_epo[:, :, e], rowvar=False)
    else:
        Cxxe = X_epo

    Ne = Cxxe.shape[2]
    ev = np.zeros(Ne)
    
    for n in range(Ne):
        # eigvalsh быстрее и надежнее для расчета только значений симметричных матриц
        evals = la.eigvalsh(Cxxe[:, :, n])
        ev[n] = np.max(evals)

    if threshold is None:
        p = np.percentile(ev, [50, 95])
        threshold = p[0] + perc_factor * (p[1] - p[0])

    rm_idx = np.where(ev > threshold)[0]

    if show_plot:
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 5))
            plt.plot(ev, label='Max Eigenvalue')
            plt.axhline(threshold, color='r', linestyle='--', label='Threshold')
            if len(rm_idx) > 0:
                plt.plot(rm_idx, np.full_like(rm_idx, threshold, dtype=float), '*k', label='Outliers')
            plt.xlim([0, Ne - 1])
            plt.xlabel('Epochs')
            plt.legend()
            plt.show()
        except ImportError:
            pass

    return rm_idx


def detect_high_variance_epochs(X_epo, threshold=None, show_plot=True, perc_factor=2.5):
    """
    Находит эпохи с высокой усредненной дисперсией по каналам.
    """
    if X_epo.ndim == 3:
        # MATLAB var по умолчанию использует dim=1 и ddof=1
        V = np.var(X_epo, axis=0, ddof=1)
    else:
        V = X_epo

    mean_V = np.mean(V, axis=0) # Усреднение по каналам

    if threshold is None:
        p = np.percentile(mean_V, [50, 95])
        threshold = p[0] + perc_factor * (p[1] - p[0])

    rm_idx = np.where(mean_V > threshold)[0]

    if show_plot:
        try:
            import matplotlib.pyplot as plt
            rows, cols = 4, 6
            fig = plt.figure(figsize=(12, 8))
            
            # Subplot: матрица V (heatmap)
            ax_main = plt.subplot2grid((rows, cols), (0, 0), rowspan=rows-1, colspan=cols-1)
            im = ax_main.imshow(V, aspect='auto', origin='lower', cmap='viridis')
            ax_main.set_ylabel('Channels')
            
            # Subplot: график по эпохам (внизу)
            ax_bottom = plt.subplot2grid((rows, cols), (rows-1, 0), colspan=cols-1, sharex=ax_main)
            ax_bottom.plot(mean_V)
            ax_bottom.axhline(threshold, color='r', linestyle='--')
            if len(rm_idx) > 0:
                ax_bottom.plot(rm_idx, np.full_like(rm_idx, threshold, dtype=float), '*k')
            ax_bottom.set_xlabel('Epochs')
            ax_bottom.set_xlim([0, V.shape[1] - 1])
            
            # Subplot: график по каналам (справа)
            ax_right = plt.subplot2grid((rows, cols), (0, cols-1), rowspan=rows-1, sharey=ax_main)
            ax_right.plot(np.mean(V, axis=1), np.arange(V.shape[0]))
            ax_right.set_ylim([0, V.shape[0] - 1])
            ax_right.set_yticklabels([])
            
            plt.tight_layout()
            plt.show()
        except ImportError:
            pass

    return rm_idx