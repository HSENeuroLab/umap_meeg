# -*- coding: utf-8 -*-
import numpy as np
import mne
import pickle
import os
from mne.datasets import eegbci, fetch_fsaverage
from scipy.io import savemat
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- 1. Подготовка и загрузка данных ---
fs_dir = fetch_fsaverage(verbose=True)
subjects_dir = fs_dir.parent

subject = "fsaverage"
trans = "fsaverage"  
src = fs_dir / "bem" / "fsaverage-ico-5-src.fif"
bem = fs_dir / "bem" / "fsaverage-5120-5120-5120-bem-sol.fif"

(raw_fname,) = eegbci.load_data(subjects=1, runs=[6])
raw = mne.io.read_raw_edf(raw_fname, preload=True)

eegbci.standardize(raw)

montage = mne.channels.make_standard_montage("standard_1005")
raw.set_montage(montage)
raw.set_eeg_reference(projection=True) 
raw.apply_proj()

# %%
# --- 2. Визуализация прилегания электродов через Matplotlib ---
bem_surfs = mne.read_bem_surfaces(bem)
head_surf = [s for s in bem_surfs if s['id'] == mne.io.constants.FIFF.FIFFV_BEM_SURF_ID_HEAD][0]
head_rr = head_surf['rr']  

picks_eeg = mne.pick_types(raw.info, eeg=True, exclude=[])
ch_names = [raw.info['ch_names'][i] for i in picks_eeg]
elec_rr = np.array([raw.info['chs'][i]['loc'][:3] for i in picks_eeg]) # Это координаты в Head space

# ================= НОВОЕ =================
# Загружаем матрицу трансформации Head -> MRI для fsaverage
trans_fname = fs_dir / "bem" / "fsaverage-trans.fif"
head_mri_t = mne.read_trans(trans_fname)

# Переводим координаты электродов в пространство MRI для корректной отрисовки с BEM
elec_rr_mri = mne.transforms.apply_trans(head_mri_t, elec_rr)
# =========================================

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Рисуем голову 
ax.scatter(head_rr[:, 0], head_rr[:, 1], head_rr[:, 2], 
           alpha=0.3, s=2, c='gray', label='Head Surface')

# Рисуем электроды (используем новые координаты elec_rr_mri!)
ax.scatter(elec_rr_mri[:, 0], elec_rr_mri[:, 1], elec_rr_mri[:, 2], 
           alpha=1, s=30, c='red', edgecolors='k', label='Electrodes')

ax.set_title('Проверка прилегания электродов к BEM-модели')
ax.legend()

# %%
# --- 3. Расчет прямой модели ---
fwd = mne.make_forward_solution(
    raw.info, trans=trans, src=src, bem=bem, eeg=True, mindist=5.0, n_jobs=None
)

# Явно задаем свободную ориентацию диполей (surf_ori=False, force_fixed=False)
fwd_free = mne.convert_forward_solution(
    fwd, surf_ori=False, force_fixed=False, use_cps=True
)

# %%
# --- 4. Нативное сохранение в формате FIF (MNE-Python) ---

# 1. Сохраняем ПОЛНУЮ прямую модель
output_fwd_fif = os.path.join(os.path.dirname(__file__), 'fsaverage-fwd.fif')
mne.write_forward_solution(output_fwd_fif, fwd_free, overwrite=True)

# 2. Сохраняем ОТДЕЛЬНО объект Info из raw (там есть все фидуциалы, координаты и имена каналов)
output_info_fif = os.path.join(os.path.dirname(__file__), 'fsaverage-info.fif')
mne.io.write_info(output_info_fif, raw.info, overwrite=True)

print(f"Прямая модель полностью сохранена в:\n{output_fwd_fif}")
print(f"Файл информации (Info) сохранен в:\n{output_info_fif}")