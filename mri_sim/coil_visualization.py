from .phantom import generate_diff_coil_sensitivity_maps
import matplotlib.pyplot as plt
import numpy as np

def plot_4coil_sensitivity(coil_mag, coil_phase):
    """
    涓撻棬缁樺埗銆?涓嚎鍦堛€戠殑MRI鎺ユ敹绾垮湀鐏垫晱搴﹀垎甯冨浘
    灞曠ず鍐呭锛氭瘡涓嚎鍦堢殑 鐏垫晱搴﹀箙鍊?+ 鐩镐綅鍒嗗竷
    甯冨眬锛?琛?脳 4鍒楋紙绗竴琛岋細骞呭€硷紝绗簩琛岋細鐩镐綅锛?
    """
    # 鍥哄畾缁樺埗4涓嚎鍦堬紝鑷姩鎴彇鍓?涓€氶亾
    n_coils_plot = 4
    coil_names = ["Coil1", "Coil2", "Coil3", "Coil4"]
    
    # 鍒涘缓鐢诲竷锛?琛?骞呭€?鐩镐綅) 脳 4鍒?4涓嚎鍦?
    fig, axes = plt.subplots(2, n_coils_plot, figsize=(16, 8))
    #fig.suptitle('MRI 4閫氶亾鎺ユ敹绾垮湀鐏垫晱搴﹀垎甯冨浘', fontsize=18, y=0.95)

    for c in range(n_coils_plot):
        # 鎻愬彇2D鏁版嵁锛堝師鏁版嵁缁村害 [n_coil, 1, Nx, Ny]锛屽幓鎺塏z缁村害锛?
        mag = coil_mag[c, 0, :, :]
        phase = coil_phase[c, 0, :, :]

        # ---------- 缁樺埗鐏垫晱搴﹀箙鍊硷紙绗竴琛岋級----------
        im_mag = axes[0, c].imshow(
            mag.T,        # 杞疆鍖归厤鍥惧儚鍧愭爣
            cmap='viridis',
            origin='lower'# 宸︿笅涓哄潗鏍囧師鐐?
        )
        axes[0, c].set_title(f'{coil_names[c]}\nAmplitude', fontsize=12)
        axes[0, c].axis('off')
        plt.colorbar(im_mag, ax=axes[0, c], shrink=0.7)

        # ---------- 缁樺埗鐏垫晱搴︾浉浣嶏紙绗簩琛岋級----------
        im_phase = axes[1, c].imshow(
            phase.T,
            cmap='hsv',   # 鐜舰鑹插浘锛屾渶閫傚悎鐩镐綅(0~2蟺)
            vmin=0, vmax=2*np.pi,
            origin='lower'
        )
        axes[1, c].set_title(f'{coil_names[c]}\nPhase', fontsize=12)
        axes[1, c].axis('off')
        plt.colorbar(im_phase, ax=axes[1, c], shrink=0.7)

    plt.tight_layout()
    plt.show()

# ====================== 娴嬭瘯锛氱敓鎴愬苟缁樺埗4绾垮湀鐏垫晱搴?======================

