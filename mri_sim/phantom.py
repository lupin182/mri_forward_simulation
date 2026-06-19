import nibabel as nib
import numpy as np
from .device_manager import get_xp, device_manager
xp = get_xp()

def generate_simple_asymmetric_phantom(Nz=1, Nx=64, Ny=64):
    """
    鐢熸垚涓€涓崟鍒囩墖闈炲绉颁綋妯★紝鐢ㄤ簬楠岃瘉鏂瑰悜鍜屽潗鏍囩郴,杩欓噷浣撴ā鏁版嵁鏄鏁ｇ殑,涓旂畝鍖栦负姣忎釜浣撶礌鍗曠被鍨嬪崟spin_packet
    1. 涓績鏈変竴涓ぇ鍦?鍦嗘煴 (瀵嗗害 1.0)
    2. 鏈変竴涓皬鏂瑰潡/鍦嗙偣 (瀵嗗害 2.0) -> 鐢ㄤ簬瀹氭柟鍚?
    3. 鑳屾櫙鏈夊井寮变俊鍙?(瀵嗗害 0.1)

    杩斿洖:
    Rho: (type, spin_packet, Nz, Nx, Ny) 瀵嗗害
    T1:  (type, spin_packet, Nz, Nx, Ny) T1鍊?
    T2:  (type, spin_packet, Nz, Nx, Ny) T2鍊?
    """
    
    # 1. 鍒濆鍖?(鑳屾櫙瀵嗗害 0.1, T1=2s, T2=0.5s)
    rho = np.ones((Nz, Nx, Ny)) * 0.1
    t1 = np.ones((Nz, Nx, Ny)) * 2.0
    t2 = np.ones((Nz, Nx, Ny)) * 0.005


    # 2. 鐢熸垚鍧愭爣缃戞牸 (娉ㄦ剰杩欓噷涓轰簡鐢熸垚鏁版嵁锛屾垜浠弗鏍兼寜鐓?Nz, Nx, Ny 椤哄簭)
    # 杩欓噷鐨?x_idx 瀵瑰簲 Nx 缁村害锛寉_idx 瀵瑰簲 Ny 缁村害
    _, x, y = np.meshgrid(
        np.arange(Nz), 
        np.arange(Nx) - Nx/2,  # 灞呬腑鍧愭爣: -32 鍒?+32
        np.arange(Ny) - Ny/2, 
        indexing='ij'
    )

    # 3. 涓讳綋锛氫腑蹇冨ぇ鍦?(鍗婂緞 16)
    # x^2 + y^2 < r^2
    mask_main = (x**2 + y**2) <= 16**2
    
    # 4. 鏍囪鐗╋細鍙充笂瑙掑皬鐐?(鍗槦)
    # 鏀惧湪 X 姝ｅ崐杞? Y 姝ｅ崐杞?(渚嬪 x=15, y=15 澶?
    # 杩欐槸涓€涓?6x6 鐨勫皬鏂瑰潡
    mask_marker = (x > 10) & (x < 20) & (y > 10) & (y < 20)

    # 5. 璧嬪€?
    # 涓讳綋锛氭按 (Rho=1.0, T1=1.0, T2=0.1)
    rho[mask_main] = 1.0
    t1[mask_main]  = 1.0
    t2[mask_main]  = 0.1

    # 鏍囪鐗╋細楂樹寒娌圭偣 (Rho=2.0, T1=0.5, T2=0.05 - 鐭璗1鏇翠寒(鍦ㄧ煭TR涓?, 鐭璗2)
    rho[mask_marker] = 2.0 
    t1[mask_marker]  = 0.5 
    t2[mask_marker]  = 0.05

    rho[0,10:15,50:55] = 2.0
    t1[0,10:15,50:55] = 0.5
    t2[0,10:15,50:55] = 0.05

    rho = rho[np.newaxis,np.newaxis,:,:,:]
    t1 = t1[np.newaxis,np.newaxis,:,:,:]
    t2 = t2[np.newaxis,np.newaxis,:,:,:]

    return rho, t1, t2

def generate_simple_ring_phantom(Nz=1, Nx=64, Ny=64, inner_radius=10, outer_radius=20):
    """
    鐢熸垚涓€涓腑蹇冧负鍦嗙幆/鍦嗙幆鏌辩殑浣撴ā鏁版嵁,杩欓噷浣撴ā鏁版嵁鏄鏁ｇ殑,涓旂畝鍖栦负姣忎釜浣撶礌鍗曠被鍨嬪崟spin_packet
    
    鍙傛暟:
        inner_radius: 鍐呭渾鍗婂緞 (鍍忕礌)
        outer_radius: 澶栧渾鍗婂緞 (鍍忕礌)
        
    杩斿洖:
        Rho: (type, spin_packet, Nz, Nx, Ny) 瀵嗗害, 鍦嗙幆=1.0, 鑳屾櫙=0.1
        T1:  (type, spin_packet, Nz, Nx, Ny) T1鍊?
        T2:  (type, spin_packet, Nz, Nx, Ny) T2鍊?
    """
    
    # 1. 鍒濆鍖栬儗鏅?(Nz, Nx, Ny)
    # 鑳屾櫙瀵嗗害 0.1
    rho = np.ones((Nz, Nx, Ny)) * 0.1
    # 鑳屾櫙寮涜鲍鏃堕棿 (妯℃嫙娴佷綋鎴栭暱寮涜鲍缁勭粐)
    t1 = np.ones((Nz, Nx, Ny)) * 2.0  # 2000ms
    t2 = np.ones((Nz, Nx, Ny)) * 0.5  # 500ms

    # 2. 鐢熸垚鍧愭爣缃戞牸
    # 浣跨敤 indexing='ij' 涓ユ牸鍖归厤 (Nz, Nx, Ny) 鐨勭煩闃靛舰鐘?
    _, x_idx, y_idx = np.meshgrid(
        np.arange(Nz), 
        np.arange(Nx), 
        np.arange(Ny), 
        indexing='ij'
    )

    # 3. 瀹氫箟涓績鐐?
    cx, cy = Nx // 2, Ny // 2

    # 4. 璁＄畻璺濈骞虫柟 (鍦?Nx-Ny 骞抽潰涓?
    # 蹇界暐 Z 杞磋窛绂?(鍋囪鏄渾鏌辩姸/2D鍦嗙幆)
    dist_sq = (x_idx - cx)**2 + (y_idx - cy)**2

    # 5. 鐢熸垚鍦嗙幆鎺╄啘 (Mask)
    # 閫昏緫锛氳窛绂?>= 鍐呭崐寰勫钩鏂? 涓? 璺濈 <= 澶栧崐寰勫钩鏂?
    ring_mask = (dist_sq >= inner_radius**2) & (dist_sq <= outer_radius**2)

    # 6. 璧嬪€?(鍦嗙幆閮ㄥ垎)
    # 妯℃嫙绫讳技姘寸殑鎬ц川 (楂樹俊鍙凤紝闀縏1/T2)
    rho[ring_mask] = 1.0    # 瀵嗗害 1
    t1[ring_mask]  = 1.0    # T1 1000ms
    t2[ring_mask]  = 0.1    # T2 100ms (绋嶅井鐭竴鐐癸紝妯℃嫙缁勭粐)

    rho = rho[np.newaxis,np.newaxis,:,:,:]
    t1 = t1[np.newaxis,np.newaxis,:,:,:]
    t2 = t2[np.newaxis,np.newaxis,:,:,:]

    return rho, t1, t2

def generate_simple_sphere_phantom(Nz=1, Nx=64, Ny=64, radius=16):
    """
    鐢熸垚涓€涓腑蹇冩湁鐞冧綋锛堟垨鍦嗙洏锛夌殑浣撴ā鏁版嵁,杩欓噷浣撴ā鏁版嵁鏄鏁ｇ殑,涓旂畝鍖栦负姣忎釜浣撶礌鍗曠被鍨嬪崟spin_packet
    
    杩斿洖:
        Rho: (type, spin_packet, Nz, Nx, Ny) 瀵嗗害, 鐞?1.0, 鑳屾櫙=0.1
        T1:  (type, spin_packet, Nz, Nx, Ny) T1鍊? 鐞?1.0s (1000ms), 鑳屾櫙=2.0s
        T2:  (type, spin_packet, Nz, Nx, Ny) T2鍊? 鐞?0.1s (100ms),  鑳屾櫙=0.5s
    """
    
    # 1. 鍒濆鍖栬儗鏅?(Nz, Nx, Ny)
    # 鑳屾櫙瀵嗗害 0.1
    rho = np.ones((Nz, Nx, Ny)) * 1e-7
    # 鑳屾櫙 T1 = 2000ms, T2 = 500ms (妯℃嫙绫讳技鑴戣剨娑叉垨姘磋偪鐨勯暱寮涜鲍鑳屾櫙锛屾柟渚垮姣?
    t1 = np.ones((Nz, Nx, Ny)) * 2.0
    t2 = np.ones((Nz, Nx, Ny)) * 0.5

    # 2. 鐢熸垚鍧愭爣缃戞牸
    # 娉ㄦ剰杩欓噷浣跨敤 indexing='ij' 浠ュ尮閰嶇煩闃电储寮曚範鎯?
    z, x, y = np.meshgrid(
        np.arange(Nz), 
        np.arange(Nx), 
        np.arange(Ny), 
        indexing='ij'
    )

    # 3. 瀹氫箟鐞冨績鍧愭爣
    cz, cx, cy = Nz // 2, Nx // 2, Ny // 2

    # 4. 璁＄畻璺濈骞虫柟 (Distance Squared)
    # 濡傛灉 Nz=1锛寊鏂瑰悜鐨勮窛绂讳篃灏辨槸0锛岄€€鍖栦负2D鍦?
    dist_sq = (x - cx)**2 + (y - cy)**2 + (z - cz)**2

    # 5. 鐢熸垚鐞冧綋鎺╄啘 (Mask)
    mask = dist_sq <= radius**2

    # 6. 璧嬪€?(鐞冧綋閮ㄥ垎)
    rho[mask] = 1.0   # 瀵嗗害 1
    t1[mask]  = 1.0   # T1 1000ms (鍏稿瀷姘?鑴戝疄璐?
    t2[mask]  = 0.1   # T2 100ms

    rho = rho[np.newaxis,np.newaxis,:,:,:]
    t1 = t1[np.newaxis,np.newaxis,:,:,:]
    t2 = t2[np.newaxis,np.newaxis,:,:,:]

    return rho, t1, t2


def generate_multi_spin_sphere_phantom(Nz=1, Nx=64, Ny=64, radius=16, Nspins=15):
    """
    鐢熸垚銆愬崟灞傚鑷棆鍖呫€戠悆浣撲綋妯★紝涓撶敤浜嶨RE搴忓垪鎵扮浉姊害浠跨湡瀹為獙
    姣忎釜浣撶礌鍖呭惈Nspins涓嫭绔嬭嚜鏃嬪寘锛屾柊澧瀌WRnd妯℃嫙T2*鏁堝簲
    
    鍙傛暟锛?
        Nz=1: 鍗曞眰锛?D浠跨湡锛?
        Nx,Ny: 鐭╅樀灏哄
        radius: 鐞冧綋鍗婂緞
        Nspins: 鍗曚釜浣撶礌鍐呯殑鑷棆鍖呮暟閲?
    杩斿洖锛?
        Rho:    (1, Nspins, Nz, Nx, Ny) 璐ㄥ瓙瀵嗗害
        T1:     (1, Nspins, Nz, Nx, Ny) T1寮涜鲍鏃堕棿 (s)
        T2:     (1, Nspins, Nz, Nx, Ny) T2寮涜鲍鏃堕棿 (s)
        dWRnd:  (1, Nspins, Nz, Nx, Ny) 闅忔満灞€閮ㄧ鍦哄亸绉伙紝鍗曚綅锛歳ad/s锛屾ā鎷烼2*鏁堝簲
    """
    # 1. 鍩虹浣撴ā锛堢悆浣?鑳屾櫙锛?
    rho_base = np.ones((Nz, Nx, Ny)) * 1e-7
    t1_base = np.ones((Nz, Nx, Ny)) * 2.0
    t2_base = np.ones((Nz, Nx, Ny)) * 0.5

    # 鍧愭爣缃戞牸
    z, x, y = np.meshgrid(np.arange(Nz), np.arange(Nx), np.arange(Ny), indexing='ij')
    cz, cx, cy = Nz // 2, Nx // 2, Ny // 2
    dist_sq = (x - cx)**2 + (y - cy)**2 + (z - cz)**2
    mask = dist_sq <= radius**2

    # 鐞冧綋鍖哄煙璧嬪€?
    rho_base[mask] = 1.0
    t1_base[mask] = 1.0
    t2_base[mask] = 0.1

    # 2. 鎵╁睍涓哄鑷棆鍖呯淮搴︼紙褰㈢姸缁熶竴锛?, Nspins, Nz, Nx, Ny锛?
    rho = np.tile(rho_base[np.newaxis, np.newaxis, :, :, :], (1, Nspins, 1, 1, 1))
    T1  = np.tile(t1_base[np.newaxis, np.newaxis, :, :, :],  (1, Nspins, 1, 1, 1))
    T2  = np.tile(t2_base[np.newaxis, np.newaxis, :, :, :],  (1, Nspins, 1, 1, 1))

    # 3. 鐢熸垚 dWRnd 鉁?涓ユ牸婊¤冻鎵€鏈夎姹?
    # 鍗曚綅锛歳ad/s
    # 鍒嗗竷锛氭鎬佸垎甯冿紝鍧囧€?锛堟棤鍋忥級锛屽皬鏍囧噯宸紙鐪熷疄鐗╃悊閲忕骇锛?
    # 姣忎釜鑷棆鍖?浣撶礌鐨勫亸绉婚兘鐙珛涓嶅悓
    # 褰㈢姸锛氫笌 Rho/T1/T2 瀹屽叏涓€鑷?
    dWRnd = np.random.normal(
        loc=0.0,        # 鏃犲亸锛堝潎鍊间负0锛?
        scale=35.0,      # 灏忛殢鏈烘暟锛宺ad/s 鍗曚綅锛圱2*浠跨湡鏍囧噯閲忕骇锛?
        size=rho.shape  # 褰㈢姸瀹屽叏鍖归厤
    )

    return rho, T1, T2, dWRnd


def generate_coil_sensitivity_maps(
    Nx=64, 
    Ny=64, 
    Nz=1, 
    n_coils=8,      # 绾垮湀閫氶亾鏁?
    sigma=0.8       # 楂樻柉骞虫粦搴?
):
    """
    鐢熸垚MRI澶氶€氶亾鎺ユ敹绾垮湀鐏垫晱搴﹀浘
    鉁?鍒嗙杈撳嚭锛氬箙鍊肩伒鏁忓害 + 鐩镐綅鐏垫晱搴?
    鉁?鐗╃悊姝ｇ‘锛氱幆褰㈠垎甯冮珮鏂箙搴?+ 骞虫粦闅忔満鐩镐綅
    鉁?缁村害鏍囧噯锛?n_coils, Nz, Nx, Ny)
    
    鍙傛暟锛?
        Nx, Ny: 鍥惧儚鐭╅樀灏哄
        Nz: 灞傛暟锛堥粯璁?锛?D浠跨湡锛?
        n_coils: 绾垮湀閫氶亾鏁?
        sigma: 鐏垫晱搴﹀钩婊戝害
    
    杩斿洖锛?
        coil_mag:   骞呭€肩伒鏁忓害鍥撅紝float32锛屽舰鐘?(n_coils, Nz, Nx, Ny)
        coil_phase: 鐩镐綅鐏垫晱搴﹀浘锛宖loat32锛屽舰鐘?(n_coils, Nz, Nx, Ny)锛屽崟浣嶏細寮у害(rad)
    """
    # 1. 褰掍竴鍖栧潗鏍囩綉鏍?[-1, 1]
    x = np.linspace(-1, 1, Nx)
    y = np.linspace(-1, 1, Ny)
    X, Y = np.meshgrid(x, y, indexing='ij')

    # 鍒濆鍖栧箙鍊煎拰鐩镐綅鏁扮粍
    coil_mag = np.zeros((n_coils, Nz, Nx, Ny), dtype=np.float32)
    coil_phase = np.zeros((n_coils, Nz, Nx, Ny), dtype=np.float32)

    # 2. 绾垮湀浣嶇疆锛氱幆褰㈠潎鍖€鍒嗗竷
    coil_positions = []
    for c in range(n_coils):
        angle = 2 * np.pi * c / n_coils
        cx = 0.85 * np.cos(angle)
        cy = 0.85 * np.sin(angle)
        coil_positions.append((cx, cy))

    # 3. 閫愪竴鐢熸垚姣忎釜绾垮湀鐨勫箙鍊?& 鐩镐綅
    for c in range(n_coils):
        cx, cy = coil_positions[c]
        
        # 骞呭€硷細2D楂樻柉鍒嗗竷
        amp = np.exp(-((X - cx)** 2 + (Y - cy)** 2) / (2 * sigma** 2))
        
        # 鐩镐綅锛氬钩婊戦殢鏈虹浉浣嶏紙鏃犻珮棰戝櫔澹帮級
        phase = np.random.uniform(0, 2*np.pi) + 0.2 * (X + Y)
        
        # 璧嬪€煎埌瀵瑰簲缁村害
        coil_mag[c, 0, :, :] = amp
        coil_phase[c, 0, :, :] = phase

    return coil_mag, coil_phase, n_coils


def generate_diff_coil_sensitivity_maps(
    Nx=64, 
    Ny=64, 
    Nz=1, 
    n_coils=8,      # 绾垮湀閫氶亾鏁?
    sigma=0.6       # 楂樻柉骞虫粦搴?
):
    """
    銆愰珮搴︾湡瀹炵増銆戝甫绌洪棿浣嶇疆宸紓鐨凪RI澶氶€氶亾鎺ユ敹绾垮湀鐏垫晱搴?
    鉁?姣忎釜绾垮湀浣嶄簬涓嶅悓绌洪棿浣嶇疆锛氬乏涓?宸︿笅/鍙充笂/鍙充笅/宸?鍙?涓?涓?
    鉁?闈犺繎绾垮湀鐨勫尯鍩熺伒鏁忓害鏄捐憲鏇撮珮锛堝畬鍏ㄧ鍚堢湡瀹炵墿鐞嗭級
    鉁?鍒嗙杈撳嚭骞呭€?+ 鐩镐綅
    鉁?缁村害锛?n_coils, Nz, Nx, Ny)
    """
    # 褰掍竴鍖栧潗鏍?[-1, 1]
    x = np.linspace(-1, 1, Nx)
    y = np.linspace(-1, 1, Ny)
    X, Y = np.meshgrid(x, y, indexing='ij')

    coil_mag = np.zeros((n_coils, Nz, Nx, Ny), dtype=np.float32)
    coil_phase = np.zeros((n_coils, Nz, Nx, Ny), dtype=np.float32)

    # ====================== 鏍稿績淇敼 ======================
    # 8 涓嚎鍦堢湡瀹炲垎甯冨湪 FOV 涓嶅悓浣嶇疆锛屽悇鑷湁涓撳睘楂樼伒鏁忓尯
    # 浣嶇疆鏍煎紡锛?cx, cy)锛岃秺澶ц秺闈犺繎瀵瑰簲杈圭紭
    coil_positions = [
        (-0.75,  0.75),  # 0: 宸︿笂绾垮湀 鈫?宸︿笂鏈€鐏垫晱
        (-0.75, -0.75),  # 1: 宸︿笅绾垮湀 鈫?宸︿笅鏈€鐏垫晱
        ( 0.75,  0.75),  # 2: 鍙充笂绾垮湀 鈫?鍙充笂鏈€鐏垫晱
        ( 0.75, -0.75),  # 3: 鍙充笅绾垮湀 鈫?鍙充笅鏈€鐏垫晱
        (-0.85,  0.0),   # 4: 宸︿晶绾垮湀 鈫?宸﹁竟鏈€鐏垫晱
        ( 0.85,  0.0),   # 5: 鍙充晶绾垮湀 鈫?鍙宠竟鏈€鐏垫晱
        ( 0.0,  0.85),   # 6: 涓婇儴绾垮湀 鈫?涓婅竟鏈€鐏垫晱
        ( 0.0, -0.85),   # 7: 涓嬮儴绾垮湀 鈫?涓嬭竟鏈€鐏垫晱
    ]

    # 鑻ラ€氶亾鏁颁笉鏄?锛岃嚜鍔ㄦ埅鏂垨寰幆鍙栦綅缃?
    coil_positions = coil_positions[:n_coils]

    # ====================== 涓烘瘡涓嚎鍦堢敓鎴愮伒鏁忓害 ======================
    for c in range(n_coils):
        cx, cy = coil_positions[c]
        
        # 楂樻柉骞呭害锛氫腑蹇冨湪 (cx,cy)锛岃秺闈犺繎杩欓噷鏁板€艰秺澶?
        amp = np.exp(-((X - cx)** 2 + (Y - cy)** 2) / (2 * sigma** 2))
        
        # 骞虫粦鐩镐綅锛堜繚鎸佺墿鐞嗙湡瀹烇級
        global_phase = np.random.uniform(0, 2 * np.pi)
        phase = global_phase + 0.15 * X + 0.15 * Y
        
        coil_mag[c, 0, :, :] = amp
        coil_phase[c, 0, :, :] = phase

    return coil_mag, coil_phase, n_coils

def load_simple_phantom(phantom_path, slice_num:int=None):
    """
    杞藉叆宸叉湁鐨勪綋妯℃暟鎹?杩欓噷浣撴ā鏁版嵁鏄鏁ｇ殑,涓旂畝鍖栦负姣忎釜浣撶礌鍗曠被鍨嬪崟spin_packet
    
    杩斿洖:
        Rho: (type, spin_packet, Nz, Nx, Ny) 瀵嗗害
        T1:  (type, spin_packet, Nz, Nx, Ny) T1鍊?
        T2:  (type, spin_packet, Nz, Nx, Ny) T2鍊?
    """
    phantom = nib.load(phantom_path)
    data = phantom.get_fdata()
    if slice_num is None: # 濡傛灉娌℃湁鎸囧畾鍒囩墖锛岃繑鍥炴墍鏈夊垏鐗?
        rho = data[:,:,:,0].transpose(2,0,1)
        t1 = data[:,:,:,1].transpose(2,0,1)
        t2 = data[:,:,:,2].transpose(2,0,1)
        rho = rho[np.newaxis,np.newaxis,:,:,:]
        t1 = t1[np.newaxis,np.newaxis,:,:,:]
        t2 = t2[np.newaxis,np.newaxis,:,:,:]
    else: # 濡傛灉鎸囧畾鍒囩墖锛岃繑鍥炴寚瀹氬垏鐗?
        rho = data[:,:,slice_num,0]
        t1 = data[:,:,slice_num,1]
        t2 = data[:,:,slice_num,2]
        rho = rho[np.newaxis,np.newaxis,np.newaxis,:,:]
        t1 = t1[np.newaxis,np.newaxis,np.newaxis,:,:]
        t2 = t2[np.newaxis,np.newaxis,np.newaxis,:,:]

    return rho, t1, t2

class Phantom:
    def __init__(self,rho:np.ndarray,t1:np.ndarray,t2:np.ndarray,fov_x:float=1.0,
                slice_thickness:float=1.0,fov_y:float=1.0,
                RxCoilNum=1,TxCoilNum=1,B0:float=3.0,dB0:np.ndarray=None,
                txCoilmg:np.ndarray=None,rxCoilmg:np.ndarray=None,
                txCoilpe:np.ndarray=None,rxCoilpe:np.ndarray=None,
                CS:np.ndarray=None,dWRnd:np.ndarray=None):
        """
        浣撴ā绫伙紝鏀寔CuPy GPU鍔犻€熴€?
        """
        self.fov_x = fov_x
        self.fov_y = fov_y
        self.slice_thickness = slice_thickness
        self.B0 = B0
        if len(rho.shape) == 3:
            rho = rho[np.newaxis,np.newaxis,:,:,:]
        if len(t1.shape) == 3:
            t1 = t1[np.newaxis,np.newaxis,:,:,:]
        if len(t2.shape) == 3:
            t2 = t2[np.newaxis,np.newaxis,:,:,:]

        # 灏嗘暟鎹Щ鍔ㄥ埌褰撳墠璁惧锛圕PU/GPU锛?
        self.rho = device_manager.to_device(rho)
        self.t1 = device_manager.to_device(t1)
        self.t2 = device_manager.to_device(t2)

        self.Nz = rho.shape[2]
        self.Nx = rho.shape[3]
        self.Ny = rho.shape[4]

        self.dx = self.fov_x / self.Nx
        self.dy = self.fov_y / self.Ny

        z_axis = (xp.arange(self.Nz) - self.Nz / 2 + 0.5) * self.slice_thickness
        x_axis = (xp.arange(self.Nx) - self.Nx / 2 + 0.5) * self.dx
        y_axis = (xp.arange(self.Ny) - self.Ny / 2 + 0.5) * self.dy

        self.z, self.x, self.y = xp.meshgrid(z_axis, x_axis, y_axis, indexing='ij')

        assert self.rho.shape == self.t1.shape, "rho shape must be t1 shape"
        assert self.rho.shape == self.t2.shape, "rho shape must be t2 shape"
        assert len(self.rho.shape) == 5, "input rho shape must be (TypeNum, SpinNum, Nz, Nx, Ny) or (Nz, Nx, Ny)"

        self.SpinNum = self.rho.shape[1]     # 鑷棆鏁?
        self.TypeNum = self.rho.shape[0]     # 绫诲瀷鏁?
        self.RxCoilNum = RxCoilNum     # 鎺ユ敹绾垮湀鏁?
        self.TxCoilNum = TxCoilNum     # 鍙戝皠绾垮湀鏁?


        # 楂樼骇鐜灞炴€ч粯璁ゅ€?
        self.txCoilmg = device_manager.to_device(xp.ones((TxCoilNum,self.Nz,self.Nx,self.Ny))) if txCoilmg is None else device_manager.to_device(txCoilmg)    # 鍙戝皠鍦烘晱鎰熷害
        self.txCoilpe = device_manager.to_device(xp.zeros((TxCoilNum,self.Nz,self.Nx,self.Ny))) if txCoilpe is None else device_manager.to_device(txCoilpe)    # 鍙戝皠鍦烘晱鎰熷害
        self.rxCoilmg = device_manager.to_device(xp.ones((RxCoilNum,self.Nz,self.Nx,self.Ny))) if rxCoilmg is None else device_manager.to_device(rxCoilmg)    # 鎺ユ敹鍦烘晱鎰熷害
        self.rxCoilpe = device_manager.to_device(xp.zeros((RxCoilNum,self.Nz,self.Nx,self.Ny))) if rxCoilpe is None else device_manager.to_device(rxCoilpe)    # 鎺ユ敹鍦烘晱鎰熷害
        
        assert self.txCoilmg.shape == (TxCoilNum, self.Nz, self.Nx, self.Ny), "txCoilmg shape must be (TxCoilNum, self.Nz, self.Nx, self.Ny)"
        assert self.txCoilpe.shape == (TxCoilNum, self.Nz, self.Nx, self.Ny), "txCoilpe shape must be (TxCoilNum, self.Nz, self.Nx, self.Ny)"
        assert self.rxCoilmg.shape == (RxCoilNum, self.Nz, self.Nx, self.Ny), "rxCoilmg shape must be (RxCoilNum, self.Nz, self.Nx, self.Ny)"
        assert self.rxCoilpe.shape == (RxCoilNum, self.Nz, self.Nx, self.Ny), "rxCoilpe shape must be (RxCoilNum, self.Nz, self.Nx, self.Ny)"

        # 闅忔椂闂存紨鍖栫殑鐘舵€?(鍒濆鍖栧钩琛℃€?
        self.Mx = device_manager.to_device(xp.zeros_like(self.rho))       
        self.My = device_manager.to_device(xp.zeros_like(self.rho))
        self.Mz = device_manager.to_device(xp.copy(self.rho * self.B0))

        self.Gyro = 42.576e6    # gyromagnetic ratio
        # chemical shift array
        self.CS = device_manager.to_device(xp.zeros_like(self.rho)) if CS is None else device_manager.to_device(CS)
        # B0 inhomogeneity
        self.dB0 = device_manager.to_device(xp.zeros_like(self.rho)) if dB0 is None else device_manager.to_device(dB0)
        # random off-resonance for T2*
        self.dWRnd = device_manager.to_device(xp.zeros_like(self.rho)) if dWRnd is None else device_manager.to_device(dWRnd)

        assert self.dWRnd.shape == (self.TypeNum, self.SpinNum, self.Nz, self.Nx, self.Ny), "dWRnd shape must be (self.TypeNum, self.SpinNum, self.Nz, self.Nx, self.Ny)"
        assert self.CS.shape == (self.TypeNum, self.SpinNum, self.Nz, self.Nx, self.Ny), "CS shape must be (self.TypeNum, self.SpinNum, self.Nz, self.Nx, self.Ny)"
        assert self.dB0.shape == (self.TypeNum, self.SpinNum, self.Nz, self.Nx, self.Ny), "dB0 shape must be (self.TypeNum, self.SpinNum, self.Nz, self.Nx, self.Ny)"

