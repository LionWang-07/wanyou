import matplotlib
matplotlib.use('Agg')   # 云端服务器没有显示器，必须指定Agg后端
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from scipy.interpolate import griddata

# -------------------------- session_state --------------------------
if "param_single" not in st.session_state:
    st.session_state.param_single = {
        "v_min":70.0,
        "v_max":90.0,
        "tire_r":0.536,
        "i_axle":2.85,
        "gear_text":"1.0",
        "tq_min":300.0,
        "tq_max":1800.0
    }

if "multi_text_value" not in st.session_state:
    st.session_state.multi_text_value = """70,90,0.536,2.85,1.0,300,1800
70,90,0.536,2.714,1.0,315,1892
"""

# 等效换算模块session
if "eq_data" not in st.session_state:
    st.session_state.eq_data = {
        "v_low":70.0,
        "v_high":90.0,
        "t1_low":300.0,
        "t1_high":1800.0,
        "ig1":1.0,
        "ia1":2.85,
        "r1":0.536,
        "ig2":1.27,
        "ia2":2.714,
        "r2":0.536
    }

# -------------------------- 页面配置 --------------------------
st.set_page_config(page_title="发动机万有&等效扭矩计算器V0", layout="wide")
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# -------------------------- 公共函数 --------------------------
def speed_to_engine_rpm(v_kmh, tire_r_m, ig, ia):
    """车速→发动机转速 n(rpm) = v*ig*ia / (0.377*r)"""
    return v_kmh * ig * ia / (0.377 * tire_r_m)

def calc_equivalent_torque(v_low, v_high, t1_low, t1_high, ig1, ia1, r1, ig2, ia2, r2):
    """
    等效换算：保持车轮扭矩不变，求新系统发动机扭矩
    Tw = Te1 * ig1 * ia1 / r1 = Te2 * ig2 * ia2 / r2
    → Te2 = Te1 * (ig1*ia1 / r1) / (ig2*ia2 / r2)
    返回：n1_low,n1_high, n2_low,n2_high, t2_low,t2_high, ratio
    """
    # 参照系统发动机转速区间
    n1_low = speed_to_engine_rpm(v_low, r1, ig1, ia1)
    n1_high = speed_to_engine_rpm(v_high, r1, ig1, ia1)
    # 新系统发动机转速区间
    n2_low = speed_to_engine_rpm(v_low, r2, ig2, ia2)
    n2_high = speed_to_engine_rpm(v_high, r2, ig2, ia2)

    # 扭矩换算系数
    k = (ig1 * ia1 / r1) / (ig2 * ia2 / r2)
    t2_low = t1_low * k
    t2_high = t1_high * k
    return n1_low,n1_high, n2_low,n2_high, t2_low,t2_high, k

def get_custom_be_levels():
    levels = []
    val = 170.0
    while val <= 180:
        levels.append(val)
        val += 1
    val = 180
    while val <= 186:
        levels.append(val)
        val += 2
    val = 186
    while val <= 195:
        levels.append(val)
        val += 3
    val = 195
    while val <= 210:
        levels.append(val)
        val += 5
    levels = sorted(list(set(levels)))
    return levels

def draw_const_power_lines(ax, xlim, ylim, power_step=50):
    n_min, n_max = xlim
    t_min, t_max = ylim
    n_line = np.linspace(n_min, n_max, 800)
    p_low = int((n_min * t_min)/9550)
    p_high = int((n_max * t_max)/9550)
    p_start = int(np.ceil(p_low / power_step)) * power_step
    p_end = int(np.floor(p_high / power_step)) * power_step
    power_list = np.arange(p_start, p_end + power_step, power_step)
    n_pad = (n_max - n_min) * 0.08
    t_pad = (t_max - t_min) * 0.08
    min_valid_seg_width = (n_max - n_min)*0.02
    for p in power_list:
        t_line = p * 9550.0 / n_line
        mask = (t_line >= t_min) & (t_line <= t_max)
        valid_n = n_line[mask]
        valid_t = t_line[mask]
        if len(valid_n) < 2:
            continue
        ax.plot(valid_n, valid_t, color="#666666", linestyle="-.", lw=1.0, alpha=0.65)
        seg_width = valid_n.max() - valid_n.min()
        if seg_width < min_valid_seg_width:
            continue
        mid_pos = len(valid_n) // 2
        label_n = valid_n[mid_pos]
        label_t = valid_t[mid_pos]
        label_n = np.clip(label_n, n_min + n_pad, n_max - n_pad)
        label_t = np.clip(label_t, t_min + t_pad, t_max - t_pad)
        ax.text(label_n, label_t, f"{p}kW", color="#444444", fontsize=8)

def calc_working_region_stats(N_grid, T_grid, BE_grid, rpm_low, rpm_high, tq_low, tq_high, be_bins):
    region_mask = (N_grid >= rpm_low) & (N_grid <= rpm_high) & (T_grid >= tq_low) & (T_grid <= tq_high)
    be_in_region = BE_grid[region_mask]
    be_valid = be_in_region[~np.isnan(be_in_region)]
    valid_total = len(be_valid)
    if valid_total == 0:
        return None, None, None, 0
    avg_be = float(np.mean(be_valid))
    bin_count, _ = np.histogram(be_valid, bins=be_bins)
    bin_ratio = bin_count / valid_total
    return avg_be, bin_count, bin_ratio, valid_total

# ===================== 多标签页 =====================
tab1, tab2 = st.tabs(["🔧万有特性绘图&工况统计","⚙️速比-扭矩等效换算"])

# =========标签页1：万有特性模块========
with tab1:
    with st.sidebar:
        st.header("📥发动机数据导入")
        uploaded_file = st.file_uploader("上传发动机万有Excel", type=["xlsx", "xls"])
        st.markdown("""
表格无表头：
1列：万有转速｜2列：万有扭矩｜3列：比油耗
4列：外特性转速｜5列：外特性扭矩
""")
        st.divider()
        st.header("📈绘图设置")
        fig_width = st.number_input("图表宽度 inch", value=12.0, min_value=6.0, max_value=20.0)
        fig_height = st.number_input("图表高度 inch", value=7.5, min_value=4.0, max_value=14.0)
        use_manual_axis = st.checkbox("手动设置坐标轴范围", value=False)
        col1, col2 = st.columns(2)
        with col1:
            x_min = st.number_input("转速下限 rpm", value=600.0, min_value=0.0)
            x_max = st.number_input("转速上限 rpm", value=2600.0, min_value=100.0)
        with col2:
            y_min = st.number_input("扭矩下限 N·m", value=0.0, min_value=0.0)
            y_max = st.number_input("扭矩上限 N·m", value=2200.0, min_value=100.0)

        st.divider()
        st.header("🚛整车参数模式")
        param_mode = st.radio("参数输入模式", ["单参数输入", "多参数批量输入"])
        group_list = []
        if param_mode == "单参数输入":
            v_min = st.number_input("常用车速下限 km/h", value=st.session_state.param_single["v_min"], min_value=0.0)
            v_max = st.number_input("常用车速上限 km/h", value=st.session_state.param_single["v_max"], min_value=1.0)
            tire_r = st.number_input("轮胎半径 m", value=st.session_state.param_single["tire_r"], min_value=0.1)
            i_axle = st.number_input("驱动桥速比", value=st.session_state.param_single["i_axle"], min_value=0.1)
            gear_ratio_text = st.text_input("变速箱挡位速比(英文逗号分隔)", value=st.session_state.param_single["gear_text"])
            tq_min = st.number_input("常用扭矩下限 N·m", value=st.session_state.param_single["tq_min"], min_value=0.0)
            tq_max = st.number_input("常用扭矩上限 N·m", value=st.session_state.param_single["tq_max"], min_value=10.0)
            st.session_state.param_single = {
                "v_min":v_min,"v_max":v_max,"tire_r":tire_r,"i_axle":i_axle,"gear_text":gear_ratio_text,"tq_min":tq_min,"tq_max":tq_max
            }
            group_list.append({"v_min":v_min,"v_max":v_max,"tire_r":tire_r,"i_axle":i_axle,"gear_text":gear_ratio_text,"tq_min":tq_min,"tq_max":tq_max})
        else:
            multi_text = st.text_area("多组参数输入", height=180, value=st.session_state.multi_text_value)
            st.session_state.multi_text_value = multi_text
            lines = multi_text.strip().splitlines()
            for line_idx, line in enumerate(lines):
                line = line.strip()
                if not line: continue
                parts = line.split(",")
                if len(parts)!=7:
                    st.warning(f"第{line_idx+1}行格式错误，跳过")
                    continue
                try:
                    g={"v_min":float(parts[0]),"v_max":float(parts[1]),"tire_r":float(parts[2]),"i_axle":float(parts[3]),"gear_text":parts[4],"tq_min":float(parts[5]),"tq_max":float(parts[6])}
                    group_list.append(g)
                except Exception as e:
                    st.warning(f"第{line_idx+1}行解析失败:{e}")

    color_palette = ["#ff3333","#0088ff","#00aa44","#ff9900","#9933ff","#cc2288","#00bbbb","#777777"]
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    custom_levels = get_custom_be_levels()
    axis_xlim = (x_min, x_max)
    axis_ylim = (y_min, y_max)
    stats_df = None
    be_bins = [170, 180, 190, 200, 210, 230, 250, 280, 350]
    bin_labels = [f"[{be_bins[i]},{be_bins[i+1]})" for i in range(len(be_bins)-1)]

    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file, header=None)
        map_n = df.iloc[:,0].dropna().values
        map_t = df.iloc[:,1].dropna().values
        map_be = df.iloc[:,2].dropna().values
        outer_n = df.iloc[:,3].dropna().values
        outer_t = df.iloc[:,4].dropna().values
        ax.plot(outer_n, outer_t, color="red", lw=2.8, label="外特性边界", zorder=5)
        ni = np.linspace(min(map_n), max(map_n), 120)
        ti = np.linspace(min(map_t), max(map_t), 120)
        N_grid, T_grid = np.meshgrid(ni, ti)
        BE_grid = griddata((map_n, map_t), map_be, (N_grid, T_grid), method="linear")
        max_be = np.nanmax(BE_grid)
        if max_be > 210:
            add_levels = list(np.arange(215, max_be+10,10))
            all_levels = sorted(list(set(custom_levels + add_levels)))
        else:
            all_levels = custom_levels
        contour = ax.contour(N_grid, T_grid, BE_grid, levels=all_levels, cmap="jet", alpha=0.72)
        ax.clabel(contour, inline=True, fontsize=7.5)
        draw_const_power_lines(ax, axis_xlim, axis_ylim, power_step=50)
        stat_rows = []
        for group_idx, group in enumerate(group_list):
            c = color_palette[group_idx % len(color_palette)]
            gear_str = group["gear_text"]
            gear_ratios = [float(x.strip()) for x in gear_str.split(";")]
            for ig in gear_ratios:
                rpm_low = speed_to_engine_rpm(group["v_min"], group["tire_r"], ig, group["i_axle"])
                rpm_high = speed_to_engine_rpm(group["v_max"], group["tire_r"], ig, group["i_axle"])
                rect_x = [rpm_low, rpm_high, rpm_high, rpm_low, rpm_low]
                rect_y = [group["tq_min"], group["tq_min"], group["tq_max"], group["tq_max"], group["tq_min"]]
                ax.plot(rect_x, rect_y, color=c, linestyle="--", lw=1.4, alpha=0.75)
            ig0 = gear_ratios[0]
            rpm_mid = speed_to_engine_rpm((group["v_min"]+group["v_max"])/2, group["tire_r"], ig0, group["i_axle"])
            tq_mid = (group["tq_min"]+group["tq_max"])/2
            label_name = f"G{group_idx+1}"
            ax.text(rpm_mid, tq_mid, label_name, color=c, fontweight="bold", fontsize=9)
            rpm_low = speed_to_engine_rpm(group["v_min"], group["tire_r"], ig0, group["i_axle"])
            rpm_high = speed_to_engine_rpm(group["v_max"], group["tire_r"], ig0, group["i_axle"])
            tq_low = group["tq_min"]
            tq_high = group["tq_max"]
            avg_be, bin_count, bin_ratio, valid_pts = calc_working_region_stats(N_grid, T_grid, BE_grid, rpm_low, rpm_high, tq_low, tq_high, be_bins)
            row_dict = {"工况编号":label_name, "有效网格点数":valid_pts}
            if valid_pts > 0:
                row_dict["平均比油耗(g/kWh)"] = round(avg_be,2)
                for i,lab in enumerate(bin_labels):
                    row_dict[f"占比_{lab} g/kWh"] = f"{bin_ratio[i]*100:.1f}%"
            else:
                row_dict["平均比油耗(g/kWh)"] = "无有效数据"
                for lab in bin_labels:
                    row_dict[f"占比_{lab} g/kWh"] = "-"
            stat_rows.append(row_dict)
        stats_df = pd.DataFrame(stat_rows)
        if use_manual_axis:
            ax.set_xlim(axis_xlim)
            ax.set_ylim(axis_ylim)
        else:
            ax.set_xlim(np.min(map_n), np.max(map_n))
            ax.set_ylim(np.min(map_t), np.max(map_t))
        ax.set_xlabel("发动机转速 rpm", fontsize=11)
        ax.set_ylabel("发动机扭矩 N·m", fontsize=11)
        ax.set_title("发动机万有特性｜外特性｜等功率线｜多组整车工况区", fontsize=13)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")
    else:
        ax.text(0.5,0.5,"请在左侧上传发动机万有Excel表格",ha="center",va="center",transform=ax.transAxes, fontsize=14)
        if use_manual_axis:
            ax.set_xlim(axis_xlim)
            ax.set_ylim(axis_ylim)
    st.pyplot(fig)
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    buf.seek(0)
    st.download_button("📷下载当前万有图片PNG", data=buf, file_name="engine_map_v8.png", mime="image/png")
    if uploaded_file is not None and stats_df is not None:
        st.subheader("📊工况区域比油耗统计")
        st.dataframe(stats_df, use_container_width=True)
        csv_data = stats_df.to_csv(index=False, encoding="utf-8-sig")
        st.download_button("📥导出统计表格CSV", data=csv_data, file_name="工况比油耗统计.csv", mime="text/csv")
    st.divider()
    st.subheader("📋当前参数组汇总")
    st.write(group_list)

# =========标签页2：等效扭矩换算模块========
with tab2:
    st.markdown("## ⚙️变速箱/后桥/轮胎参数等效扭矩换算")
    st.info("物理前提：**相同车速区间，维持车轮输出扭矩不变，求解新传动系统对应的发动机扭矩、转速区间**")
    colA, colB = st.columns(2)
    with colA:
        st.subheader("🔹参照系统（原始）")
        v_low = st.number_input("车速下限 km/h", value=st.session_state.eq_data["v_low"])
        v_high = st.number_input("车速上限 km/h", value=st.session_state.eq_data["v_high"])
        t1_low = st.number_input("原始发动机扭矩下限 N·m", value=st.session_state.eq_data["t1_low"])
        t1_high = st.number_input("原始发动机扭矩上限 N·m", value=st.session_state.eq_data["t1_high"])
        ig1 = st.number_input("原始变速箱速比ig1", value=st.session_state.eq_data["ig1"])
        ia1 = st.number_input("原始驱动桥速比ia1", value=st.session_state.eq_data["ia1"])
        r1 = st.number_input("原始轮胎半径 m", value=st.session_state.eq_data["r1"])

    with colB:
        st.subheader("🔹调整后系统（待求解）")
        ig2 = st.number_input("新变速箱速比ig2", value=st.session_state.eq_data["ig2"])
        ia2 = st.number_input("新驱动桥速比ia2", value=st.session_state.eq_data["ia2"])
        r2 = st.number_input("新轮胎半径 m", value=st.session_state.eq_data["r2"])

    # 更新session
    st.session_state.eq_data = {
        "v_low":v_low,"v_high":v_high,"t1_low":t1_low,"t1_high":t1_high,
        "ig1":ig1,"ia1":ia1,"r1":r1,"ig2":ig2,"ia2":ia2,"r2":r2
    }

    n1_low,n1_high, n2_low,n2_high, t2_low,t2_high, k = calc_equivalent_torque(v_low, v_high, t1_low, t1_high, ig1, ia1, r1, ig2, ia2, r2)

    st.divider()
    st.subheader("📝计算结果（车轮扭矩不变）")
    res_rows = [
        {"项目":"参照系统发动机转速下限(rpm)", "数值":round(n1_low,1)},
        {"项目":"参照系统发动机转速上限(rpm)", "数值":round(n1_high,1)},
        {"项目":"调整后发动机转速下限(rpm)", "数值":round(n2_low,1)},
        {"项目":"调整后发动机转速上限(rpm)", "数值":round(n2_high,1)},
        {"项目":"原始发动机扭矩下限(N·m)", "数值":round(t1_low,1)},
        {"项目":"原始发动机扭矩上限(N·m)", "数值":round(t1_high,1)},
        {"项目":"等效发动机扭矩下限(N·m)", "数值":round(t2_low,1)},
        {"项目":"等效发动机扭矩上限(N·m)", "数值":round(t2_high,1)},
        {"项目":"扭矩换算系数 k", "数值":round(k,4)},
    ]
    res_df = pd.DataFrame(res_rows)
    st.dataframe(res_df, use_container_width=True)
    csv_res = res_df.to_csv(index=False, encoding="utf-8-sig")
    st.download_button("📥导出本次换算结果CSV", data=csv_res, file_name="速比扭矩等效换算结果.csv", mime="text/csv")

    st.markdown(r"""
> 换算公式：
> $T_{e2}=T_{e1} \cdot \dfrac{i_{g1}\cdot i_{a1}/r_1}{i_{g2}\cdot i_{a2}/r_2}$
>
> 说明：忽略传动效率差异；该结果含义：**要输出和原来一样的车轮扭矩，新传动系统发动机需要提供的扭矩区间**。
""")

