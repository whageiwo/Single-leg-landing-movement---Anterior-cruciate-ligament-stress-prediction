import os
import streamlit as st
import joblib
import numpy as np
import shap
import streamlit.components.v1 as components

# ------------------ 页面配置 ------------------
st.set_page_config(page_title="单腿落地动作-前交叉韧带应力预测", layout="wide")

# ------------------ 字体优先列表（修复未定义 & 缺字库时的回退） ------------------
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 优先加载本地中文字体（可把 SimHei.ttf 放到脚本同目录）
cn_font_path_candidates = [
    "SimHei.ttf",                          # 当前目录
    os.path.join("fonts", "SimHei.ttf"),   # fonts/ 目录
]
cn_font_name = None
for p in cn_font_path_candidates:
    if os.path.exists(p):
        try:
            font_manager.fontManager.addfont(p)
            cn_font_name = font_manager.FontProperties(fname=p).get_name()
            break
        except Exception:
            pass

# 若本地未找到，就尝试系统内常见中文字体名称
if cn_font_name is None:
    common_cn_names = ["SimHei", "Microsoft YaHei", "Noto Sans SC", "Source Han Sans SC"]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in common_cn_names:
        if name in available:
            cn_font_name = name
            break

# 如果还没有，就用 DejaVu Sans（英文+部分 CJK），中文可能不全，但不报错
if cn_font_name is None:
    cn_font_name = "DejaVu Sans"

en_font_name = "DejaVu Sans"

plt.rcParams["font.family"] = [cn_font_name, en_font_name]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 12
plt.rcParams["font.weight"] = "bold"

# ------------------ 页面标题 ------------------
st.markdown(
    "<h1 style='text-align: center; color: darkred; margin-bottom: 40px;'>单腿落地动作-前交叉韧带应力预测</h1>",
    unsafe_allow_html=True
)

# ------------------ 加载模型 ------------------
MODEL_PATH = "DJT_XGJ_model.bin"
if not os.path.exists(MODEL_PATH):
    st.error(f"未找到模型文件：{MODEL_PATH}")
    st.stop()
model = joblib.load(MODEL_PATH)

# ------------------ 特征名称（6个） ------------------
feature_names = [
    "踝关节外翻角度",
    "踝关节屈伸角度",
    "膝关节屈伸角度",
    "髋关节屈伸角度",
    "髋关节内收角度",
    "躯干屈曲角度"
]

# ------------------ 页面布局 ------------------
col1, col2 = st.columns([1.2, 1.2])
inputs = []

# 左列输入（前5个）
with col1:
    for name in feature_names[:5]:
        st.markdown(f"<p style='font-size:16px'>{name}</p>", unsafe_allow_html=True)
        val = st.number_input("", value=0.0, step=0.1, format="%.2f", key=name)
        inputs.append(val)

# 右列输入（第6个）
with col2:
    for name in feature_names[5:]:
        st.markdown(f"<p style='font-size:16px'>{name}</p>", unsafe_allow_html=True)
        val = st.number_input("", value=0.0, step=0.1, format="%.2f", key=name)
        inputs.append(val)

# 组装输入
X_input = np.array([inputs], dtype=float)

# -------- 预测结果 --------
try:
    pred = float(model.predict(X_input)[0])
except Exception as e:
    st.exception(e)
    st.stop()

with col2:
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(f"<h3 style='color:darkgreen;'>预测结果</h3>", unsafe_allow_html=True)
    st.markdown(
        f"<p style='color:blue; font-size:40px; font-weight:bold;'>前交叉韧带应力: {pred:.2f}</p>",
        unsafe_allow_html=True
    )

# ---------------- SHAP 力图 ----------------
st.markdown("<h3 style='color:purple; text-align:center;'>SHAP 力图</h3>", unsafe_allow_html=True)

explainer = shap.TreeExplainer(model)
shap_values = explainer(X_input)

# 兼容不同 SHAP 版本的 expected_value（可能是标量或数组）
try:
    expected_value = float(np.ravel(explainer.expected_value)[0])
except Exception:
    try:
        expected_value = float(np.ravel(shap_values.base_values)[0])
    except Exception:
        expected_value = 0.0

force_plot = shap.force_plot(
    expected_value,
    shap_values.values[0],
    X_input[0],
    feature_names=[str(f) for f in feature_names]
)

components.html(
    f"<head>{shap.getjs()}</head>{force_plot.html()}",
    height=400,
    scrolling=True
)

