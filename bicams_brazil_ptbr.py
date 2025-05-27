import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import streamlit as st
from datetime import datetime
from fpdf import FPDF
import tempfile

# Map English month names to Portuguese month names
months_pt = {
    "January": "Janeiro", "February": "Fevereiro", "March": "Março",
    "April": "Abril", "May": "Maio", "June": "Junho",
    "July": "Julho", "August": "Agosto", "September": "Setembro",
    "October": "Outubro", "November": "Novembro", "December": "Dezembro"
}

def format_date(date):
    day = date.strftime("%-d")
    month = months_pt[date.strftime("%B")]
    year = date.strftime("%Y")
    return f"{day} {month} {year}"

# Regression-based norms for raw scores
regression_models = {
    'CVLT_totaldeacertos': {
        'constant': 8.512324, 'age': -0.14798, 'age2': 0.001373,
        'sex': 0.176426, 'education': 0.364315, 'residual_sd': 2.527166
    },
    'BVMT_Total': {
        'constant': 11.58455, 'age': -0.14752, 'age2': 0.000896,
        'sex': -0.19042, 'education': 0.22895, 'residual_sd': 2.626665
    },
    'SDMT': {
        'constant': 9.248778, 'age': -0.01094, 'age2': -0.00086,
        'sex': -0.4714, 'education': 0.263055, 'residual_sd': 2.48323
    }
}

# Discrete lookup tables for scaled scores
conversion_table = {
    'CVLT_totaldeacertos': {
        1: (-np.inf, 19), 2: (20, 28), 3: (29, 31), 4: (32, 35), 5: (36, 39), 6: (40, 41),
        7: (42, 44), 8: (45, 48), 9: (49, 52), 10: (53, 56), 11: (57, 60), 12: (61, 64),
        13: (65, 66), 14: (67, 69), 15: (70, 71), 16: (72, 72), 17: (73, 74), 18: (75, 75), 19: (76, np.inf)
    },
    'BVMT_Total': {
        1: (-np.inf, 2), 2: (3, 5), 3: (6, 8), 4: (9, 12), 5: (13, 17), 6: (18, 20),
        7: (21, 23), 8: (24, 26), 9: (27, 28), 10: (29, 30), 11: (31, 32), 12: (33, 34), 13: (35, 35),
        14: (36, np.inf)
    },
    'SDMT': {
        1: (-np.inf, 9), 2: (10, 17), 3: (18, 23), 4: (24, 29), 5: (30, 36), 6: (37, 43),
        7: (44, 49), 8: (50, 53), 9: (54, 58), 10: (59, 62), 11: (63, 68), 12: (69, 74), 13: (75, 79),
        14: (80, 93), 15: (94, 107), 16: (108, np.inf)
    }
}

# Convert raw to discrete scaled score
def convert_to_scaled_score(raw_score, measure):
    for score, (low, high) in conversion_table[measure].items():
        if low <= raw_score <= high:
            return score
    return np.nan

# Predict raw score based on demographics
def calculate_predicted_raw_score(age, sex, education, measure):
    m = regression_models[measure]
    age2 = age**2
    sex_code = 1 if sex == 'M' else 2
    return m['constant'] + m['age']*age + m['age2']*age2 + m['sex']*sex_code + m['education']*education

# Interpret percentile
def interpret_percentile(pct):
    if pct>=98: return ">130",">98","Excepcionalmente Alto","#00008B"
    if pct>=90: return "120-129","91-97","Acima da Média","#0000FF"
    if pct>=75: return "110-119","75-90","Médio-Alto","#00FFFF"
    if pct>=25: return "90-109","25-74","Médio","#00FF00"
    if pct>=9:  return "80-89","9-24","Médio-Baixo","#FFD700"
    if pct>=2:  return "70-79","2-8","Abaixo da Média","#FF4500"
    return "<70","<2","Excepcionalmente Baixo","#FF0000"

# Plot distribution with Z marker
def plot_normal_distribution(z, name, pct, label, color):
    fig, ax = plt.subplots(figsize=(8,3), dpi=100)
    x=np.linspace(-4,4,200); y=norm.pdf(x)
    ax.plot(x,y); ax.scatter([z],[norm.pdf(z)],color=color,edgecolor='black',linewidth=1.5,s=100)
    ax.set_xlabel("Z-score",fontsize=8); ax.set_ylabel("Densidade",fontsize=8)
    ax.set_title(f"Normas BICAMS: {name}",fontsize=10)
    ax.legend([f"Z={z:.2f}, Pct={pct:.1f}%, {label}"],fontsize=8)
    ax.grid(True); fig.tight_layout()
    return fig

# PDF report
def save_report_as_pdf(data, name, sex, age, edu, date):
    pdf=FPDF(); pdf.set_auto_page_break(True,10); pdf.add_page()
    pdf.set_font("Arial","B",12)
    pdf.multi_cell(0,8,"Avaliação Cognitiva Normativa - BICAMS para EM",align='C')
    pdf.ln(4)
    pdf.set_font("Arial","",10)
    pdf.multi_cell(0,6,f"Nome: {name} | Sexo: {sex} | Idade: {age} | Escolaridade: {edu} | Data: {format_date(date)}",align='C')
    for name_m, raw, scaled, z, pct, label, fig in data:
        pdf.ln(4); pdf.set_font("Arial","B",10); pdf.cell(0,6,name_m,ln=True,align='C')
        pdf.set_font("Arial","",10); pdf.cell(0,6,f"Bruto: {raw} | Escala: {scaled} | Z: {z:.2f} | Pct: {pct:.1f}% | {label}",ln=True,align='C')
        with tempfile.NamedTemporaryFile(suffix='.png',delete=False) as tmp:
            fig.savefig(tmp.name,dpi=100); pdf.image(tmp.name,w=150,x=(pdf.w-150)/2); os.unlink(tmp.name)
    pdf.ln(4)
    pdf.set_font("Arial","I",8)
    pdf.multi_cell(0,4,"Normas: Spedo et al. 2022. doi:10.1590/0004-282X-ANP-2020-0526",align='C')
    out=tempfile.NamedTemporaryFile(delete=False,suffix='.pdf')
    pdf.output(out.name); return out.name, f"{name}_BICAMS_{date}.pdf"

# Main Streamlit app
def main():
    st.title("Calculadora Normativa BICAMS - Brasil")
    name=st.text_input("Nome/Código")
    sex=st.selectbox("Sexo",["M","F"])
    age=st.slider("Idade",18,100,40)
    edu=st.slider("Escolaridade",1,20,12)
    date=st.date_input("Data",value=datetime.today())
    report=[]
    for key, title, maxr in [("SDMT","Symbol Digit Modalities Test",120),("CVLT_totaldeacertos","CVLT-II",80),("BVMT_Total","BVMT-R",36)]:
        st.write("---"); st.write(f"### {title}")
        na=st.checkbox(f"Não se aplica - {title}",key=key)
        if not na:
            raw=st.number_input(f"Pontuação bruta {title}",0,maxr,int(maxr/2),1)
            scaled=convert_to_scaled_score(raw,key)
            pred=calculate_predicted_raw_score(age,sex,edu,key)
            z=(raw-pred)/regression_models[key]['residual_sd']
            pct=norm.cdf(z)*100
            scale_lbl,pct_lbl,label,color=interpret_percentile(pct)
            st.write(f"Bruto: {raw} | Escala: {scaled} | Z: {z:.2f} | Pct: {pct:.1f}% | {label}")
            fig=plot_normal_distribution(z,title,pct,label,color)
            st.pyplot(fig)
            report.append((title,raw,scaled,z,pct,label,fig))
    if st.button("Salvar PDF") and report:
        path,fname=save_report_as_pdf(report,name,sex,age,edu,date)
        with open(path,'rb') as f: st.download_button("Baixar PDF",f,file_name=fname,mime='application/pdf')
        os.remove(path)

if __name__=="__main__": main()
