import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import streamlit as st
from datetime import datetime
from fpdf import FPDF
from PIL import Image
import io
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

# Definir coeficientes do modelo de regressão e desvios padrão residuais para as medidas BICAMS
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

conversion_table = {
    'CVLT_totaldeacertos': {
        1: (-np.inf, 19), 2: (20, 28), 3: (29, 31), 4: (32, 35), 5: (36, 39), 6: (40, 41),
        7: (42, 44), 8: (45, 48), 9: (49, 52), 10: (53, 56), 11: (57, 60), 12: (61, 64),
        13: (65, 66), 14: (67, 69), 15: (70, 71), 16: (72, 72), 17: (73, 74), 18: (75, 75), 19: (76, np.inf)
    },
    'BVMT_Total': {
        1: (-np.inf, 2), 2: (3, 5), 3: (6, 8), 4: (9, 12), 5: (13, 17), 6: (18, 20),
        7: (21, 23), 8: (24, 26), 9: (27, 28), 10: (29, 30), 11: (31, 32), 12: (33, 34), 13: (35, 35),
        14: (36, 36)
    },
    'SDMT': {
        1: (-np.inf, 9), 2: (10, 17), 3: (18, 23), 4: (24, 29), 5: (30, 36), 6: (37, 43),
        7: (44, 49), 8: (50, 53), 9: (54, 58), 10: (59, 62), 11: (63, 68), 12: (69, 74), 13: (75, 79),
        14: (80, 93), 15: (94, 107), 16: (108, np.inf)
    }
}

# Função para converter pontuações brutas em pontuações escaladas
def convert_to_scaled_score(raw_score, measure):
    for scaled_score, (low, high) in conversion_table[measure].items():
        if low <= raw_score <= high:
            return scaled_score
    return np.nan

# Função para calcular pontuações escaladas previstas
def calculate_predicted_scaled_score(age, sex, education, measure):
    model = regression_models[measure]
    age2 = age ** 2
    sex_for_model = 1 if sex == 'M' else 2
    pss = (model['constant'] + model['age'] * age + model['age2'] * age2 +
           model['sex'] * sex_for_model + model['education'] * education)
    return pss

# Função para interpretar o percentil de acordo com a tabela
def interpret_percentile(percentile):
    if percentile > 98:
        return ">130", ">98", "Pontuação Excepcionalmente Alta", "Exceptionally High Score"
    elif 91 <= percentile <= 97:
        return "120-129", "91-97", "Pontuação Acima da Média", "Above Average Score"
    elif 75 <= percentile <= 90:
        return "110-119", "75-90", "Pontuação Média Superior", "High Average Score"
    elif 25 <= percentile <= 74:
        return "90-109", "25-74", "Pontuação Média", "Average Score"
    elif 9 <= percentile <= 24:
        return "80-89", "9-24", "Pontuação Média Inferior", "Low Average Score"
    elif 2 <= percentile <= 8:
        return "70-79", "2-8", "Pontuação Abaixo da Média", "Below Average Score"
    else:
        return "<70", "<2", "Pontuação Excepcionalmente Baixa", "Exceptionally Low Score"

def plot_normal_distribution(z_score, measure, measure_name):
    # Set a consistent figure size and DPI for uniformity
    fig, ax = plt.subplots(figsize=(6, 3), dpi=100)  # Set a consistent figure size and DPI

    x = np.linspace(-4, 4, 100)
    y = norm.pdf(x)
    ax.plot(x, y, label="Distribuição Normal", zorder=1)

    def custom_cmap(val):
        norm_val = val / 4
        if norm_val > 0:
            color = (0, 1 - norm_val, 1)
        else:
            color = (1, 1 + norm_val, 0)

        if val == 0:
            color = (0, 1, 0)

        return color

    dot_color = custom_cmap(z_score)
    ax.scatter([z_score], [norm.pdf(z_score)], color=dot_color, label="Z-score", s=100, zorder=2)

    ax.legend()
    ax.set_xlabel("Z-score", fontsize=8)
    ax.set_ylabel("Densidade de Probabilidade", fontsize=8)
    ax.set_title(f"Valores normativos para {measure_name}", fontsize=10)

    ax.grid()

    # Adjust layout to fit all content
    fig.tight_layout(rect=[0, 0, 0.8, 1])
    
    text_box = fig.add_axes([0.82, 0.1, 0.16, 0.8])  # Further reduce the text box width
    text_box.axis('off')
    
    percentile = norm.cdf(z_score) * 100
    score_label = interpret_percentile(percentile)
    
    text_box.text(0.1, 0.8, f"Z-score: {z_score:.2f}", fontsize=8, verticalalignment='top')
    text_box.text(0.1, 0.6, f"Percentil: {percentile:.1f}%", fontsize=8, verticalalignment='top')
    text_box.text(0.1, 0.4, f"Classificação: {score_label[2]}", fontsize=8, verticalalignment='top')

    return fig

def save_report_as_pdf(report_data, patient_name, sex, age, education, test_date):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=10)
    pdf.add_page()

    pdf.set_font("Arial", "B", 12)
    pdf.cell(190, 8, txt="Relatório BICAMS", ln=True, align="C")
    
    pdf.set_font("Arial", size=10)
    formatted_date = format_date(test_date)
    header_text = f"Nome ou Código: {patient_name} | Sexo: {sex} | Idade: {age} anos | Escolaridade: {education} anos | Data do Teste: {formatted_date}"
    pdf.multi_cell(190, 6, txt=header_text)
    pdf.cell(190, 6, txt="", ln=True)

    for data in report_data:
        measure, z_score, percentile, fig, score_label = data
        pdf.set_font("Arial", "B", size=10)
        pdf.cell(190, 6, txt=f"Teste: {measure}", ln=True)
        pdf.set_font("Arial", size=10)
        pdf.cell(190, 6, txt=f"Z-score: {z_score:.2f} | Percentil: {percentile:.1f}% | Classificação: {score_label[2]}", ln=True)
        pdf.cell(190, 6, txt="", ln=True)
        
        # Save figure to a temporary file with consistent size
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
            fig.set_size_inches(6, 3)  # Set consistent figure size for saving
            fig.savefig(tmpfile.name, format="png", dpi=100)
            pdf.image(tmpfile.name, x=10, y=None, w=190)  # Fit to page width without cropping
            os.unlink(tmpfile.name)  # Remove the temporary file after use
        pdf.cell(190, 6, txt="", ln=True)

    # Add the citation
    pdf.cell(190, 6, txt="", ln=True)
    pdf.set_font("Arial", "I", size=8)
    pdf.multi_cell(190, 4, txt="Conversão normativa utilizando a *Calculadora Normativa do BICAMS para a População Brasileira*, desenvolvida por Jonadab dos Santos Silva. Disponível em ", align="L")
    pdf.set_text_color(0, 0, 255)
    pdf.set_font("Arial", "U", 8)
    pdf.cell(0, 4, "https://bicams-brazil-calculator.streamlit.app/", ln=True, link="https://bicams-brazil-calculator.streamlit.app/")
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Arial", "I", size=8)
    pdf.multi_cell(190, 4, txt="Fonte dos dados normativos: Spedo CT, Pereira DA, Frndak SE, Marques VD, Barreira AA, Smerbeck A, Silva PHRD, Benedict RHB. Brief International Cognitive Assessment for Multiple Sclerosis (BICAMS): discrete and regression-based norms for the Brazilian context. Arq Neuropsiquiatr. 2022 Jan;80(1):62-68. doi: 10.1590/0004-282X-ANP-2020-0526.", align="L")
    
    file_name = f"{patient_name.replace(' ', '_')}_BICAMS_Report_{test_date.strftime('%Y-%m-%d')}.pdf"
    
    # Save PDF to a temporary file
    temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    pdf.output(temp_pdf.name)
    return temp_pdf.name, file_name

def main():
    st.title("Calculadora Normativa do BICAMS para a População Brasileira")

    patient_name = st.text_input("Nome ou Código do Paciente")
    sex_display = st.selectbox("Sexo", ["Masculino", "Feminino"])
    sex = "M" if sex_display == "Masculino" else "F"
    age = st.slider("Idade em anos", min_value=18, max_value=100, value=40, step=1)
    education = st.slider("Escolaridade em anos", min_value=1, max_value=20, value=12, step=1)
    test_date = st.date_input("Data do Teste", value=datetime.today())

    st.write("### CVLT (total de acertos)")
    cvlt_not_applicable = st.checkbox("Não se aplica", key="cvlt_na")
    if not cvlt_not_applicable:
        cvlt_input_method = st.radio("Como deseja inserir a pontuação?", ["Deslizar", "Digite"], key="cvlt_input")
        if cvlt_input_method == "Deslizar":
            cvlt_raw = st.slider("Pontuação Total CVLT", min_value=0, max_value=80, value=50, step=1)
        else:
            cvlt_raw = st.number_input("Pontuação Total CVLT", min_value=0, max_value=80, value=50, step=1)
    else:
        cvlt_raw = None

    st.write("### BVMT (total)")
    bvmt_not_applicable = st.checkbox("Não se aplica", key="bvmt_na")
    if not bvmt_not_applicable:
        bvmt_input_method = st.radio("Como deseja inserir a pontuação?", ["Deslizar", "Digite"], key="bvmt_input")
        if bvmt_input_method == "Deslizar":
            bvmt_raw = st.slider("Pontuação Total BVMT", min_value=0, max_value=36, value=20, step=1)
        else:
            bvmt_raw = st.number_input("Pontuação Total BVMT", min_value=0, max_value=36, value=20, step=1)
    else:
        bvmt_raw = None

    st.write("### SDMT")
    sdmt_not_applicable = st.checkbox("Não se aplica", key="sdmt_na")
    if not sdmt_not_applicable:
        sdmt_input_method = st.radio("Como deseja inserir a pontuação?", ["Deslizar", "Digite"], key="sdmt_input")
        if sdmt_input_method == "Deslizar":
            sdmt_raw = st.slider("Pontuação SDMT", min_value=0, max_value=120, value=60, step=1)
        else:
            sdmt_raw = st.number_input("Pontuação SDMT", min_value=0, max_value=120, value=60, step=1)
    else:
        sdmt_raw = None

    z_scores = []
    report_data = []

    if cvlt_raw is not None:
        cvlt_scaled = convert_to_scaled_score(cvlt_raw, 'CVLT_totaldeacertos')
        if not np.isnan(cvlt_scaled):
            cvlt_pss = calculate_predicted_scaled_score(age, sex, education, 'CVLT_totaldeacertos')
            cvlt_z = (cvlt_scaled - cvlt_pss) / regression_models['CVLT_totaldeacertos']['residual_sd']

            percentile = norm.cdf(cvlt_z) * 100
            score_label = interpret_percentile(percentile)
            st.write(f"Z-score: {cvlt_z:.2f}")
            st.write(f"Percentil: {percentile:.1f}%")
            st.write(f"Classificação: {score_label[2]}")

            fig_cvlt = plot_normal_distribution(cvlt_z, 'CVLT_totaldeacertos', "CVLT (total de acertos)")
            st.pyplot(fig_cvlt)

            z_scores.append(cvlt_z)
            report_data.append(("CVLT (total de acertos)", cvlt_z, percentile, fig_cvlt, score_label))

    if bvmt_raw is not None:
        bvmt_scaled = convert_to_scaled_score(bvmt_raw, 'BVMT_Total')
        if not np.isnan(bvmt_scaled):
            bvmt_pss = calculate_predicted_scaled_score(age, sex, education, 'BVMT_Total')
            bvmt_z = (bvmt_scaled - bvmt_pss) / regression_models['BVMT_Total']['residual_sd']

            percentile = norm.cdf(bvmt_z) * 100
            score_label = interpret_percentile(percentile)
            st.write(f"Z-score: {bvmt_z:.2f}")
            st.write(f"Percentil: {percentile:.1f}%")
            st.write(f"Classificação: {score_label[2]}")

            fig_bvmt = plot_normal_distribution(bvmt_z, 'BVMT_Total', "BVMT (total)")
            st.pyplot(fig_bvmt)

            z_scores.append(bvmt_z)
            report_data.append(("BVMT (total)", bvmt_z, percentile, fig_bvmt, score_label))

    if sdmt_raw is not None:
        sdmt_scaled = convert_to_scaled_score(sdmt_raw, 'SDMT')
        if not np.isnan(sdmt_scaled):
            sdmt_pss = calculate_predicted_scaled_score(age, sex, education, 'SDMT')
            sdmt_z = (sdmt_scaled - sdmt_pss) / regression_models['SDMT']['residual_sd']

            percentile = norm.cdf(sdmt_z) * 100
            score_label = interpret_percentile(percentile)
            st.write(f"Z-score: {sdmt_z:.2f}")
            st.write(f"Percentil: {percentile:.1f}%")
            st.write(f"Classificação: {score_label[2]}")

            fig_sdmt = plot_normal_distribution(sdmt_z, 'SDMT', "SDMT")
            st.pyplot(fig_sdmt)

            z_scores.append(sdmt_z)
            report_data.append(("SDMT", sdmt_z, percentile, fig_sdmt, score_label))

    if st.button("Salvar Relatório como PDF"):
        if report_data:
            temp_pdf_path, file_name = save_report_as_pdf(report_data, patient_name, sex, age, education, test_date)
            with open(temp_pdf_path, "rb") as f:
                st.download_button(label="Baixar Relatório PDF", data=f, file_name=file_name, mime="application/pdf")
            os.remove(temp_pdf_path)
        else:
            st.warning("Nenhum teste foi realizado.")

if __name__ == "__main__":
    main()
