import os
import re
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                Image, Table, TableStyle, HRFlowable, PageBreak)
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.utils import ImageReader

W, H = A4
OUT = 'CBIR_CEDAR_Relatorio.pdf'

def ler_sumario(caminho):
    map_score = "0.0000"
    linhas_tabela = []
    if os.path.exists(caminho):
        with open(caminho, 'r') as f:
            linhas = f.readlines()
        for linha in linhas:
            if linha.startswith("Query"):
                match = re.search(r'Query (\d+) \((.*?)\): AP=([\d.]+) P@1=([\d.]+) P@3=([\d.]+) P@5=([\d.]+)', linha)
                if match:
                    id_q, classe, ap, p1, p3, p5 = match.groups()
                    linhas_tabela.append([f'Q{id_q}', classe, ap, p1, p3, p5])
            elif linha.startswith("MAP"):
                map_score = linha.split("=")[1].strip()
    return linhas_tabela, map_score

def make_pdf():
    doc = SimpleDocTemplate(
        OUT, pagesize=A4,
        leftMargin=3*cm, rightMargin=2*cm,
        topMargin=3*cm, bottomMargin=2*cm
    )
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle('Title2', parent=styles['Title'],
        fontSize=16, textColor=colors.black, spaceAfter=12, alignment=TA_CENTER, fontName='Times-Bold')
    subtitle_style = ParagraphStyle('Subtitle', parent=styles['Normal'],
        fontSize=12, textColor=colors.black, spaceAfter=6, alignment=TA_CENTER, fontName='Times-Roman')
    h1 = ParagraphStyle('H1', parent=styles['Heading1'],
        fontSize=14, textColor=colors.black, spaceBefore=18, spaceAfter=8, fontName='Times-Bold', leading=16)
    h2 = ParagraphStyle('H2', parent=styles['Heading2'],
        fontSize=12, textColor=colors.black, spaceBefore=12, spaceAfter=6, fontName='Times-Bold')
    body = ParagraphStyle('Body', parent=styles['Normal'],
        fontSize=11, textColor=colors.black, leading=14, alignment=TA_JUSTIFY, spaceAfter=8, fontName='Times-Roman')
    code_style = ParagraphStyle('Code', parent=styles['Code'],
        fontSize=9, leading=11, fontName='Courier',
        leftIndent=12, rightIndent=12, spaceBefore=6, spaceAfter=6)
    caption = ParagraphStyle('Caption', parent=styles['Normal'],
        fontSize=10, textColor=colors.black, alignment=TA_CENTER, spaceBefore=4, spaceAfter=12, fontName='Times-Italic')

    def img_block(path, target_width=14*cm, cap=''):
        items = []
        if os.path.exists(path):
            img_reader = ImageReader(path)
            orig_width, orig_height = img_reader.getSize()
            
            aspect_ratio = orig_height / orig_width
            target_height = target_width * aspect_ratio
            
            items.append(Image(path, width=target_width, height=target_height))
        
        if cap:
            items.append(Paragraph(cap, caption))
        return items

    story = []

    story += [
        Spacer(1, 2*cm),
        Paragraph("Sistema de Content-Based Image Retrieval (CBIR) com Contexto Espacial", title_style),
        Spacer(1, 0.5*cm),
        Paragraph("Disciplina: Recuperação de Informação Multimídia", subtitle_style),
        Paragraph("Iuri Pedroso &nbsp;|&nbsp; Herich Gabriel de Campos", subtitle_style),
        Paragraph("Universidade Estadual do Centro-Oeste (UNICENTRO)", subtitle_style),
        Paragraph("Maio de 2026", subtitle_style),
        Spacer(1, 1.5*cm),
    ]

    story += [
        Paragraph("1. Introdução", h1),
        Paragraph(
            "Este relatório documenta a implementação de um sistema de Content-Based Image "
            "Retrieval (CBIR) aplicado a imagens de assinaturas do dataset CEDAR. O sistema "
            "tem como objetivo recuperar imagens visualmente similares a uma consulta (query), "
            "utilizando uma abordagem que pondera a extração clássica de características visuais "
            "e a posição espacial (contexto) das regiões de interesse identificadas.", body),
        Paragraph(
            "A arquitetura prescinde de métodos de aprendizado profundo (Deep Learning), adequando-se "
            "ao escopo de indexação e recuperação clássica por similaridade geométrica e de textura.", body),
    ]

    story += [
        Paragraph("2. Dataset e Pré-processamento", h1),
        Paragraph(
            "Foi selecionado o dataset CEDAR, composto por imagens reais de assinaturas originais "
            "e falsificadas. As imagens foram redimensionadas preservando a proporção original e "
            "inseridas em um canvas padrão. Em seguida, aplicou-se binarização adaptativa para "
            "realçar os traços contra o fundo.", body),
    ]

    tdata = [
        ['Parâmetro', 'Valor'],
        ['Total de documentos indexados', '25'],
        ['Total de queries', '5'],
        ['Redimensionamento base', '128 × 256 pixels'],
    ]
    t = Table(tdata, colWidths=[7*cm, 7*cm])
    t.setStyle(TableStyle([
        ('FONTNAME',   (0,0), (-1,0), 'Times-Bold'),
        ('FONTSIZE',   (0,0), (-1,-1), 11),
        ('GRID',       (0,0), (-1,-1), 0.5, colors.black),
        ('ALIGN',      (0,0), (-1,-1), 'LEFT'),
        ('VALIGN',     (0,0), (-1,-1), 'MIDDLE'),
        ('PADDING',    (0,0), (-1,-1), 4),
    ]))
    story += [t, Spacer(1, 0.5*cm)]
    story += img_block('./output_cbir/fig1_dataset_overview.png', 14*cm,
                       'Figura 1: Amostra dos documentos do dataset CEDAR pré-processados.')

    story += [
        Paragraph("3. Propostas de Regiões", h1),
        Paragraph(
            "Para a identificação dos objetos de interesse, implementou-se o algoritmo de Sliding "
            "Window. Três escalas de janelas varrem a imagem analisando a densidade de pixels pretos. "
            "Posteriormente, aplica-se o filtro de Non-Maximum Suppression (NMS) com limiar de IoU "
            "de 0.4 para eliminar redundâncias e isolar a assinatura principal.", body),
    ]
    story += img_block('./output_cbir/fig2_proposals.png', 14*cm,
                       'Figura 2: Processo de geração de regiões candidatas e seleção via NMS.')

    story += [
        Paragraph("4. Extração de Descritores e Indexação", h1),
        Paragraph(
            "A representação vetorial de cada região foi construída através da concatenação de "
            "três extratores de características:", body),
        Paragraph("1. HOG (Histogram of Oriented Gradients): Para captação das distribuições de bordas.", body),
        Paragraph("2. Momentos de Hu: Visando capturar invariantes geométricas.", body),
        Paragraph("3. LBP (Local Binary Pattern): Para análise de textura.", body),
        Paragraph(
            "Os descritores combinados foram submetidos a redução de dimensionalidade (PCA) para 64 "
            "componentes e indexados em uma estrutura BallTree (espaço Euclidiano).", body),
    ]

    story += [
        Paragraph("5. Ranqueamento", h1),
        Paragraph(
            "A recuperação pondera a distância no espaço vetorial e a coerência espacial da posição "
            "do traço. O cálculo é regido por:", body),
        Paragraph(
            "Score = (0.7 * Similaridade_Visual) + (0.3 * IoU_Espacial)", code_style),
        Paragraph(
            "Essa abordagem garante que traços com formatos semelhantes, mas localizados em posições "
            "incompatíveis do canvas, recebam penalidades estruturais.", body),
    ]

    story += [
        Paragraph("6. Resultados", h1),
    ]

    linhas_sumario, map_score = ler_sumario('./output_cbir/summary.txt')
    
    metrics_table = [['Query', 'Classe', 'AP', 'P@1', 'P@3', 'P@5']]
    
    if linhas_sumario:
        for linha in linhas_sumario:
            metrics_table.append(linha)
    else:
        for i in range(5):
            metrics_table.append([f'Q{i}', 'N/A', '0.000', '0.00', '0.00', '0.00'])

    metrics_table.append(['—', 'MAP', map_score, '—', '—', '—'])

    mt = Table(metrics_table, colWidths=[2*cm, 4*cm, 2*cm, 2*cm, 2*cm, 2*cm])
    mt.setStyle(TableStyle([
        ('FONTNAME',   (0,0), (-1,0), 'Times-Bold'),
        ('FONTNAME',   (1,-1), (-1,-1), 'Times-Bold'),
        ('FONTSIZE',   (0,0), (-1,-1), 11),
        ('GRID',       (0,0), (-1,-1), 0.5, colors.black),
        ('ALIGN',      (2,0), (-1,-1), 'CENTER'),
        ('VALIGN',     (0,0), (-1,-1), 'MIDDLE'),
        ('PADDING',    (0,0), (-1,-1), 4),
    ]))
    story += [mt, Spacer(1, 0.5*cm)]
    story += img_block('./output_cbir/fig4_metrics.png', 14*cm,
                       'Figura 3: Gráficos de Average Precision e Precision@K.')

    story.append(PageBreak())

    story.append(Paragraph("6.1. Consultas", h2))
    
    if linhas_sumario:
        for i, linha in enumerate(linhas_sumario):
            id_q = linha[0].replace('Q', '')
            path = f'./output_cbir/fig3_query_{id_q}.png'
            story += img_block(path, 14*cm, f'Figura {4+i}: Resultados recuperados para a {linha[0]}.')
            if i < len(linhas_sumario) - 1:
                story.append(Spacer(1, 0.5*cm))
    else:
        for qi in range(5):
            path = f'./output_cbir/fig3_query_{qi}.png'
            story += img_block(path, 14*cm, f'Figura {4+qi}: Resultados recuperados para a Query {qi}.')
            if qi < 4:
                story.append(Spacer(1, 0.5*cm))

    story += [
        Paragraph("7. Conclusão", h1),
        Paragraph(
            "O pipeline implementado cumpre as diretrizes estabelecidas. A combinação de algoritmos "
            "clássicos de visão computacional demonstrou ser uma ferramenta válida para a estruturação "
            "e recuperação de dados puramente baseada no conteúdo e na coerência da posição dos traços.", body),
        Spacer(1, 0.5*cm),
        Paragraph("Link do repositório: https://github.com/iuripedroso/TopicosP-Computacao", code_style),
    ]

    doc.build(story)
    print(f" PDF acadêmico gerado: {OUT}")

if __name__ == '__main__':
    make_pdf()