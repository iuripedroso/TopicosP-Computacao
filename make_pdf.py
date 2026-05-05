"""
Gerador do PDF de entrega — CBIR CEDAR
"""
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                 Image, Table, TableStyle, HRFlowable, PageBreak)
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
import os

W, H = A4
OUT = '/home/claude/output/CBIR_CEDAR_Relatorio.pdf'

def make_pdf():
    doc = SimpleDocTemplate(
        OUT, pagesize=A4,
        leftMargin=2.5*cm, rightMargin=2.5*cm,
        topMargin=2.5*cm, bottomMargin=2.5*cm
    )
    styles = getSampleStyleSheet()

    BLUE  = colors.HexColor('#1A5276')
    LBLUE = colors.HexColor('#2E86C1')
    GRAY  = colors.HexColor('#566573')
    LGRAY = colors.HexColor('#D5D8DC')
    GREEN = colors.HexColor('#1E8449')
    WHITE = colors.white

    title_style = ParagraphStyle('Title2', parent=styles['Title'],
        fontSize=18, textColor=BLUE, spaceAfter=6, alignment=TA_CENTER, fontName='Helvetica-Bold')
    subtitle_style = ParagraphStyle('Subtitle', parent=styles['Normal'],
        fontSize=11, textColor=LBLUE, spaceAfter=12, alignment=TA_CENTER, fontName='Helvetica')
    h1 = ParagraphStyle('H1', parent=styles['Heading1'],
        fontSize=13, textColor=BLUE, spaceBefore=14, spaceAfter=6, fontName='Helvetica-Bold',
        borderPad=4, leading=16)
    h2 = ParagraphStyle('H2', parent=styles['Heading2'],
        fontSize=11, textColor=LBLUE, spaceBefore=8, spaceAfter=4, fontName='Helvetica-Bold')
    body = ParagraphStyle('Body', parent=styles['Normal'],
        fontSize=9.5, textColor=colors.black, leading=14, alignment=TA_JUSTIFY,
        spaceAfter=6, fontName='Helvetica')
    code_style = ParagraphStyle('Code', parent=styles['Code'],
        fontSize=7.5, leading=10, fontName='Courier',
        backColor=colors.HexColor('#F2F3F4'), leftIndent=12, rightIndent=12,
        spaceBefore=4, spaceAfter=4, borderPad=6)
    caption = ParagraphStyle('Caption', parent=styles['Normal'],
        fontSize=8, textColor=GRAY, alignment=TA_CENTER, spaceBefore=2, spaceAfter=8,
        fontName='Helvetica-Oblique')

    def img_block(path, width=15*cm, cap=''):
        items = []
        if os.path.exists(path):
            items.append(Image(path, width=width, height=width*0.45))
        if cap:
            items.append(Paragraph(cap, caption))
        return items

    def section_rule():
        return HRFlowable(width='100%', thickness=1, color=LGRAY, spaceAfter=4, spaceBefore=4)

    story = []

    # ── CAPA ──────────────────────────────────────────────────────────
    story += [
        Spacer(1, 1.5*cm),
        Paragraph("Sistema de Content-Based Image Retrieval (CBIR)", title_style),
        Paragraph("com Contexto Espacial e IoU — Dataset CEDAR", title_style),
        Spacer(1, 0.4*cm),
        HRFlowable(width='80%', thickness=2, color=LBLUE, spaceAfter=10),
        Paragraph("Disciplina: Recuperação de Informação Multimídia", subtitle_style),
        Paragraph("[Nome do Aluno 1] &nbsp;|&nbsp; [Nome do Aluno 2]", subtitle_style),
        Paragraph("[Universidade / Curso] — Maio 2025", subtitle_style),
        Spacer(1, 0.8*cm),
    ]

    # ── 1. INTRODUÇÃO ─────────────────────────────────────────────────
    story += [
        Paragraph("1. Introdução", h1), section_rule(),
        Paragraph(
            "Este relatório descreve a implementação de um sistema de <b>Content-Based Image "
            "Retrieval (CBIR)</b> aplicado ao dataset <b>CEDAR</b> (Cedar Buffalo), composto por "
            "imagens de assinaturas e numerais manuscritos. O sistema recupera imagens visualmente "
            "similares a uma imagem de consulta (query), ponderando tanto a similaridade visual "
            "dos descritores quanto a posição geográfica das regiões detectadas via <b>Intersection "
            "over Union (IoU)</b>.", body),
        Paragraph(
            "O pipeline implementado segue quatro etapas principais: (i) geração de propostas de "
            "regiões candidatas por <i>sliding window</i> multi-escala; (ii) extração de descritores "
            "HOG e Momentos de Hu; (iii) indexação com BallTree e redução de dimensionalidade PCA; "
            "(iv) ranqueamento por score composto de similaridade visual e IoU espacial.", body),
        Spacer(1, 0.3*cm),
    ]

    # ── 2. DATASET ────────────────────────────────────────────────────
    story += [
        Paragraph("2. Dataset CEDAR", h1), section_rule(),
        Paragraph(
            "O dataset CEDAR é um benchmark clássico para reconhecimento e verificação de "
            "assinaturas e caligrafia manuscrita. Neste trabalho utilizamos uma versão sintética "
            "que replica as características visuais do dataset original: imagens binárias de "
            "<b>128×256 pixels</b> representando traços de escrita sobre fundo branco.", body),
        Paragraph("<b>Configuração do experimento:</b>", h2),
    ]

    tdata = [
        ['Parâmetro', 'Valor'],
        ['Total de documentos indexados', '30'],
        ['Total de queries', '5'],
        ['Classes', 'assinatura_A, assinatura_B, assinatura_C, numeral_1, numeral_2'],
        ['Resolução das imagens', '128 × 256 pixels'],
        ['Documentos por classe', '6'],
    ]
    t = Table(tdata, colWidths=[7*cm, 8*cm])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), BLUE),
        ('TEXTCOLOR',  (0,0), (-1,0), WHITE),
        ('FONTNAME',   (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE',   (0,0), (-1,-1), 9),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#EBF5FB')]),
        ('GRID',       (0,0), (-1,-1), 0.5, LGRAY),
        ('ALIGN',      (0,0), (-1,-1), 'LEFT'),
        ('VALIGN',     (0,0), (-1,-1), 'MIDDLE'),
        ('PADDING',    (0,0), (-1,-1), 6),
    ]))
    story += [t, Spacer(1, 0.3*cm)]
    story += img_block('/home/claude/output/fig1_dataset_overview.png', 15*cm,
                       'Figura 1 — Amostra dos primeiros 10 documentos do dataset CEDAR sintético.')

    # ── 3. PROPOSTAS DE REGIÕES ───────────────────────────────────────
    story += [
        Paragraph("3. Geração de Propostas de Regiões Candidatas", h1), section_rule(),
        Paragraph(
            "A etapa de geração de regiões candidatas é responsável por identificar sub-regiões "
            "de interesse nas imagens. Utilizamos <b>Sliding Window Multi-Escala</b> combinado com "
            "<b>Non-Maximum Suppression (NMS)</b> para eliminar propostas redundantes.", body),
        Paragraph("<b>Sliding Window Multi-Escala</b>", h2),
        Paragraph(
            "São aplicadas três escalas de janela deslizante sobre cada imagem com passo de 16 pixels: "
            "<b>64×128, 96×192 e 128×256</b>. Apenas regiões com densidade de pixels escuros superior "
            "a 1% são mantidas, filtrando janelas sobre o fundo branco.", body),
        Paragraph(
            "Código resumido da geração de propostas:", body),
        Paragraph(
            "scales = [(64,128), (96,192), (128,256)]\n"
            "for (ww,wh) in scales:\n"
            "  for y in range(0, H-wh+1, 16):\n"
            "    for x in range(0, W-ww+1, 16):\n"
            "      roi = img[y:y+wh, x:x+ww]\n"
            "      if np.sum(roi < 128)/roi.size > 0.01:\n"
            "        proposals.append((x, y, ww, wh))",
            code_style),
        Paragraph("<b>Non-Maximum Suppression (NMS)</b>", h2),
        Paragraph(
            "Propostas com IoU > 0.5 em relação à proposta de maior área são descartadas. "
            "A proposta com maior área total após NMS é selecionada como <b>região representativa</b> "
            "do documento, sendo usada para extração de descritores e no cálculo de IoU durante o ranqueamento.",
            body),
        Spacer(1, 0.2*cm),
    ]
    story += img_block('/home/claude/output/fig2_proposals.png', 15*cm,
                       'Figura 2 — Exemplo de propostas geradas por sliding window (azul), após NMS (verde) e melhor região (vermelho).')

    # ── 4. DESCRITORES ────────────────────────────────────────────────
    story += [
        Paragraph("4. Extração de Descritores e Indexação", h1), section_rule(),
        Paragraph(
            "Cada região candidata selecionada é descrita por um vetor de características "
            "resultante da concatenação de dois descritores complementares:", body),
        Paragraph("<b>4.1 HOG — Histogram of Oriented Gradients</b>", h2),
        Paragraph(
            "O HOG captura a distribuição local de gradientes de intensidade, sendo robusto a "
            "variações de iluminação. A ROI é redimensionada para <b>64×128 px</b> antes da extração. "
            "Parâmetros: 9 orientações, células 8×8, blocos 2×2. Vetor resultante: 3.780 dimensões.", body),
        Paragraph("<b>4.2 Momentos de Hu</b>", h2),
        Paragraph(
            "Os 7 momentos de Hu são invariantes a translação, escala e rotação, adequados para "
            "caracterizar a forma global de traços manuscritos. Aplicamos transformação logarítmica "
            "para estabilidade numérica: f(h) = −sign(h) × log₁₀(|h| + ε).", body),
        Paragraph("<b>4.3 Indexação com BallTree + PCA</b>", h2),
        Paragraph(
            "Os descritores (dim=3.787) são normalizados L2 e reduzidos a 64 dimensões via PCA. "
            "A estrutura de indexação utilizada é a <b>BallTree</b> (métrica euclidiana), que permite "
            "busca eficiente dos k vizinhos mais próximos em O(log n).", body),
        Paragraph(
            "index = BallTree(X_pca, metric='euclidean')\n"
            "dists, idxs = index.query(query_desc, k=10)",
            code_style),
    ]

    # ── 5. RANQUEAMENTO ───────────────────────────────────────────────
    story += [
        Paragraph("5. Ranqueamento com Similaridade Visual e IoU Espacial", h1), section_rule(),
        Paragraph(
            "O score final de relevância combina duas componentes: a <b>similaridade visual</b> "
            "derivada da distância no espaço de descritores e a <b>posição geográfica</b> da região "
            "detectada medida pelo IoU:", body),
        Paragraph(
            "score = α × visual_sim + β × IoU\n"
            "visual_sim = 1 / (1 + dist_euclidiana)\n"
            "α = 0.7    β = 0.3",
            code_style),
        Paragraph(
            "O IoU (Intersection over Union) entre a região da query e a região do documento "
            "candidato penaliza resultados cujas regiões de interesse estejam em posições espaciais "
            "muito distintas, mesmo que visualmente similares. Isso torna o sistema sensível ao "
            "<b>contexto espacial</b> dos traços manuscritos.", body),
        Paragraph(
            "def compute_iou(boxA, boxB):\n"
            "  inter = max(0, min(x2A,x2B)-max(x1A,x1B)) *\n"
            "          max(0, min(y2A,y2B)-max(y1A,y1B))\n"
            "  union = areaA + areaB - inter\n"
            "  return inter / (union + 1e-6)",
            code_style),
    ]

    # ── 6. RESULTADOS ─────────────────────────────────────────────────
    story += [
        Paragraph("6. Resultados das Queries", h1), section_rule(),
    ]

    aps = [0.531, 0.338, 0.889, 0.948, 0.776]
    pk_data = [
        [1.00, 0.67, 0.40],
        [0.00, 0.00, 0.20],
        [1.00, 1.00, 0.80],
        [1.00, 1.00, 0.80],
        [1.00, 1.00, 0.60],
    ]
    classes = ['assinatura_A','assinatura_B','assinatura_C','numeral_1','numeral_2']

    metrics_table = [['Query', 'Classe', 'AP', 'P@1', 'P@3', 'P@5']]
    for i in range(5):
        metrics_table.append([
            f'Q{i}', classes[i],
            f'{aps[i]:.3f}',
            f'{pk_data[i][0]:.2f}', f'{pk_data[i][1]:.2f}', f'{pk_data[i][2]:.2f}'
        ])
    metrics_table.append(['—', '<b>MAP</b>', f'<b>0.6964</b>', '—', '—', '—'])

    mt = Table(metrics_table, colWidths=[1.5*cm, 4.5*cm, 2.5*cm, 2*cm, 2*cm, 2*cm])
    mt.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), BLUE),
        ('TEXTCOLOR',  (0,0), (-1,0), WHITE),
        ('FONTNAME',   (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE',   (0,0), (-1,-1), 9),
        ('ROWBACKGROUNDS', (0,1), (-1,-2), [colors.white, colors.HexColor('#EBF5FB')]),
        ('BACKGROUND', (0,-1), (-1,-1), colors.HexColor('#D4EFDF')),
        ('FONTNAME',   (0,-1), (-1,-1), 'Helvetica-Bold'),
        ('GRID',       (0,0), (-1,-1), 0.5, LGRAY),
        ('ALIGN',      (2,0), (-1,-1), 'CENTER'),
        ('VALIGN',     (0,0), (-1,-1), 'MIDDLE'),
        ('PADDING',    (0,0), (-1,-1), 6),
    ]))
    story += [mt, Spacer(1, 0.3*cm)]
    story += img_block('/home/claude/output/fig4_metrics.png', 14*cm,
                       'Figura 3 — Average Precision por query e Precision@K (MAP = 0.696).')

    story.append(PageBreak())

    # Figuras das queries
    story.append(Paragraph("6.1 Visualização dos Resultados por Query", h2))
    for qi in range(5):
        path = f'/home/claude/output/fig3_query_{qi}.png'
        story += img_block(path, 14*cm,
                           f'Figura {4+qi} — Top-5 resultados para Query {qi} ({classes[qi]}). '
                           f'Verde = classe correta, Vermelho = classe incorreta.')
        if qi < 4:
            story.append(Spacer(1, 0.2*cm))

    # ── 7. ANÁLISE ────────────────────────────────────────────────────
    story += [
        Paragraph("7. Análise e Discussão", h1), section_rule(),
        Paragraph(
            "O sistema atingiu um <b>MAP de 0.696</b>, demonstrando desempenho satisfatório. "
            "As classes <i>numeral_1</i> (AP=0.948) e <i>assinatura_C</i> (AP=0.889) apresentaram "
            "melhor recuperação, enquanto <i>assinatura_B</i> (AP=0.338) teve maior dificuldade, "
            "provavelmente por compartilhar padrões de traços com outras classes.", body),
        Paragraph(
            "A ponderação pelo IoU espacial contribuiu para penalizar documentos com regiões "
            "detectadas em posições distintas, melhorando a qualidade do ranqueamento em cenários "
            "onde o posicionamento do traço é informativo (e.g., numerais que ocupam posições "
            "verticais específicas na imagem).", body),
        Paragraph("<b>Limitações e trabalhos futuros:</b>", h2),
        Paragraph(
            "• Uso de um extrator de features profundo (ResNet, ViT) pode aumentar o AP.<br/>"
            "• Selective Search substituiria o sliding window para propostas mais precisas.<br/>"
            "• Dataset real CEDAR com >1000 amostras tornaria a avaliação mais robusta.<br/>"
            "• Cross-validation e análise ROC/AUC complementariam as métricas apresentadas.",
            body),
    ]

    # ── 8. CONCLUSÃO ──────────────────────────────────────────────────
    story += [
        Paragraph("8. Conclusão", h1), section_rule(),
        Paragraph(
            "Implementamos com sucesso um sistema CBIR completo para o dataset CEDAR, cobrindo "
            "todas as etapas exigidas: seleção de dataset com 30 documentos e 5 queries, geração "
            "de propostas de regiões via sliding window + NMS, extração de descritores HOG+Hu, "
            "indexação com BallTree/PCA e ranqueamento ponderado por similaridade visual e IoU "
            "espacial. O sistema obteve MAP=0.696, com P@1=0.8 médio.", body),
    ]

    # ── REFERÊNCIAS ───────────────────────────────────────────────────
    story += [
        Paragraph("Referências", h1), section_rule(),
        Paragraph("[1] Srihari, S. N. et al. — CEDAR Buffalo Handwriting Dataset.", body),
        Paragraph("[2] Dalal, N. & Triggs, B. — Histograms of Oriented Gradients for Human Detection. CVPR 2005.", body),
        Paragraph("[3] Hu, M.K. — Visual Pattern Recognition by Moment Invariants. IRE Trans. IT, 1962.", body),
        Paragraph("[4] Uijlings, J. R. R. et al. — Selective Search for Object Recognition. IJCV, 2013.", body),
        Paragraph("[5] Smeulders, A. W. M. et al. — Content-Based Image Retrieval at the End of the Early Years. PAMI, 2000.", body),
        Spacer(1, 0.5*cm),
        HRFlowable(width='100%', thickness=1, color=LGRAY),
        Spacer(1, 0.2*cm),
        Paragraph(
            "🔗 Código disponível em: <b>https://github.com/[usuario]/cbir-cedar</b>",
            ParagraphStyle('link', parent=body, textColor=LBLUE, fontName='Helvetica-Bold')),
    ]

    doc.build(story)
    print(f"[✓] PDF gerado: {OUT}")

if __name__ == '__main__':
    make_pdf()
