import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np


class FluxogramaReciclagemEletronica:
    """
    Classe para criar fluxograma profissional dos processos de reciclagem eletrônica.
    """

    def __init__(self):
        # Configurações de estilo
        self.colors = {
            'coleta': '#E57373',
            'desmontagem': '#64B5F6',
            'separacao': '#4DB6AC',
            'refino': '#81C784',
            'arrow': '#2C3E50',
            'text_primary': '#2C3E50',
            'text_secondary': '#34495E',
            'background': '#FFFFFF',
            'border': '#34495E'
        }

        self.fonts = {
            'title': {'size': 20, 'weight': 'bold'},
            'main_box': {'size': 12, 'weight': 'bold'},
            'sub_text': {'size': 10, 'weight': 'normal'}
        }

        # Dados do fluxograma
        self.etapas_principais = [
            {
                'pos': (2, 5),
                'text': 'COLETA E\nPRÉ-PROCESSO',
                'color': self.colors['coleta'],
                'width': 3.2,
                'height': 1.8
            },
            {
                'pos': (6, 5),
                'text': 'DESMONTAGEM\nE TRIAGEM',
                'color': self.colors['desmontagem'],
                'width': 3.2,
                'height': 1.8
            },
            {
                'pos': (10, 5),
                'text': 'SEPARAÇÃO\nMECÂNICA',
                'color': self.colors['separacao'],
                'width': 3.2,
                'height': 1.8
            },
            {
                'pos': (14, 5),
                'text': 'REFINO E\nPURIFICAÇÃO',
                'color': self.colors['refino'],
                'width': 3.2,
                'height': 1.8
            }
        ]

        self.subcategorias = [
            {
                'pos': (2, 2.5),
                'items': ['• Logística reversa', '• Classificação inicial', '• Armazenamento seguro']
            },
            {
                'pos': (6, 2.5),
                'items': ['• Processo manual', '• Automatização', '• Robótica avançada']
            },
            {
                'pos': (10, 2.5),
                'items': ['• Separação magnética', '• Por densidade', '• Flotação seletiva']
            },
            {
                'pos': (14, 2.5),
                'items': ['• Hidrometalurgia', '• Pirometalurgia', '• Bio-processamento']
            }
        ]

    def configurar_matplotlib(self):
        """Configura o estilo do matplotlib para um visual mais profissional."""
        plt.style.use('default')
        plt.rcParams.update({
            'font.family': 'DejaVu Sans',
            'axes.spines.top': False,
            'axes.spines.right': False,
            'axes.spines.bottom': False,
            'axes.spines.left': False,
            'figure.facecolor': self.colors['background']
        })

    def criar_caixa_principal(self, ax, etapa):
        """Cria uma caixa principal do fluxograma com estilo profissional."""
        x, y = etapa['pos']
        width, height = etapa['width'], etapa['height']

        # Sombra
        shadow = FancyBboxPatch(
            (x - width / 2 + 0.05, y - height / 2 - 0.05),
            width, height,
            boxstyle="round,pad=0.15,rounding_size=0.1",
            facecolor='lightgray',
            alpha=0.3,
            zorder=1
        )
        ax.add_patch(shadow)

        # Caixa principal
        box = FancyBboxPatch(
            (x - width / 2, y - height / 2),
            width, height,
            boxstyle="round,pad=0.15,rounding_size=0.1",
            facecolor=etapa['color'],
            edgecolor=self.colors['border'],
            linewidth=1.5,
            zorder=2
        )
        ax.add_patch(box)

        # Texto
        ax.text(
            x, y, etapa['text'],
            ha='center', va='center',
            fontsize=self.fonts['main_box']['size'],
            fontweight=self.fonts['main_box']['weight'],
            color='white',
            zorder=3
        )

    def criar_seta_conexao(self, ax, inicio, fim):
        """Cria setas de conexão entre as etapas."""
        arrow = FancyArrowPatch(
            inicio, fim,
            arrowstyle='-|>',
            lw=2.5,
            color=self.colors['arrow'],
            mutation_scale=20,
            zorder=2
        )
        ax.add_patch(arrow)

    def criar_caixa_subcategoria(self, ax, subcategoria):
        """Cria caixas de subcategorias com informações detalhadas."""
        x, y = subcategoria['pos']
        text = '\n'.join(subcategoria['items'])

        # Caixa de fundo
        bbox_props = dict(
            boxstyle="round,pad=0.4",
            facecolor='white',
            edgecolor=self.colors['border'],
            linewidth=1,
            alpha=0.95
        )

        ax.text(
            x, y, text,
            ha='center', va='top',
            fontsize=self.fonts['sub_text']['size'],
            fontweight=self.fonts['sub_text']['weight'],
            color=self.colors['text_secondary'],
            bbox=bbox_props,
            zorder=3
        )

    def criar_seta_subcategoria(self, ax, etapa_pos, sub_pos):
        """Cria setas conectando etapas principais às subcategorias."""
        start_x, start_y = etapa_pos
        end_x, end_y = sub_pos

        arrow = FancyArrowPatch(
            (start_x, start_y - 0.9),
            (end_x, end_y + 0.8),
            arrowstyle='-|>',
            lw=1.5,
            color=self.colors['arrow'],
            alpha=0.7,
            connectionstyle="arc3,rad=0.1",
            mutation_scale=15,
            zorder=1
        )
        ax.add_patch(arrow)

    def criar_fluxograma(self, salvar_arquivo=True, nome_arquivo='fluxograma_reciclagem_profissional.png'):
        """Método principal para criar o fluxograma completo."""
        self.configurar_matplotlib()

        # Criar figura
        fig, ax = plt.subplots(1, 1, figsize=(18, 10))

        # Título principal
        ax.text(
            8, 7.5,
            'PANORAMA GERAL DOS PROCESSOS DE RECICLAGEM ELETRÔNICA',
            ha='center', va='center',
            fontsize=self.fonts['title']['size'],
            fontweight=self.fonts['title']['weight'],
            color=self.colors['text_primary']
        )

        # Criar caixas principais
        for etapa in self.etapas_principais:
            self.criar_caixa_principal(ax, etapa)

        # Criar setas entre etapas principais
        for i in range(len(self.etapas_principais) - 1):
            start_pos = self.etapas_principais[i]['pos']
            end_pos = self.etapas_principais[i + 1]['pos']

            start_x = start_pos[0] + self.etapas_principais[i]['width'] / 2
            end_x = end_pos[0] - self.etapas_principais[i + 1]['width'] / 2

            self.criar_seta_conexao(
                ax,
                (start_x, start_pos[1]),
                (end_x, end_pos[1])
            )

        # Criar subcategorias
        for subcategoria in self.subcategorias:
            self.criar_caixa_subcategoria(ax, subcategoria)

        # Criar setas para subcategorias
        for i, (etapa, subcategoria) in enumerate(zip(self.etapas_principais, self.subcategorias)):
            self.criar_seta_subcategoria(ax, etapa['pos'], subcategoria['pos'])

        # Configurações do gráfico
        ax.set_xlim(0, 16)
        ax.set_ylim(0.5, 8)
        ax.axis('off')

        # Layout e salvamento
        plt.tight_layout()

        if salvar_arquivo:
            plt.savefig(
                nome_arquivo,
                dpi=300,
                bbox_inches='tight',
                facecolor=self.colors['background'],
                edgecolor='none',
                format='png'
            )
            print(f"Fluxograma salvo como: {nome_arquivo}")

        plt.show()
        return fig, ax

    def gerar_relatorio_tecnico(self):
        """Gera um relatório técnico das etapas do processo."""
        relatorio = """
        RELATÓRIO TÉCNICO - PROCESSOS DE RECICLAGEM ELETRÔNICA
        ========================================================

        1. COLETA E PRÉ-PROCESSO:
           - Implementação de logística reversa eficiente
           - Classificação inicial por tipo de equipamento
           - Sistemas de armazenamento seguro e ambientalmente correto

        2. DESMONTAGEM E TRIAGEM:
           - Processos manuais para componentes complexos
           - Automatização para maior eficiência
           - Tecnologias robóticas para precisão e segurança

        3. SEPARAÇÃO MECÂNICA:
           - Separação magnética para materiais ferromagnéticos
           - Separação por densidade para diferentes metais
           - Flotação seletiva para materiais específicos

        4. REFINO E PURIFICAÇÃO:
           - Hidrometalurgia para recuperação de metais preciosos
           - Pirometalurgia para processamento térmico
           - Bio-processamento utilizando microorganismos
        """
        return relatorio


# Exemplo de uso
def main():
    """Função principal para executar o gerador de fluxograma."""
    fluxograma = FluxogramaReciclagemEletronica()

    # Criar o fluxograma
    fig, ax = fluxograma.criar_fluxograma(
        salvar_arquivo=True,
        nome_arquivo='panorama_reciclagem_eletronica_v2.png'
    )

    # Gerar relatório técnico (opcional)
    relatorio = fluxograma.gerar_relatorio_tecnico()
    print(relatorio)


if __name__ == "__main__":
    main()