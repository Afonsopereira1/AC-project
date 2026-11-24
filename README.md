# Deteção de sopros cardíacos (CirCor DigiScope) — CC3043 Machine Learning II

Classificação de sopros (Present / Absent / Unknown) a partir de fonocardiogramas, com uma previsão
por paciente. Dois classificadores: **CNN 2D** (espectrograma mel) e **RNN** (GRU sobre MFCCs).

## Conteúdo

| Ficheiro | Descrição |
|---|---|
| `MLII_atualizado.ipynb` | Notebook principal: análise exploratória, split, pré-processamento, modelos, experiências, avaliação final, guardar modelos e gerar previsões. |
| `murmur_pipeline.py` | Inferência por paciente sem labels (usada para o CSV do dataset escondido) e métricas (WAmurmur, matriz de confusão da Tabela 1). |
| `artefactos/` | Modelos finais e estatísticas de normalização (criados pela secção 7.4 do notebook). |
| `requirements.txt` | Dependências. |

O dataset **não** está incluído.

## Preparação

1. Python 3.11 (ou superior) e as dependências:
   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. Descarregar a parte pública do CirCor DigiScope (https://physionet.org/content/circor-heart-sound/1.0.3/)
   e colocá-la em `circor/`, ao lado do notebook:
   - `circor/training_data.csv`
   - `circor/training_data/` (ficheiros `.wav` e `.tsv`)

## Reproduzir os resultados

1. Abrir `MLII_atualizado.ipynb` e correr todas as células por ordem (Restart + Run All).
2. Na primeira execução são criadas caches na pasta do notebook (`signals.pkl`, `mel.npy`, `mfcc.npy`,
   cerca de 2,5 GB; também `rec.csv` e `windows.csv`). São reutilizadas nas execuções seguintes;
   se o pré-processamento ou as features mudarem, têm de ser apagadas.
3. Em CPU, a execução completa demora cerca de 3 horas, quase todas nos treinos. As sementes estão
   fixas (`SEED = 42`). A RNN reproduz exatamente os mesmos resultados; o treino da CNN, porém, pode variar
   com a versão das bibliotecas e o hardware (numa execução anterior do mesmo código, noutro ambiente, a
   WAmurmur da CNN no teste foi 0,793 em vez de 0,758). A secção 7.6 do notebook regista as versões usadas.
4. A secção 7.4 guarda os modelos finais em `artefactos/`.

### Protocolo de avaliação
Split por paciente (70 / 15 / 15 %), estratificado por `Murmur`. As crianças com dois registos
(`Additional ID`) ficam sempre no mesmo conjunto. Todas as decisões (hiperparâmetros, época, limiar,
arquitetura) usam só a validação; o teste é avaliado uma única vez, no fim. A métrica principal é a
WAmurmur por paciente, com a matriz de confusão por paciente.

### Resultados no teste (145 pacientes)

| Modelo | WAmurmur |
|---|---:|
| CNN (exp. 5) | 0,758 |
| RNN (R3) | 0,775 |

(Baseline "tudo Absent": 0,365. Intervalos de confiança a 95 % por bootstrap: CNN 0,67–0,85; RNN 0,68–0,86.
Diferença RNN − CNN: +0,019, IC 95 % de −0,037 a 0,079, não significativa.)

O modelo usado para o dataset escondido é a CNN: foi escolhida na validação (WAmurmur 0,767 contra 0,764 da RNN)
antes de se ver o teste, e a escolha não foi alterada depois.

## Gerar o CSV de previsões (dataset escondido)

Com os modelos já guardados em `artefactos/` (secção 7.4 do notebook), numa pasta com os ficheiros
`IDPACIENTE_VÁLVULA.wav` (sem labels nem `.tsv`):

```bash
python murmur_pipeline.py --wav-dir <pasta_com_wavs> --out predictions.csv --model cnn
```

O resultado tem as colunas `patient_id,murmur` e uma linha por paciente. A previsão por paciente
é obtida em dois passos: média das probabilidades das janelas de 5 s de cada gravação e, depois, regra
hierárquica sobre as gravações do paciente (Present se alguma for Present; senão Unknown se alguma for
Unknown; senão Absent). O mesmo pode ser feito na secção 7.5 do notebook. `--model` aceita `cnn`
(por defeito), `rnn` e `ensemble`.
