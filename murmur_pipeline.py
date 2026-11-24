"""Pipeline de inferência para a deteção de sopros cardíacos (CirCor DigiScope).

Reproduz exatamente o pré-processamento, as arquiteturas e a agregação por paciente
do notebook MLII_atualizado.ipynb, mas sem precisar de labels nem de ficheiros .tsv,
para poder ser aplicado a pacientes novos (por exemplo, o dataset escondido).

Uso na linha de comandos:
    # CSV simples (colunas patient_id,murmur)
    python murmur_pipeline.py --wav-dir <pasta_com_wavs> --out predictions.csv --model cnn

    # CSV no formato do template do bónus (colunas ID,Murmur; mesmas linhas e ordem do template)
    python murmur_pipeline.py --wav-dir CirCor_test_students/test_data \
        --template CirCor_test_students/PLX_GYY_test_results.csv \
        --out PL3_G06_test_results.csv --model cnn

Os modelos e as estatísticas de normalização são lidos de `artefactos/` (criados pela
secção 7.4 do notebook, ver `save_artifacts`).

Também contém as métricas (WAmurmur e matriz de confusão na orientação da Tabela 1 do
enunciado), usadas pelo notebook.
"""
import argparse
import glob
import json
import os
import re

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Parâmetros (iguais aos do notebook)
# --------------------------------------------------------------------------- #
SR = 4000                      # frequência de amostragem (Hz)
WINDOW_SEC = 5                 # duração de cada janela (s)
HOP_SEC = 2.5                  # passo entre janelas (s), 50 % de sobreposição
WIN = WINDOW_SEC * SR          # 20000 amostras
HOP = int(HOP_SEC * SR)        # 10000 amostras

LOW_HZ, HIGH_HZ = 25, 400      # filtro passa-banda
N_FFT, HOP_LENGTH = 512, 128
N_MELS, N_MFCC, FMAX = 64, 20, 800

CLASSES = ["Absent", "Present", "Unknown"]   # ordem usada nos modelos (id 0, 1, 2)
DEFAULT_ARTIFACTS = "artefactos"

# Ordem do enunciado (Tabela 1 e fórmula da WA)
ORDEM_ENUNCIADO = ["Present", "Unknown", "Absent"]
PESOS = {"Present": 5, "Unknown": 3, "Absent": 1}


# --------------------------------------------------------------------------- #
# Métricas
# --------------------------------------------------------------------------- #
def weighted_accuracy(y_true, y_pred):
    """WAmurmur: cada paciente Present conta 5, Unknown 3 e Absent 1 (igual à do notebook)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    acertos = sum(PESOS[t] for t, p in zip(y_true, y_pred) if t == p)
    total = sum(PESOS[t] for t in y_true)
    return acertos / total


def confusion_enunciado(y_true, y_pred):
    """Matriz de confusão na orientação da Tabela 1 do enunciado.

    Linhas = classe PREVISTA, colunas = classe REAL, ordem Present, Unknown, Absent.
    O elemento [i, j] é m_ij: nº de pacientes previstos como i cuja classe real é j.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    cm = pd.DataFrame(0, index=ORDEM_ENUNCIADO, columns=ORDEM_ENUNCIADO)
    for t, p in zip(y_true, y_pred):
        cm.loc[p, t] += 1
    cm.index.name, cm.columns.name = "previsto", "real"
    return cm


def weighted_accuracy_from_cm(cm):
    """Fórmula literal do enunciado, a partir da matriz da Tabela 1 (linhas = previsto,
    colunas = real, ordem Present, Unknown, Absent; DataFrame ou array 3x3)."""
    m = cm.values if hasattr(cm, "values") else np.asarray(cm)
    (mPP, mPU, mPA), (mUP, mUU, mUA), (mAP, mAU, mAA) = m
    num = 5 * mPP + 3 * mUU + mAA
    den = 5 * (mPP + mUP + mAP) + 3 * (mPU + mUU + mAU) + (mPA + mUA + mAA)
    return num / den


# --------------------------------------------------------------------------- #
# Pré-processamento
# --------------------------------------------------------------------------- #
def _sos():
    from scipy.signal import butter
    return butter(4, [LOW_HZ, HIGH_HZ], btype="bandpass", fs=SR, output="sos")


def load_clean(path, sos=None):
    """Carrega, filtra (passa-banda 25–400 Hz, filtfilt) e normaliza (z-score) uma gravação.

    Se a gravação for mais curta do que uma janela (5 s), faz padding com zeros no fim,
    para dar sempre pelo menos uma janela (as gravações do dataset completo vão desde 4,8 s).
    """
    import librosa
    from scipy.signal import sosfiltfilt
    sos = _sos() if sos is None else sos
    y, _ = librosa.load(path, sr=SR)
    if len(y) == 0:
        raise ValueError("gravação vazia")
    y = sosfiltfilt(sos, y)
    y = (y - y.mean()) / (y.std() + 1e-8)
    y = y.astype(np.float32)
    if len(y) < WIN:
        y = np.pad(y, (0, WIN - len(y)))
    return y


def make_windows(y):
    """Janelas de 5 s com passo de 2,5 s; devolve um array (n_janelas, 20000)."""
    return np.stack([y[s:s + WIN] for s in range(0, len(y) - WIN + 1, HOP)])


def window_features(windows):
    """Espectrograma mel (dB, ref=max) e MFCCs (não normalizados) de cada janela.

    Devolve mel (N, 1, 64, 157) e mfcc (N, 157, 20), pelo mesmo caminho da célula 130 do notebook.
    """
    import librosa
    mels, mfccs = [], []
    for w in windows:
        S = librosa.feature.melspectrogram(y=w, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH,
                                           n_mels=N_MELS, fmax=FMAX)
        mels.append(librosa.power_to_db(S, ref=np.max))
        mfccs.append(librosa.feature.mfcc(S=librosa.power_to_db(S), n_mfcc=N_MFCC).T)
    mel = np.stack(mels)[:, np.newaxis].astype(np.float32)
    return mel, np.stack(mfccs).astype(np.float32)


# --------------------------------------------------------------------------- #
# Modelos (cópia fiel das células 182 e 189 do notebook; nomes de camadas iguais)
# --------------------------------------------------------------------------- #
def _torch_nn():
    import torch.nn as nn
    return nn


def build_cnn(n_classes=3, filtros=(16, 32, 64)):
    nn = _torch_nn()

    class CNN(nn.Module):
        def __init__(self):
            super().__init__()

            def bloco(c_in, c_out):
                return nn.Sequential(
                    nn.Conv2d(c_in, c_out, kernel_size=3, padding=1),
                    nn.BatchNorm2d(c_out),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                )

            f1, f2, f3 = filtros
            self.features = nn.Sequential(bloco(1, f1), bloco(f1, f2), bloco(f2, f3))
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(f3, n_classes))

        def forward(self, x):
            x = self.features(x)
            x = self.pool(x).flatten(1)
            return self.classifier(x)

    return CNN()


def build_rnn(n_features=N_MFCC, hidden=64, n_layers=2, n_classes=3):
    nn = _torch_nn()

    class RNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.gru = nn.GRU(input_size=n_features, hidden_size=hidden, num_layers=n_layers,
                              batch_first=True, dropout=0.3 if n_layers > 1 else 0)
            self.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(hidden, n_classes))

        def forward(self, x):
            _, h = self.gru(x)
            return self.classifier(h[-1])

    return RNN()


# --------------------------------------------------------------------------- #
# Artefactos (modelos finais + estatísticas de normalização dos MFCCs)
# --------------------------------------------------------------------------- #
def save_artifacts(cnn, rnn, mu, sd, out_dir=DEFAULT_ARTIFACTS,
                   cnn_filtros=(16, 32, 64), rnn_hidden=32, rnn_layers=1):
    """Guarda os modelos finais (CNN exp. 5 e RNN R3) e `mu`/`sd` do treino."""
    import torch
    os.makedirs(out_dir, exist_ok=True)
    torch.save({k: v.cpu() for k, v in cnn.state_dict().items()}, os.path.join(out_dir, "cnn_exp5.pt"))
    torch.save({k: v.cpu() for k, v in rnn.state_dict().items()}, os.path.join(out_dir, "rnn_r3.pt"))
    np.savez(os.path.join(out_dir, "mfcc_norm.npz"), mu=np.asarray(mu), sd=np.asarray(sd))
    config = {"classes": CLASSES, "sr": SR, "window_sec": WINDOW_SEC, "hop_sec": HOP_SEC,
              "cnn_filtros": list(cnn_filtros), "rnn_hidden": rnn_hidden, "rnn_layers": rnn_layers}
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)


def load_artifacts(artifacts_dir=DEFAULT_ARTIFACTS):
    """Carrega CNN, RNN (em modo eval, em CPU) e `mu`/`sd`."""
    import torch
    with open(os.path.join(artifacts_dir, "config.json")) as f:
        cfg = json.load(f)
    if cfg["classes"] != CLASSES:
        raise ValueError(f"Ordem de classes dos artefactos {cfg['classes']} != {CLASSES}")
    cnn = build_cnn(filtros=tuple(cfg["cnn_filtros"]))
    cnn.load_state_dict(torch.load(os.path.join(artifacts_dir, "cnn_exp5.pt"), map_location="cpu"))
    rnn = build_rnn(hidden=cfg["rnn_hidden"], n_layers=cfg["rnn_layers"])
    rnn.load_state_dict(torch.load(os.path.join(artifacts_dir, "rnn_r3.pt"), map_location="cpu"))
    norm = np.load(os.path.join(artifacts_dir, "mfcc_norm.npz"))
    return cnn.eval(), rnn.eval(), norm["mu"], norm["sd"]


# --------------------------------------------------------------------------- #
# Inferência e agregação por paciente
# --------------------------------------------------------------------------- #
def aggregate_patient(rec_preds):
    """Regra hierárquica (a mesma do notebook): se alguma gravação for Present, o paciente é
    Present; senão, se alguma for Unknown, é Unknown; senão, Absent."""
    rec_preds = list(rec_preds)
    if "Present" in rec_preds:
        return "Present"
    if "Unknown" in rec_preds:
        return "Unknown"
    return "Absent"


def _patient_id(path):
    """O nome dos ficheiros é IDPACIENTE_VÁLVULA[_n].wav."""
    pid = os.path.basename(path).split("_")[0]
    return int(pid) if re.fullmatch(r"\d+", pid) else pid


def recording_probs(path, cnn, rnn, mu, sd, model="cnn", sos=None, batch_size=64):
    """Probabilidades (3,) de uma gravação: média das probabilidades (softmax) das suas janelas."""
    import torch
    mel, mfcc = window_features(make_windows(load_clean(path, sos)))
    probs = []
    with torch.no_grad():
        if model in ("cnn", "ensemble"):
            p_cnn = torch.cat([torch.softmax(cnn(torch.from_numpy(mel[i:i + batch_size])), 1)
                               for i in range(0, len(mel), batch_size)]).numpy()
            probs.append(p_cnn)
        if model in ("rnn", "ensemble"):
            mfcc_n = ((mfcc - mu) / (sd + 1e-8)).astype(np.float32)
            p_rnn = torch.cat([torch.softmax(rnn(torch.from_numpy(mfcc_n[i:i + batch_size])), 1)
                               for i in range(0, len(mfcc_n), batch_size)]).numpy()
            probs.append(p_rnn)
    return np.mean(probs, axis=0).mean(axis=0)      # janelas (média CNN/RNN) → gravação


def predict_patients_df(wav_dir, artifacts_dir=DEFAULT_ARTIFACTS, model="cnn",
                        only_patients=None, verbose=True):
    """Uma previsão por paciente para todos os .wav de `wav_dir` (não usa labels nem .tsv).

    Devolve um DataFrame com as colunas patient_id e murmur."""
    if model not in ("cnn", "rnn", "ensemble"):
        raise ValueError("model deve ser 'cnn', 'rnn' ou 'ensemble'")
    cnn, rnn, mu, sd = load_artifacts(artifacts_dir)
    sos = _sos()
    paths = sorted(glob.glob(os.path.join(wav_dir, "*.wav")))
    if only_patients is not None:
        only = set(only_patients)
        paths = [p for p in paths if _patient_id(p) in only]
    if not paths:
        raise FileNotFoundError(f"Sem ficheiros .wav em {wav_dir}")

    por_paciente = {}
    for i, path in enumerate(paths, 1):
        pid = _patient_id(path)
        por_paciente.setdefault(pid, [])
        try:
            p = recording_probs(path, cnn, rnn, mu, sd, model, sos)
            por_paciente[pid].append(CLASSES[int(np.argmax(p))])
        except Exception as e:                                  # gravação ilegível: ignorada
            print(f"AVISO: gravação ignorada ({os.path.basename(path)}): {e}")
        if verbose and i % 200 == 0:
            print(f"{i}/{len(paths)} gravações")

    linhas = []
    for pid, preds in por_paciente.items():
        if not preds:
            print(f"AVISO: paciente {pid} sem gravações válidas; previsto Absent.")
        linhas.append({"patient_id": pid, "murmur": aggregate_patient(preds)})
    return pd.DataFrame(linhas).sort_values("patient_id").reset_index(drop=True)


def predict_patients(wav_dir, out_csv, artifacts_dir=DEFAULT_ARTIFACTS, model="cnn", **kw):
    """Gera o CSV de submissão (patient_id,murmur), com uma linha por paciente."""
    df = predict_patients_df(wav_dir, artifacts_dir, model, **kw)
    df.to_csv(out_csv, index=False)
    print(f"{len(df)} pacientes → {out_csv}")
    print(df["murmur"].value_counts().to_string())
    return df


VALID_LABELS = ("Present", "Unknown", "Absent")


def fill_template(template_csv, preds, out_csv):
    """Preenche o template de submissão do bónus (colunas ID,Murmur) com as previsões.

    `preds` é o DataFrame de `predict_patients_df` (patient_id, murmur). O ficheiro de saída tem
    exatamente os IDs e a ordem do template, com as colunas ID,Murmur. Dá erro se faltar uma
    previsão para algum ID do template, se sobrar alguma, se houver IDs repetidos ou se algum
    valor não for Present, Unknown ou Absent.
    """
    tpl = pd.read_csv(template_csv, dtype={"ID": str})
    if list(tpl.columns) != ["ID", "Murmur"]:
        raise ValueError(f"O template devia ter as colunas ID,Murmur e tem {list(tpl.columns)}")
    if tpl["ID"].duplicated().any():
        raise ValueError("O template tem IDs repetidos")
    if preds["patient_id"].astype(str).duplicated().any():
        raise ValueError("As previsões têm pacientes repetidos")
    mapa = dict(zip(preds["patient_id"].astype(str), preds["murmur"]))
    em_falta = [i for i in tpl["ID"] if i not in mapa]
    a_mais = sorted(set(mapa) - set(tpl["ID"]))
    if em_falta:
        raise ValueError(f"{len(em_falta)} IDs do template sem previsão: {em_falta[:10]}")
    if a_mais:
        raise ValueError(f"{len(a_mais)} pacientes previstos que não estão no template: {a_mais[:10]}")
    out = pd.DataFrame({"ID": tpl["ID"], "Murmur": [mapa[i] for i in tpl["ID"]]})
    invalidos = out[~out["Murmur"].isin(VALID_LABELS)]
    if len(invalidos):
        raise ValueError(f"Valores inválidos em Murmur: {invalidos.head().to_dict('records')}")
    out.to_csv(out_csv, index=False)
    return out


def main():
    ap = argparse.ArgumentParser(description="Previsão de sopros cardíacos por paciente.")
    ap.add_argument("--wav-dir", required=True, help="pasta com os ficheiros IDPACIENTE_VÁLVULA.wav")
    ap.add_argument("--out", default="predictions.csv",
                    help="CSV de saída (patient_id,murmur; ou ID,Murmur se for dado --template)")
    ap.add_argument("--template", default=None,
                    help="CSV template do bónus (ID,Murmur): se dado, a saída segue o template "
                         "(mesmos IDs e ordem)")
    ap.add_argument("--artifacts", default=DEFAULT_ARTIFACTS, help="pasta com os modelos guardados")
    ap.add_argument("--model", default="cnn", choices=["cnn", "rnn", "ensemble"])
    args = ap.parse_args()
    if args.template is None:
        predict_patients(args.wav_dir, args.out, args.artifacts, args.model)
    else:
        df = predict_patients_df(args.wav_dir, args.artifacts, args.model)
        out = fill_template(args.template, df, args.out)
        print(f"{len(out)} pacientes → {args.out}")
        print(out["Murmur"].value_counts().to_string())


if __name__ == "__main__":
    main()
