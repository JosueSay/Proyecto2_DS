import os, json, yaml, argparse, logging
import torch, numpy as np, pandas as pd
from transformers import AutoTokenizer, AutoConfig

try:
    from .model import CrossEncoder
except ImportError:
    from model import CrossEncoder

# Preferimos paralelismo del tokenizer en local (desactívalo si satura CPU)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
useSlowTokenizer = os.getenv("USE_SLOW_TOKENIZER", "0") == "1"


def resolveBackboneName(modelDir: str) -> str:
    """
    Resuelve el backbone usado en train:
    1) train_config.yaml -> model_name
    2) config.json (guardado del backbone) -> _name_or_path
    3) AutoConfig.from_pretrained(modelDir) -> _name_or_path
    4) Fallback razonable
    """
    yamlPath = os.path.join(modelDir, "train_config.yaml")
    if os.path.isfile(yamlPath):
        try:
            yamlCfg = yaml.safe_load(open(yamlPath))
            if isinstance(yamlCfg, dict) and yamlCfg.get("model_name"):
                return yamlCfg["model_name"]
        except Exception:
            pass

    jsonPath = os.path.join(modelDir, "config.json")
    if os.path.isfile(jsonPath):
        try:
            jsonCfg = json.load(open(jsonPath))
            if isinstance(jsonCfg, dict) and jsonCfg.get("_name_or_path"):
                return jsonCfg["_name_or_path"]
        except Exception:
            pass

    try:
        autoCfg = AutoConfig.from_pretrained(modelDir)
        name = getattr(autoCfg, "_name_or_path", None)
        if name:
            return name
    except Exception:
        pass

    return "microsoft/deberta-v3-base"


class PairRanker:
    def __init__(self, modelDir, maxLenPrompt=256, maxLenResp=700):
        logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

        self.deviceType = "cuda" if torch.cuda.is_available() else "cpu"
        if self.deviceType == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            logging.info("GPU: %s | CUDA %s", torch.cuda.get_device_name(0), torch.version.cuda)
        else:
            logging.info("CPU")

        # Tokenizer (usa el guardado si existe)
        tokDir = os.path.join(modelDir, "tokenizer")
        tokSrc = tokDir if os.path.isdir(tokDir) else modelDir
        self.tokenizer = AutoTokenizer.from_pretrained(tokSrc, use_fast=not useSlowTokenizer)

        # Modelo: inicializa backbone correcto y carga pesos finos
        backboneName = resolveBackboneName(modelDir)
        self.model = CrossEncoder(backboneName).to(self.deviceType).eval()

        statePath = os.path.join(modelDir, "model.bin")
        if not os.path.isfile(statePath):
            raise FileNotFoundError(f"No se encontró {statePath}. Asegúrate de que el entrenamiento guardó model.bin.")
        stateDict = torch.load(statePath, map_location=self.deviceType)
        self.model.load_state_dict(stateDict, strict=True)

        # Longitudes
        tokMax = getattr(self.tokenizer, "model_max_length", 512) or 512
        self.seqMax = min(int(maxLenPrompt) + int(maxLenResp) + 3, tokMax if tokMax > 0 else 512)
        logging.info("Inferencia seqMax=%d | backbone=%s", self.seqMax, backboneName)

        # AMP en inferencia
        self.useAmp = (self.deviceType == "cuda")
        self.autocastKwargs = dict(device_type="cuda", dtype=torch.bfloat16) if self.useAmp else dict(enabled=False)

    @torch.inference_mode()
    def predictOne(self, promptText, responseA, responseB):
        def encodePair(responseText):
            enc = self.tokenizer.encode_plus(
                str(promptText), str(responseText),
                truncation=True,
                max_length=self.seqMax,
                padding="max_length",
                return_tensors="pt"
            )
            return {k: v.to(self.deviceType, non_blocking=True) for k, v in enc.items()}

        encA, encB = encodePair(responseA), encodePair(responseB)

        if self.useAmp:
            with torch.amp.autocast(**self.autocastKwargs):
                scoreA, scoreB = self.model(encA, encB)
        else:
            scoreA, scoreB = self.model(encA, encB)

        # Prob A vs B + empate heurístico
        delta = (scoreA - scoreB).item()
        probA = torch.sigmoid(torch.tensor(delta)).item()
        probB = 1.0 - probA
        probTie = float(np.clip(0.5 - abs(delta) / 6.0, 0.0, 0.5))
        norm = probA + probB + probTie
        return probA / norm, probB / norm, probTie / norm

    def predictCsv(self, testCsvPath, outCsvPath):
        dataFrame = pd.read_csv(testCsvPath)
        rows = []
        for _, row in dataFrame.iterrows():
            probA, probB, probTie = self.predictOne(row["prompt"], row["response_a"], row["response_b"])
            rows.append([row["id"], probA, probB, probTie])
        submission = pd.DataFrame(rows, columns=["id", "winner_model_a", "winner_model_b", "winner_tie"])
        submission.to_csv(outCsvPath, index=False)
        logging.info("✅ Guardado predicciones: %s", outCsvPath)
        return outCsvPath

    @staticmethod
    def computeMetrics(mergedFrame: pd.DataFrame) -> pd.DataFrame:
        labelCols = ["winner_model_a_y", "winner_model_b_y", "winner_tie_y"]
        probCols = ["winner_model_a_p", "winner_model_b_p", "winner_tie_p"]

        yTrue = mergedFrame[labelCols].values.argmax(1)
        yPred = mergedFrame[probCols].values.argmax(1)

        classNames = ["A", "B", "TIE"]
        metricRows = []

        accuracy = (yTrue == yPred).mean()
        metricRows.append(dict(metric="accuracy", value=float(accuracy)))

        for classIndex, className in enumerate(classNames):
            tp = int(((yPred == classIndex) & (yTrue == classIndex)).sum())
            fp = int(((yPred == classIndex) & (yTrue != classIndex)).sum())
            fn = int(((yPred != classIndex) & (yTrue == classIndex)).sum())
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            metricRows += [
                dict(metric=f"precision_{className}", value=float(precision)),
                dict(metric=f"recall_{className}", value=float(recall)),
                dict(metric=f"f1_{className}", value=float(f1)),
            ]

        macroF1 = float(np.mean([m["value"] for m in metricRows if m["metric"].startswith("f1_")]))
        metricRows.append(dict(metric="macro_f1", value=macroF1))
        return pd.DataFrame(metricRows)

    def maybeMetrics(self, labeledCsvPath, predsCsvPath, metricsOutPath=None):
        labelFrame = pd.read_csv(labeledCsvPath)
        neededCols = {"id", "winner_model_a", "winner_model_b", "winner_tie"}
        if not neededCols.issubset(set(labelFrame.columns)):
            logging.info("El CSV no tiene columnas de etiqueta winner_*. Métricas omitidas.")
            return None

        predsFrame = pd.read_csv(predsCsvPath)
        merged = labelFrame.merge(
            predsFrame[["id", "winner_model_a", "winner_model_b", "winner_tie"]],
            on="id", suffixes=("_y", "_p")
        )
        metricsFrame = self.computeMetrics(merged)
        if metricsOutPath:
            metricsFrame.to_csv(metricsOutPath, index=False)
            logging.info("✅ Guardado métricas: %s", metricsOutPath)
        else:
            logging.info("Métricas (no guardadas, usa --metrics_out para CSV):\n%s", metricsFrame.to_string(index=False))
        return metricsFrame

    @classmethod
    def load(cls, modelDir, maxLenPrompt=256, maxLenResp=700):
        return cls(modelDir, maxLenPrompt, maxLenResp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True, help="Directorio con model.bin, tokenizer/, train_config.yaml")
    parser.add_argument("--test", required=True, help="CSV con columnas: id,prompt,response_a,response_b (+ opcional winner_*)")
    parser.add_argument("--out", required=True, help="CSV de salida con probabilidades")
    parser.add_argument("--metrics_out", default=None, help="(Opcional) CSV para guardar métricas si test tiene winner_*")
    parser.add_argument("--max_len_prompt", type=int, default=256)
    parser.add_argument("--max_len_resp", type=int, default=700)
    args = parser.parse_args()

    ranker = PairRanker.load(args["model_dir"] if isinstance(args, dict) else args.model_dir,
                             args["max_len_prompt"] if isinstance(args, dict) else args.max_len_prompt,
                             args["max_len_resp"] if isinstance(args, dict) else args.max_len_resp)
    outPath = ranker.predictCsv(args["test"] if isinstance(args, dict) else args.test,
                                args["out"] if isinstance(args, dict) else args.out)

    ranker.maybeMetrics(args["test"] if isinstance(args, dict) else args.test,
                        outPath,
                        args["metrics_out"] if isinstance(args, dict) else args.metrics_out)
    print(f"Saved to {outPath}")
