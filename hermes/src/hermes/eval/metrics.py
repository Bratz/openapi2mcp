"""Multi-label metrics — the paper's suite (docs/02-TEST-PLAN §3, 05-PAPER-FACTS §1).

Inputs are {operation_id: set(smell_ids)} for predictions and gold labels over
the same operation ids. LABELS is the fixed 9-smell space.
"""

from __future__ import annotations

from dataclasses import dataclass

LABELS = (
    "LAZY", "BLOATED", "TANGLED", "FRAGMENTED", "EXCESS_STRUCTURED",
    "PATH_AND_METHOD", "INPUT", "RESPONSE", "SECURITY",
)


@dataclass
class SmellPR:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 1.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 1.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0


@dataclass
class EvalMetrics:
    jaccard: float
    f1_micro: float
    f1_macro: float
    hamming_loss: float
    cardinality_diff: float
    per_smell: dict[str, SmellPR]
    clean_ops_with_predictions: int
    n_operations: int

    def as_dict(self) -> dict:
        return {
            "jaccard": round(self.jaccard, 4),
            "f1_micro": round(self.f1_micro, 4),
            "f1_macro": round(self.f1_macro, 4),
            "hamming_loss": round(self.hamming_loss, 4),
            "cardinality_diff": round(self.cardinality_diff, 4),
            "clean_ops_with_predictions": self.clean_ops_with_predictions,
            "n_operations": self.n_operations,
            "per_smell": {
                s: {"tp": pr.tp, "fp": pr.fp, "fn": pr.fn,
                    "precision": round(pr.precision, 4), "recall": round(pr.recall, 4),
                    "f1": round(pr.f1, 4)}
                for s, pr in self.per_smell.items()
            },
        }


def compute_metrics(predicted: dict[str, set[str]], gold: dict[str, set[str]]) -> EvalMetrics:
    if set(predicted) != set(gold):
        missing = set(gold) ^ set(predicted)
        raise ValueError(f"prediction/gold operation ids differ: {sorted(missing)[:5]}")
    known = set(LABELS)
    for name, sets in (("gold", gold), ("predicted", predicted)):
        unknown = {label for labels in sets.values() for label in labels} - known
        if unknown:
            raise ValueError(f"unknown smell labels in {name}: {sorted(unknown)}")

    ops = sorted(gold)
    jaccard_sum = 0.0
    hamming_wrong = 0
    cardinality_sum = 0
    per_smell = {label: SmellPR() for label in LABELS}
    clean_with_preds = 0

    for op in ops:
        p, g = predicted[op], gold[op]
        union = p | g
        jaccard_sum += (len(p & g) / len(union)) if union else 1.0  # both-empty = 1.0
        hamming_wrong += len(p ^ g)
        cardinality_sum += len(p) - len(g)
        if not g and p:
            clean_with_preds += 1
        for label in LABELS:
            in_p, in_g = label in p, label in g
            if in_p and in_g:
                per_smell[label].tp += 1
            elif in_p:
                per_smell[label].fp += 1
            elif in_g:
                per_smell[label].fn += 1

    n = len(ops)
    tp = sum(pr.tp for pr in per_smell.values())
    fp = sum(pr.fp for pr in per_smell.values())
    fn = sum(pr.fn for pr in per_smell.values())
    f1_micro = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 1.0
    # macro over labels that occur in gold OR predictions (a label absent from
    # both everywhere scores a perfect-but-vacuous 1.0 and would inflate macro).
    active = [label for label in LABELS
              if per_smell[label].tp + per_smell[label].fp + per_smell[label].fn > 0]
    f1_macro = sum(per_smell[label].f1 for label in active) / len(active) if active else 1.0

    return EvalMetrics(
        jaccard=jaccard_sum / n if n else 1.0,
        f1_micro=f1_micro,
        f1_macro=f1_macro,
        hamming_loss=hamming_wrong / (n * len(LABELS)) if n else 0.0,
        cardinality_diff=cardinality_sum / n if n else 0.0,
        per_smell=per_smell,
        clean_ops_with_predictions=clean_with_preds,
        n_operations=n,
    )


# Gates (docs/02-TEST-PLAN §3) and the paper's best-model reference row.
GATES = {"jaccard": 0.75, "f1_micro": 0.85, "hamming_loss": 0.12, "clean_ops_max": 2}
PAPER_REFERENCE = {"model": "gpt-oss:120b", "jaccard": 0.85, "f1_micro": 0.92,
                   "f1_macro": 0.73, "hamming_loss": 0.07, "cardinality_diff": -0.53}


def gate_failures(metrics: EvalMetrics) -> list[str]:
    failures = []
    if metrics.jaccard < GATES["jaccard"]:
        failures.append(f"jaccard {metrics.jaccard:.3f} < {GATES['jaccard']}")
    if metrics.f1_micro < GATES["f1_micro"]:
        failures.append(f"f1_micro {metrics.f1_micro:.3f} < {GATES['f1_micro']}")
    if metrics.hamming_loss > GATES["hamming_loss"]:
        failures.append(f"hamming_loss {metrics.hamming_loss:.3f} > {GATES['hamming_loss']}")
    if metrics.clean_ops_with_predictions > GATES["clean_ops_max"]:
        failures.append(
            f"{metrics.clean_ops_with_predictions} clean operations got predictions (max {GATES['clean_ops_max']})"
        )
    return failures
