"""Proof-of-concept concept learning simulator for CS6795.

This module implements a small, interpretable prototype that compares:
1. Top-down rule learning from a written concept definition
2. Bottom-up incremental feature learning from labeled examples
3. Case-based reasoning over stored examples
4. A hybrid learner that combines rule priors with experience

The proof-of-concept uses a structured but cognitively plausible toy domain:
the ambiguous concept "soup". Examples are represented with binary features
that a human learner could plausibly attend to (e.g., savory, spoonable,
drink-like, meal-like).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from collections import Counter, defaultdict
from typing import Dict, List, Sequence, Tuple
import json
import math


FEATURES: List[str] = [
    "liquid_base",
    "savory",
    "sweet",
    "spoonable",
    "drink_like",
    "meal_like",
    "contains_solids",
    "served_hot",
    "cold_ok",
    "broth_based",
]


@dataclass(frozen=True)
class Example:
    """Single labeled concept example."""

    name: str
    features: Dict[str, int]
    label: int  # 1=in concept, 0=not in concept


@dataclass(frozen=True)
class HybridConfig:
    """Configuration that controls how the hybrid learner mixes evidence."""

    early_rule_weight: float = 0.50
    early_feature_weight: float = 0.20
    late_rule_weight: float = 0.35
    late_feature_weight: float = 0.40
    stage_switch_examples: int = 3
    threshold: float = 0.58
    negative_exception_penalty: float = 0.10
    positive_exception_bonus: float = 0.05
    similarity_margin: float = 0.15


DATASET: List[Example] = [
    Example("tomato_soup", {"liquid_base": 1, "savory": 1, "sweet": 0, "spoonable": 1, "drink_like": 0, "meal_like": 1, "contains_solids": 0, "served_hot": 1, "cold_ok": 0, "broth_based": 1}, 1),
    Example("chicken_noodle_soup", {"liquid_base": 1, "savory": 1, "sweet": 0, "spoonable": 1, "drink_like": 0, "meal_like": 1, "contains_solids": 1, "served_hot": 1, "cold_ok": 0, "broth_based": 1}, 1),
    Example("miso_soup", {"liquid_base": 1, "savory": 1, "sweet": 0, "spoonable": 1, "drink_like": 0, "meal_like": 1, "contains_solids": 1, "served_hot": 1, "cold_ok": 0, "broth_based": 1}, 1),
    Example("gazpacho", {"liquid_base": 1, "savory": 1, "sweet": 0, "spoonable": 1, "drink_like": 0, "meal_like": 1, "contains_solids": 1, "served_hot": 0, "cold_ok": 1, "broth_based": 0}, 1),
    Example("lentil_stew", {"liquid_base": 0, "savory": 1, "sweet": 0, "spoonable": 1, "drink_like": 0, "meal_like": 1, "contains_solids": 1, "served_hot": 1, "cold_ok": 0, "broth_based": 0}, 0),
    Example("ramen_broth", {"liquid_base": 1, "savory": 1, "sweet": 0, "spoonable": 1, "drink_like": 0, "meal_like": 1, "contains_solids": 1, "served_hot": 1, "cold_ok": 0, "broth_based": 1}, 1),
    Example("smoothie", {"liquid_base": 1, "savory": 0, "sweet": 1, "spoonable": 0, "drink_like": 1, "meal_like": 0, "contains_solids": 0, "served_hot": 0, "cold_ok": 1, "broth_based": 0}, 0),
    Example("hot_chocolate", {"liquid_base": 1, "savory": 0, "sweet": 1, "spoonable": 0, "drink_like": 1, "meal_like": 0, "contains_solids": 0, "served_hot": 1, "cold_ok": 0, "broth_based": 0}, 0),
    Example("cereal_with_milk", {"liquid_base": 1, "savory": 0, "sweet": 1, "spoonable": 1, "drink_like": 0, "meal_like": 1, "contains_solids": 1, "served_hot": 0, "cold_ok": 1, "broth_based": 0}, 0),
    Example("oatmeal", {"liquid_base": 0, "savory": 0, "sweet": 1, "spoonable": 1, "drink_like": 0, "meal_like": 1, "contains_solids": 1, "served_hot": 1, "cold_ok": 0, "broth_based": 0}, 0),
    Example("vegetable_broth", {"liquid_base": 1, "savory": 1, "sweet": 0, "spoonable": 1, "drink_like": 1, "meal_like": 0, "contains_solids": 0, "served_hot": 1, "cold_ok": 0, "broth_based": 1}, 0),
    Example("clam_chowder", {"liquid_base": 1, "savory": 1, "sweet": 0, "spoonable": 1, "drink_like": 0, "meal_like": 1, "contains_solids": 1, "served_hot": 1, "cold_ok": 0, "broth_based": 0}, 1),
    Example("fruit_soup", {"liquid_base": 1, "savory": 0, "sweet": 1, "spoonable": 1, "drink_like": 0, "meal_like": 0, "contains_solids": 1, "served_hot": 0, "cold_ok": 1, "broth_based": 0}, 0),
    Example("gravy", {"liquid_base": 1, "savory": 1, "sweet": 0, "spoonable": 0, "drink_like": 0, "meal_like": 0, "contains_solids": 0, "served_hot": 1, "cold_ok": 0, "broth_based": 0}, 0),
    Example("split_pea_soup", {"liquid_base": 1, "savory": 1, "sweet": 0, "spoonable": 1, "drink_like": 0, "meal_like": 1, "contains_solids": 1, "served_hot": 1, "cold_ok": 0, "broth_based": 0}, 1),
    Example("iced_coffee", {"liquid_base": 1, "savory": 0, "sweet": 0, "spoonable": 0, "drink_like": 1, "meal_like": 0, "contains_solids": 0, "served_hot": 0, "cold_ok": 1, "broth_based": 0}, 0),
]

TRAIN_NAMES = {
    "tomato_soup",
    "chicken_noodle_soup",
    "miso_soup",
    "gazpacho",
    "smoothie",
    "hot_chocolate",
    "cereal_with_milk",
    "vegetable_broth",
}
TRAIN_SET: List[Example] = [ex for ex in DATASET if ex.name in TRAIN_NAMES]
TEST_SET: List[Example] = [ex for ex in DATASET if ex.name not in TRAIN_NAMES]

SEQUENCES: Dict[str, List[str]] = {
    "canonical_first": [
        "tomato_soup",
        "chicken_noodle_soup",
        "miso_soup",
        "smoothie",
        "hot_chocolate",
        "cereal_with_milk",
        "vegetable_broth",
        "gazpacho",
    ],
    "boundary_first": [
        "gazpacho",
        "vegetable_broth",
        "cereal_with_milk",
        "tomato_soup",
        "smoothie",
        "miso_soup",
        "hot_chocolate",
        "chicken_noodle_soup",
    ],
    "negatives_first": [
        "smoothie",
        "hot_chocolate",
        "cereal_with_milk",
        "vegetable_broth",
        "tomato_soup",
        "chicken_noodle_soup",
        "miso_soup",
        "gazpacho",
    ],
}

DEFAULT_DEFINITION = (
    "Soup is primarily a savory liquid dish, usually spoonable and often served "
    "as part of a meal. It is not mainly sweet or a beverage, though some soups "
    "can be served cold."
)


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def hamming_similarity(a: Dict[str, int], b: Dict[str, int]) -> float:
    return sum(1 for feature in FEATURES if a[feature] == b[feature]) / float(len(FEATURES))


class TopDownRuleLearner:
    """Extracts a lightweight rule representation from a written definition."""

    def __init__(self, definition: str | Dict[str, float]) -> None:
        if isinstance(definition, dict):
            self.rule_weights = definition
            self.definition = json.dumps(definition, indent=2)
        else:
            self.definition = definition
            self.rule_weights = self.parse_definition(definition)

    @staticmethod
    def parse_definition(text: str) -> Dict[str, float]:
        text = text.lower()
        weights: Dict[str, float] = defaultdict(float)

        positive_map = {
            "liquid": "liquid_base",
            "savory": "savory",
            "spoon": "spoonable",
            "meal": "meal_like",
            "broth": "broth_based",
            "solid": "contains_solids",
            "hot": "served_hot",
            "cold": "cold_ok",
        }
        negative_map = {
            "sweet": "sweet",
            "beverage": "drink_like",
            "drink": "drink_like",
        }

        for keyword, feature in positive_map.items():
            if keyword in text:
                weights[feature] += 1.0
        for keyword, feature in negative_map.items():
            if keyword in text:
                weights[feature] -= 1.0

        # These cues are useful but should behave more like preferences than hard constraints.
        for feature in ("served_hot", "meal_like", "broth_based"):
            if feature in weights:
                weights[feature] *= 0.5

        return dict(weights)

    def score(self, example_features: Dict[str, int]) -> float:
        raw = 0.0
        total_weight = 0.0
        for feature, weight in self.rule_weights.items():
            total_weight += abs(weight)
            raw += weight * (1 if example_features[feature] == 1 else -1)
        if total_weight == 0:
            return 0.5
        return (raw / total_weight + 1.0) / 2.0

    def predict(self, example_features: Dict[str, int], threshold: float = 0.5) -> int:
        return int(self.score(example_features) >= threshold)


class IncrementalFeatureLearner:
    """Bottom-up learner that updates a contrastive feature representation."""

    def __init__(self, features: Sequence[str]) -> None:
        self.features = list(features)
        self.examples_seen: List[Example] = []
        self.positive_counts: Counter[str] = Counter()
        self.negative_counts: Counter[str] = Counter()
        self.num_positive = 0
        self.num_negative = 0
        self.history: List[Dict[str, object]] = []

    def update(self, example: Example) -> None:
        self.examples_seen.append(example)
        if example.label == 1:
            self.num_positive += 1
            for feature, value in example.features.items():
                self.positive_counts[feature] += value
        else:
            self.num_negative += 1
            for feature, value in example.features.items():
                self.negative_counts[feature] += value
        self.history.append(self.current_representation())

    def current_representation(self) -> Dict[str, object]:
        common_positive: List[str] = []
        forbidden: List[str] = []
        contrast: Dict[str, float] = {}

        for feature in self.features:
            pos_rate = self.positive_counts[feature] / self.num_positive if self.num_positive else 0.5
            neg_rate = self.negative_counts[feature] / self.num_negative if self.num_negative else 0.5
            contrast[feature] = round(pos_rate - neg_rate, 3)

            if self.num_positive and pos_rate >= 0.75 and neg_rate <= 0.50:
                common_positive.append(feature)
            if self.num_negative and neg_rate >= 0.75 and pos_rate <= 0.50:
                forbidden.append(feature)

        return {
            "examples_seen": len(self.examples_seen),
            "common_positive": common_positive,
            "forbidden": forbidden,
            "contrast": contrast,
        }

    def score(self, example_features: Dict[str, int]) -> float:
        raw = 0.0
        for feature in self.features:
            positive_rate = (self.positive_counts[feature] + 1) / (self.num_positive + 2)
            negative_rate = (self.negative_counts[feature] + 1) / (self.num_negative + 2)
            feature_weight = math.log(positive_rate / negative_rate)
            feature_value = 1 if example_features[feature] == 1 else -1
            raw += feature_weight * feature_value

        bias = math.log((self.num_positive + 1) / (self.num_negative + 1))
        raw += 0.3 * bias
        return sigmoid(raw / 3.0)

    def predict(self, example_features: Dict[str, int], threshold: float = 0.5) -> int:
        return int(self.score(example_features) >= threshold)


class CaseBasedReasoner:
    """Memory-based learner using similarity-weighted nearest neighbours."""

    def __init__(self, k: int = 3) -> None:
        self.k = k
        self.memory: List[Example] = []

    def update(self, example: Example) -> None:
        self.memory.append(example)

    def score(self, example_features: Dict[str, int]) -> float:
        if not self.memory:
            return 0.5

        scored = [
            (hamming_similarity(example_features, mem.features), mem.label)
            for mem in self.memory
        ]
        scored.sort(key=lambda item: item[0], reverse=True)
        neighbours = scored[: self.k]
        numerator = sum(similarity * label for similarity, label in neighbours)
        denominator = sum(similarity for similarity, _ in neighbours) or 1.0
        return numerator / denominator

    def predict(self, example_features: Dict[str, int], threshold: float = 0.5) -> int:
        return int(self.score(example_features) >= threshold)

    def nearest_positive_similarity(self, example_features: Dict[str, int]) -> float:
        return max(
            (hamming_similarity(example_features, mem.features) for mem in self.memory if mem.label == 1),
            default=0.0,
        )

    def nearest_negative_similarity(self, example_features: Dict[str, int]) -> float:
        return max(
            (hamming_similarity(example_features, mem.features) for mem in self.memory if mem.label == 0),
            default=0.0,
        )


class HybridLearner:
    """Combines rule priors with feature abstraction and exemplar memory."""

    def __init__(
        self,
        rule_learner: TopDownRuleLearner,
        feature_learner: IncrementalFeatureLearner,
        case_reasoner: CaseBasedReasoner,
        config: HybridConfig | None = None,
    ) -> None:
        self.rule_learner = rule_learner
        self.feature_learner = feature_learner
        self.case_reasoner = case_reasoner
        self.config = config or HybridConfig()

    def component_weights(self) -> Tuple[float, float, float]:
        memory_size = len(self.case_reasoner.memory)
        if memory_size < self.config.stage_switch_examples:
            rule_weight = self.config.early_rule_weight
            feature_weight = self.config.early_feature_weight
        else:
            rule_weight = self.config.late_rule_weight
            feature_weight = self.config.late_feature_weight
        case_weight = 1.0 - rule_weight - feature_weight
        return rule_weight, feature_weight, case_weight

    def explain_score(self, example_features: Dict[str, int]) -> Dict[str, float]:
        rule_weight, feature_weight, case_weight = self.component_weights()
        rule_score = self.rule_learner.score(example_features)
        feature_score = self.feature_learner.score(example_features)
        case_score = self.case_reasoner.score(example_features)
        combined = (
            rule_weight * rule_score
            + feature_weight * feature_score
            + case_weight * case_score
        )

        nearest_positive = self.case_reasoner.nearest_positive_similarity(example_features)
        nearest_negative = self.case_reasoner.nearest_negative_similarity(example_features)
        exception_adjustment = 0.0
        if nearest_negative - nearest_positive > self.config.similarity_margin:
            exception_adjustment -= self.config.negative_exception_penalty
        elif nearest_positive - nearest_negative > self.config.similarity_margin:
            exception_adjustment += self.config.positive_exception_bonus

        final_score = min(1.0, max(0.0, combined + exception_adjustment))
        return {
            "rule_weight": round(rule_weight, 3),
            "feature_weight": round(feature_weight, 3),
            "case_weight": round(case_weight, 3),
            "rule_score": round(rule_score, 3),
            "feature_score": round(feature_score, 3),
            "case_score": round(case_score, 3),
            "combined_score": round(combined, 3),
            "nearest_positive": round(nearest_positive, 3),
            "nearest_negative": round(nearest_negative, 3),
            "exception_adjustment": round(exception_adjustment, 3),
            "final_score": round(final_score, 3),
        }

    def score(self, example_features: Dict[str, int]) -> float:
        return self.explain_score(example_features)["final_score"]

    def predict(self, example_features: Dict[str, int]) -> int:
        return int(self.score(example_features) >= self.config.threshold)


def evaluate_model(model: object, dataset: Sequence[Example]) -> Dict[str, object]:
    gold = [ex.label for ex in dataset]
    preds = [model.predict(ex.features) for ex in dataset]
    scores = [round(model.score(ex.features), 3) for ex in dataset]

    accuracy = sum(int(gold_label == prediction) for gold_label, prediction in zip(gold, preds)) / float(len(dataset))
    tp = sum(1 for gold_label, prediction in zip(gold, preds) if gold_label == 1 and prediction == 1)
    fp = sum(1 for gold_label, prediction in zip(gold, preds) if gold_label == 0 and prediction == 1)
    fn = sum(1 for gold_label, prediction in zip(gold, preds) if gold_label == 1 and prediction == 0)
    precision = tp / float(tp + fp) if tp + fp else 0.0
    recall = tp / float(tp + fn) if tp + fn else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0

    misclassified = [
        {
            "name": ex.name,
            "gold": ex.label,
            "prediction": prediction,
            "score": score,
        }
        for ex, prediction, score in zip(dataset, preds, scores)
        if ex.label != prediction
    ]

    return {
        "accuracy": round(accuracy, 3),
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "predictions": preds,
        "scores": scores,
        "misclassified": misclassified,
    }


def build_final_predictions(final_models: Dict[str, object], hybrid_learner: HybridLearner) -> Dict[str, Dict[str, object]]:
    final_predictions: Dict[str, Dict[str, object]] = {}
    for model_name, model in final_models.items():
        model_predictions: Dict[str, object] = {}
        for example in TEST_SET:
            entry = {
                "label": example.label,
                "score": round(model.score(example.features), 3),
                "prediction": model.predict(example.features),
            }
            if model_name == "hybrid":
                entry["explanation"] = hybrid_learner.explain_score(example.features)
            model_predictions[example.name] = entry
        final_predictions[model_name] = model_predictions
    return final_predictions


def build_code_insights(traces: List[Dict[str, object]], hybrid_learner: HybridLearner) -> Dict[str, object]:
    final_trace = traces[-1]
    model_rank_by_final_accuracy = sorted(
        (
            {
                "model": model_name,
                "final_accuracy": metrics["accuracy"],
                "mean_accuracy": round(
                    sum(trace["evaluations"][model_name]["accuracy"] for trace in traces) / len(traces),
                    3,
                ),
                "final_errors": [item["name"] for item in metrics["misclassified"]],
            }
            for model_name, metrics in final_trace["evaluations"].items()
        ),
        key=lambda item: (-item["final_accuracy"], -item["mean_accuracy"], item["model"]),
    )

    hybrid_error_names = [item["name"] for item in final_trace["evaluations"]["hybrid"]["misclassified"]]
    strongest_positive = sorted(
        final_trace["representation"]["contrast"].items(),
        key=lambda item: item[1],
        reverse=True,
    )[:3]
    strongest_negative = sorted(
        final_trace["representation"]["contrast"].items(),
        key=lambda item: item[1],
    )[:3]

    return {
        "hybrid_config": asdict(hybrid_learner.config),
        "model_rank_by_final_accuracy": model_rank_by_final_accuracy,
        "hybrid_only_error": hybrid_error_names,
        "dominant_positive_features": strongest_positive,
        "dominant_negative_features": strongest_negative,
        "final_common_positive_features": final_trace["representation"]["common_positive"],
        "final_forbidden_features": final_trace["representation"]["forbidden"],
    }


def run_sequence(
    sequence_name: str,
    definition: str = DEFAULT_DEFINITION,
    hybrid_config: HybridConfig | None = None,
) -> Dict[str, object]:
    if sequence_name not in SEQUENCES:
        raise KeyError(f"Unknown sequence: {sequence_name}")

    train_lookup = {example.name: example for example in TRAIN_SET}
    rule_learner = TopDownRuleLearner(definition)
    feature_learner = IncrementalFeatureLearner(FEATURES)
    case_reasoner = CaseBasedReasoner(k=3)
    hybrid_learner = HybridLearner(rule_learner, feature_learner, case_reasoner, config=hybrid_config)

    traces: List[Dict[str, object]] = []
    for step, example_name in enumerate(SEQUENCES[sequence_name], start=1):
        example = train_lookup[example_name]
        feature_learner.update(example)
        case_reasoner.update(example)

        models = {
            "top_down": rule_learner,
            "bottom_up": feature_learner,
            "case_based": case_reasoner,
            "hybrid": hybrid_learner,
        }
        evaluations = {model_name: evaluate_model(model, TEST_SET) for model_name, model in models.items()}
        traces.append(
            {
                "step": step,
                "example_name": example.name,
                "example_label": example.label,
                "representation": feature_learner.current_representation(),
                "hybrid_weights": {
                    "rule": round(hybrid_learner.component_weights()[0], 3),
                    "feature": round(hybrid_learner.component_weights()[1], 3),
                    "case": round(hybrid_learner.component_weights()[2], 3),
                },
                "evaluations": evaluations,
            }
        )

    final_models = {
        "top_down": rule_learner,
        "bottom_up": feature_learner,
        "case_based": case_reasoner,
        "hybrid": hybrid_learner,
    }

    return {
        "sequence_name": sequence_name,
        "definition": definition,
        "train_sequence": SEQUENCES[sequence_name],
        "trace": traces,
        "final_feature_representation": feature_learner.current_representation(),
        "final_predictions": build_final_predictions(final_models, hybrid_learner),
        "code_insights": build_code_insights(traces, hybrid_learner),
    }


def run_all_sequences(
    definition: str = DEFAULT_DEFINITION,
    hybrid_config: HybridConfig | None = None,
) -> Dict[str, object]:
    return {
        sequence_name: run_sequence(sequence_name, definition=definition, hybrid_config=hybrid_config)
        for sequence_name in SEQUENCES
    }


def save_json(data: Dict[str, object], output_path: str) -> None:
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


if __name__ == "__main__":
    results = run_all_sequences()
    save_json(results, "week8_results.json")
    print("Saved week8_results.json")
