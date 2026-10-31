"""Locked v3 health rules. Only final Z can trigger a health pause.

The training probe and validation stream establish their own learning evidence.
No gradients, logits thresholds, or historical best IoU substitute for that evidence.
"""
from __future__ import annotations
import copy


class LearningEvidence:
    def __init__(self):
        self.learned = [False] * 4
        self.learning_streak = [0] * 4
        self.lost_streak = [0] * 4
        self.zero_tp_streak = [0] * 4
        self.last_update = 0

    def observe(self, result, update, *, stream, recent_positive=None):
        if update <= self.last_update:
            raise ValueError("Health nodes must be strictly increasing")
        self.last_update = update
        counts, recalls = result["counts"], result["recall_percent"]
        if len(counts) != 4 or len(recalls) != 4:
            raise ValueError("Four class health evidence is required")
        reasons = []
        for c, name in enumerate(("EX", "HE", "SE", "MA")):
            if counts[c][0] + counts[c][2] <= 0:
                if stream == "probe":
                    raise ValueError("Fixed training probe must contain every class")
                self.learning_streak[c] = self.lost_streak[c] = self.zero_tp_streak[c] = 0
                continue
            recall = float(recalls[c]) / 100
            if not 0 <= recall <= 1:
                raise ValueError("Nonfinite or invalid recall")
            # Consecutive zero detections can begin before 4k; the alarm cannot.
            self.zero_tp_streak[c] = self.zero_tp_streak[c] + 1 if counts[c][0] == 0 else 0
            if update < 4000:
                self.learning_streak[c] = self.lost_streak[c] = 0
                continue
            if not self.learned[c]:
                self.learning_streak[c] = self.learning_streak[c] + 1 if recall >= .05 else 0
                if self.learning_streak[c] >= 2:
                    self.learned[c] = True
                    self.lost_streak[c] = 0
                elif (stream == "probe" and self.zero_tp_streak[c] >= 3
                      and recent_positive is not None and recent_positive[c]):
                    reasons.append(f"probe_persistently_unlearned_{name}")
            else:
                self.lost_streak[c] = self.lost_streak[c] + 1 if recall <= .001 else 0
                if self.lost_streak[c] >= 2:
                    reasons.append(f"{stream}_learned_then_lost_{name}")
        return reasons

    def state_dict(self):
        return copy.deepcopy(vars(self))

    def load_state_dict(self, state):
        if set(state) != set(vars(self)):
            raise ValueError("Incomplete learning evidence state")
        for key in ("learned", "learning_streak", "lost_streak", "zero_tp_streak"):
            if len(state[key]) != 4:
                raise ValueError("Incomplete class health state")
        self.__dict__.update(copy.deepcopy(state))


class HealthGuard:
    def __init__(self, protocol=None):
        self.probe, self.validation = LearningEvidence(), LearningEvidence()
        self.supervision = []
        self.last_supervision_update = 0
        self.validation_peak = None

    def note_supervision(self, update, positive_pixels):
        if update != self.last_supervision_update + 1 or len(positive_pixels) != 4:
            raise ValueError("Supervision must be recorded once for every SGD update")
        if any(int(v) < 0 for v in positive_pixels):
            raise ValueError("Negative positive-pixel count")
        self.last_supervision_update = update
        self.supervision.append((update, [int(v) for v in positive_pixels]))
        self.supervision = [v for v in self.supervision if v[0] > update - 1000]

    def observe_probe(self, result, update):
        if self.last_supervision_update != update:
            raise ValueError("Probe is missing current SGD supervision history")
        recent = [any(p[c] > 0 for u, p in self.supervision if u > update - 1000)
                  for c in range(4)]
        return self.probe.observe(result, update, stream="probe", recent_positive=recent)

    def observe_validation(self, result, update):
        value = float(result["mIoU_percent"])
        if not 0 <= value <= 100:
            raise ValueError("Nonfinite or invalid validation mIoU")
        reasons = self.validation.observe(result, update, stream="validation")
        # The specification imposes no grace period for a 15 percentage-point drop.
        if self.validation_peak is not None and self.validation_peak - value >= 15:
            reasons.append("validation_miou_drop_at_least_15pp")
        self.validation_peak = value if self.validation_peak is None else max(self.validation_peak, value)
        return reasons

    def state_dict(self):
        return {"version": 3, "probe": self.probe.state_dict(),
                "validation": self.validation.state_dict(),
                "supervision": copy.deepcopy(self.supervision),
                "last_supervision_update": self.last_supervision_update,
                "validation_peak": self.validation_peak}

    def load_state_dict(self, state):
        required = {"version", "probe", "validation", "supervision", "last_supervision_update", "validation_peak"}
        if set(state) != required or state["version"] != 3:
            raise ValueError("Incomplete or incompatible health state")
        self.probe.load_state_dict(state["probe"])
        self.validation.load_state_dict(state["validation"])
        self.supervision = copy.deepcopy(state["supervision"])
        self.last_supervision_update = state["last_supervision_update"]
        self.validation_peak = state["validation_peak"]
