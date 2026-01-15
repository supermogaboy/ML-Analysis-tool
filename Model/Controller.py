from Calm_run import score_calm_specialist
from trend_risk_run import score_trend_risk_specialist  
from volatile_event_run import score_volatile_event_specialist
from crisis_run import score_crisis_specialist



SPECIALISTS = {
    "calm": score_calm_specialist,
    "trend_risk": score_trend_risk_specialist,
    "volatile_event": score_volatile_event_specialist,
    "crisis": score_crisis_specialist,
}


def controller(regime: str):
    regime = regime.strip().lower()

    fn = SPECIALISTS.get(regime)
    if fn is None:
        raise ValueError(f"Invalid regime: {regime}. Choose from {list(SPECIALISTS)}")

    score = fn()          # or fn(csv_path=..., model_dir=...)
    return score

if __name__ == "__main__":
    controller(input("enter regime: "))