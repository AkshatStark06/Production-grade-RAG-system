import json
import sys

# Thresholds (production rules)
THRESHOLDS = {
    "correctness": 0.50,
    "faithfulness": 0.50,
    "hit_rate": 0.80
}

def check_metrics():
    with open("evaluation/results.json", "r") as f:
        results = json.load(f)

    failed = False

    print("\n🔍 Checking Metrics Against Thresholds...\n")

    for key, threshold in THRESHOLDS.items():
        value = results.get(key, 0)

        print(f"{key}: {value:.4f} (Required: {threshold})")

        if value < threshold:
            print(f"❌ FAILED: {key} below threshold!")
            failed = True
        else:
            print(f"✅ PASSED")

    if failed:
        print("\n🚨 BUILD FAILED due to low performance!")
        sys.exit(1)
    else:
        print("\n🎉 All checks passed!")
        sys.exit(0)


if __name__ == "__main__":
    check_metrics()