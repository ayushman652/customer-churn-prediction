import pandas as pd
import matplotlib.pyplot as plt

from .config import FEATURES, TARGET, OUTPUT_DIR

def plot_coefficients(model):
    coefficients = pd.Series(
        model.coef_[0],
        index=FEATURES
    )
    
    coefficients.sort_values().plot(kind="barh")
    
    plt.title("feature coefficients")
    plt.xlabel("coefficient value")
    plt.tight_layout()
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_DIR / "feature_coefficients.png")
    plt.show()
    