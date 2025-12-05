import numpy as np
import pandas as pd

from typing import Tuple, Dict, Any, List

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

import matplotlib.pyplot as plt
import os


# figures directory for future use
os.makedirs("figures", exist_ok=True)


# ============================================================
# 1. Load data and basic cleaning
# ============================================================

def load_data(path: str) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Load the California Housing dataset and return feature matrix X
    and target vector y.
    """
    df = pd.read_csv(path)
    # Drop rows with missing values
    df = df.dropna()

    target_col = "Median_House_Value"
    y = df[target_col].values
    X = df.drop(columns=[target_col])

    return X, y


# ============================================================
# 2. Train/Val/Test split (70/15/15)
# ============================================================

def split_data(X: pd.DataFrame, y: np.ndarray, random_state: int = 42):
    """
    Split the data into train (70%), validation (15%), and test (15%).
    """
    # First split: train (70%) + temp (30%)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=random_state
    )
    # Second split: val (15%) + test (15%) from the temp set
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=random_state
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


# ============================================================
# 3. Standardization
# ============================================================

def standardize_splits(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
):

    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_val_std = scaler.transform(X_val)
    X_test_std = scaler.transform(X_test)
    return X_train_std, X_val_std, X_test_std, scaler


# ============================================================
# 4. Linear regression baseline
# ============================================================

def train_linear_regression(
    X_train_std: np.ndarray,
    y_train: np.ndarray,
    X_val_std: np.ndarray,
    y_val: np.ndarray,
) -> Tuple[LinearRegression, Dict[str, float]]:
    """
    Train an ordinary least squares linear regressor and report
    train/validation metrics.
    """
    lin_reg = LinearRegression()
    lin_reg.fit(X_train_std, y_train)

    y_train_pred = lin_reg.predict(X_train_std)
    y_val_pred = lin_reg.predict(X_val_std)

    metrics = {
        "train_mse": mean_squared_error(y_train, y_train_pred),
        "train_mae": mean_absolute_error(y_train, y_train_pred),
        "val_mse": mean_squared_error(y_val, y_val_pred),
        "val_mae": mean_absolute_error(y_val, y_val_pred),
    }

    print(
        f"[Linear] Train MSE: {metrics['train_mse']:.2f}, "
        f"Train MAE: {metrics['train_mae']:.2f}"
    )
    print(
        f"[Linear] Val   MSE: {metrics['val_mse']:.2f}, "
        f"Val   MAE: {metrics['val_mae']:.2f}"
    )

    return lin_reg, metrics


# ============================================================
# 5. Neural network regressor with small hyperparameter search
#    + early stopping
# ============================================================

def train_mlp_with_tuning(
    X_train_std: np.ndarray,
    y_train: np.ndarray,
    X_val_std: np.ndarray,
    y_val: np.ndarray,
    hidden_layer_grid: Tuple[tuple, ...] = ((32,), (64,), (64, 32)),
    lr_grid: Tuple[float, ...] = (1e-3, 5e-4),
    max_iter: int = 500,
    random_state: int = 42,
) -> Tuple[MLPRegressor, Dict[str, Any], List[Dict[str, Any]]]:
    """
    Perform a small grid search over hidden layer sizes and learning rates.
    For each configuration we:
      - train an MLPRegressor with early stopping
      - evaluate MSE on the external validation split

    We return the best model, its metrics, and the full search history.
    """
    best_mlp = None
    best_val_mse = np.inf
    best_config = None
    search_history: List[Dict[str, Any]] = []

    for hidden in hidden_layer_grid:
        for lr in lr_grid:
            mlp = MLPRegressor(
                hidden_layer_sizes=hidden,
                activation="relu",
                solver="adam",
                learning_rate_init=lr,
                max_iter=max_iter,
                random_state=random_state,
                early_stopping=True,
                n_iter_no_change=10,
                validation_fraction=0.1,  # internal validation for early stopping
            )

            mlp.fit(X_train_std, y_train)

            y_train_pred = mlp.predict(X_train_std)
            y_val_pred = mlp.predict(X_val_std)

            train_mse = mean_squared_error(y_train, y_train_pred)
            val_mse = mean_squared_error(y_val, y_val_pred)

            search_history.append(
                {
                    "hidden_layers": hidden,
                    "learning_rate": lr,
                    "train_mse": train_mse,
                    "val_mse": val_mse,
                }
            )

            print(
                f"[MLP config hidden={hidden}, lr={lr}] "
                f"Train MSE: {train_mse:.2f}, Val MSE: {val_mse:.2f}"
            )

            if val_mse < best_val_mse:
                best_val_mse = val_mse
                best_mlp = mlp
                best_config = {"hidden_layers": hidden, "learning_rate": lr}

    assert best_mlp is not None

    # Recompute metrics for the best configuration
    y_train_pred = best_mlp.predict(X_train_std)
    y_val_pred = best_mlp.predict(X_val_std)
    metrics = {
        "train_mse": mean_squared_error(y_train, y_train_pred),
        "train_mae": mean_absolute_error(y_train, y_train_pred),
        "val_mse": mean_squared_error(y_val, y_val_pred),
        "val_mae": mean_absolute_error(y_val, y_val_pred),
        "hidden_layers": best_config["hidden_layers"],
        "learning_rate": best_config["learning_rate"],
    }

    print(
        f"[MLP best] hidden={best_config['hidden_layers']}, "
        f"lr={best_config['learning_rate']}"
    )
    print(
        f"[MLP best] Train MSE: {metrics['train_mse']:.2f}, "
        f"Train MAE: {metrics['train_mae']:.2f}"
    )
    print(
        f"[MLP best] Val   MSE: {metrics['val_mse']:.2f}, "
        f"Val   MAE: {metrics['val_mae']:.2f}"
    )

    return best_mlp, metrics, search_history


# ============================================================
# 6. Conformal prediction (split-conformal)
# ============================================================

def fit_conformal(
    model,
    X_calib_std: np.ndarray,
    y_calib: np.ndarray,
    alpha: float = 0.1
) -> float:
    """
    Split-conformal: use a calibration set (here: the validation split)
    to compute the residual quantile for level 1 - alpha.
    """
    y_calib_pred = model.predict(X_calib_std)
    residuals = np.abs(y_calib - y_calib_pred)
    q_hat = np.quantile(residuals, 1 - alpha)
    return q_hat


def predict_conformal(
    model,
    X_std: np.ndarray,
    q_hat: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return point predictions and symmetric prediction intervals.
    """
    y_pred = model.predict(X_std)
    lower = y_pred - q_hat
    upper = y_pred + q_hat
    return y_pred, lower, upper


def coverage_and_width(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray
) -> Tuple[float, float]:
    """
    Compute empirical coverage and average interval width.
    """
    covered = (y_true >= lower) & (y_true <= upper)
    coverage = covered.mean()
    avg_width = np.mean(upper - lower)
    return coverage, avg_width


# ============================================================
# 7. Diagnostic plotting
# ============================================================

def plot_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    filename: str
) -> None:
    """
    Scatter plot of true vs predicted values with y = x reference line.
    Saves the figure to 'filename'.
    """
    plt.figure()
    plt.scatter(y_true, y_pred, alpha=0.3)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")
    plt.xlabel("True median house value")
    plt.ylabel("Predicted median house value")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()


# ============================================================
# 8. Main experiment
# ============================================================

def main() -> None:
    # 1. Load data
    X, y = load_data("California_Houses.csv")

    # 2. Split into train/val/test
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    # 3. Standardize
    X_train_std, X_val_std, X_test_std, scaler = standardize_splits(
        X_train, X_val, X_test
    )

    # 4. Linear regression baseline
    lin_reg, lin_metrics = train_linear_regression(
        X_train_std, y_train, X_val_std, y_val
    )
    y_test_pred_lin = lin_reg.predict(X_test_std)
    mse_lin_test = mean_squared_error(y_test, y_test_pred_lin)
    mae_lin_test = mean_absolute_error(y_test, y_test_pred_lin)
    print(f"[Linear] Test  MSE: {mse_lin_test:.2f}, Test MAE: {mae_lin_test:.2f}")

    # 5. Neural network with tuning + early stopping
    mlp, mlp_metrics, search_history = train_mlp_with_tuning(
        X_train_std, y_train, X_val_std, y_val
    )
    y_test_pred_mlp = mlp.predict(X_test_std)
    mse_mlp_test = mean_squared_error(y_test, y_test_pred_mlp)
    mae_mlp_test = mean_absolute_error(y_test, y_test_pred_mlp)
    print(f"[MLP]    Test  MSE: {mse_mlp_test:.2f}, Test MAE: {mae_mlp_test:.2f}")

    # 6. Conformal prediction using validation as calibration
    alpha = 0.1  # 90% prediction interval
    q_hat = fit_conformal(mlp, X_val_std, y_val, alpha=alpha)
    _, lower_mlp, upper_mlp = predict_conformal(mlp, X_test_std, q_hat)
    coverage, avg_width = coverage_and_width(y_test, lower_mlp, upper_mlp)
    print(f"[Conformal MLP] Coverage: {coverage:.3f}, Avg width: {avg_width:.2f}")

    # 7. Diagnostic plots for the report
    plot_predictions(
        y_test,
        y_test_pred_lin,
        "Linear regression: test predictions",
        "figures/linear_test_scatter.png",
    )

    plot_predictions(
        y_test,
        y_test_pred_mlp,
        "Neural network: test predictions",
        "figures/mlp_test_scatter.png",
    )

if __name__ == "__main__":
    main()
