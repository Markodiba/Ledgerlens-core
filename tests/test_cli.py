import os

from typer.testing import CliRunner

from cli import app

runner = CliRunner()


def test_generate_data_writes_csvs(tmp_path):
    out_dir = str(tmp_path / "synthetic")
    result = runner.invoke(
        app,
        [
            "generate-data",
            "--out-dir", out_dir,
            "--n-normal-accounts", "3",
            "--n-wash-rings", "1",
            "--ring-size", "2",
        ],
    )

    assert result.exit_code == 0, result.output
    assert os.path.exists(os.path.join(out_dir, "trades.csv"))
    assert os.path.exists(os.path.join(out_dir, "order_book_events.csv"))
    assert os.path.exists(os.path.join(out_dir, "labels.csv"))


def test_train_saves_models(tmp_path, monkeypatch):
    model_dir = str(tmp_path / "models")
    monkeypatch.setenv("MODEL_DIR", model_dir)

    import config.settings as settings_module

    object.__setattr__(settings_module.settings, "model_dir", model_dir)

    result = runner.invoke(
        app,
        [
            "train",
            "--n-normal-accounts", "30",
            "--n-wash-rings", "8",
            "--ring-size", "3",
        ],
    )

    assert result.exit_code == 0, result.output
    assert os.path.exists(os.path.join(model_dir, "random_forest.joblib"))
    assert os.path.exists(os.path.join(model_dir, "xgboost.joblib"))
    assert os.path.exists(os.path.join(model_dir, "lightgbm.joblib"))
