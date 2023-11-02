import datetime
import torch
import pandas as pd
from model import get_diabetes_model
from data import get_diabetes_full_data
from config import OUT_PATH


def get_device() -> torch.device:
    return torch.device(
        device=(
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    )


def get_model_from_path(model_path: str, device: torch.device) -> torch.nn.Module:
    model = get_diabetes_model(device)
    model.load_state_dict(torch.load(model_path))
    return model


def main() -> None:
    # load model
    device = get_device()
    model = get_model_from_path("model/2023-11-02 09:21:54.851095.pt", device)
    # load data
    data = get_diabetes_full_data(device)
    preds = []
    pred_round = []
    # predict
    for i in range(len(data)):
        x, y = data[i]
        x = x.unsqueeze(0)
        pred = model(x)
        preds.append(pred.item())
        pred_round.append(pred.round().item())

    # save original and prediction
    df = pd.DataFrame(data.m_x.cpu().numpy())
    df["y"] = data.m_y.cpu().numpy()
    df["pred_round"] = pred_round
    df["pred"] = preds
    file_name = f"{OUT_PATH}/{datetime.datetime.now()}.csv"
    df.to_csv(file_name, index=False)


if __name__ == "__main__":
    main()
