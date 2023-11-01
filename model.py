import torch


class DiabetesModel(torch.nn.Module):
    m_linear_stack: torch.nn.Sequential
    m_flatten: torch.nn.Flatten

    def __init__(self) -> None:
        super().__init__()
        # use less than 5 linear layers
        self.m_linear_stack = torch.nn.Sequential(
            torch.nn.Linear(8, 6),
            torch.nn.ReLU(),
            torch.nn.Linear(6, 4),
            torch.nn.ReLU(),
            torch.nn.Linear(4, 2),
            torch.nn.ReLU(),
            torch.nn.Linear(2, 1),
            torch.nn.Sigmoid(),
        )
        self.m_flatten = torch.nn.Flatten()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.m_flatten(x)
        logits = self.m_linear_stack(x)
        return logits

def get_diabetes_model(device: torch.device) -> DiabetesModel:
    model = DiabetesModel()
    model.to(device)
    return model

# test
if __name__ == "__main__":
    model = get_diabetes_model("cpu")
    print(model)
    print(model.m_linear_stack)
    print(model.m_flatten)
    print(model.forward(torch.randn(1, 8)))
