import torch
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from config import BATCH_SIZE, LEARNING_RATE, EPOCHS
from model import get_diabetes_model
from data import get_diabetes_data


def train(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_data_loader: torch.utils.data.DataLoader,
) -> None:
    total_size = len(train_data_loader.dataset)
    model.train()
    for batch, (x, y) in enumerate(train_data_loader):
        # Compute prediction and loss
        pred = model(x)
        loss = loss_fn(pred, y)
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Print progress (batch size is huge)
        # if batch % 100 == 0:

        loss, current = loss.item(), batch * len(x)
        print(f"loss: {loss:>7f}  [{current:>5d}/{total_size:>5d}]")


def test(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    test_data_loader: torch.utils.data.DataLoader,
) -> float:
    total_size = len(test_data_loader.dataset)
    num_batches = len(test_data_loader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for x, y in test_data_loader:
            pred = model(x)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.round() == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= total_size
    print(
        f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
    )
    return test_loss


def get_current_device() -> torch.device:
    return torch.device(
        device=(
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    )


def main() -> None:
    device = get_current_device()
    model = get_diabetes_model(device)
    train_data, test_data = get_diabetes_data(device)
    train_data_loader = DataLoader(train_data, batch_size=BATCH_SIZE)
    test_data_loader = DataLoader(test_data, batch_size=BATCH_SIZE)
    loss_fn = torch.nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

    test_loss_seq = []

    for epoch in range(EPOCHS):
        # clear last epoch's output
        print("\033[H\033[J")
        print(f"Epoch {epoch+1}\n-------------------------------")
        train(model, loss_fn, optimizer, train_data_loader)
        test_loss_seq.append(test(model, loss_fn, test_data_loader))

    plt.plot(test_loss_seq)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()
    
    print("Done!")


if __name__ == "__main__":
    main()
