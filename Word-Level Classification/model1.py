from imports import *
from model2 import ASLDataLM



class ASLClassifier(nn.Module):
    def __init__(self, num_features, num_labels, num_frames):
        super(ASLClassifier, self).__init__()
        input_size = num_features * num_labels * num_frames
        self.flatten = nn.Flatten()
        self.resnet = nn.Sequential(
            nn.Conv2d(input_size, 256),
            nn.ReLU(),
            nn.Linear(256,256),
            nn.ReLU(),
            nn.Linear(256,256),
            nn.ReLU(),
            nn.Linear(256, num_labels),
        )

    def forward(self, x):
        logits = self.resnet(x)
        return logits


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


def overfitCheck(dataloader, epochs, model, loss_fn, optimizer):
    data, targets = next(iter(dataloader))
    for epoch in range(epochs):
        # Compute prediction and loss
        data = data
        targets = targets
        pred = model(data)

        loss = loss_fn(pred, targets)
        loss = loss.item()
        print(f"loss: {loss:>7f}")

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def main():
    data_path = path.join('data', 'split_data')  #'load the data of (videos/frames, label) here'
    dataModule = ASLDataLM(data_path,8,8)
    train_dataset = dataModule.train_dataloader()
    train_loader = DATA.DataLoader(train_dataset, shuffle = True)
    test_dataset = dataModule.test_dataloader()
    test_loader = DATA.DataLoader(test_dataset, shuffle = True)

    sizes = 0   # should be in shape: ((x,y, fps, secs), label)
                # x=pixel width of frames,
                # y=pixel height of frames,
                # fps= number of frames per second in vid
                # secs= the duration of the vid
    model = ASLClassifier(sizes).to(device)


    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for t in range(epochs):
        print(f"Generated Data{' '.replace(' ','-'*17)}\n-------------------------------")
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_loader, model, loss_fn, optimizer)
        test_loop(test_loader, model, loss_fn)
    print("Done!")

    #Save the weights in a dictionary
    torch.save(model.state_dict(), "./model1params.pth")


if __name__=="__main__":
    # Hyperparameters
    learning_rate = 1e-3    # η from log_reg
    batch_size = 64         # the size of the samples that are used during training
    epochs = 30             # specifies the desired number of iterations tweaking 
                            # weights through the training set (epochs < 100)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    main()
