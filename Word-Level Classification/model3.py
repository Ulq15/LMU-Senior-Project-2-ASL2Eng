from imports import *

# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(1, 6, 3)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, 6, 3)
#         self.conv3 = nn.Conv2d(6, 16, 3)
#         self.fc1 = nn.Linear(16 * 5 * 5, 120)
#         self.fc2 = nn.Linear(120, 48)
#         self.fc3 = nn.Linear(48, 24)

#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = self.pool(F.relu(self.conv3(x)))
#         x = x.view(-1, 16 * 5 * 5)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x

class ASLDataset(DATA.Dataset):
    def __init__(self, dataFile, directory, transform=None, target_transform=None):
        self.path = dataFile
        self.dir = directory
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.vids)

    def __getitem__(self, idx):
        label = self.labels[idx]
        vid = self.vids[idx]
        sample = {"VideoPath":vid, "Label":label}
        return sample

    # # takes in a set of frames then creates a tensor of each frame
    # # then
    # def vectorize_img(image):
    #     indexes = []
    #     for px_row in image:
    #         for px_cell in px_row:
    #             pass
    #             indexes.append()
    #     onehot = F.one_hot(torch.tensor(indexes), ).float()
    #     return torch.flatten(onehot)



def train_loop(dataloader:DATA.Dataset, model:torch.nn.Module, loss_fn:torch.nn.CrossEntropyLoss, optimizer:torch.optim.Optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 10 == 0:
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


def main():
    data_path = path.join('data', 'split_data')  #'load the data of (videos/frames, label) here'

    asl_dataset = ASLDataset(data_path)
    data_loader = DATA.DataLoader(asl_dataset, shuffle = True)

    model = vmodels.r2plus1d_18(pretrained=True, progress=True)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        print(f"Generated Data"+'-'*17+"\n"+'-'*31)
        print(f"Epoch {epoch+1}\n"+'-'*31)
        train_loop(data_loader, model, loss_fn, optimizer)
        test_loop(data_loader, model, loss_fn)
    print("Done!")

    #Save the weights in a dictionary
    torch.save(model.state_dict(), "./model3params.pth")


if __name__=="__main__":
    # Hyperparameters
    learning_rate = 1e-3    # η from log_reg
    batch_size = 64         # the size of the samples that are used during training
    epochs = 30             # specifies the desired number of iterations tweaking 
                            # weights through the training set (epochs < 100)
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using {device}")

    main()

