from turtle import forward
from imports import *
from VideoResNet import r2plus1d_18

class ASLDataLM(pl.LightningDataModule):

    def __init__(self, data_path, batch_size, num_workers):
        self.batch_size = batch_size
        self.data_path = data_path
        self.num_workers = num_workers      # Number of parallel processes fetching data
        self.clip_length = 1                # Duration of sampled clip for each video
        self.vocab = None
        self.onehot = preprocessing.OneHotEncoder(sparse=False)

    def setup(self, stage=None):
        # Assign Train/val split(s) for use in Dataloaders
        if stage in (None, "fit", "validate"):
            train_transform = Compose([
                ApplyTransformToKey(
                    key="video",
                    transform=Compose(
                        [
                            UniformTemporalSubsample(8),
                            Lambda(lambda x: x / 255.0),
                            Normalize((0.45, 0.45, 0.45),(0.225, 0.225, 0.225)),
                            RandomShortSideScale(min_size=256, max_size=320),
                            RandomCrop(244),
                            RandomHorizontalFlip(p=0.5),
                        ]
                    ),
                ),
            ])
            self.train_dataset = LVDS(
                labeled_video_paths=self._load_csv(self.data_path+'\\train.csv'),
                clip_sampler=UCS(clip_duration=self.clip_length),
                transform=train_transform,
                decode_audio=False,
            )

        # Assign Test split(s) for use in Dataloaders
        if stage in (None, "test", "predict"):
            test_transform = Compose([
                ApplyTransformToKey(
                    key="video",
                    transform=Compose(
                        [
                            UniformTemporalSubsample(8),
                            Lambda(lambda x: x / 255.0),
                            Normalize((0.45, 0.45, 0.45),(0.225, 0.225, 0.225))
                        ]
                    ),
                ),
            ])
            self.test_dataset = LVDS(
                labeled_video_paths=self._load_csv(self.data_path+'\\test.csv'),
                clip_sampler=UCS(clip_duration=self.clip_length),
                transform=test_transform,
                decode_audio=False,
            )

    def train_dataloader(self):
        return DATA.DataLoader(self.train_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DATA.DataLoader(self.test_dataset, batch_size=self.batch_size)

    def _load_csv(self, csv_name):
        video_labels = []
        if self.vocab is None:
            self.load_vocab(csv_name)
        with open(csv_name, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                folderPath = os.getcwd()
                vpath = path.join(folderPath,row[0])
                vpath = path.normpath(vpath)
                label = row[1]
                if not path.exists(vpath) or not path.isfile(vpath):
                    continue
                vector = self.onehot.transform(np.array([label]).reshape(-1, 1)).flatten()
                singleton = (vpath, {'word': vector})
                # print(f'DataMember:\t'+str(singleton))
                video_labels.append(singleton)
        return video_labels

    def load_vocab(self, csv_name):
        self.vocab = []
        with open(csv_name, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                label = row[1]
                if label not in self.vocab:
                    self.vocab.append(label)
        self.onehot.fit(np.array(self.vocab).reshape(-1, 1))


class ASLClassifierLM(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.loss_module = nn.CrossEntropyLoss()
        self.model = r2plus1d_18()


    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        # We will support Adam or SGD as optimizers.
        # if self.hparams.optimizer_name == "Adam":
        # AdamW is Adam with a correct implementation of weight decay
        optimizer = optim.AdamW(self.model.parameters(), lr=1e-1)

        # We will reduce the learning rate by 0.1 after 100 and 150 epochs
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)
        return [optimizer], [scheduler]
        # return optimizer

    def training_step(self, batch, batch_idx):
        # "batch" is the output of the training data loader.
        vid_input = batch['video']
        label = batch['word']
        preds = self.model(vid_input)
        loss = self.loss_module(preds, label)
        acc = (preds.argmax(dim=-1) == label).float().mean()

        # Logs the accuracy per epoch to tensorboard (weighted average over batches)
        self.log("train_acc", acc, on_step=False, on_epoch=True)
        self.log("train_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        vid_input = batch['video']
        label = batch['word'][0]
        preds = self.model(vid_input).argmax(dim=-1)
        acc = (label == preds).float().mean()
        # By default logs it per epoch (weighted average over batches), and returns it afterwards
        self.log("test_acc", acc)


def main():
    # Function for setting the seed
    # pl.seed_everything(42)

    # Ensure that all operations are deterministic on GPU (if used) for reproducibility
    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using {device}")

    batch_size = 1

    data_path = path.join('data', 'split_data')
    dataModule = ASLDataLM(data_path, batch_size, 8)

    model = ASLClassifierLM()
    # reload_dataloaders_every_epoch=False, auto_select_gpus=True,
    trainer = Trainer(gpus=1, accelerator="gpu", auto_lr_find=True, default_root_dir='./stateSaves/')
    trainer.fit(model, dataModule)
    trainer.save_checkpoint("fit#1.ckpt")
    # trainer.test(model, dataModule)


if __name__ == '__main__':
    main()
