from imports import *
import VideoResNet as VRN
from typing import Tuple, Optional, Callable, List, Sequence, Type, Any, Union

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
        block=VRN.BasicBlock
        conv_makers=[VRN.Conv2Plus1D] * 4
        layers=[2, 2, 2, 2]
        num_classes = 2000
        zero_init_residual = False

        self.inplanes = 64

        self.stem = VRN.R2Plus1dStem()

        self.layer1 = self._make_layer(block, conv_makers[0], 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, conv_makers[1], 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, conv_makers[2], 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, conv_makers[3], 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # init weights
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, VRN.Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[union-attr, arg-type]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        # Flatten the layer to fc
        x = x.flatten(1)
        x = self.fc(x)
        return x

    def _make_layer(
        self,
        block: Type[Union[VRN.BasicBlock, VRN.Bottleneck]],
        conv_builder: Type[Union[VRN.Conv3DSimple, VRN.Conv3DNoTemporal, VRN.Conv2Plus1D]],
        planes: int,
        blocks: int,
        stride: int = 1,
    ) -> nn.Sequential:
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            ds_stride = conv_builder.get_downsample_stride(stride)
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion, kernel_size=1, stride=ds_stride, bias=False),
                nn.BatchNorm3d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, conv_builder, stride, downsample))

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, conv_builder))

        return nn.Sequential(*layers)

    def configure_optimizers(self):
        # We will support Adam or SGD as optimizers.
        # if self.hparams.optimizer_name == "Adam":
        # AdamW is Adam with a correct implementation of weight decay
        optimizer = optim.AdamW(self.parameters())

        # We will reduce the learning rate by 0.1 after 100 and 150 epochs
        # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)
        # return [optimizer], [scheduler]
        return optimizer

    def training_step(self, batch, batch_idx):
        # "batch" is the output of the training data loader.
        vid_input = batch['video']
        label = batch['word']
        preds = self.forward(vid_input)
        loss = self.loss_module(preds, label)

        acc = (preds.argmax(dim=-1) == label.argmax(dim=-1)).float().mean()

        # Logs the accuracy per epoch to tensorboard (weighted average over batches)
        self.log("train_acc", acc, on_step=False, on_epoch=True)
        self.log("train_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        vid_input = batch['video']
        label = batch['word'][0]
        preds = self.forward(vid_input).argmax(dim=-1)
        acc = (label == preds).float().mean()
        # By default logs it per epoch (weighted average over batches), and returns it afterwards
        self.log("test_acc", acc)

    def backward(self, loss: torch.Tensor, optimizer: Optional[optim.Optimizer], optimizer_idx: Optional[int], *args, **kwargs) -> None:
        return super().backward(loss, optimizer, optimizer_idx, *args, **kwargs)

    def optimizer_step(self, epoch: int, batch_idx: int, optimizer: Union[optim.Optimizer, LightningOptimizer], optimizer_idx: int = 0, optimizer_closure: Optional[Callable[[], Any]] = None, on_tpu: bool = False, using_native_amp: bool = False, using_lbfgs: bool = False) -> None:
        return super().optimizer_step(epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure, on_tpu, using_native_amp, using_lbfgs)



def main():
    # # Function for setting the seed
    # pl.seed_everything(42)

    # # Ensure that all operations are deterministic on GPU (if used) for reproducibility
    # torch.backends.cudnn.determinstic = True
    # torch.backends.cudnn.benchmark = False

    # # Set device
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print(f"Using {device}")

    batch_size = 8

    data_path = path.join('data', 'split_data')
    dataModule = ASLDataLM(data_path, batch_size, num_workers=8)

    model = ASLClassifierLM()
    es = EarlyStopping(monitor="train_loss", mode="min", check_on_train_epoch_end=True)
    # reload_dataloaders_every_epoch=False, auto_select_gpus=True,
    trainer = Trainer(gpus=1, callbacks=[es], accelerator="gpu", auto_lr_find=True, default_root_dir='./stateSaves/')
    trainer.fit(model=model, datamodule=dataModule)#, ckpt_path='./stateSaves/lightning_logs/')
    trainer.save_checkpoint("fit#2.ckpt")
    # trainer.test(model, dataModule)


if __name__ == '__main__':
    main()
