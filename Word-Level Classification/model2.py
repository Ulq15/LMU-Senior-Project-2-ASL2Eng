from imports import *
import VideoResNet as VRN

warnings.filterwarnings(
    "ignore", ".*Trying to infer the `batch_size` from an ambiguous collection.*"
)
warnings.filterwarnings(
    "ignore", ".*The dataloader, train_dataloader, does not have many workers which may be a bottleneck.*"
)

class ASLDataLM(pl.LightningDataModule):

    def __init__(self, data_path, batch_size, num_workers):
        super().__init__()
        self.batch_size = batch_size
        self.data_path = data_path
        self.num_workers = num_workers      # Number of parallel processes fetching data
        self.clip_length = 3                # Duration of sampled clip for each video
        self.vocab = None
        self.onehot = preprocessing.OneHotEncoder(sparse=False)
        self.save_hyperparameters()

    def setup(self, stage=None):
        # Assign Train/val split(s) for use in Dataloaders
        # print(stage)
        if stage in (None, "fit"):
            train_transform = Compose([
                ApplyTransformToKey(
                    key="video",
                    transform=Compose(
                        [
                            UniformTemporalSubsample(8),
                            Lambda(lambda x: x / 255.0),
                            Normalize((0.45, 0.45, 0.45),(0.225, 0.225, 0.225)),
                            RandomShortSideScale(min_size=256, max_size=320),
                            # RandomCrop(244),
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

        # if stage in (None, "validate"):
        #     val_transform = Compose([
        #         ApplyTransformToKey(
        #             key="video",
        #             transform=Compose(
        #                 [
        #                     UniformTemporalSubsample(8),
        #                     Lambda(lambda x: x / 255.0),
        #                     Normalize((0.45, 0.45, 0.45),(0.225, 0.225, 0.225)),
        #                     RandomShortSideScale(min_size=256, max_size=320),
        #                     # RandomCrop(244),
        #                     RandomHorizontalFlip(p=0.5),
        #                 ]
        #             ),
        #         ),
        #     ])
        #     self.val_dataset = LVDS(
        #         labeled_video_paths=self._load_csv(self.data_path+'\\val.csv'),
        #         clip_sampler=UCS(clip_duration=self.clip_length),
        #         transform=val_transform,
        #         decode_audio=False,
        #     )

        # Assign Test split(s) for use in Dataloaders
        if stage in (None, "test"):
            test_transform = Compose([
                ApplyTransformToKey(
                    key="video",
                    transform=Compose(
                        [
                            UniformTemporalSubsample(8),
                            Lambda(lambda x: x / 255.0),
                            Normalize((0.45, 0.45, 0.45),(0.225, 0.225, 0.225)),
                            ShortSideScale(256)
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

    # def val_dataloader(self):
    #     return DATA.DataLoader(self.val_dataset, batch_size=self.batch_size)

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
                singleton = (vpath, {'label': vector, 'word':label})
                # print(f'DataMember:\t'+str(singleton))
                video_labels.append(singleton)
        return video_labels

    def load_vocab(self, csv_name):
        pklPath = './savedVars/vocab.pkl'
        oneHotPath = './savedVars/onehot.pkl'
        if path.exists(pklPath):
            with open(pklPath,'rb') as f:
                self.vocab = pkl.load(f)
        else:
            self.vocab = []
            with open(csv_name, 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    label = row[1]
                    if label not in self.vocab:
                        self.vocab.append(label)
            with open(pklPath, 'xb') as file:
                pkl.dump(self.vocab, file)
        if path.exists(oneHotPath):
            with open(oneHotPath, 'rb') as f:
                self.onehot = pkl.load(f)
        else:
            arr = np.array(self.vocab).reshape(-1, 1)
            self.onehot.fit(arr)
            with open(oneHotPath, 'xb') as file:
                pkl.dump(self.onehot, file)


class ASLClassifierLM(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.accuracy = metrics.Accuracy()
        self.lr = 1e-1
        # block=VRN.BasicBlock
        # conv_makers=[VRN.Conv2Plus1D] * 4
        # layers=[2, 2, 2, 2]
        # num_classes = 2000
        # zero_init_residual = False

        # self.inplanes = 64

        # self.stem = VRN.R2Plus1dStem()

        # self.layer1 = self._make_layer(block, conv_makers[0], 64, layers[0], stride=1)
        # self.layer2 = self._make_layer(block, conv_makers[1], 128, layers[1], stride=2)
        # self.layer3 = self._make_layer(block, conv_makers[2], 256, layers[2], stride=2)
        # self.layer4 = self._make_layer(block, conv_makers[3], 512, layers[3], stride=2)

        # self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        # self.fc = nn.Linear(512 * block.expansion, num_classes)

        # # init weights
        # for m in self.modules():
        #     if isinstance(m, nn.Conv3d):
        #         nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        #         if m.bias is not None:
        #             nn.init.constant_(m.bias, 0)
        #     elif isinstance(m, nn.BatchNorm3d):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)
        #     elif isinstance(m, nn.Linear):
        #         nn.init.normal_(m.weight, 0, 0.01)
        #         nn.init.constant_(m.bias, 0)

        # if zero_init_residual:
        #     for m in self.modules():
        #         if isinstance(m, VRN.Bottleneck):
        #             nn.init.constant_(m.bn3.weight, 0)  # type: ignore[union-attr, arg-type]
        self.model = VRN.r2plus1d_18(num_classes=2000)

    def forward(self, x: Tensor) -> Tensor:
        # x = self.stem(x)

        # x = self.layer1(x)
        # x = self.layer2(x)
        # x = self.layer3(x)
        # x = self.layer4(x)

        # x = self.avgpool(x)
        # # Flatten the layer to fc
        # x = x.flatten(1)
        # x = self.fc(x)
        # return x
        logits = self.model(x)
        return logits

    # def _make_layer(
    #     self,
    #     block: Type[Union[VRN.BasicBlock, VRN.Bottleneck]],
    #     conv_builder: Type[Union[VRN.Conv3DSimple, VRN.Conv3DNoTemporal, VRN.Conv2Plus1D]],
    #     planes: int,
    #     blocks: int,
    #     stride: int = 1,
    # ) -> nn.Sequential:
    #     downsample = None

    #     if stride != 1 or self.inplanes != planes * block.expansion:
    #         ds_stride = conv_builder.get_downsample_stride(stride)
    #         downsample = nn.Sequential(
    #             nn.Conv3d(self.inplanes, planes * block.expansion, kernel_size=1, stride=ds_stride, bias=False),
    #             nn.BatchNorm3d(planes * block.expansion),
    #         )
    #     layers = []
    #     layers.append(block(self.inplanes, planes, conv_builder, stride, downsample))

    #     self.inplanes = planes * block.expansion
    #     for i in range(1, blocks):
    #         layers.append(block(self.inplanes, planes, conv_builder))

    #     return nn.Sequential(*layers)

    def configure_optimizers(self):
        # We will support Adam or SGD as optimizers.
        # if self.hparams.optimizer_name == "Adam":
        # AdamW is Adam with a correct implementation of weight decay
        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr)
        # return optimizer
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1, verbose=True)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        # "batch" is the output of the training data loader.
        vid_input:Tensor = batch['video']
        label_vect:Tensor = batch['label']
        # label:str = batch['word']
        preds:Tensor = self.model(vid_input)
        # print((label, label_vect.argmax(dim=-1), preds.argmax(dim=-1)))
        loss:Tensor = F.cross_entropy(preds, label_vect)
        self.accuracy(preds, label_vect.argmax(dim=-1))
        # Logs the accuracy per epoch to tensorboard (weighted average over batches)
        self.log("train_acc_step", self.accuracy, on_epoch=False, on_step=True)
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        self.log("train_acc_epoch", self.accuracy, on_epoch=True, on_step=False)
        return loss

    # def validation_step(self, batch, batch_idx):
    #     vid_input:Tensor = batch['video']
    #     label_vect:Tensor = batch['label']
    #     # label:str = batch['word']
    #     preds:Tensor = self.model(vid_input)
    #     val_loss:Tensor = F.cross_entropy(preds, label_vect)
    #     self.log("val_loss", val_loss)
    #     self.accuracy(preds, label_vect)
    #     # By default logs it per epoch (weighted average over batches), and returns it afterwards
    #     self.log("val_acc", self.accuracy, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        vid_input:Tensor = batch['video']
        label_vect:Tensor = batch['label']
        # label:str = batch['word']
        preds:Tensor = self.model(vid_input)
        test_loss:Tensor = F.cross_entropy(preds, label_vect)
        self.log("test_loss", test_loss)
        self.accuracy(preds, label_vect)
        # By default logs it per epoch (weighted average over batches), and returns it afterwards
        self.log("test_acc", self.accuracy)
        self.log('test_acc_epoch', self.accuracy)

    def backward(self, loss: Tensor, optimizer: Optional[optim.Optimizer], optimizer_idx: Optional[int], *args, **kwargs) -> None:
        return super().backward(loss, optimizer, optimizer_idx, *args, **kwargs)

    def optimizer_step(self, epoch: int, batch_idx: int, optimizer: Union[optim.Optimizer, LightningOptimizer], optimizer_idx: int = 0, optimizer_closure: Optional[Callable[[], Any]] = None, on_tpu: bool = False, using_native_amp: bool = False, using_lbfgs: bool = False) -> None:
        return super().optimizer_step(epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure, on_tpu, using_native_amp, using_lbfgs)



def main():
    data_path = './split_data/'
    var_Path = './savedVars/'
    batch_size = 8
    dataModule = ASLDataLM(data_path, batch_size, 8)
    model = ASLClassifierLM()
    # model_name = 'i3d_r50'
    # model = torch.hub.load("facebookresearch/pytorchvideo:main", model=model_name, pretrained=False, model_depth=101, model_num_class=2000)
    es = EarlyStopping(monitor="train_loss", mode="min", check_on_train_epoch_end=True)
    # auto_select_gpus=True,
    trainer = Trainer(max_epochs=10, callbacks=[es], enable_checkpointing=True,  gpus=1, accelerator="gpu", auto_lr_find=True, default_root_dir=var_Path)
    trainer.fit(model=model, datamodule=dataModule)#, ckpt_path='./stateSaves/')
    trainer.save_checkpoint(path.join(var_Path,"fit#2.ckpt"))
    # trainer.test(model, dataModule)


if __name__ == '__main__':
    main()
