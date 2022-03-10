from imports import *

class ASLDataLM(pl.LightningDataModule):
    
    def __init__(self, data_path, batch_size, num_workers=0):
        self.batch_size = batch_size
        self.data_path = data_path
        self.num_workers = num_workers      # Number of parallel processes fetching data
        self.clip_length = 1            # Duration of sampled clip for each video
        
    def train_dataloader(self):
        train_transform = Compose(
            [
            ApplyTransformToKey(
              key="video",
              transform=Compose(
                  [
                    UniformTemporalSubsample(8),
                    Lambda(lambda x: x / 255.0),
                    Normalize((0.45, 0.45, 0.45), (0.225, 0.225, 0.225)),
                    RandomShortSideScale(min_size=256, max_size=320),
                    RandomCrop(244),
                    RandomHorizontalFlip(p=0.5),
                  ]
                ),
              ),
            ]
        )
        train_dataset = LVDS(
            labeled_video_paths=self._load_csv(self.data_path+'\\train.csv'),
            clip_sampler=UCS(
                clip_duration=self.clip_length,
                backpad_last=True
            ),
            transform=train_transform,
            decode_audio=False,
        )
        return DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
    
    def test_dataloader(self):
        test_transform = Compose(
            [
            ApplyTransformToKey(
              key="video",
              transform=Compose(
                  [
                    UniformTemporalSubsample(8),
                    Lambda(lambda x: x / 255.0),
                    Normalize((0.45, 0.45, 0.45), (0.225, 0.225, 0.225))
                  ]
                ),
              ),
            ]
        )
        test_dataset = LVDS(
            labeled_video_paths=self._load_csv(self.data_path+'\\test.csv'),
            clip_sampler=UCS(
                clip_duration=self.clip_length,
                backpad_last=True
            ),
            transform=test_transform,
            decode_audio=False,
        )
        return DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def _load_csv(self, csv_name):
        video_labels = []
        with open(csv_name, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                vpath = row[0]
                label = row[1]
                #print(f'Row:\t'+str(({vpath:label})))
                video_labels.append({vpath:label})
        return video_labels
        

class ASLClassifierLM(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # self.save_hyperparameters()
        # self.model = create_model(model_name, model_hparams)
        self.loss_module = nn.CrossEntropyLoss()
        #********* START MAKING THIS
        self.model = models.create_resnet(
            input_channel=3,
            model_depth=152,
            model_num_class=2000,
            norm=nn.BatchNorm3d,
            activation=nn.ReLU,
        )
        
    def forward(self, vid):
        return self.model(vid)
    
    def configure_optimizers(self):
        # We will support Adam or SGD as optimizers.
        # if self.hparams.optimizer_name == "Adam":
            # AdamW is Adam with a correct implementation of weight decay
        optimizer = optim.AdamW(self.parameters())
        # elif self.hparams.optimizer_name == "SGD":
        #     optimizer = optim.SGD(self.parameters())
        # else:
        #     assert False, f'Unknown optimizer: "{self.hparams.optimizer_name}"'

        # We will reduce the learning rate by 0.1 after 100 and 150 epochs
        # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)
        # return [optimizer], [scheduler]
        return optimizer
    
    def training_step(self, batch, batch_idx):
        # "batch" is the output of the training data loader.
        
       
            
        vid_input, label = batch[batch_idx]
        preds = self(vid_input)
        loss = F.cross_entropy(preds, label)
        acc = (preds.argmax(dim=-1) == label).float().mean()
        
        # Logs the accuracy per epoch to tensorboard (weighted average over batches)
        self.log("train_acc", acc, on_step=False, on_epoch=True)
        self.log("train_loss", loss)
        return loss
        
    def test_step(self, batch, batch_idx):
        vid_input, labels = batch
        preds = self.model(vid_input).argmax(dim=-1)
        acc = (labels == preds).float().mean()
        # By default logs it per epoch (weighted average over batches), and returns it afterwards
        self.log("test_acc", acc)



def main():
    data_path = path.join('data', 'split_data')
    dm = ASLDataLM(data_path, batch_size, 0)
    
    model = ASLClassifierLM().cuda()
    
    trainer = Trainer( gpus=1, accelerator="gpu", auto_lr_find=True) # gpus=-1, auto_select_gpus=True,
    trainer.accelerator
    trainer.fit(model, dm)
    trainer.test(model, dm)
    
    

if __name__=='__main__':
    # Function for setting the seed
    pl.seed_everything(42)

    # Ensure that all operations are deterministic on GPU (if used) for reproducibility
    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False
    
    # Set device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model_dict = {}    
    act_fn_by_name = {"tanh": nn.Tanh, "relu": nn.ReLU, "leakyrelu": nn.LeakyReLU, "gelu": nn.GELU}


    # Hyperparameters
    learning_rate = 1e-3    # η from log_reg
    batch_size = 64         # the size of the samples that are used during training
    epochs = 30             # specifies the desired number of iterations tweaking 
                            # weights through the training set (epochs < 100)


    main()
    
    
# Len(Batch[0]):  64
#                 ['neighbor', 'calm', 'box', 'grammar', 'move', 'common sense', 
#                   'train', 'librarian', 'color', 'dry', 'care', 'shopping', 'pain', 
#                   'front', 'spray', 'tomorrow', 'microscope', 'dirt', 'will', 'borrow', 
#                   'admit', 'divorce', 'philadelphia', 'quote', 'vacation', 'demonstrate', 
#                   'smell', 'bald', 'horse', 'lesson', 'camp', 'ten', 'french fries', 
#                   'trouble', 'sequence', 'play', 'replace', 'pet', 'silver', 'spain', 
#                   'skeleton', 'large', 'cereal', 'sit', 'book', 'sugar', 'flatter', 
#                   'stare', 'relationship', 'debate', 'attitude', 'resign', 'noon', 
#                   'question', 'almost', 'wonderful', 'helmet', 'december', 'program', 
#                   'diabetes', 'boast', 'careless', 'blanket', 'early']