class Config:
    def __init__(self, dataset="mnist", **kwargs):
        self.seed = kwargs.get("seed", 42)
        self.data_dir = kwargs.get("data_dir", "data")
        self.output_dir = kwargs.get("output_dir", "output")
        self.batch_size = kwargs.get("batch_size", 256)
        self.epochs = kwargs.get("epochs", 50)
        self.lr = kwargs.get("lr", 2e-4)
        self.device = kwargs.get("device", "cuda")

        if dataset == "mnist":
            self.img_size = 28
            self.channels = 1
            self.latent_dim = 32
            self.encoder_channels = [32, 64, 128]
            self.decoder_channels = kwargs.get("decoder_channels", [128, 64, 32])
            self.codebook_size = kwargs.get("codebook_size", 64)
        else:
            self.img_size = 64
            self.channels = 3
            self.latent_dim = 128
            self.encoder_channels = [64, 128, 256, 512]
            self.decoder_channels = kwargs.get("decoder_channels", [512, 256, 128, 64])
            self.codebook_size = kwargs.get("codebook_size", 512)

        self.train_split = kwargs.get("train_split", 0.8)
        self.wandb_project = kwargs.get("wandb_project", "ece285-fpvae")
        self.wandb_run = kwargs.get("wandb_run", None)
        self.use_wandb = kwargs.get("use_wandb", True)
        self.fid_batches = kwargs.get("fid_batches", 10)
        self.fid_every_n_epochs = kwargs.get("fid_every_n_epochs", 10)
        self.beta = kwargs.get("beta", 1.0)


def get_mnist_config(**kwargs):
    return Config(dataset="mnist", **kwargs)


def get_abo_config(**kwargs):
    return Config(dataset="abo", **kwargs)
