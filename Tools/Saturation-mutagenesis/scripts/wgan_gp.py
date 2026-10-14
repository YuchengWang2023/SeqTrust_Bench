import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad
from tqdm import trange


# =========================================================
# Residual block
# =========================================================
class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 5, padding=2)
        self.act = nn.ReLU()

    def forward(self,x):
        return x + self.conv(self.act(x))


# =========================================================
# Generator
# =========================================================
class Generator(nn.Module):
    def __init__(self, seq_len, latent_dim, hidden_dim, seq_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.fc = nn.Linear(latent_dim, seq_len*hidden_dim)

        self.blocks = nn.Sequential(
            ResBlock(hidden_dim),
            ResBlock(hidden_dim),
            ResBlock(hidden_dim),
            ResBlock(hidden_dim),
            ResBlock(hidden_dim),
        )

        self.conv_out = nn.Conv1d(hidden_dim, seq_dim, 1)

    def forward(self,z):
        x = self.fc(z)
        x = x.view(x.size(0), self.hidden_dim, self.seq_len)
        x = self.blocks(x)
        x = self.conv_out(x)
        x = torch.softmax(x,dim=1)
        return x.transpose(1,2)   # (B, L, 4)


# =========================================================
# Critic
# =========================================================
class Critic(nn.Module):
    def __init__(self, seq_len, hidden_dim, seq_dim):
        super().__init__()
        self.conv = nn.Conv1d(seq_dim, hidden_dim, 1)

        self.blocks = nn.Sequential(
            ResBlock(hidden_dim),
            ResBlock(hidden_dim),
            ResBlock(hidden_dim),
            ResBlock(hidden_dim),
            ResBlock(hidden_dim),
        )

        self.fc = nn.Linear(hidden_dim*seq_len,1)

    def forward(self,x):
        x = x.transpose(1,2)
        x = self.conv(x)
        x = self.blocks(x)
        x = x.reshape(x.size(0),-1)
        return self.fc(x)


# =========================================================
# WGANGP main class
# =========================================================
class WGANGP:

    def __init__(
        self,
        seq_len=500,
        seq_dim=4,
        latent_dim=128,
        hidden_dim=128,
        batch_size=128,
        critic_iters=10,
        gp_lambda=10,
        lr=1e-4,
        device=None
    ):

        self.seq_len = seq_len
        self.seq_dim = seq_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.critic_iters = critic_iters
        self.gp_lambda = gp_lambda

        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available()
            else torch.device("cpu")
        )

        self.G = Generator(seq_len, latent_dim, hidden_dim, seq_dim).to(self.device)
        self.D = Critic(seq_len, hidden_dim, seq_dim).to(self.device)

        self.opt_G = optim.Adam(self.G.parameters(),lr=lr,betas=(0.5,0.9))
        self.opt_D = optim.Adam(self.D.parameters(),lr=lr,betas=(0.5,0.9))


    # -----------------------------------------------------
    # FASTA reader
    # -----------------------------------------------------
    def read_fasta(self, filename):
        ids, seqs = [], []
        with open(filename) as f:
            seq=[]
            for line in f:
                if line[0]=='>':
                    ids.append(line[1:].strip())
                    if seq:
                        seqs.append("".join(seq))
                    seq=[]
                else:
                    seq.append(line.strip().upper())
            if seq:
                seqs.append("".join(seq))
        return ids, seqs


    def one_hot(self, seq):
        mapping = {'A':0,'C':1,'G':2,'T':3}
        arr = np.zeros((len(seq),4))
        for i,s in enumerate(seq):
            if s in mapping:
                arr[i,mapping[s]]=1
        return arr


    # -----------------------------------------------------
    # Gradient penalty
    # -----------------------------------------------------
    def gradient_penalty(self, real, fake):

        alpha = torch.rand(real.size(0),1,1).to(self.device)
        inter = alpha*real + (1-alpha)*fake
        inter.requires_grad_(True)

        d_inter = self.D(inter)
        grads = grad(
            outputs=d_inter,
            inputs=inter,
            grad_outputs=torch.ones_like(d_inter),
            create_graph=True
        )[0]

        grads = grads.reshape(grads.size(0), -1)
        gp = ((grads.norm(2,dim=1)-1)**2).mean()
        return gp


    # -----------------------------------------------------
    # Training
    # -----------------------------------------------------
    def train(
        self,
        regions,
        iters=400000,
        log_interval=1000,
        save_interval=10000,
        outdir="./output"
    ):

        os.makedirs(outdir,exist_ok=True)

        # ids, seqs = self.read_fasta(fasta_file)
        # X = np.array([self.one_hot(s) for s in seqs])
        # X = torch.tensor(X,dtype=torch.float32).to(self.device)
        X = torch.from_numpy(regions).to(dtype=torch.float32, device=self.device)

        pbar = trange(iters, desc="Training WGAN-GP")

        # for epoch in range(iters):
        for epoch in pbar:

            # ---- Train critic ----
            for _ in range(self.critic_iters):

                idx = np.random.randint(0,X.shape[0],self.batch_size)
                real = X[idx]

                z = torch.randn(self.batch_size,self.latent_dim).to(self.device)
                fake = self.G(z).detach()

                d_real = self.D(real).mean()
                d_fake = self.D(fake).mean()

                gp = self.gradient_penalty(real,fake)

                d_loss = d_fake - d_real + self.gp_lambda*gp

                self.opt_D.zero_grad()
                d_loss.backward()
                self.opt_D.step()


            # ---- Train generator ----
            z = torch.randn(self.batch_size,self.latent_dim).to(self.device)
            fake = self.G(z)
            g_loss = -self.D(fake).mean()

            self.opt_G.zero_grad()
            g_loss.backward()
            self.opt_G.step()


            # ---- logging ----
            if epoch%log_interval==0:
                print(f"[{epoch}]  D={d_loss.item():.4f}  G={g_loss.item():.4f}")


            # ---- save ----
            # if epoch%save_interval==0:
            #     torch.save(self.G.state_dict(),
            #                f"{outdir}/gen_{epoch}.pt")
            #     torch.save(self.D.state_dict(),
            #                f"{outdir}/disc_{epoch}.pt")

            if epoch % save_interval == 0:

                d_val = d_loss.item()
                g_val = g_loss.item()

                gen_path  = f"{outdir}/gen_e{epoch}_D{d_val:.4f}_G{g_val:.4f}.pt"
                disc_path = f"{outdir}/disc_e{epoch}_D{d_val:.4f}_G{g_val:.4f}.pt"

                torch.save(self.G.state_dict(), gen_path)
                torch.save(self.D.state_dict(), disc_path)

                print(f"Saved: {gen_path}")
                print(f"Saved: {disc_path}")


    # -----------------------------------------------------
    # Load model
    # -----------------------------------------------------
    def load(self, gen_ckpt, disc_ckpt=None):

        self.G.load_state_dict(torch.load(gen_ckpt,map_location=self.device))
        if disc_ckpt:
            self.D.load_state_dict(torch.load(disc_ckpt,map_location=self.device))


    # -----------------------------------------------------
    # Generate sequences
    # -----------------------------------------------------
    def generate(self, n=128):

        char = ['A','C','G','T']
        z = torch.randn(n,self.latent_dim).to(self.device)
        x = self.G(z).cpu().detach().numpy()

        idx = x.argmax(axis=2)

        seqs=[]
        for s in idx:
            seqs.append("".join([char[i] for i in s]))
        return seqs
