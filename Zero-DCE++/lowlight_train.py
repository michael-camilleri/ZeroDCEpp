import torch
import torch.utils.data as tud
import torch.optim
import os
import argparse
import dataloader
import model
import Myloss
import numpy as np
from tqdm import tqdm


def train(cfg):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    scale_factor = cfg.scale_factor
    DCE_net = model.enhance_net_nopool(scale_factor).cuda()

    # === Setup Model === #
    # DCE_net.apply(weights_init)
    if cfg.pretrain_model is not None:
        DCE_net.load_state_dict(torch.load(cfg.pretrain_model))
    dataset = dataloader.StructuredLoader(cfg.images, cfg.scale_factor)
    v_len = int(cfg.validation_ratio * len(dataset))
    t_len = len(dataset) - v_len
    print(f'Training on {t_len} and evaluating on {v_len}.')
    train_ds, valid_ds = tud.random_split(
        dataset, [t_len, v_len], generator=torch.Generator().manual_seed(cfg.random_seed)
    )
    train_loader = tud.DataLoader(
        train_ds, cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True
    )
    valid_loader = tud.DataLoader(
        valid_ds, cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True
    )

    # === Setup Loss Functions ===
    #  Note that I have removed the color loss: L_color = Myloss.L_color()
    L_spa = Myloss.L_spa()
    L_exp = Myloss.L_exp(16)
    L_TV = Myloss.L_TV()
    optimizer = torch.optim.Adam(DCE_net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    best_model = np.PINF

    # Iterate over epochs
    for epoch in range(1, cfg.num_epochs+1):
        # First Train
        DCE_net.train()
        epoch_loss = 0
        for b_id, batch in enumerate(tqdm(train_loader, miniters=cfg.display_iter)):
            b_id += 1; batch = batch.cuda(); E = 0.6
            enhanced_image, A = DCE_net(batch)

            # Compute Loss
            Loss_TV = 1600 * L_TV(A)
            loss_spa = torch.mean(L_spa(enhanced_image, batch))
            loss_exp = 10 * torch.mean(L_exp(enhanced_image, E))
            # loss_col = 5 * torch.mean(L_color(enhanced_image))
            loss = Loss_TV + loss_spa + loss_exp # + loss_col
            epoch_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(DCE_net.parameters(), cfg.grad_clip_norm)
            optimizer.step()

        # Record Batch Loss
        print(f'\rBatch Loss: Training: [{epoch}]: {epoch_loss}')

        # Now Evaluate
        DCE_net.eval()
        with torch.no_grad():
            epoch_loss = 0
            for batch in tqdm(valid_loader, miniters=cfg.display_iter):
                batch = batch.cuda(); E = 0.6
                enhanced_image, A = DCE_net(batch)
                Loss_TV = 1600 * L_TV(A)
                loss_spa = torch.mean(L_spa(enhanced_image, batch))
                loss_exp = 10 * torch.mean(L_exp(enhanced_image, E))
                # loss_col = 5 * torch.mean(L_color(enhanced_image))
                epoch_loss += (Loss_TV + loss_spa + loss_exp).item() # + loss_col

            print(f'\rBatch Loss: Validation: [{epoch}]: {epoch_loss}')

            if epoch_loss < best_model:
                print(f'Found new best Validation Loss: Storing Model.')
                best_model = epoch_loss
                torch.save(
                    DCE_net.state_dict(), os.path.join(cfg.snapshots_folder, "model.best.pth")
                )
                with open(os.path.join(cfg.snapshots_folder, 'model.best.txt'), 'w') as f:
                    f.write(f'Epoch: {epoch}')


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Input Parameters
    parser.add_argument(
        '--images',
        type=str,
        default="/media/veracrypt4/Q1/Snippets/Curated/Behaviour/Train/Frames_Raw"
    )
    parser.add_argument('--validation_ratio', type=float, default=0.3)
    parser.add_argument('--lr', type=float, default=0.00001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--grad_clip_norm', type=float, default=0.1)
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--display_iter', type=int, default=50)
    parser.add_argument('--scale_factor', type=int, default=12)
    parser.add_argument('--random_seed', type=int, default=101)
    parser.add_argument(
        '--snapshots_folder',
        type=str,
        default="/home/s1238640/Documents/DataSynced/PhD Project/Data/MRC Harwell/Scratch/ZeroDCE++"
    )
    parser.add_argument(
        '--pretrain_model',
        type=str,
        default="/media/veracrypt5/MRC_Data/Models/ZeroDCE/Base/ZeroDCE++.pth"
    )

    trn_config = parser.parse_args()

    if not os.path.exists(trn_config.snapshots_folder):
        os.mkdir(trn_config.snapshots_folder)

    train(trn_config)
