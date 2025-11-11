import os
import argparse
import torch
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from datas.dataset import StableDataset, collate_fn
from text import symbols
from config import MelConfig, ModelConfig, TrainConfig
from models.model import StableTTS
from dataclasses import asdict
from utils.load import continue_training
from utils.scheduler import get_cosine_schedule_with_warmup


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--max-steps', type=int, default=None, help='Optional: stop after this many optimizer steps')
    p.add_argument('--save-dir', default=None, help='Optional: override model_save_path from config')
    p.add_argument('--resume-from', type=str, default=None, help='Path to model checkpoint .pt to resume training')
    p.add_argument('--start-epoch', type=int, default=None, help='Epoch to start training from manually')
    return p.parse_args()


def main():
    args = parse_args()

    device = torch.device(args.device)
    mel_config = MelConfig()
    model_config = ModelConfig()
    train_config = TrainConfig()

    if args.save_dir:
        train_config.model_save_path = args.save_dir

    os.makedirs(train_config.model_save_path, exist_ok=True)

    # dataset / loader
    dataset = StableDataset(train_config.train_dataset_path, mel_config.hop_length)
    loader = DataLoader(dataset, batch_size=train_config.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4, pin_memory=True)

    model = StableTTS(len(symbols), mel_config.n_mels, **asdict(model_config)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.learning_rate)

    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(train_config.warmup_steps), num_training_steps=train_config.num_epochs * len(loader))

    # resume if any
    current_epoch = continue_training(train_config.model_save_path, model, optimizer)

    writer = SummaryWriter(train_config.log_dir)

    step = 0
    try:
        model.train()
        for epoch in range(current_epoch, train_config.num_epochs):
            for batch_idx, datas in enumerate(loader):
                datas = [d.to(device, non_blocking=True) for d in datas]
                x, x_lengths, y, y_lengths, z, z_lengths = datas

                optimizer.zero_grad()
                dur_loss, diff_loss, prior_loss, _ = model(x, x_lengths, y, y_lengths, z, z_lengths)
                loss = dur_loss + diff_loss + prior_loss
                loss.backward()
                optimizer.step()
                scheduler.step()

                if step % train_config.log_interval == 0:
                    writer.add_scalar('training/diff_loss', diff_loss.item(), step)
                    writer.add_scalar('training/dur_loss', dur_loss.item(), step)
                    writer.add_scalar('training/prior_loss', prior_loss.item(), step)

                if step % (train_config.save_interval * len(loader)) == 0:
                    # save checkpoint
                    torch.save(model.state_dict(), os.path.join(train_config.model_save_path, f'checkpoint_step_{step}.pt'))
                    torch.save(optimizer.state_dict(), os.path.join(train_config.model_save_path, f'optimizer_step_{step}.pt'))

                step += 1
                if args.max_steps and step >= args.max_steps:
                    print('Reached max-steps, exiting')
                    return

            print(f'Epoch {epoch} finished, step={step}, last_loss={loss.item()}')

    except KeyboardInterrupt:
        print('Interrupted — saving model...')
        torch.save(model.state_dict(), os.path.join(train_config.model_save_path, f'checkpoint_interrupt.pt'))


if __name__ == '__main__':
    main()
