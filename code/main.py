import world
import utils
from world import cprint
import torch
import time
import Procedure

# ==============================
utils.set_seed(world.seed)
print(">>SEED:", world.seed)
# ==============================
import register
from register import dataset

Recmodel = register.MODELS[world.model_name](world.config, dataset)
Recmodel = Recmodel.to(world.device)
bpr = utils.BPRLoss(Recmodel, world.config)

weight_file = utils.getFileName()
print(f"load and save to {weight_file}")
if world.LOAD:
    try:
        Recmodel.load_state_dict(torch.load(weight_file, map_location=torch.device(world.device)))
        world.cprint(f"loaded model weights from {weight_file}")
    except FileNotFoundError:
        print(f"{weight_file} not exists, start from beginning")

Neg_k = 1

patience = 1000
best_loss = float('inf')
no_improvement_epochs = 0

try:
    for epoch in range(world.TRAIN_epochs):
        start_time = time.time()
        if world.model_name == 'ncl':
            Recmodel.e_step()
        if epoch % 1 == 0:
            cprint("[TEST]")
            Procedure.Test(dataset, Recmodel, epoch, None, world.config['multicore'])  # Remove `w`

        output_information, current_loss = Procedure.BPR_train_original(
            dataset, Recmodel, bpr, epoch, neg_k=Neg_k, w=None  # Remove `w`
        )

        end_time = time.time()
        epoch_duration = end_time - start_time

        print(f'EPOCH[{epoch + 1}/{world.TRAIN_epochs}] {output_information} [Time: {epoch_duration:.2f}s]')

        if current_loss < best_loss:
            best_loss = current_loss
            no_improvement_epochs = 0
            torch.save(Recmodel.state_dict(), weight_file)
        else:
            no_improvement_epochs += 1

        if no_improvement_epochs >= patience:
            print(f"Early stopping at epoch {epoch + 1}. Best loss: {best_loss:.6f}")
            break

finally:
    print("Training finished.")
