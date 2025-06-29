import world
import dataloader
import model
from pprint import pprint

if world.dataset in ['nyc', 'bkk', 'tko', 'gowalla', 'brightkite']:
    dataset = dataloader.Loader(path="../data/" + world.dataset)
elif world.dataset == 'lastfm':
    dataset = dataloader.LastFM()

print('===========config================')
pprint(world.config)
print("cores for test:", world.CORES)
print("comment:", world.comment)
print("tensorboard:", world.tensorboard)
print("LOAD:", world.LOAD)
print("Weight path:", world.PATH)
print("Test Topks:", world.topks)
print("using bpr loss")
print('===========end===================')

MODELS = {
    'sepgcn': model.SEPGCN,
}