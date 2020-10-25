#%%
import matplotlib as plt
import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
# %%
tfds.list_builders()
#%%
builder=tfds.builder('rock_paper_scissors')
builder.info
# %%
train=tfds.load(name='rock_paper_scissors', split='train')
test=tfds.load(name='rock_paper_scissors', split='test')
# %%
