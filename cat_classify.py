
# %%
from fastai.vision import *
from fastai import *
# get_ipython().run_line_magic('reload_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')
# get_ipython().run_line_magic('matplotlib', 'inline')


# %%


# %%
path = untar_data(URLs.PETS)
path


# %%
path_anno = path/'annotations'
path_img = path/'images'


# %%
fnames = get_image_files(path_img)
fnames[:5]


# %%
np.random.seed(2)
pat = r'/([^/]+)_\d+.jpg$'


# %%
data = ImageDataBunch.from_name_re(
    path_img, fnames, pat, ds_tfms=get_transforms(), size=224)
data.normalize(imagenet_stats)


# %%
data.show_batch(rows=3, figsize=(7, 6))


# %%
data.classes


# %%
len(data.classes), data.c


# %%
learn = cnn_learner(data, models.resnet34, metrics=error_rate)


# %%
learn.fit_one_cycle(4)


# %%
learn.save('stage-1')

# %%
interp = ClassificationInterpretation.from_learner(learn)

# %%
interp.plot_top_losses(9, figsize=(15, 11))

# %%
