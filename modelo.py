
# coding: utf-8

# In[1]:


from fastai.vision import *


# In[ ]:


## This first lines brings all the tools from fast ai


# In[2]:


folder = 'lebron'
file = 'urls_lebron.csv'


# In[ ]:


## I think this creates a folder and file


# In[3]:


path = Path('data/Goats')
dest = path/folder
dest.mkdir(parents=True, exist_ok=True)


# In[ ]:


## I think this puts the folder and file into the right path


# In[4]:


download_images(path/file, dest, max_pics=200)


# In[ ]:


##And this download the images frm the csv file 


# In[5]:


folder = 'jordan'
file = 'urls_jordan.csv'


# In[6]:


path = Path('data/Goats')
dest = path/folder
dest.mkdir(parents=True, exist_ok=True)


# In[7]:


download_images(path/file, dest, max_pics=200)


# In[ ]:


## This just repits the process for the other folder and file


# In[8]:


classes = ['lebron','jordan']


# In[ ]:


## This creates the two clasess


# In[9]:


for c in classes:
    print(c)
    verify_images(path/c, delete=True, max_size=500)


# In[ ]:


## I am not really sure what this does, ithink it is simply to verify that the images are there


# In[10]:


np.random.seed(42)
data = ImageDataBunch.from_folder(path, train=".", valid_pct=0.2,
        ds_tfms=get_transforms(), size=224, num_workers=4).normalize(imagenet_stats)


# In[ ]:


## This separates the data set into train and validation set, limits the size and normalizes the data set


# In[11]:


data.classes


# In[ ]:


## This just prints the data classes ther are


# In[12]:


data.show_batch(rows=3, figsize=(7,8))


# In[ ]:


## This should should some examples of the data points


# In[13]:


data.classes, data.c, len(data.train_ds), len(data.valid_ds)


# In[ ]:


## This tells you what the data classes are the lenght of the train set and thev valitdaion ste


# In[14]:


learn = cnn_learner(data, models.resnet34, metrics=error_rate)


# In[ ]:


## This chooses the model


# In[15]:


learn.fit_one_cycle(4)


# In[ ]:


## This train the model 4 times,  the fit one cycle uses the last layers


# In[16]:


learn.save('stage-1')


# In[ ]:


#This saves the model


# In[17]:


learn.unfreeze()


# In[ ]:


## this lets the model run


# In[18]:


learn.lr_find(8)


# In[ ]:


## I am not sure what this does 


# In[19]:


learn.recorder.plot()


# In[ ]:


## This plots the learning rate and loss at each rate


# In[20]:


learn.fit_one_cycle(2, max_lr=slice(3e-5,3e-4))


# In[ ]:


#This trains the model again but with different learning rates


# In[21]:


learn.save('stage-2')


# In[ ]:


## This saves anther stage of the model


# In[22]:


learn.load('stage-2');


# In[ ]:


##This load the the second version of the model


# In[23]:


interp = ClassificationInterpretation.from_learner(learn)


# In[ ]:


## This creates a variable to intepret from the dat


# In[24]:


interp.plot_confusion_matrix()


# In[ ]:


##This shows what mistakes the model is making


# In[25]:


from fastai.widgets import *


# In[ ]:


## Import some widgets to clean up data


# In[26]:


db = (ImageList.from_folder(path)
                   .split_none()
                   .label_from_folder()
                   .transform(get_transforms(), size=224)
                   .databunch()
     )


# In[ ]:


##This reverts the split of the dat we made before to clean up the data


# In[27]:


learn_cln = cnn_learner(db, models.resnet34, metrics=error_rate)

learn_cln.load('stage-2');


# In[ ]:


## This creates a new learner


# In[28]:


ds, idxs = DatasetFormatter().from_toplosses(learn_cln)


# In[ ]:


## I think this is to create a variable to clean data


# In[29]:


ImageCleaner(ds, idxs, path)


# In[ ]:


#This should show images which you can delete to clean data 


# In[30]:


ds, idxs = DatasetFormatter().from_similars(learn_cln)


# In[ ]:


## This is to find duplicates 


# In[31]:


ImageCleaner(ds, idxs, path, duplicates=True)


# In[ ]:


#This is to delete duplicates


# In[32]:


learn.export()


# In[ ]:


##This is to export the model


# In[33]:


defaults.device = torch.device('cpu')


# In[ ]:


##This should export the model  into a file that you can run in acpu


# In[35]:


img = open_image(path/'jordan'/'00000022.jpg')
img


# In[ ]:


## This opens up an image


# In[36]:


learn = load_learner(path)


# In[ ]:


##This loads up the model for cpu


# In[37]:


pred_class,pred_idx,outputs = learn.predict(img)
pred_class


# In[ ]:


## This is the line to predict if the image is or jordan or lebron


# In[38]:


learn.path, learn.model_dir

