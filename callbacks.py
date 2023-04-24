import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import os
import matplotlib.pyplot as plt
import io
import PIL.Image
from torchvision.transforms import ToTensor
from spec2rgb import ColourSystem


def plot_images(ground_truth, prediction, cs):
    ground_truth = (ground_truth - ground_truth.min())/(ground_truth.max() - ground_truth.min())
    prediction = (prediction - prediction.min())/(prediction.max() - prediction.min())

    
    mse = np.mean((ground_truth - prediction)**2)
    # Calculate PSNR
    psnr = 10 * np.log10(1 / mse)


    
    prediction = cs.spec_to_rgb(prediction)
    ground_truth = cs.spec_to_rgb(ground_truth)
    
    figure, axs = plt.subplots(1, 2, figsize=(10, 10))
    axs[0].imshow(ground_truth);axs[0].set_title('Ground truth');axs[0].set_yticks([]);axs[0].set_xticks([])
    axs[1].imshow(prediction);axs[1].set_title('Prediction');axs[1].set_xlabel(f"MSE={mse:.2f},PSNR={psnr:.2f}");axs[1].set_yticks([]);axs[1].set_xticks([])     
    return figure

def plot_channelwise(ground_truth, prediction, n_channels=3):

    ground_truth = ground_truth[...,  np.linspace(0, ground_truth.shape[2], n_channels, dtype=int, endpoint=False).tolist()]
    prediction = prediction[...,  np.linspace(0, prediction.shape[2], n_channels, dtype=int, endpoint=False).tolist()]
    
    ground_truth = (ground_truth - ground_truth.min())/(ground_truth.max() - ground_truth.min())
    prediction = (prediction - prediction.min())/(prediction.max() - prediction.min())
    
    figure, axs = plt.subplots(2, ground_truth.shape[-1], figsize=(12, 8))

    for i in range(ground_truth.shape[-1]):
        msech = np.mean((ground_truth[:,:,i] - prediction[:,:,i])**2)
        psnrch = 10 * np.log10(1 / msech)
        axs[0,i].imshow(ground_truth[:,:,i]);axs[0, 0].set_ylabel("Ground Truth");axs[0,i].set_title('{} channel'.format(i));axs[0,i].set_yticks([]);axs[0,i].set_xticks([]);

        axs[1,i].imshow(prediction[:,:,i]);axs[1,0].set_ylabel('Prediction');axs[1,i].set_xlabel(f"MSE={msech:.2f},PSNR={psnrch:.2f}");axs[1,i].set_yticks([]);axs[1,i].set_xticks([])     
    return figure
    

def plot_to_image(figure):
    """
    Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call.
    """
    
    buf = io.BytesIO()
    
    # Use plt.savefig to save the plot to a PNG in memory.
    figure.savefig(buf, format='png', bbox_inches='tight', transparent = False, dpi = 300)
    
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    
    image = PIL.Image.open(buf)
    image = np.transpose(np.array(image), [2, 0, 1])
    
    return image


def save_checkpoint(epoch, model, optimizer, loss, isbest, path, filename = 'checkpoint.pth'):
    '''
    
    Save the model and optimizer state at checkpoint to be used in case of failure
    :param exp_name: name of the experiment
    :param epoch: current epoch
    :param model: model to be saved
    :param optimizer: optimizer to be saved
    :param loss: loss to be saved
    :param isbest: boolean to indicate if the current model is the best model
    :return: None
    '''

    state = {'epoch': epoch,
                'model': model,
                'optimizer': optimizer,
                'loss': loss}


    
    torch.save(state, os.path.join(path, 'LAST_' + filename))
    # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
    if isbest:
        torch.save(state, os.path.join(path, 'BEST_' + filename))





class LogTrainingCallback():

    def __init__(self, full_dataset, exp_folder,shape_data,eval_loss, log_scalars_freq = 1, log_images_freq = 1, write_images = True, write_scalars = True, device = None,start=0,end=0,number=0):

        #create exp_folder if path not exists
        if not os.path.exists(exp_folder):
            os.makedirs(exp_folder)
        self.writer = SummaryWriter(os.path.join(exp_folder, "tensorboard"))
        self.write_scalars = write_scalars
        self.write_images = write_images
        self.log_scalars_freq = log_scalars_freq
        self.log_images_freq = log_images_freq
        self.full_dataset = full_dataset
        self.device = device
        self.best_loss = np.inf
        self.shape_data = shape_data
        self.write_histogram = True
        self.log_histogram_freq = 1
        self.eval_loss=eval_loss
        self.exp_folder=exp_folder
        self.start=start
        self.end=end
        self.number_bands=number
        self.color_space="sRGB"
        
    def step(self, model, loss, epoch):

        if self.write_histogram and epoch % self.log_histogram_freq == 0 and epoch >= 1:

            
            names, outputs = model.get_histogram_of_layers()

            for idx, output in enumerate(outputs):
                self.writer.add_histogram(f"Layer {idx}, {names[idx]}", output.clone().cpu().data.numpy(), epoch, bins='doane')

            for name, weight in model.named_parameters():
                self.writer.add_histogram(name, weight, epoch)
                #self.writer.add_histogram(f'{name}.grad', weight.grad, epoch)

        eval_loss_result = {}
        if self.write_images and epoch % self.log_images_freq == 0 and epoch >= 0:

            prediction = []#np.zeros((len(self.full_dataset.dataset), 1))
            ground_truth = []#np.zeros((len(self.full_dataset.dataset), 1))
            for i, vdata in enumerate(self.full_dataset):
                print("Evaluated {}/{}".format(i+1, len(self.full_dataset)))
                inputs, outputs_gt = vdata
                
                inputs  = inputs.to(self.device)
                outputs_gt = outputs_gt.to(self.device)

                with torch.no_grad():
                    outputs_pred = model(inputs)
                    # prediction[i*outputs_gt.shape[0]:(i+1)*outputs_gt.shape[0],...] = outputs_pred.cpu().detach().numpy()
                    # ground_truth[i*outputs_gt.shape[0]:(i+1)*outputs_gt.shape[0],...] = outputs_gt.cpu().detach().numpy()
                    prediction.append(outputs_pred.cpu().detach().numpy())
                    ground_truth.append(outputs_gt.cpu().detach().numpy())
                
            prediction = np.concatenate(prediction, 0)
            ground_truth = np.concatenate(ground_truth, 0)
            prediction = prediction.reshape(self.shape_data);
            prediction = (prediction - np.min(prediction))/(np.max(prediction) - np.min(prediction))

            ground_truth = ground_truth.reshape(self.shape_data)
            ground_truth = (ground_truth - np.min(ground_truth))/(np.max(ground_truth) - np.min(ground_truth))
            
            


            for key in self.eval_loss.keys():      
                eval_loss_result[key] = self.eval_loss[key](prediction, ground_truth)





            cs = ColourSystem(cs=self.color_space, start=self.start, end=self.end, num=self.number_bands)
            figure = plot_images(ground_truth, prediction, cs)

            image = plot_to_image(figure)

            self.writer.add_image('predictions', image, epoch)

            figure = plot_channelwise(ground_truth, prediction)

            image = plot_to_image(figure)

            self.writer.add_image('channelwise', image, epoch)

        best=False
        if self.write_scalars and epoch % self.log_scalars_freq == 0 and epoch >= 1:
            # Iterate over keys in the loss dictionary
            for key in loss.keys():
                self.writer.add_scalar("Train " + key, loss[key], epoch)

            for key in eval_loss_result.keys():
                self.writer.add_scalar(f"Test {key}", eval_loss_result[key], epoch)

            monitor = list(eval_loss_result.keys())[0]
            if eval_loss_result[monitor] < self.best_loss:
                self.best_loss = eval_loss_result[monitor]
                best = True
        save_checkpoint(epoch, model, None, eval_loss_result, best, path = self.exp_folder, filename =  'checkpoint.pth')
if __name__ == "__main__":
    ### Load images and plot them
   #y=plot_images(np.random.rand(2,2),np.random.rand(2,2),3)
   #print("ASALSKDJLKASJD")
   #plt.show()
   pass