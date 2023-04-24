import torch
import torch.nn as nn

class TrainingLoop():

    def __init__(self, model, loss_fn, optimizer, train_loader, model_regularizations, losses_weights, regularization_weights, full_dataset, schedulers = [], callbacks = [], device=None,forward = None):
        self.forward = forward
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.device = device
        self.schedulers = schedulers
        self.callbacks = callbacks
        self.model_regularizations = model_regularizations
        self.losses_weights = losses_weights
        self.regularization_weights = regularization_weights
        self.full_dataset = full_dataset

    def train_one_epoch(self, freq=1, steps_per_epoch = None):

        running_fidelity = 0.
        last_fidelity = 0.

        for idx_sensor, traindata in enumerate(self.train_loader):
            for i, data in enumerate(traindata):
                # Every data instance is an input + outputs pair
                
                coords, imagesensor = data
                coords = coords.to(self.device)
                
                imagesensor = imagesensor.to(self.device)
            
                coords  = coords.to(self.device)
                # Zero your gradients for every batch!
                self.optimizer.zero_grad(set_to_none=True)

                # Make inference 
                x_pred = self.model(coords)
                y_pred = self.forward(intensities=x_pred,coords=coords, sensor = idx_sensor)


                final_loss = 0.0
                loss_values = loss_values = { key: 0.0 for key in self.loss_fn.keys()}
                for idx, key in enumerate(self.loss_fn.keys()):
                    
                    loss_values[key] += self.loss_fn[key](y_pred, imagesensor) * self.losses_weights[idx]
                    final_loss += loss_values[key]



                final_loss.backward()

                # Adjust learning weights
                self.optimizer.step()

                # Keep track of loss
                running_fidelity += torch.sum(torch.stack(list(loss_values.values()), 0)).item()


                if i % freq == 0:  # print every freq mini-batches

                    last_fidelity = running_fidelity / freq
                    print(f'  batch {i+1}/{len(traindata)} fidelity loss: {last_fidelity:.5E}')
                    # show previous line in scientific notation
                    running_fidelity = 0.
                if steps_per_epoch != None and i >= steps_per_epoch:
                    return loss_values
        return loss_values

    def reg_one_epoch(self, freq=1, steps_per_epoch = None):
        running_reg = 0.
        last_reg = 0.

        for i, data in enumerate(self.full_dataset):
            # Every data instance is an input + outputs pair

            coords, _ = data

            coords  = coords.to(self.device)

            # Zero your gradients for every batch!
            self.optimizer.zero_grad(set_to_none=True)

            final_loss = 0.0    
            # Iterate over the dictionary of regularizations
            regularization_values = {}#regularization_values = { key: 0.0 for key in self.model_regularizations.keys()}
            for idx, key in enumerate(self.model_regularizations.keys()):
                regularization_values[key] = self.model_regularizations[key](self.model, coords) * self.regularization_weights[idx]
                final_loss += regularization_values[key]



            final_loss.backward()

            # Adjust learning weights
            self.optimizer.step()


            running_reg += torch.sum(torch.stack(list(regularization_values.values()), 0)).item()
            if i % freq == 0:  # print every freq mini-batches
                last_reg = running_reg / freq
                print(f'   {i + 1}/{len(self.full_dataset)} reg loss: {last_reg:.5E}')
                # show previous line in scientific notation
                running_reg = 0.
            if steps_per_epoch != None and i >= steps_per_epoch:
                return regularization_values
        return regularization_values


    def fit(self, n_epochs, freq=1, steps_per_epoch = None):
        
        for epoch in range(n_epochs):

            print('Epoch {}/{}'.format(epoch + 1, n_epochs))

            self.model.train(True)
            if self.loss_fn:
                resuts_fidelities  = self.train_one_epoch(freq = freq, steps_per_epoch = steps_per_epoch)
            else:
                resuts_fidelities = {}
            if self.model_regularizations:
                results_reg = self.reg_one_epoch(freq = freq, steps_per_epoch = steps_per_epoch)
            else:
                results_reg = {}
            resuts_losses = {**resuts_fidelities, **results_reg}
            self.model.train(False)

            for s in self.schedulers:
                s.step()

            for c in self.callbacks:
                c.step(model = self.model, loss= resuts_losses,epoch =  epoch)
        return self.model

    
