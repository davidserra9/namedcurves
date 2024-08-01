import logging

class Trainer():
    def __init__(self, model, optimizer, criterion, scheduler, train_loader, valid_evaluator, test_evaluator, config_train, config_eval):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.valid_evaluator = valid_evaluator
        self.test_evaluator = test_evaluator
        self.config_train = config_train
        self.config_eval = config_eval

    def _train_step(self, input_image, target_image):
        self.optimizer.zero_grad()

        prediction, x_backbone = self.model(input_image.cuda(), return_backbone=True)
        loss = self.criterion(x_backbone, prediction, target_image.cuda())

        loss.backward()
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        return loss.item()

    def _train_epoch(self):
        epoch_loss = 0
        self.model.train()
        for data in self.train_loader:
            input_image, target_image, name = data['input_image'], data['target_image'], data['name']
            loss = self._train_step(input_image, target_image)
            epoch_loss += loss

        return epoch_loss / len(self.train_loader)

    def train(self):
        for epoch in range(self.config_train.epochs):
            epoch_loss = self._train_epoch()
            logging.info(f"Epoch {epoch+1}/{self.config_train.epochs} | Loss: {epoch_loss}")

            if self.valid_evaluator is not None and (epoch+1) % self.config_train.valid_every == 0:
                self.valid_evaluator(self.model)

        self.test_evaluator(self.model, save_results=True if self.valid_evaluator is None else False)
        logging.info("Training finished.")