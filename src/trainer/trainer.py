from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer
import torch
from tqdm.auto import tqdm
import torch.nn.functional as F


class Trainer(BaseTrainer):
    """
    Trainer class. Defines the logic of batch logging and processing.
    """

    def process_batch(self, batch, metrics: MetricTracker):
        """
        Run batch through the model, compute metrics, compute loss,
        and do training step (during training stage).

        The function expects that criterion aggregates all losses
        (if there are many) into a single one defined in the 'loss' key.

        Args:
            batch (dict): dict-based batch containing the data from
                the dataloader.
            metrics (MetricTracker): MetricTracker object that computes
                and aggregates the metrics. The metrics depend on the type of
                the partition (train or inference).
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform),
                model outputs, and losses.
        """
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)  # transform batch on device -- faster

        metric_funcs = self.metrics["inference"]
        if self.is_train:
            metric_funcs = self.metrics["train"]
            self.optimizer.zero_grad()

        embeddings = self.model(batch["data_object"], aug=True)
        batch.update({"embeddings": embeddings})
        results = self.criterion(embeddings, batch["labels"])
        batch.update(results)

        if self.is_train:
            batch["loss"].backward()  # sum of all losses is always called loss
            self._clip_grad_norm()
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        # update metrics for each loss (in case of multiple losses)
        for loss_name in self.config.writer.loss_names:
            metrics.update(loss_name, batch[loss_name].item())

        for met in metric_funcs:
            metrics.update(met.name, met(**batch))
        return batch
    
    def inference_batch(self, batch_idx, batch, metrics, part):
        """
        Run batch through the model, compute metrics, and
        save predictions to disk.

        Save directory is defined by save_path in the inference
        config and current partition.

        Args:
            batch_idx (int): the index of the current batch.
            batch (dict): dict-based batch containing the data from
                the dataloader.
            metrics (MetricTracker): MetricTracker object that computes
                and aggregates the metrics. The metrics depend on the type
                of the partition (train or inference).
            part (str): name of the partition. Used to define proper saving
                directory.
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform)
                and model outputs.
        """
        batch = self.move_batch_to_device(batch, part)
        batch = self.transform_batch(batch)  # transform batch on device -- faster

        outputs_1 = self.model(batch["data_object_1"])
        embedding_1 = F.normalize(outputs_1, p=2, dim=1)

        outputs_2 = self.model(batch["data_object_2"])
        embedding_2 = F.normalize(outputs_2, p=2, dim=1)

        outputs = {"embedding1" : embedding_1, "embedding2" : embedding_2}

        batch.update(outputs)
        return batch
    
    def count_scores(self, pairs, embeddings):
        labels = []
        scores = []
        for elem in pairs:
            label, idx1, idx2 = elem.tolist()
            labels.append(label)
            embedding_11, embedding_12 = embeddings[idx1][0], embeddings[idx1][1]
            embedding_21, embedding_22 = embeddings[idx2][0], embeddings[idx2][1]
            score_1 = torch.mean(torch.matmul(embedding_11, embedding_21.T))
            score_2 = torch.mean(torch.matmul(embedding_12, embedding_22.T))
            score = (score_1 + score_2) / 2
            scores.append(score)

        output = {
            "scores": scores,
            "labels": labels,
        }
        # if self.save_path is not None:
        #     torch.save(output, self.save_path / part / f"output.pth")

        return labels, scores
    
    def _inference_part(self, epoch, part, dataloader):
        """
        Run inference on a given partition and save predictions

        Args:
            part (str): name of the partition.
            dataloader (DataLoader): dataloader for the given partition.
        Returns:
            logs (dict): metrics, calculated on the partition.
        """
        embeddings = {}
        pairs = []
        self.is_train = False
        self.model.eval()
        self.evaluation_metrics.reset()
        with torch.no_grad():
            for batch_idx, batch in tqdm(
                enumerate(dataloader),
                desc=part,
                total=len(dataloader),
            ):
                batch = self.inference_batch(
                    batch_idx=batch_idx,
                    batch=batch,
                    part=part,
                    metrics=self.evaluation_metrics,
                )
                
                for i in range(batch["embedding1"].shape[0]):
                    embeddings[batch["index"][i].item()] = [batch["embedding1"][i], batch["embedding2"][i*5 : (i + 1)*5, :]]
                pairs = batch["test_pairs"][0]


        labels, scores = self.count_scores(pairs, embeddings)     
        results = {}
        if self.evaluation_metrics is not None:
            for met in self.metrics["inference"]:
                name = part + '_' + met.name
                results[name] = met(torch.tensor(scores), torch.tensor(labels))
                self.writer.add_scalar(
                    name, results[name]
                )
        return results
       

    def _evaluation_epoch(self, epoch, part, dataloader):
        """
        Evaluate model on the partition after training for an epoch.

        Args:
            epoch (int): current training epoch.
            part (str): partition to evaluate on
            dataloader (DataLoader): dataloader for the partition.
        Returns:
            logs (dict): logs that contain the information about evaluation.
        """
        if part != "train":
            return self._inference_part(epoch, part, dataloader)
        self.is_train = False
        self.model.eval()
        self.evaluation_metrics.reset()
        with torch.no_grad():
            for batch_idx, batch in tqdm(
                enumerate(dataloader),
                desc=part,
                total=len(dataloader),
            ):
                batch = self.process_batch(
                    batch,
                    metrics=self.evaluation_metrics,
                )
            self.writer.set_step(epoch * self.epoch_len, part)
            self._log_scalars(self.evaluation_metrics)
            self._log_batch(
                batch_idx, batch, part
            )  # log only the last batch during inference

        return self.evaluation_metrics.result()

    def _log_batch(self, batch_idx, batch, mode="train"):
        """
        Log data from batch. Calls self.writer.add_* to log data
        to the experiment tracker.

        Args:
            batch_idx (int): index of the current batch.
            batch (dict): dict-based batch after going through
                the 'process_batch' function.
            mode (str): train or inference. Defines which logging
                rules to apply.
        """
        # method to log data from you batch
        # such as audio, text or images, for example

        # logging scheme might be different for different partitions
        if mode == "train":  # the method is called only every self.log_step steps
            # Log Stuff
            pass
        else:
            # Log Stuff
            pass
