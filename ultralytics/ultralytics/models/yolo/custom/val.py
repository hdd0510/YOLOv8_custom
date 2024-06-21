from ultralytics.engine.validator import BaseValidator
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.models.yolo.classify import ClassificationValidator
from ultralytics.data import build_dataloader
from ultralytics.data.dataset import CustomClsDataset, YOLOConcatDataset
from ultralytics.utils.torch_utils import torch_distributed_zero_first
class CustomValidator(BaseValidator):
    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)
        self.det_validator = DetectionValidator(dataloader, save_dir, pbar, args, _callbacks)
        self.vtgp_validator = ClassificationValidator(dataloader, save_dir, pbar, args, _callbacks)
        self.args.task = 'custom'
    def build_dataset(self, img_path, mode="val", batch=None):
        det_dataset = DetectionValidator.build_dataset(self, img_path[f"{mode}_det"], mode, batch)
        vtgp_dataset = CustomClsDataset(self.args, data=img_path[f"{mode}_vtgp"], data_name="cls_vtgp", prefix=mode)
        return YOLOConcatDataset([det_dataset, vtgp_dataset])
    
    def get_dataloader(self, dataset_path, batch_size=8):
        dataset = self.build_dataset(dataset_path, batch=batch_size)
        dataloader = build_dataloader(dataset, batch_size, workers=4)
        return dataloader
        # if mode == "train":
        #     dataloader = build_dataloader(dataset, batch_size, self.args.workers, mode == "train", rank)
        # else:
        #     dataloader = {"val_det": build_dataloader(dataset["val_det"], batch_size, self.args.workers, mode == "train", rank),
        #                   "val_vtgp": build_dataloader(dataset["val_vtgp"], batch_size, self.args.workers, mode == "train", rank)}
        # return dataloader
    # def __call__(self, trainer=None, model=None):
    #     if self.args.task == "custom":
    #         self.det_validator(model=model)
    #         self.vtgp_validator(model=model)
        # else:
        #     super().__call__(model)