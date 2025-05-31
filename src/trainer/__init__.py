from .stablemtl_trainer import StableMTLTrainer


trainer_cls_name_dict = {
    "StableMTLTrainer": StableMTLTrainer,
}


def get_trainer_cls(trainer_name):
    return trainer_cls_name_dict[trainer_name]
