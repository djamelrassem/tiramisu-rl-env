from core.services.converting_service import ConvertService
from utils.config.config import Config
from ..models.prediction_model import Model_Recursive_LSTM_v2
import torch

MAX_DEPTH = 6

class PredictionService():

    def __init__(self):
        self.model = Model_Recursive_LSTM_v2()
        self.model.load_state_dict(
            torch.load(Config.config.tiramisu.model_checkpoint, map_location="cpu"))
        # Set the model in evaluation mode
        self.model.eval()

    def get_speedup(self,schedule_object):
        computations_tensor, loops_tensor = ConvertService.get_schedule_representation(
            schedule_object.prog.annotations,
            schedule_object.schedule_dict,
            schedule_object.templates["comps_repr_templates_list"],
            schedule_object.templates["loops_repr_templates_list"],
            schedule_object.templates["comps_placeholders_indices_dict"],
            schedule_object.templates["loops_placeholders_indices_dict"],
            max_depth=MAX_DEPTH - 1)
        tree_tensors = (schedule_object.templates["prog_tree"],
                        computations_tensor, loops_tensor)
        with torch.no_grad():
            predicted_speedup = self.model(
                tree_tensors,
                num_matrices=MAX_DEPTH - 1).item()
            return predicted_speedup


        