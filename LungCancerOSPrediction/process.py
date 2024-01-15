from pathlib import Path
import json
import random

from evalutils import ClassificationAlgorithm
from evalutils.validators import (
    UniquePathIndicesValidator,
    UniqueImagesValidator,
)


class Lungcancerosprediction(ClassificationAlgorithm):
    def __init__(self):
        super().__init__(
            validators=dict(
                input_image=(
                    UniqueImagesValidator(),
                    UniquePathIndicesValidator(),
                )
            ),
        )

        # path to image file
        self.image_input_dir = "/input/images/chest-ct/"
        self.image_input_path = list(Path(self.image_input_dir).glob("*.mha"))[0]

        # load clinical information
        # dictionary with patient_age and psa information
        # with open("/input/clinical-information-lung-ct.json") as fp:
        #     self.clinical_info = json.load(fp)

        # path to output files
        self.os_output_file = Path("/output/overall-survival-months.json")

    def predict(self):
        """
        Your algorithm goes here
        """        
        
        # read image
        # clinical_info = self.clinical_info
        # print('Clinical info: ')
        # print(clinical_info)

        overall_survival = float(27.8)  # mean of train data
        print('OS (months): ', overall_survival)

        # save case-level class
        with open(str(self.os_output_file), 'w') as f:
            json.dump(overall_survival, f)


if __name__ == "__main__":
    Lungcancerosprediction().predict()
